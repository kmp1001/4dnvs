# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training arguments and the canonical DrivenVS manifest dataset."""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

from .camera_calibration import (
    resolve_sample_calibration,
    resolve_sample_time_offsets,
)
from .temporal import (
    TEMPORAL_HISTORY_FRAMES,
    temporal_frame_index,
    temporal_sequence_key,
)


def parse_training_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train the two-stage reference-conditioned DrivenVS bridge.")

    # Losses.
    parser.add_argument("--lambda_latent", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=0.0, type=float)
    parser.add_argument("--lambda_lpips", default=0.0, type=float)
    parser.add_argument("--lambda_temporal", default=0.05, type=float)
    parser.add_argument("--temporal_loss_warmup_steps", default=2000, type=int)
    parser.add_argument("--temporal_loss_use_small_raft", action="store_true")
    parser.add_argument("--flow_estimator", choices=("raft", "waft"), default="raft")
    parser.add_argument("--waft_repo", default=None)
    parser.add_argument("--waft_config", default=None)
    parser.add_argument("--waft_checkpoint", default=None)
    parser.add_argument("--waft_iterations", type=int, default=None)

    # Dataset and curriculum.
    parser.add_argument("--dataset_folder", required=True)
    parser.add_argument("--validation_dataset", default=None)
    parser.add_argument("--world_scenario_mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--train_image_prep", default="resize_576x1024")
    parser.add_argument("--test_image_prep", default="resize_576x1024")
    parser.add_argument("--training_stage", choices=("stage1", "stage2"), default="stage1")
    parser.add_argument(
        "--temporal_sample_probability",
        default=0.75,
        type=float,
        help="Stage-2 probability of V=5; the complement uses V=1.",
    )
    parser.add_argument("--num_reference_images", default=8, type=int)

    # Reference conditioning.
    parser.add_argument("--camera_embed", action="store_true")
    parser.add_argument("--time_embed", action="store_true")
    parser.add_argument("--camera_translation_scale", default=1.0, type=float)
    parser.add_argument("--reference_time_embedding_dim", default=32, type=int)
    parser.add_argument("--reference_time_scale", default=1.0, type=float)

    # ArtiFixer-style rectified flow used in both stages.
    parser.add_argument("--bridge_timestep_shift", default=5.0, type=float)
    parser.add_argument("--rf_num_train_timesteps", default=1000, type=int)
    parser.add_argument("--validation_steps", default=4, type=int)
    parser.add_argument("--training_objective", choices=("rectified_flow", "meanflow"), default="rectified_flow")
    parser.add_argument("--meanflow_interval_probability", default=0.75, type=float)
    parser.add_argument("--meanflow_endpoint_probability", default=0.0, type=float)
    parser.add_argument("--meanflow_time_sampler", choices=("uniform", "logit_normal"), default="logit_normal")
    parser.add_argument("--meanflow_time_mu", default=-0.4, type=float)
    parser.add_argument("--meanflow_time_sigma", default=1.0, type=float)
    parser.add_argument("--meanflow_adaptive_p", default=1.0, type=float)
    parser.add_argument("--meanflow_jvp_backend", choices=("autograd", "func"), default="func")
    parser.add_argument("--meanflow_ema_decay", default=0.9999, type=float)
    parser.add_argument("--meanflow_ema", action=argparse.BooleanOptionalAction, default=True)

    # Model initialization.
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--train_full_unet", action="store_true")
    parser.add_argument("--freeze_vae", action="store_true")
    parser.add_argument("--vae_skip_connection", action="store_true")

    # Optimization.
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=10_000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")

    # Sampling, checkpointing, and logging.
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--viz_freq", type=int, default=500)
    parser.add_argument("--checkpointing_steps", type=int, default=0)
    parser.add_argument("--num_samples_eval", type=int, default=20)
    parser.add_argument("--tracker_project_name", default="cosmos_harmonizer")
    parser.add_argument("--tracker_run_name", required=True)
    parser.add_argument("--report_to", default="wandb")
    return parser.parse_args(input_args)


RESIZE_IMAGE_PREPS = {
    "resize_200x360": (200, 360),
    "resize_256": (256, 256),
    "resize_256x256": (256, 256),
    "resize_288x512": (288, 512),
    "resize_384x672": (384, 672),
    "resize_384x704": (384, 704),
    "resize_416x736": (416, 736),
    "resize_448x832": (448, 832),
    "resize_512": (512, 512),
    "resize_512x512": (512, 512),
    "resize_544x960": (544, 960),
    "resize_nusc": (576, 1024),  # Nuscenes 450 - 850
    "resize_waymo": (640, 960),  # Waymo
    "resize_av2_stereo": (576, 1024),  # AV2 - Stereo
    "resize_576x1024": (576, 1024),
    "resize_768x1360": (768, 1360),
    "resize_1024x576": (1024, 576),
    "resize_1088x1920": (1088, 1920),
    "resize_2176x3840": (2176, 3840),
}


def build_transform(image_prep, interpolation=transforms.InterpolationMode.LANCZOS):
    if image_prep not in RESIZE_IMAGE_PREPS:
        raise ValueError(
            "DrivenVS requires deterministic pure resize so paired pixels and "
            f"camera intrinsics remain aligned; got {image_prep!r}"
        )
    return transforms.Resize(RESIZE_IMAGE_PREPS[image_prep], interpolation=interpolation)


_CONFIDENT_COVERAGE_FIELDS = (
    "opacity",
    "opacity_image",
    "coverage_image",
    "confident_coverage",
)
_LR_RL_SEQUENCE_RE = re.compile(r"^(?:LR|RL)(?:_|$)", re.IGNORECASE)


def resolve_manifest_media_paths(entry, manifest_path):
    """Resolve all supported media paths relative to their JSON manifest."""
    manifest_dir = Path(str(manifest_path)).expanduser().resolve().parent

    def resolve_path(value):
        path = Path(str(value)).expanduser()
        return str(path if path.is_absolute() else manifest_dir / path)

    resolved = dict(entry)
    for field in ("image", "target_image", "source_image"):
        if resolved.get(field) not in (None, ""):
            resolved[field] = resolve_path(resolved[field])
    for field in ("reference_images", "temporal_history_images"):
        if field in resolved:
            resolved[field] = [resolve_path(path) for path in resolved[field]]
    return resolved


def _coverage_frame_name(entry, image_path):
    for value in (
        entry.get("idx"),
        entry.get("frame_idx"),
        entry.get("frame_index"),
        entry.get("_sample_key"),
        image_path.stem,
    ):
        if value in (None, ""):
            continue
        stem = Path(str(value)).stem
        if stem.isdigit():
            return f"{stem.zfill(5)}.png"
    return f"{image_path.stem}.png"


def resolve_confident_coverage_path(entry):
    """Resolve white-valid GS coverage from an explicit or conventional path."""
    image_path = Path(str(entry["image"])).expanduser()
    for field in _CONFIDENT_COVERAGE_FIELDS:
        if entry.get(field) in (None, ""):
            continue
        path = Path(str(entry[field])).expanduser()
        if not path.is_absolute() and entry.get("_source_manifest"):
            path = Path(str(entry["_source_manifest"])).resolve().parent / path
        if not path.is_file():
            raise FileNotFoundError(f"Coverage path from field {field!r} does not exist: {path}")
        return str(path)

    sequence_name = entry.get("coverage_sequence")
    if sequence_name in (None, ""):
        sequence_name = next(
            (parent.name for parent in image_path.parents if _LR_RL_SEQUENCE_RE.match(parent.name)),
            image_path.parent.name,
        )
    frame_name = _coverage_frame_name(entry, image_path)
    attempted = []
    for dataset_root in image_path.parents:
        candidate = dataset_root / "confident_coverage" / str(sequence_name) / frame_name
        attempted.append(candidate)
        if candidate.is_file():
            return str(candidate)
    attempted_text = "\n  ".join(str(path) for path in attempted)
    raise FileNotFoundError(
        f"Could not resolve coverage for {image_path}. Provide an explicit coverage "
        f"field or create confident_coverage/{sequence_name}/{frame_name}. Tried:\n  "
        f"{attempted_text}"
    )


def resolve_reference_visibility_scores(entry, num_reference_images):
    """Return one visibility score per reference for Option-A selection.

    Canonical manifest field: ``reference_visibility_scores``.  Two short
    aliases are accepted for convenience.  If no score list is present, the
    manifest order is treated as already visibility-ranked, so index 0 becomes
    the joint reference.  Returning equal zeros is intentional: ``argmax``
    deterministically selects the first item while preserving the original
    order of the remaining references.
    """
    value = None
    used_field = None
    for field in (
        "reference_visibility_scores",
        "reference_visibility",
        "visibility_scores",
    ):
        if entry.get(field) is not None:
            value = entry[field]
            used_field = field
            break
    if value is None:
        return np.zeros(num_reference_images, dtype=np.float32)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{used_field} must be a list of {num_reference_images} scores")
    if len(value) != num_reference_images:
        raise ValueError(f"{used_field} must contain {num_reference_images} values, got {len(value)}")
    scores = np.asarray(value, dtype=np.float32)
    if scores.shape != (num_reference_images,) or not np.isfinite(scores).all():
        raise ValueError(f"{used_field} must contain finite numeric visibility scores")
    return scores


def world_scenario_path(entry, rgb_path):
    """Canonical double-reprojection controls share the scene and RGB filename."""
    return str(Path(entry["world_scenario_image"]).with_name(Path(rgb_path).name))


def reference_world_scenario_paths(entry):
    """Stereo references use their own camera; older manifests share one view."""
    if "reference_world_scenario_images" in entry:
        return entry["reference_world_scenario_images"]
    return [world_scenario_path(entry, path) for path in entry["reference_images"]]


class DrivenVSDataset(torch.utils.data.Dataset):
    """Return V=1 Stage-1 or mixed V=1/V=5 Stage-2 training samples.

    Temporal input is ``[warped(t), GT(t-1), ..., GT(t-4)]`` and its target is
    ``[GT(t), GT(t-1), ..., GT(t-4)]``. Source references remain a separate
    ``[R,C,H,W]`` tensor for frozen-VAE encoding; the model later promotes the
    highest-visibility view to joint self-attention and keeps the rest as K/V.
    """

    REQUIRED_FIELDS = ("image", "target_image", "reference_images")

    def __init__(
        self,
        dataset_folder,
        split,
        image_prep,
        num_reference_images,
        camera_embed=False,
        time_embed=False,
        training_stage="stage1",
        temporal_sample_probability=0.75,
        world_scenario=False,
    ):
        super().__init__()
        if training_stage not in {"stage1", "stage2"}:
            raise ValueError(f"Invalid training_stage={training_stage!r}")
        if num_reference_images < 1:
            raise ValueError("num_reference_images must be positive")
        if not 0.0 <= temporal_sample_probability <= 1.0:
            raise ValueError("temporal_sample_probability must be in [0, 1]")

        self.world_scenario = world_scenario
        self.center_index = 0
        self.num_reference_images = int(num_reference_images)
        self.camera_embed = bool(camera_embed)
        self.time_embed = bool(time_embed)
        self.training_stage = training_stage
        self.temporal_sample_probability = float(temporal_sample_probability)
        self.resolve_calibration = self.camera_embed or self.time_embed
        self.image_transform = build_transform(image_prep)
        self.opacity_transform = build_transform(
            image_prep,
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        self.data = {}
        self.img_names = []

        sources = [item.strip() for item in dataset_folder.split(",") if item.strip()]
        if not sources:
            raise ValueError("dataset_folder contains no JSON manifest")
        for source_index, source in enumerate(sources):
            if not source.endswith(".json"):
                raise ValueError(f"Expected a JSON manifest, got {source}")
            with open(source) as handle:
                manifest = json.load(handle)
            if split not in manifest or not isinstance(manifest[split], dict):
                raise KeyError(f"Split {split!r} is missing or invalid in {source}")

            for sample_key, raw_entry in manifest[split].items():
                if not isinstance(raw_entry, dict):
                    raise TypeError(f"Sample {sample_key!r} must be an object")
                missing = [field for field in self.REQUIRED_FIELDS if field not in raw_entry]
                if missing:
                    raise KeyError(f"Sample {sample_key!r} is missing {missing}")
                references = raw_entry["reference_images"]
                if not isinstance(references, list) or len(references) != self.num_reference_images:
                    raise ValueError(
                        f"Sample {sample_key!r} must contain exactly {self.num_reference_images} reference_images"
                    )

                name = f"{source_index}:{sample_key}"
                entry = resolve_manifest_media_paths(raw_entry, source)
                entry.update(
                    {
                        "_sample_key": sample_key,
                        "_source_index": source_index,
                        "_source_manifest": source,
                    }
                )
                entry["_opacity_path"] = resolve_confident_coverage_path(entry)
                entry["_reference_visibility_scores"] = resolve_reference_visibility_scores(
                    entry,
                    self.num_reference_images,
                )
                entry["_frame_index"] = temporal_frame_index(entry, sample_key)
                entry["_sequence_key"] = temporal_sequence_key(entry, sample_key)
                self.data[name] = entry
                self.img_names.append(name)

        self._index_temporal_history()

    def _index_temporal_history(self):
        temporal_index = {
            (entry["_source_index"], entry["_sequence_key"], entry["_frame_index"]): name
            for name, entry in self.data.items()
            if entry["_frame_index"] is not None
        }
        path_index = {
            (entry["_source_index"], str(Path(entry["target_image"]).resolve())): name
            for name, entry in self.data.items()
        }
        for entry in self.data.values():
            explicit = entry.get("temporal_history_images")
            if explicit is not None:
                if len(explicit) != TEMPORAL_HISTORY_FRAMES:
                    raise ValueError(f"temporal_history_images must contain {TEMPORAL_HISTORY_FRAMES} paths")
                history_entries = []
                for path in explicit:
                    resolved_path = str(Path(path).resolve())
                    matched_name = path_index.get((entry["_source_index"], resolved_path))
                    if matched_name is not None:
                        history_entries.append(self.data[matched_name])
                        continue

                    # Explicit history is allowed to point at a clean target
                    # that has no standalone manifest entry. Clone only the
                    # sequence/calibration context needed to resolve its camera.
                    history = dict(entry)
                    history["image"] = resolved_path
                    history["target_image"] = resolved_path
                    for field in ("frame_index", "frame_idx", "idx"):
                        history.pop(field, None)
                    history["_frame_index"] = temporal_frame_index(history)
                    history_entries.append(history)
                history_indices = [history.get("_frame_index") for history in history_entries]
                if entry["_frame_index"] is not None and all(index is not None for index in history_indices):
                    expected = [entry["_frame_index"] - offset for offset in range(1, TEMPORAL_HISTORY_FRAMES + 1)]
                    if history_indices != expected:
                        raise ValueError(
                            f"Sample {entry['_sample_key']!r} temporal_history_images "
                            f"resolve to {history_indices}; expected {expected}"
                        )
            elif entry["_frame_index"] is not None:
                names = [
                    temporal_index.get(
                        (
                            entry["_source_index"],
                            entry["_sequence_key"],
                            entry["_frame_index"] - offset,
                        )
                    )
                    for offset in range(1, TEMPORAL_HISTORY_FRAMES + 1)
                ]
                history_entries = [self.data[name] for name in names] if names and all(names) else []
            else:
                history_entries = []
            entry["_history_entries"] = history_entries

        self.temporal_eligible_names = [name for name, entry in self.data.items() if entry["_history_entries"]]
        if (
            self.training_stage == "stage2"
            and self.temporal_sample_probability > 0
            and not self.temporal_eligible_names
        ):
            raise ValueError(
                "Stage 2 found no sample with four consecutive past targets. "
                "Check frame indices/temporal_sequence or temporal_history_images."
            )

    def __len__(self):
        return len(self.img_names)

    def _load_image(self, path):
        with Image.open(path) as image:
            tensor = TF.to_tensor(self.image_transform(image.convert("RGB")))
        return TF.normalize(tensor, mean=[0.5], std=[0.5])

    def _load_opacity(self, path):
        with Image.open(path) as image:
            tensor = TF.to_tensor(self.opacity_transform(image.convert("L")))
        # Keep continuous opacity exactly like ArtiFixer.  Max pooling to latent
        # resolution happens inside DrivenVSModel.prepare_opacity_latent().
        return tensor.float().clamp_(0.0, 1.0)

    def __getitem__(self, index):
        name = self.img_names[index]
        entry = self.data[name]
        use_temporal = self.training_stage == "stage2" and random.random() < self.temporal_sample_probability
        if use_temporal and not entry["_history_entries"]:
            name = random.choice(self.temporal_eligible_names)
            entry = self.data[name]
        history_entries = entry["_history_entries"] if use_temporal else []

        history_targets = [self._load_image(history["target_image"]) for history in history_entries]
        current_input = self._load_image(entry["image"])
        current_target = self._load_image(entry["target_image"])
        current_opacity = self._load_opacity(entry["_opacity_path"])
        target_path = torch.stack([current_input, *history_targets], dim=0)
        targets = torch.stack([current_target, *history_targets], dim=0)
        opacity = torch.stack(
            [current_opacity, *[torch.ones_like(current_opacity) for _ in history_targets]],
            dim=0,
        )
        references = torch.stack(
            [self._load_image(path) for path in entry["reference_images"]],
            dim=0,
        )
        sample = {
            "conditioning_pixel_values": target_path,
            "reference_pixel_values": references,
            "reference_visibility_scores": torch.tensor(entry["_reference_visibility_scores"], dtype=torch.float32),
            "output_pixel_values": targets,
            "opacity_pixel_values": opacity,
            "center_index": self.center_index,
        }
        if self.world_scenario:
            sample["world_scenario_pixel_values"] = torch.stack(
                [
                    self._load_image(world_scenario_path(entry, path))
                    for path in [entry["target_image"], *[history["target_image"] for history in history_entries]]
                ]
            )
            sample["reference_world_scenario_pixel_values"] = torch.stack(
                [self._load_image(path) for path in reference_world_scenario_paths(entry)]
            )
        if self.resolve_calibration:
            resolver = resolve_sample_calibration if self.camera_embed else resolve_sample_time_offsets
            kwargs = {"resolve_time": self.time_embed} if self.camera_embed else {}
            camera = resolver(
                entry,
                entry["_source_manifest"],
                reference_images=entry["reference_images"],
                **kwargs,
            )
            if self.camera_embed:
                target_c2ws = [camera["target_c2w"]]
                target_Ks = [camera["target_K"]]
                for history in history_entries:
                    history_camera = resolve_sample_calibration(
                        history,
                        history["_source_manifest"],
                        reference_images=history["reference_images"],
                        resolve_time=False,
                    )
                    target_c2ws.append(history_camera["target_c2w"])
                    target_Ks.append(history_camera["target_K"])
                sample.update(
                    {
                        "target_c2w": torch.tensor(np.stack(target_c2ws), dtype=torch.float32),
                        "target_K": torch.tensor(np.stack(target_Ks), dtype=torch.float32),
                        "reference_c2ws": torch.tensor(camera["reference_c2ws"], dtype=torch.float32),
                        "reference_Ks": torch.tensor(camera["reference_Ks"], dtype=torch.float32),
                    }
                )
            if self.time_embed:
                sample.update(
                    {
                        "reference_time_offsets": torch.tensor(camera["reference_time_offsets"], dtype=torch.float32),
                        "reference_time_unit": str(camera["reference_time_unit"]),
                    }
                )
        return sample
