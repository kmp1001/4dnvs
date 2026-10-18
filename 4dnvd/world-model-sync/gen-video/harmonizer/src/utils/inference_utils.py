# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared manifest preparation for DrivenVS inference and evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .camera_calibration import (
    resolve_sample_calibration,
    resolve_sample_time_offsets,
)
from .temporal import (
    TEMPORAL_HISTORY_FRAMES,
    select_consecutive_prediction_history,
    temporal_frame_index,
)
from .training_utils import (
    reference_world_scenario_paths,
    resolve_confident_coverage_path,
    resolve_manifest_media_paths,
    resolve_reference_visibility_scores,
)

RESOLUTION_MAP = {
    "resize_nusc": (1024, 576),  # Nuscenes 450, 800
    "resize_288x512": (512, 288),  # low-resolution self-forcing smoke
    "resize_waymo": (960, 640),  # Waymo
    "resize_av2_stereo": (1024, 576),  # AV2 - Stereo
    "1024": (1024, 576),
}


def load_manifest_split(manifest_path: str, split: str) -> list[tuple[str, dict]]:
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    if split not in manifest or not isinstance(manifest[split], dict):
        raise KeyError(f"Split {split!r} is missing or is not an object in {manifest_path}")
    return [
        (sample_key, resolve_manifest_media_paths(entry, manifest_path))
        for sample_key, entry in manifest[split].items()
    ]


def load_image_tensor(path: str, size, device, dtype) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        tensor = transforms.ToTensor()(image)
    return transforms.Normalize([0.5], [0.5])(tensor).to(
        device=device,
        dtype=dtype,
    )


def load_opacity_tensor(path: str, size, device, dtype) -> torch.Tensor:
    with Image.open(path) as opacity:
        opacity = opacity.convert("L").resize(size, Image.Resampling.NEAREST)
        tensor = transforms.ToTensor()(opacity)
    # Preserve soft opacity instead of thresholding it to a binary mask.
    return tensor.clamp_(0.0, 1.0).to(device=device, dtype=dtype)


def safe_output_path(output_dir: Path, sample_key: str, entry: dict) -> Path:
    relative = Path(
        str(
            entry.get(
                "relative_path",
                entry.get("frame_name", Path(entry["image"]).name),
            )
        )
    )
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        digest = hashlib.sha1(f"{sample_key}:{entry['image']}".encode("utf-8")).hexdigest()[:16]
        relative = Path(f"{digest}_{Path(entry['image']).name}")
    return output_dir / relative


def build_reference_conditioning(
    entry: dict,
    manifest_path: str,
    reference_paths: list[str],
    *,
    camera_embed: bool,
    time_embed: bool,
    device: torch.device,
    target_history: list[dict],
) -> tuple[dict[str, torch.Tensor], dict | None]:
    """Build hybrid-reference metadata and target-path camera geometry."""
    if not camera_embed and not time_embed:
        return {}, None

    resolver = resolve_sample_calibration if camera_embed else resolve_sample_time_offsets
    resolver_kwargs = {"resolve_time": time_embed} if camera_embed else {}
    camera = resolver(
        entry,
        manifest_path,
        reference_images=reference_paths,
        **resolver_kwargs,
    )

    conditioning = {}
    current_target_camera = None
    if camera_embed:
        current_target_camera = {
            "target_c2w": camera["target_c2w"],
            "target_K": camera["target_K"],
        }
        target_c2ws = [camera["target_c2w"]] + [item["target_c2w"] for item in target_history]
        target_Ks = [camera["target_K"]] + [item["target_K"] for item in target_history]
        conditioning.update(
            {
                "target_c2w": torch.tensor(np.stack(target_c2ws), device=device, dtype=torch.float32).unsqueeze(0),
                "target_K": torch.tensor(np.stack(target_Ks), device=device, dtype=torch.float32).unsqueeze(0),
                "reference_c2ws": torch.tensor(camera["reference_c2ws"], device=device, dtype=torch.float32).unsqueeze(
                    0
                ),
                "reference_Ks": torch.tensor(camera["reference_Ks"], device=device, dtype=torch.float32).unsqueeze(0),
            }
        )
    if time_embed:
        conditioning["reference_time_offsets"] = torch.tensor(
            camera["reference_time_offsets"], device=device, dtype=torch.float32
        ).unsqueeze(0)
    return conditioning, current_target_camera


def prepare_inference_sample(
    sample_key: str,
    entry: dict,
    *,
    manifest_path: str,
    num_reference_images: int,
    size,
    device: torch.device,
    dtype: torch.dtype,
    camera_embed: bool,
    time_embed: bool,
    prediction_history: list[dict],
    temporal: bool,
    world_scenario: bool = False,
) -> dict:
    """Prepare either V=1 or autoregressive V=5 target-path input."""
    missing = [name for name in ("image", "target_image", "reference_images") if name not in entry]
    if missing:
        raise KeyError(f"Sample {sample_key!r} is missing fields: {missing}")
    reference_paths = entry["reference_images"]
    if not isinstance(reference_paths, list):
        raise TypeError(f"Sample {sample_key!r} reference_images must be a list")
    if len(reference_paths) != num_reference_images:
        raise ValueError(
            f"Sample {sample_key!r} has {len(reference_paths)} references; expected {num_reference_images}"
        )

    current_index = temporal_frame_index(entry, sample_key)
    target_history = select_consecutive_prediction_history(prediction_history, current_index) if temporal else []
    coverage_entry = dict(entry)
    coverage_entry.setdefault("_sample_key", sample_key)
    coverage_entry.setdefault("_source_manifest", manifest_path)
    opacity_path = resolve_confident_coverage_path(coverage_entry)

    with Image.open(entry["image"]) as input_image:
        original_size = input_image.size
    warped_current = load_image_tensor(entry["image"], size, device, dtype)
    opacity_current = load_opacity_tensor(opacity_path, size, device, dtype)
    references = torch.stack(
        [load_image_tensor(path, size, device, dtype) for path in reference_paths],
        dim=0,
    )
    target_frames = [warped_current] + [item["prediction"] for item in target_history]
    opacity_frames = [opacity_current] + [torch.ones_like(opacity_current) for _ in target_history]
    conditioning, current_target_camera = build_reference_conditioning(
        entry,
        manifest_path,
        reference_paths,
        camera_embed=camera_embed,
        time_embed=time_embed,
        device=device,
        target_history=target_history,
    )
    conditioning["reference_visibility_scores"] = torch.tensor(
        resolve_reference_visibility_scores(entry, num_reference_images),
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    current_world = None
    if world_scenario:
        current_world = load_image_tensor(entry["world_scenario_image"], size, device, dtype)
        conditioning["world_scenario"] = torch.stack(
            [current_world, *[item["world_scenario"] for item in target_history]], dim=1
        ).unsqueeze(0)
        conditioning["reference_world_scenario"] = torch.stack(
            [load_image_tensor(path, size, device, dtype) for path in reference_world_scenario_paths(entry)]
        ).unsqueeze(0)
    return {
        "current_world_scenario": current_world,
        "target_path": torch.stack(target_frames, dim=1).unsqueeze(0),
        "opacity": torch.stack(opacity_frames, dim=1).unsqueeze(0),
        "references": references.unsqueeze(0),
        "conditioning": conditioning,
        "current_target_camera": current_target_camera,
        "current_frame_index": current_index,
        "num_target_views": len(target_frames),
        "original_size": original_size,
        "opacity_path": opacity_path,
    }


def record_prediction(
    history: list[dict],
    prediction: torch.Tensor,
    prepared: dict,
    dtype: torch.dtype,
) -> None:
    history.append(
        {
            "prediction": prediction[0, :, 0].detach().to(dtype=dtype),
            "frame_index": prepared["current_frame_index"],
            **(
                {"world_scenario": prepared["current_world_scenario"]}
                if prepared.get("current_world_scenario") is not None
                else {}
            ),
            **(prepared["current_target_camera"] or {}),
        }
    )
    del history[:-TEMPORAL_HISTORY_FRAMES]
