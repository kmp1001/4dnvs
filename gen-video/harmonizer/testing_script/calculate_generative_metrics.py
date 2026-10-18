# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calculate FID, KID, and FVD from completed evaluation output.

Supports ordinary single-process execution and one-process-per-GPU execution
through ``torchrun``. Work is partitioned by complete scene so FVD clips never
cross ranks. Rank 0 reconstructs the global feature distributions before
calculating KID, FID, and FVD; per-rank scores are never averaged.

With ``--group-by-prefix``, the script additionally combines all sequence
folders beginning with L, R, or U into three independent distribution pools.
Pool metrics are calculated from the pooled features, never by averaging
per-sequence or per-scene scores.

Clean-FID keeps ``num_workers=0`` so its local resize callable is never sent
through multiprocessing pickle. Each worker owns its models on one device.
"""

import argparse
import json
import os
import pickle
import random
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from datetime import timedelta
from inspect import signature
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as F
import yaml
from cleanfid import fid as fid_backend
from cleanfid.features import build_feature_extractor
from PIL import Image
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

from fvd import (
    FVD_METHODS,
    default_model_path,
    extract_fvd_features,
    frechet_distance,
    load_i3d_pretrained,
)


DIRECTIONAL_POOL_PREFIXES = ("D", "L", "R", "U")


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(requested_device: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested = torch.device(requested_device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for metric calculation")
        device = torch.device("cuda", local_rank) if world_size > 1 else requested
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = requested
        backend = "gloo"
    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(hours=24),
        )
    return DistributedContext(rank, local_rank, world_size, device)


def barrier(context: DistributedContext) -> None:
    if context.is_distributed:
        dist.barrier()


def rank_print(context: DistributedContext, message: str) -> None:
    prefix = f"[rank {context.rank}/{context.world_size}] " if context.is_distributed else ""
    print(prefix + message, flush=True)


def image_pair_batches(prediction, target, device):
    """Match the original KID preprocessing, including prediction resizing."""
    if prediction.size != target.size:
        prediction = prediction.resize(target.size, Image.Resampling.BILINEAR)
    prediction_batch = F.to_tensor(prediction).unsqueeze(0).to(device)
    target_batch = F.to_tensor(target).unsqueeze(0).to(device)
    return prediction_batch, target_batch


def make_kid_extractor(args, device):
    return KernelInceptionDistance(
        feature=2048,
        subsets=args.kid_subsets,
        subset_size=args.kid_subset_size,
        normalize=False,
        sync_on_compute=False,
        dist_sync_on_step=False,
    ).to(device)


def extract_kid_feature(extractor, image_batch):
    uint8_batch = (image_batch * 255).clamp(0, 255).to(torch.uint8)
    with torch.inference_mode():
        return extractor.inception(uint8_batch).detach()


def kid_index_plan(count, subsets, subset_size, seed):
    """Match TorchMetrics' real-then-fake ``randperm`` sequence exactly."""
    if count < subset_size:
        return []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    plan = []
    for _ in range(subsets):
        real = torch.randperm(count, generator=generator)[:subset_size].numpy()
        fake = torch.randperm(count, generator=generator)[:subset_size].numpy()
        plan.append((real, fake))
    return plan


def polynomial_mmd(real_features, fake_features):
    gamma = 1.0 / real_features.shape[1]
    k_real = (real_features @ real_features.T * gamma + 1.0) ** 3
    k_fake = (fake_features @ fake_features.T * gamma + 1.0) ** 3
    k_cross = (real_features @ fake_features.T * gamma + 1.0) ** 3
    size = k_real.shape[0]
    within_real = k_real.sum() - torch.diag(k_real).sum()
    within_fake = k_fake.sum() - torch.diag(k_fake).sum()
    return (within_real + within_fake) / (size * (size - 1)) - 2.0 * k_cross.sum() / (size**2)


def compute_kid_from_tensors(real_features, fake_features, plan):
    if not plan:
        return None, None
    scores = []
    for real_indices, fake_indices in plan:
        real_indices = torch.as_tensor(real_indices, device=real_features.device)
        fake_indices = torch.as_tensor(fake_indices, device=fake_features.device)
        scores.append(polynomial_mmd(real_features[real_indices], fake_features[fake_indices]))
    scores = torch.stack(scores)
    return float(scores.mean().item()), float(scores.std(unbiased=False).item())


def compute_kid_from_lookup(real_lookup, fake_lookup, plan, device):
    if not plan:
        return None, None
    scores = []
    for real_indices, fake_indices in plan:
        real = torch.from_numpy(np.stack([real_lookup[int(i)] for i in real_indices])).to(device)
        fake = torch.from_numpy(np.stack([fake_lookup[int(i)] for i in fake_indices])).to(device)
        scores.append(polynomial_mmd(real, fake))
    scores = torch.stack(scores)
    return float(scores.mean().item()), float(scores.std(unbiased=False).item())


def selected_indices_from_plan(plan):
    selected = set()
    for real_indices, fake_indices in plan:
        selected.update(int(i) for i in real_indices)
        selected.update(int(i) for i in fake_indices)
    return selected


def sequence_pool(sequence):
    """Return D/L/R/U from the sequence folder basename, or ``None``."""
    folder_name = str(sequence).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if not folder_name:
        return None
    prefix = folder_name[0].upper()
    return prefix if prefix in DIRECTIONAL_POOL_PREFIXES else None


def build_prefix_pool_specs(grouped_records, args):
    """Build independent KID plans and sample membership for D/L/R/U pools."""
    if not args.group_by_prefix:
        return OrderedDict()
    indices = {prefix: [] for prefix in DIRECTIONAL_POOL_PREFIXES}
    sequences = {prefix: set() for prefix in DIRECTIONAL_POOL_PREFIXES}
    for scene_id, records in grouped_records.items():
        for record in records:
            prefix = sequence_pool(record["sequence"])
            if prefix is None:
                continue
            indices[prefix].append(record["_global_index"])
            sequences[prefix].add((scene_id, str(record["sequence"])))
    return OrderedDict(
        (
            prefix,
            {
                "global_indices": indices[prefix],
                "kid_plan": kid_index_plan(
                    len(indices[prefix]),
                    args.kid_subsets,
                    args.kid_subset_size,
                    args.seed,
                ),
                "num_sequences": len(sequences[prefix]),
            },
        )
        for prefix in DIRECTIONAL_POOL_PREFIXES
    )


def selected_pool_global_indices(pool_spec):
    selected_local_indices = selected_indices_from_plan(pool_spec["kid_plan"])
    return {pool_spec["global_indices"][index] for index in selected_local_indices}


def build_fvd_clips(records, clip_length, clip_stride, calculate_for_input, include_pool_labels=False):
    """Create clips without crossing temporal sequence boundaries."""
    sequences = OrderedDict()
    for record in records:
        frames = sequences.setdefault(record["sequence"], {"real": [], "fake": [], "input": []})
        frames["real"].append(record["target_path"])
        frames["fake"].append(record["resolved_prediction_path"])
        if calculate_for_input:
            frames["input"].append(record["input_path"])
    clips = {"real": [], "fake": [], "input": []}
    stride = clip_stride or clip_length
    pool_labels = []
    for sequence_id, frames_by_kind in sequences.items():
        active = [kind for kind in clips if frames_by_kind[kind]]
        frame_counts = {len(frames_by_kind[kind]) for kind in active}
        if len(frame_counts) != 1:
            raise RuntimeError(f"Mismatched FVD sequence lengths: {frame_counts}")
        frame_count = frame_counts.pop()
        if frame_count < 10:
            continue
        if clip_length == 0:
            starts, current_length = [0], frame_count
        elif frame_count < clip_length:
            starts, current_length = [0], frame_count
        else:
            starts = range(0, frame_count - clip_length + 1, stride)
            current_length = clip_length
        for start in starts:
            if include_pool_labels:
                pool_labels.append(sequence_pool(sequence_id))
            for kind in active:
                clips[kind].append(frames_by_kind[kind][start : start + current_length])
    if include_pool_labels:
        clips["pool_labels"] = pool_labels
    return clips


def concatenate_fvd_features(batches):
    return np.concatenate(batches, axis=0) if batches else np.empty((0, 400), dtype=np.float32)


def resolve_records(manifest, prediction_dir, max_frames_per_scene=None):
    if manifest.get("format_version") != 1 or not manifest.get("complete"):
        raise ValueError("The evaluation manifest is incomplete or has an unsupported format")
    records = manifest.get("samples")
    if not isinstance(records, list) or len(records) != manifest.get("num_samples"):
        raise ValueError("The evaluation manifest sample count is invalid")
    grouped = OrderedDict()
    for source_record in records:
        record = dict(source_record)
        relative = Path(record["prediction_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe prediction path in manifest: {relative}")
        prediction_path = prediction_dir / relative
        record["resolved_prediction_path"] = str(prediction_path)
        required = {
            "prediction": prediction_path,
            "target": Path(record["target_path"]),
            "input": Path(record["input_path"]),
        }
        for kind, path in required.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {kind} image for sample {record['sample_key']}: {path}")
        grouped.setdefault(record["scene"], []).append(record)
    if not grouped:
        raise ValueError("The evaluation manifest contains no samples")

    # Limit to the first k frames per scene if requested (0 or None evaluates all)
    if max_frames_per_scene is not None and max_frames_per_scene > 0:
        for scene in grouped:
            grouped[scene] = grouped[scene][:max_frames_per_scene]

    evaluation_index = 0
    for scene_records in grouped.values():
        for record in scene_records:
            record["_global_index"] = evaluation_index
            evaluation_index += 1
    return grouped


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FID, KID, and FVD on one GPU or multiple torchrun workers.")
    parser.add_argument("--prediction_dir", "--output_dir", dest="prediction_dir", required=True)
    parser.add_argument("--prediction_manifest", default=None)
    parser.add_argument("--metrics_output", default=None)
    parser.add_argument("--device", default="cuda:0", help="Single-process device; torchrun uses cuda:LOCAL_RANK.")
    parser.add_argument("--calculate-for-input", action="store_true")
    parser.add_argument(
        "--group-by-prefix",
        action="store_true",
        help=(
            "Also compute independent pooled FID/KID/FVD for sequence-folder basenames beginning with D, L, R, and U."
        ),
    )
    parser.add_argument(
        "--max-frames-per-scene",
        type=int,
        default=0,
        help="Limit calculation to the first k frames of each scene (default: 81). Set to 0 to process all frames.",
    )
    parser.add_argument("--fid-mode", default="clean")
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=20)
    parser.add_argument("--disable-fvd", action="store_true")
    parser.add_argument("--fvd-method", choices=FVD_METHODS, default="styleganv")
    parser.add_argument("--fvd-model-path", default=None)
    parser.add_argument("--fvd-clip-length", type=int, default=0)
    parser.add_argument("--fvd-clip-stride", type=int, default=0)
    parser.add_argument("--fvd-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args):
    if args.kid_subset_size < 2:
        raise ValueError("--kid-subset-size must be at least 2")
    if args.kid_subsets < 1:
        raise ValueError("--kid-subsets must be positive")
    if args.fid_batch_size < 1:
        raise ValueError("--fid-batch-size must be positive")
    if args.max_frames_per_scene < 0:
        raise ValueError("--max-frames-per-scene must be non-negative")
    if 0 < args.fvd_clip_length < 10:
        raise ValueError("--fvd-clip-length must be 0 or at least 10")
    if args.fvd_clip_stride < 0:
        raise ValueError("--fvd-clip-stride must be non-negative")
    if args.fvd_clip_length == 0 and args.fvd_clip_stride:
        raise ValueError("--fvd-clip-stride requires a positive --fvd-clip-length")
    if args.fvd_batch_size < 1:
        raise ValueError("--fvd-batch-size must be positive")
    if not args.disable_fvd:
        args.fvd_model_path = str(Path(args.fvd_model_path or default_model_path(args.fvd_method)).resolve())
        if not Path(args.fvd_model_path).is_file():
            raise FileNotFoundError(f"PyTorch FVD weights not found: {args.fvd_model_path}")


def assigned_scenes(grouped_records, context):
    """Assign whole scenes with deterministic greedy load balancing."""
    buckets = [[] for _ in range(context.world_size)]
    loads = [0 for _ in range(context.world_size)]
    indexed_scenes = list(enumerate(grouped_records.items()))
    for scene_index, scene_item in sorted(
        indexed_scenes,
        key=lambda indexed_item: (-len(indexed_item[1][1]), indexed_item[0]),
    ):
        owner = min(range(context.world_size), key=lambda rank: (loads[rank], rank))
        buckets[owner].append((scene_index, scene_item))
        loads[owner] += len(scene_item[1])
    return [scene_item for _, scene_item in sorted(buckets[context.rank])]


def calculate_local_kid(args, context, local_scenes, selected_global_indices):
    if not local_scenes:
        return {}, []
    extractor = make_kid_extractor(args, context.device)
    scene_metrics, selected_features = {}, []
    for scene_id, records in local_scenes:
        rank_print(context, f"KID features for scene {scene_id}: {len(records)} images")
        real_features, fake_features, input_features = [], [], []
        for record in tqdm(records, desc=scene_id, disable=context.is_distributed):
            with Image.open(record["resolved_prediction_path"]) as handle:
                prediction_image = handle.convert("RGB")
            with Image.open(record["target_path"]) as handle:
                target_image = handle.convert("RGB")
            prediction_batch, target_batch = image_pair_batches(prediction_image, target_image, context.device)
            real_feature = extract_kid_feature(extractor, target_batch)
            fake_feature = extract_kid_feature(extractor, prediction_batch)
            real_features.append(real_feature)
            fake_features.append(fake_feature)
            input_feature = None
            if args.calculate_for_input:
                with Image.open(record["input_path"]) as handle:
                    input_image = handle.convert("RGB")
                input_batch, _ = image_pair_batches(input_image, target_image, context.device)
                input_feature = extract_kid_feature(extractor, input_batch)
                input_features.append(input_feature)
            if record["_global_index"] in selected_global_indices:
                selected_features.append(
                    (
                        record["_global_index"],
                        real_feature[0].float().cpu().numpy(),
                        fake_feature[0].float().cpu().numpy(),
                        input_feature[0].float().cpu().numpy() if input_feature is not None else None,
                    )
                )
        count = len(records)
        plan = kid_index_plan(count, args.kid_subsets, args.kid_subset_size, args.seed)
        real_tensor, fake_tensor = torch.cat(real_features), torch.cat(fake_features)
        kid_mean, kid_std = compute_kid_from_tensors(real_tensor, fake_tensor, plan)
        output = {
            "fid": None,
            "kid_mean": kid_mean,
            "kid_std": kid_std,
            "num_images": count,
        }
        if args.calculate_for_input:
            input_kid_mean, input_kid_std = compute_kid_from_tensors(real_tensor, torch.cat(input_features), plan)
            output["input"] = {
                "fid": None,
                "kid_mean": input_kid_mean,
                "kid_std": input_kid_std,
            }
        scene_metrics[scene_id] = output
    del extractor
    if context.device.type == "cuda":
        torch.cuda.empty_cache()
    return scene_metrics, selected_features


def feature_moments(features):
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    centered = features - mean
    return {"count": len(features), "mean": mean, "m2": centered.T @ centered}


def merge_feature_moments(destination, source):
    if source is None:
        return destination
    if destination is None:
        return {
            "count": int(source["count"]),
            "mean": np.array(source["mean"], copy=True),
            "m2": np.array(source["m2"], copy=True),
        }
    destination_count = int(destination["count"])
    source_count = int(source["count"])
    combined_count = destination_count + source_count
    delta = source["mean"] - destination["mean"]
    destination["m2"] += source["m2"]
    destination["m2"] += np.outer(delta, delta) * destination_count * source_count / combined_count
    destination["mean"] += delta * source_count / combined_count
    destination["count"] = combined_count
    return destination


def fid_from_moments(first, second):
    if first is None or second is None or first["count"] < 2 or second["count"] < 2:
        return None
    n1, n2 = first["count"], second["count"]
    covariance1 = first["m2"] / (n1 - 1)
    covariance2 = second["m2"] / (n2 - 1)
    return float(fid_backend.frechet_distance(first["mean"], covariance1, second["mean"], covariance2))


def extract_cleanfid_features(paths, model, args, context):
    return fid_backend.get_files_features(
        paths,
        model=model,
        num_workers=0,
        batch_size=args.fid_batch_size,
        device=context.device,
        mode=args.fid_mode,
        verbose=False,
    )


def build_cleanfid_extractor(mode, device):
    """Build one-device Clean-FID models across old and new API versions."""
    parameters = signature(build_feature_extractor).parameters
    if "use_dataparallel" in parameters:
        return build_feature_extractor(mode, device, use_dataparallel=False)
    model = build_feature_extractor(mode, device)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module.to(device)
    return model


def empty_feature_moments():
    return {"real": None, "fake": None, "input": None}


def calculate_local_fid(args, context, local_scenes, scene_metrics):
    fid_moments = {
        "overall": empty_feature_moments(),
        "pools": {prefix: empty_feature_moments() for prefix in DIRECTIONAL_POOL_PREFIXES if args.group_by_prefix},
    }
    if not local_scenes:
        return fid_moments
    rank_print(context, "Loading one Clean-FID Inception extractor")
    model = build_cleanfid_extractor(args.fid_mode, context.device)
    for scene_id, records in local_scenes:
        rank_print(context, f"Clean-FID features for scene {scene_id}")
        real = extract_cleanfid_features([r["target_path"] for r in records], model, args, context)
        fake = extract_cleanfid_features([r["resolved_prediction_path"] for r in records], model, args, context)
        real_moments, fake_moments = feature_moments(real), feature_moments(fake)
        scene_metrics[scene_id]["fid"] = fid_from_moments(fake_moments, real_moments)
        fid_moments["overall"]["real"] = merge_feature_moments(fid_moments["overall"]["real"], real_moments)
        fid_moments["overall"]["fake"] = merge_feature_moments(fid_moments["overall"]["fake"], fake_moments)
        input_features = None
        if args.calculate_for_input:
            input_features = extract_cleanfid_features([r["input_path"] for r in records], model, args, context)
            input_moments = feature_moments(input_features)
            scene_metrics[scene_id]["input"]["fid"] = fid_from_moments(input_moments, real_moments)
            fid_moments["overall"]["input"] = merge_feature_moments(fid_moments["overall"]["input"], input_moments)
        if args.group_by_prefix:
            real_array = np.asarray(real)
            fake_array = np.asarray(fake)
            input_array = np.asarray(input_features) if input_features is not None else None
            for prefix in DIRECTIONAL_POOL_PREFIXES:
                record_indices = [
                    index for index, record in enumerate(records) if sequence_pool(record["sequence"]) == prefix
                ]
                if not record_indices:
                    continue
                pool_real = feature_moments(real_array[record_indices])
                pool_fake = feature_moments(fake_array[record_indices])
                fid_moments["pools"][prefix]["real"] = merge_feature_moments(
                    fid_moments["pools"][prefix]["real"], pool_real
                )
                fid_moments["pools"][prefix]["fake"] = merge_feature_moments(
                    fid_moments["pools"][prefix]["fake"], pool_fake
                )
                if input_array is not None:
                    pool_input = feature_moments(input_array[record_indices])
                    fid_moments["pools"][prefix]["input"] = merge_feature_moments(
                        fid_moments["pools"][prefix]["input"], pool_input
                    )
    del model
    if context.device.type == "cuda":
        torch.cuda.empty_cache()
    return fid_moments


def calculate_local_fvd(args, context, local_scenes, scene_metrics):
    clips_by_scene = {
        scene_id: build_fvd_clips(
            records,
            args.fvd_clip_length,
            args.fvd_clip_stride,
            args.calculate_for_input,
            include_pool_labels=args.group_by_prefix,
        )
        for scene_id, records in local_scenes
    }
    scene_features = {}
    for scene_id, clips in clips_by_scene.items():
        count = len(clips["real"])
        scene_metrics[scene_id].update({"fvd": None, "fvd_num_clips": count})
        if args.calculate_for_input:
            scene_metrics[scene_id]["input"].update({"fvd": None, "fvd_num_clips": count})
    if not sum(len(clips["real"]) for clips in clips_by_scene.values()):
        return scene_features
    rank_print(context, f"Loading PyTorch {args.fvd_method} FVD I3D from {args.fvd_model_path}")
    detector = load_i3d_pretrained(args.fvd_method, device=context.device, model_path=args.fvd_model_path)
    for scene_id, clips in clips_by_scene.items():
        count = len(clips["real"])
        if not count:
            continue
        real = extract_fvd_features(
            args.fvd_method,
            clips["real"],
            detector,
            context.device,
            batch_size=args.fvd_batch_size,
        )
        fake = extract_fvd_features(
            args.fvd_method,
            clips["fake"],
            detector,
            context.device,
            batch_size=args.fvd_batch_size,
        )
        scene_metrics[scene_id]["fvd"] = frechet_distance(args.fvd_method, fake, real)
        features = {"real": real, "fake": fake}
        if args.group_by_prefix:
            if len(clips["pool_labels"]) != len(real):
                raise RuntimeError(
                    f"FVD pool-label count mismatch for scene {scene_id}: "
                    f"{len(clips['pool_labels'])} labels versus {len(real)} features"
                )
            features["pool_labels"] = clips["pool_labels"]
        if args.calculate_for_input:
            input_features = extract_fvd_features(
                args.fvd_method,
                clips["input"],
                detector,
                context.device,
                batch_size=args.fvd_batch_size,
            )
            scene_metrics[scene_id]["input"]["fvd"] = frechet_distance(args.fvd_method, input_features, real)
            features["input"] = input_features
        scene_features[scene_id] = features
        rank_print(
            context,
            f"Scene {scene_id}: FVD={scene_metrics[scene_id]['fvd']} ({count} clips)",
        )
    del detector
    if context.device.type == "cuda":
        torch.cuda.empty_cache()
    return scene_features


def write_rank_payload(path, payload):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)


def load_and_merge_rank_payloads(metric_cache, world_size, pool_prefixes):
    scene_metrics, fvd_features = {}, {}
    kid_real, kid_fake, kid_input = {}, {}, {}
    fid_moments = {
        "overall": empty_feature_moments(),
        "pools": {prefix: empty_feature_moments() for prefix in pool_prefixes},
    }
    for rank in range(world_size):
        with (metric_cache / f"rank_{rank:05d}.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        overlap = set(scene_metrics).intersection(payload["scene_metrics"])
        if overlap:
            raise RuntimeError(f"Scenes were evaluated on multiple ranks: {sorted(overlap)}")
        scene_metrics.update(payload["scene_metrics"])
        fvd_features.update(payload["fvd_features"])
        for index, real, fake, input_feature in payload["selected_kid_features"]:
            if index in kid_real:
                raise RuntimeError(f"Duplicate selected KID feature for sample {index}")
            kid_real[index], kid_fake[index] = real, fake
            if input_feature is not None:
                kid_input[index] = input_feature
        for kind in fid_moments["overall"]:
            fid_moments["overall"][kind] = merge_feature_moments(
                fid_moments["overall"][kind], payload["fid_moments"]["overall"][kind]
            )
        for prefix in pool_prefixes:
            for kind in fid_moments["pools"][prefix]:
                fid_moments["pools"][prefix][kind] = merge_feature_moments(
                    fid_moments["pools"][prefix][kind],
                    payload["fid_moments"]["pools"][prefix][kind],
                )
    return scene_metrics, kid_real, kid_fake, kid_input, fid_moments, fvd_features


def build_final_results(
    args,
    context,
    manifest,
    grouped_records,
    metric_cache,
    overall_kid_plan,
    pool_specs,
):
    merged = load_and_merge_rank_payloads(metric_cache, context.world_size, pool_specs.keys())
    unordered_metrics, kid_real, kid_fake, kid_input, fid_moments, fvd_features = merged
    expected_scenes = set(grouped_records)
    if set(unordered_metrics) != expected_scenes:
        missing = sorted(expected_scenes - set(unordered_metrics))
        extra = sorted(set(unordered_metrics) - expected_scenes)
        raise RuntimeError(f"Distributed scene merge mismatch; missing={missing}, extra={extra}")
    per_scene = OrderedDict((scene_id, unordered_metrics[scene_id]) for scene_id in grouped_records)
    total_images = sum(len(records) for records in grouped_records.values())
    overall_selected_indices = selected_indices_from_plan(overall_kid_plan)
    expected_indices = set(overall_selected_indices)
    for pool_spec in pool_specs.values():
        expected_indices.update(selected_pool_global_indices(pool_spec))
    if set(kid_real) != expected_indices or set(kid_fake) != expected_indices:
        raise RuntimeError("Distributed KID merge is missing selected real or fake features")
    kid_mean, kid_std = compute_kid_from_lookup(kid_real, kid_fake, overall_kid_plan, context.device)
    config = manifest["config"]
    overall_fid_moments = fid_moments["overall"]
    overall = {
        "fid": fid_from_moments(overall_fid_moments["fake"], overall_fid_moments["real"]),
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "total_images": total_images,
        "split": manifest["split"],
        "resolution": config["resolution"],
        "num_gpus": manifest["num_gpus"],
        "metric_num_gpus": context.world_size,
        "gpu_memory": manifest["inference_summary"]["gpu_memory"],
    }
    latency = manifest["inference_summary"].get("speed_benchmark_latency_per_sample")
    if latency is not None:
        overall["speed_benchmark_latency_per_sample"] = float(latency)
        overall["speed_benchmark_throughput_samples_per_sec"] = float(1.0 / latency)
    if args.calculate_for_input:
        if set(kid_input) != expected_indices:
            raise RuntimeError("Distributed KID merge is missing selected input features")
        input_kid_mean, input_kid_std = compute_kid_from_lookup(kid_real, kid_input, overall_kid_plan, context.device)
        overall["input"] = {
            "fid": fid_from_moments(overall_fid_moments["input"], overall_fid_moments["real"]),
            "kid_mean": input_kid_mean,
            "kid_std": input_kid_std,
        }
    pools = OrderedDict()
    for prefix, pool_spec in pool_specs.items():
        selected_local_indices = selected_indices_from_plan(pool_spec["kid_plan"])
        pool_real_lookup = {
            local_index: kid_real[pool_spec["global_indices"][local_index]] for local_index in selected_local_indices
        }
        pool_fake_lookup = {
            local_index: kid_fake[pool_spec["global_indices"][local_index]] for local_index in selected_local_indices
        }
        pool_kid_mean, pool_kid_std = compute_kid_from_lookup(
            pool_real_lookup,
            pool_fake_lookup,
            pool_spec["kid_plan"],
            context.device,
        )
        pool_fid_moments = fid_moments["pools"][prefix]
        pool_result = {
            "fid": fid_from_moments(pool_fid_moments["fake"], pool_fid_moments["real"]),
            "kid_mean": pool_kid_mean,
            "kid_std": pool_kid_std,
            "num_images": len(pool_spec["global_indices"]),
            "num_sequences": pool_spec["num_sequences"],
        }
        if args.calculate_for_input:
            pool_input_lookup = {
                local_index: kid_input[pool_spec["global_indices"][local_index]]
                for local_index in selected_local_indices
            }
            pool_input_kid_mean, pool_input_kid_std = compute_kid_from_lookup(
                pool_real_lookup,
                pool_input_lookup,
                pool_spec["kid_plan"],
                context.device,
            )
            pool_result["input"] = {
                "fid": fid_from_moments(pool_fid_moments["input"], pool_fid_moments["real"]),
                "kid_mean": pool_input_kid_mean,
                "kid_std": pool_input_kid_std,
            }
        pools[prefix] = pool_result
    if not args.disable_fvd:
        total_clips = sum(per_scene[scene_id]["fvd_num_clips"] for scene_id in grouped_records)
        all_real, all_fake, all_input = [], [], []
        pool_fvd_batches = {prefix: {"real": [], "fake": [], "input": []} for prefix in pool_specs}
        for scene_id in grouped_records:
            features = fvd_features.get(scene_id)
            if features is None:
                continue
            all_real.append(features["real"])
            all_fake.append(features["fake"])
            if args.calculate_for_input:
                all_input.append(features["input"])
            if pool_specs:
                labels = np.asarray(features["pool_labels"])
                if len(labels) != len(features["real"]) or len(labels) != len(features["fake"]):
                    raise RuntimeError(f"Merged FVD pool-label mismatch for scene {scene_id}")
                for prefix in pool_specs:
                    mask = labels == prefix
                    if not np.any(mask):
                        continue
                    pool_fvd_batches[prefix]["real"].append(features["real"][mask])
                    pool_fvd_batches[prefix]["fake"].append(features["fake"][mask])
                    if args.calculate_for_input:
                        pool_fvd_batches[prefix]["input"].append(features["input"][mask])
        real = concatenate_fvd_features(all_real)
        fake = concatenate_fvd_features(all_fake)
        overall.update(
            {
                "fvd": frechet_distance(args.fvd_method, fake, real),
                "fvd_num_clips": total_clips,
                "fvd_backend": f"pytorch_{args.fvd_method}_i3d",
                "fvd_clip_length": args.fvd_clip_length or "full_sequence",
                "fvd_clip_stride": (args.fvd_clip_stride or args.fvd_clip_length if args.fvd_clip_length else None),
            }
        )
        if args.calculate_for_input:
            input_features = concatenate_fvd_features(all_input)
            overall["input"].update(
                {
                    "fvd": frechet_distance(args.fvd_method, input_features, real),
                    "fvd_num_clips": total_clips,
                }
            )
        for prefix, batches in pool_fvd_batches.items():
            pool_real = concatenate_fvd_features(batches["real"])
            pool_fake = concatenate_fvd_features(batches["fake"])
            pool_clip_count = len(pool_real)
            pools[prefix].update(
                {
                    "fvd": (frechet_distance(args.fvd_method, pool_fake, pool_real) if pool_clip_count else None),
                    "fvd_num_clips": pool_clip_count,
                }
            )
            if args.calculate_for_input:
                pool_input = concatenate_fvd_features(batches["input"])
                pools[prefix]["input"].update(
                    {
                        "fvd": (frechet_distance(args.fvd_method, pool_input, pool_real) if pool_clip_count else None),
                        "fvd_num_clips": pool_clip_count,
                    }
                )
    return overall, per_scene, pools


def print_overall(overall, pools, metrics_path, disable_fvd):
    print("\nOverall generative results")
    print(f"  Metric GPUs: {overall['metric_num_gpus']}")
    print(f"  Images: {overall['total_images']}")
    print(f"  FID: {overall['fid']}")
    print(f"  KID: {overall['kid_mean']} +/- {overall['kid_std']}")
    if not disable_fvd:
        print(f"  FVD: {overall['fvd']} ({overall['fvd_num_clips']} clips)")
    for prefix, values in pools.items():
        print(f"\n{prefix} sequence pool ({values['num_sequences']} sequences)")
        print(f"  Images: {values['num_images']}")
        print(f"  FID: {values['fid']}")
        print(f"  KID: {values['kid_mean']} +/- {values['kid_std']}")
        if not disable_fvd:
            print(f"  FVD: {values['fvd']} ({values['fvd_num_clips']} clips)")
    print(f"Metrics saved to {metrics_path}")


def main():
    args = parse_args()
    validate_args(args)
    context = setup_distributed(args.device)
    try:
        random.seed(args.seed)
        np.random.seed(args.seed % (2**32))
        torch.manual_seed(args.seed)
        if context.device.type == "cuda":
            torch.cuda.manual_seed(args.seed)
        torch.set_grad_enabled(False)
        prediction_dir = Path(args.prediction_dir).resolve()
        manifest_path = Path(args.prediction_manifest or prediction_dir / "evaluation_manifest.json").resolve()
        metrics_path = Path(args.metrics_output or prediction_dir / "generative_metrics.yaml").resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        grouped_records = resolve_records(manifest, prediction_dir, max_frames_per_scene=args.max_frames_per_scene)
        total_images = sum(len(records) for records in grouped_records.values())
        overall_kid_plan = kid_index_plan(total_images, args.kid_subsets, args.kid_subset_size, args.seed)
        selected_indices = selected_indices_from_plan(overall_kid_plan)
        pool_specs = build_prefix_pool_specs(grouped_records, args)
        for pool_spec in pool_specs.values():
            selected_indices.update(selected_pool_global_indices(pool_spec))
        local_scenes = assigned_scenes(grouped_records, context)
        rank_print(
            context,
            f"Assigned {len(local_scenes)} of {len(grouped_records)} scenes on {context.device}",
        )
        metric_cache = prediction_dir / "_generative_metric_cache"
        if context.is_main:
            if metric_cache.exists():
                shutil.rmtree(metric_cache)
            metric_cache.mkdir(parents=True, exist_ok=True)
        barrier(context)
        scene_metrics, selected_kid_features = calculate_local_kid(args, context, local_scenes, selected_indices)
        fid_moments = calculate_local_fid(args, context, local_scenes, scene_metrics)
        fvd_features = {} if args.disable_fvd else calculate_local_fvd(args, context, local_scenes, scene_metrics)
        write_rank_payload(
            metric_cache / f"rank_{context.rank:05d}.pkl",
            {
                "scene_metrics": scene_metrics,
                "selected_kid_features": selected_kid_features,
                "fid_moments": fid_moments,
                "fvd_features": fvd_features,
            },
        )
        barrier(context)
        if context.is_main:
            overall, per_scene, pools = build_final_results(
                args,
                context,
                manifest,
                grouped_records,
                metric_cache,
                overall_kid_plan,
                pool_specs,
            )
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            output = {"overall": overall}
            if pools:
                output["pools"] = dict(pools)
            output["per_scene"] = dict(per_scene)
            with metrics_path.open("w") as handle:
                yaml.safe_dump(output, handle, sort_keys=False)
            print_overall(overall, pools, metrics_path, args.disable_fvd)
        barrier(context)
        if context.is_main:
            shutil.rmtree(metric_cache)
        barrier(context)
    finally:
        if context.is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
