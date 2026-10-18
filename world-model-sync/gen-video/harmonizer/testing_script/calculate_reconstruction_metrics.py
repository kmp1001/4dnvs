# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calculate PSNR, LPIPS, and SSIM from completed evaluation output with live output logging.

Supports ordinary single-process execution and one-process-per-GPU execution
through ``torchrun``. Work is partitioned by complete scene, and rank 0 merges
the per-image values before calculating the global mean and standard deviation.

With ``--group-by-prefix``, the script additionally combines all sequence
folders beginning with D, L, R, or U into three independent metric pools. Pool
statistics are calculated from all image-level scores in the pool, never by
averaging per-sequence or per-scene summaries.
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
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as F
import yaml
from PIL import Image
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio
from tqdm import tqdm


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


def ssim_channel(img1: np.ndarray, img2: np.ndarray) -> float:
    c1, c2 = 0.01**2, 0.03**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float((numerator / denominator).mean())


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    if img1.shape != img2.shape or img1.ndim != 3:
        raise ValueError(f"SSIM expects matching CHW images, got {img1.shape} and {img2.shape}")
    return float(np.mean([ssim_channel(img1[c].numpy(), img2[c].numpy()) for c in range(img1.shape[0])]))


def pair_metrics(prediction, target, psnr_metric, lpips_metric, device):
    if prediction.size != target.size:
        prediction = prediction.resize(target.size, Image.Resampling.BILINEAR)
    prediction_tensor, target_tensor = F.to_tensor(prediction), F.to_tensor(target)
    prediction_batch = prediction_tensor.unsqueeze(0).to(device)
    target_batch = target_tensor.unsqueeze(0).to(device)
    psnr = float(psnr_metric(prediction_batch, target_batch).item())
    psnr_metric.reset()
    lpips = float(lpips_metric(prediction_batch, target_batch).item())
    lpips_metric.reset()
    return {
        "psnr": psnr,
        "lpips": lpips,
        "ssim": calculate_ssim(prediction_tensor, target_tensor),
    }


def empty_metric_lists():
    return {"psnr": [], "lpips": [], "ssim": []}


def append_metrics(storage, values):
    for name in storage:
        storage[name].append(values[name])


def extend_metrics(storage, values):
    for name in storage:
        storage[name].extend(values[name])


def summarize_metrics(values):
    return {
        f"{name}_{statistic}": float(function(items)) if items else None
        for name, items in values.items()
        for statistic, function in (("mean", np.mean), ("std", np.std))
    }


def sequence_pool(sequence):
    """Return D/L/R/U from the sequence folder basename, or ``None``."""
    folder_name = str(sequence).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if not folder_name:
        return None
    prefix = folder_name[0].upper()
    return prefix if prefix in DIRECTIONAL_POOL_PREFIXES else None


def build_prefix_pool_specs(grouped_records, enabled):
    if not enabled:
        return OrderedDict()
    image_counts = {prefix: 0 for prefix in DIRECTIONAL_POOL_PREFIXES}
    sequences = {prefix: set() for prefix in DIRECTIONAL_POOL_PREFIXES}
    for scene_id, records in grouped_records.items():
        for record in records:
            prefix = sequence_pool(record["sequence"])
            if prefix is None:
                continue
            image_counts[prefix] += 1
            sequences[prefix].add((scene_id, str(record["sequence"])))
    return OrderedDict(
        (
            prefix,
            {
                "num_images": image_counts[prefix],
                "num_sequences": len(sequences[prefix]),
            },
        )
        for prefix in DIRECTIONAL_POOL_PREFIXES
    )


def empty_pool_metric_values(enabled):
    if not enabled:
        return {}
    return {
        prefix: {"prediction": empty_metric_lists(), "input": empty_metric_lists()}
        for prefix in DIRECTIONAL_POOL_PREFIXES
    }


def resolve_records(manifest, prediction_dir):
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
    return grouped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate PSNR, LPIPS, and SSIM on one GPU or multiple torchrun workers."
    )
    parser.add_argument("--prediction_dir", "--output_dir", dest="prediction_dir", required=True)
    parser.add_argument("--prediction_manifest", default=None)
    parser.add_argument("--metrics_output", default=None)
    parser.add_argument("--device", default="cuda:0", help="Single-process device; torchrun uses cuda:LOCAL_RANK.")
    parser.add_argument("--calculate-for-input", action="store_true")
    parser.add_argument(
        "--group-by-prefix",
        action="store_true",
        help=(
            "Also compute independent pooled PSNR/LPIPS/SSIM for sequence-folder basenames beginning with D, L, R, and U."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def calculate_local_metrics(args, context, local_scenes):
    if not local_scenes:
        return {}, {}, empty_pool_metric_values(args.group_by_prefix)
    metric_kwargs = {"sync_on_compute": False, "dist_sync_on_step": False}
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0, **metric_kwargs).to(context.device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True, **metric_kwargs).to(
        context.device
    )
    scene_metrics, scene_values = {}, {}
    pool_values = empty_pool_metric_values(args.group_by_prefix)
    for scene_id, records in local_scenes:
        rank_print(context, f"Reconstruction metrics for scene {scene_id}: {len(records)} images")
        values, input_values = empty_metric_lists(), empty_metric_lists()

        # Enabled tqdm progress display with dynamic live statistics
        pbar = tqdm(
            records,
            desc=f"[Rank {context.rank}] {scene_id}" if context.is_distributed else scene_id,
            position=context.rank if context.is_distributed else 0,
            leave=not context.is_distributed,
        )
        for record in pbar:
            with Image.open(record["resolved_prediction_path"]) as handle:
                prediction_image = handle.convert("RGB")
            with Image.open(record["target_path"]) as handle:
                target_image = handle.convert("RGB")
            prediction_metrics = pair_metrics(prediction_image, target_image, psnr_metric, lpips_metric, context.device)
            append_metrics(values, prediction_metrics)

            # Update live stats on terminal output
            pbar.set_postfix(
                psnr=f"{np.mean(values['psnr']):.2f}dB",
                lpips=f"{np.mean(values['lpips']):.4f}",
                ssim=f"{np.mean(values['ssim']):.4f}",
            )

            prefix = sequence_pool(record["sequence"]) if args.group_by_prefix else None
            if prefix is not None:
                append_metrics(pool_values[prefix]["prediction"], prediction_metrics)
            if args.calculate_for_input:
                with Image.open(record["input_path"]) as handle:
                    input_image = handle.convert("RGB")
                input_metrics = pair_metrics(input_image, target_image, psnr_metric, lpips_metric, context.device)
                append_metrics(input_values, input_metrics)
                if prefix is not None:
                    append_metrics(pool_values[prefix]["input"], input_metrics)
        output = {**summarize_metrics(values), "num_images": len(records)}
        if args.calculate_for_input:
            output["input"] = summarize_metrics(input_values)
        scene_metrics[scene_id] = output
        scene_values[scene_id] = values
        if args.calculate_for_input:
            scene_values[scene_id + "\0input"] = input_values
    del psnr_metric, lpips_metric
    if context.device.type == "cuda":
        torch.cuda.empty_cache()
    return scene_metrics, scene_values, pool_values


def write_rank_payload(path, payload):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)


def load_and_merge_rank_payloads(metric_cache, world_size, pool_prefixes):
    scene_metrics, scene_values = {}, {}
    pool_values = {
        prefix: {"prediction": empty_metric_lists(), "input": empty_metric_lists()} for prefix in pool_prefixes
    }
    for rank in range(world_size):
        with (metric_cache / f"rank_{rank:05d}.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        overlap = set(scene_metrics).intersection(payload["scene_metrics"])
        if overlap:
            raise RuntimeError(f"Scenes were evaluated on multiple ranks: {sorted(overlap)}")
        scene_metrics.update(payload["scene_metrics"])
        scene_values.update(payload["scene_values"])
        for prefix in pool_prefixes:
            extend_metrics(
                pool_values[prefix]["prediction"],
                payload["pool_values"][prefix]["prediction"],
            )
            extend_metrics(pool_values[prefix]["input"], payload["pool_values"][prefix]["input"])
    return scene_metrics, scene_values, pool_values


def build_final_results(args, context, manifest, grouped_records, metric_cache, pool_specs):
    unordered_metrics, scene_values, pool_values = load_and_merge_rank_payloads(
        metric_cache, context.world_size, pool_specs.keys()
    )
    expected_scenes = set(grouped_records)
    if set(unordered_metrics) != expected_scenes:
        missing = sorted(expected_scenes - set(unordered_metrics))
        extra = sorted(set(unordered_metrics) - expected_scenes)
        raise RuntimeError(f"Distributed scene merge mismatch; missing={missing}, extra={extra}")
    per_scene = OrderedDict((scene_id, unordered_metrics[scene_id]) for scene_id in grouped_records)
    overall_values, overall_input_values = empty_metric_lists(), empty_metric_lists()
    for scene_id in grouped_records:
        extend_metrics(overall_values, scene_values[scene_id])
        if args.calculate_for_input:
            extend_metrics(overall_input_values, scene_values[scene_id + "\0input"])
    config = manifest["config"]
    overall = {
        **summarize_metrics(overall_values),
        "total_images": sum(len(records) for records in grouped_records.values()),
        "split": manifest["split"],
        "resolution": config["resolution"],
        "num_gpus": manifest["num_gpus"],
        "metric_num_gpus": context.world_size,
        "gpu_memory": manifest["inference_summary"]["gpu_memory"],
    }
    latency = manifest["inference_summary"].get("speed_latency_per_sample")
    if latency is not None:
        overall["speed_benchmark_latency_per_sample"] = float(latency)
        overall["speed_benchmark_throughput_samples_per_sec"] = float(1.0 / latency)
    if args.calculate_for_input:
        overall["input"] = summarize_metrics(overall_input_values)
    pools = OrderedDict()
    for prefix, pool_spec in pool_specs.items():
        pool_result = {
            **summarize_metrics(pool_values[prefix]["prediction"]),
            **pool_spec,
        }
        if args.calculate_for_input:
            pool_result["input"] = summarize_metrics(pool_values[prefix]["input"])
        pools[prefix] = pool_result
    return overall, per_scene, pools


def format_mean_std(values, name, unit=""):
    mean, std = values[f"{name}_mean"], values[f"{name}_std"]
    if mean is None or std is None:
        return "None"
    return f"{mean:.4f} +/- {std:.4f}{unit}"


def print_overall(overall, pools, metrics_path):
    print("\nOverall reconstruction results")
    print(f"  Metric GPUs: {overall['metric_num_gpus']}")
    print(f"  Images: {overall['total_images']}")
    print(f"  PSNR: {format_mean_std(overall, 'psnr', ' dB')}")
    print(f"  LPIPS: {format_mean_std(overall, 'lpips')}")
    print(f"  SSIM: {format_mean_std(overall, 'ssim')}")
    for prefix, values in pools.items():
        print(f"\n{prefix} sequence pool ({values['num_sequences']} sequences)")
        print(f"  Images: {values['num_images']}")
        print(f"  PSNR: {format_mean_std(values, 'psnr', ' dB')}")
        print(f"  LPIPS: {format_mean_std(values, 'lpips')}")
        print(f"  SSIM: {format_mean_std(values, 'ssim')}")
    print(f"Metrics saved to {metrics_path}")


def main():
    args = parse_args()
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
        metrics_path = Path(args.metrics_output or prediction_dir / "reconstruction_metrics.yaml").resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")
        with manifest_path.open() as handle:
            manifest = json.load(handle)

        grouped_records = resolve_records(manifest, prediction_dir)
        pool_specs = build_prefix_pool_specs(grouped_records, args.group_by_prefix)
        local_scenes = assigned_scenes(grouped_records, context)
        rank_print(
            context,
            f"Assigned {len(local_scenes)} of {len(grouped_records)} scenes on {context.device}",
        )
        metric_cache = prediction_dir / "_reconstruction_metric_cache"
        if context.is_main:
            if metric_cache.exists():
                shutil.rmtree(metric_cache)
            metric_cache.mkdir(parents=True, exist_ok=True)
        barrier(context)
        scene_metrics, scene_values, pool_values = calculate_local_metrics(args, context, local_scenes)
        write_rank_payload(
            metric_cache / f"rank_{context.rank:05d}.pkl",
            {
                "scene_metrics": scene_metrics,
                "scene_values": scene_values,
                "pool_values": pool_values,
            },
        )
        barrier(context)
        if context.is_main:
            overall, per_scene, pools = build_final_results(
                args, context, manifest, grouped_records, metric_cache, pool_specs
            )
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            output = {"overall": overall}
            if pools:
                output["pools"] = dict(pools)
            output["per_scene"] = dict(per_scene)
            with metrics_path.open("w") as handle:
                yaml.safe_dump(output, handle, sort_keys=False)
            print_overall(overall, pools, metrics_path)
        barrier(context)
        if context.is_main:
            shutil.rmtree(metric_cache)
        barrier(context)
    finally:
        if context.is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
