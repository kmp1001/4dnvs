# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the test dataset with scene-parallel multi-GPU inference.

This script performs inference only. It deliberately does not import or run
Clean-FID, TorchMetrics, or either FVD network inside spawned GPU workers.
After every worker completes, it writes ``evaluation_manifest.json`` in the
output directory. Run ``calculate_test_metrics.py`` on that manifest to
calculate PSNR, LPIPS, SSIM, FID, KID, and PyTorch FVD in one process.

The inference path matches ``inference_drivenvs.py``:

* the first four frames of a sequence use ``[current warped]`` (V=1);
* later frames use ``[current warped, pred(t-1), ..., pred(t-4)]`` (V=5);
* the highest-visibility reference joins target self-attention;
* remaining references are encoded as reference K/V memory;
* camera and signed-time conditioning are read from the dataset manifest; and
* complete scenes stay on one GPU so temporal history is never split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from utils.inference_utils import (
    RESOLUTION_MAP,
    load_manifest_split,
    prepare_inference_sample,
    record_prediction,
    safe_output_path,
)
from utils.temporal import temporal_frame_index, temporal_sequence_key


def scene_name(sample_key: str, entry: dict) -> str:
    explicit_scene = str(entry.get("scene", "")).strip()
    if explicit_scene:
        return explicit_scene
    relative = Path(str(entry.get("relative_path", entry.get("frame_name", ""))))
    if len(relative.parts) > 1 and relative.parts[0] not in {".", ".."}:
        return relative.parts[0]
    return "default"


def run_bridge_sample(
    model: DrivenVSModel,
    prepared: dict,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> torch.Tensor:
    return model.sample_bridge(
        prepared["target_path"],
        reference_images=prepared["references"],
        opacity=prepared["opacity"],
        num_steps=args.num_inference_steps,
        timestep_shift=args.timestep_shift,
        generator=generator,
        **prepared["conditioning"],
    ).float()


def prediction_to_pil(prediction: torch.Tensor) -> Image.Image:
    image_tensor = prediction[0, :, 0].cpu().mul(0.5).add(0.5).clamp(0.0, 1.0)
    return transforms.ToPILImage()(image_tensor)


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def benchmark_bridge(
    model: DrivenVSModel,
    prepared: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    for _ in tqdm(range(args.warmup_iters), desc="Speed warmup", leave=False):
        run_bridge_sample(model, prepared, args, generator)

    torch.cuda.synchronize(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in tqdm(range(args.speed_test_iters), desc="Speed test", leave=False):
        run_bridge_sample(model, prepared, args, generator)
    end.record()
    torch.cuda.synchronize(device=device)
    return start.elapsed_time(end) / 1000.0 / args.speed_test_iters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reference-conditioned predictions with scene-parallel multi-GPU inference."
    )
    parser.add_argument(
        "--dataset_json",
        "--input",
        dest="dataset_json",
        required=True,
        help="Training-format JSON manifest.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--model_path",
        "--model",
        dest="model_path",
        required=True,
        help="Reference-conditioned .pkl checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        "--output",
        dest="output_dir",
        required=True,
    )
    parser.add_argument("--num_reference_images", type=int, required=True)
    parser.add_argument("--world_scenario_mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument(
        "--meanflow",
        action="store_true",
        help="Load a Stage-II MeanFlow checkpoint; one step evaluates u(z_1,0,1).",
    )
    parser.add_argument(
        "--meanflow_use_ema",
        action="store_true",
        help="Use the CPU-maintained EMA parameters saved in a MeanFlow checkpoint.",
    )
    parser.add_argument(
        "--resampling_history_kv",
        action="store_true",
        help="Mode-A resampling checkpoint: use causal per-layer history K/V and V=1 target input.",
    )
    parser.add_argument("--timestep_shift", type=float, default=5.0)
    parser.add_argument("--camera_embed", action="store_true")
    parser.add_argument("--time_embed", action="store_true")
    parser.add_argument(
        "--nontemporal",
        action="store_true",
        help="Disable autoregressive history injection and run V=1 for every frame.",
    )
    parser.add_argument("--camera_translation_scale", type=float, default=1.0)
    parser.add_argument("--reference_time_embedding_dim", type=int, default=32)
    parser.add_argument("--reference_time_scale", type=float, default=1.0)
    parser.add_argument(
        "--resolution",
        type=str,
        default="resize_nusc",
        choices=tuple(RESOLUTION_MAP),
    )
    parser.add_argument("--vae_skip_connection", "--vae-skip-connection", action="store_true")
    parser.add_argument("--start_sample", "--start-frame", type=int, default=0)
    parser.add_argument("--sample_stride", "--skip-frames", type=int, default=1)
    parser.add_argument("--max_samples", "--max-frames", type=int, default=-1)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic evaluation and opacity-region Gaussian endpoints.",
    )
    parser.add_argument("--test-speed", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--speed-test-iters", type=int, default=50)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of visible GPUs to use. 0 uses every visible GPU.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.resampling_history_kv and (not args.meanflow or args.num_inference_steps != 1):
        raise ValueError("--resampling_history_kv requires --meanflow --num_inference_steps 1")
    if args.resampling_history_kv and args.nontemporal:
        raise ValueError("History K/V inference cannot be combined with --nontemporal")
    if args.meanflow_use_ema and not args.meanflow:
        raise ValueError("--meanflow_use_ema requires --meanflow")
    if args.num_reference_images < 1:
        raise ValueError("--num_reference_images must be positive")
    if args.num_inference_steps < 1:
        raise ValueError("--num_inference_steps must be positive")
    if not math.isfinite(args.timestep_shift) or args.timestep_shift <= 0:
        raise ValueError("--timestep_shift must be positive")
    if args.sample_stride < 1:
        raise ValueError("--sample_stride must be positive")
    if args.start_sample < 0:
        raise ValueError("--start_sample must be non-negative")
    if args.warmup_iters < 0 or args.speed_test_iters < 1:
        raise ValueError("--warmup-iters must be non-negative and --speed-test-iters positive")
    if args.camera_embed and (not math.isfinite(args.camera_translation_scale) or args.camera_translation_scale <= 0):
        raise ValueError("--camera_translation_scale must be positive")
    if args.time_embed and (args.reference_time_embedding_dim <= 0 or args.reference_time_embedding_dim % 2):
        raise ValueError("--reference_time_embedding_dim must be a positive even integer")
    if args.time_embed and (not math.isfinite(args.reference_time_scale) or args.reference_time_scale <= 0):
        raise ValueError("--reference_time_scale must be positive")
    if args.num_gpus < 0:
        raise ValueError("--num-gpus must be non-negative")


def partition_scenes(
    grouped_samples: OrderedDict[str, list[tuple[str, dict]]],
    world_size: int,
) -> list[list[tuple[str, list[tuple[str, dict]]]]]:
    """Split ordered scenes into contiguous, balanced per-rank partitions."""
    scene_items = list(grouped_samples.items())
    base_count, remainder = divmod(len(scene_items), world_size)
    partitions = []
    start = 0
    for rank in range(world_size):
        count = base_count + (1 if rank < remainder else 0)
        partitions.append(scene_items[start : start + count])
        start += count
    return partitions


def stable_scene_seed(seed: int, scene_id: str) -> int:
    """Make inference noise independent of which GPU receives a scene."""
    digest = hashlib.sha1(scene_id.encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:8], byteorder="big")) % (2**63 - 1)


def inference_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    grouped_samples: OrderedDict[str, list[tuple[str, dict]]],
    output_dir_string: str,
) -> None:
    """Run inference for complete scenes assigned to one logical CUDA device."""
    torch.cuda.set_device(rank)
    from drivenvs_model import DrivenVSModel
    from meanflow_model import MeanFlowDrivenVSModel
    from resampling_forcing_model import ResamplingMeanFlowModel

    device = torch.device("cuda", rank)
    set_random_seed(args.seed + rank)
    torch.set_grad_enabled(False)

    assigned_scenes = partition_scenes(grouped_samples, world_size)[rank]
    output_dir = Path(output_dir_string)
    partial_dir = output_dir / "_inference_workers"
    dtype = torch.bfloat16
    size = RESOLUTION_MAP[args.resolution]
    torch.cuda.reset_peak_memory_stats(device=device)
    torch.cuda.empty_cache()

    print(f"[GPU {rank}] Loading reference-conditioned model from {args.model_path}")
    if args.resampling_history_kv:
        model_class = ResamplingMeanFlowModel
        checkpoint_kwargs = {
            "mode": "a",
            "resampling_checkpoint": args.model_path,
            "use_ema": args.meanflow_use_ema,
        }
    elif args.meanflow:
        model_class = MeanFlowDrivenVSModel
        checkpoint_kwargs = {"meanflow_checkpoint": args.model_path, "meanflow_use_ema": args.meanflow_use_ema}
    else:
        model_class = DrivenVSModel
        checkpoint_kwargs = {"pretrained_path": args.model_path}
    model = model_class(
        **checkpoint_kwargs,
        train_full_unet=True,
        freeze_vae=True,
        vae_skip_connection=args.vae_skip_connection,
        num_reference_images=args.num_reference_images,
        world_scenario_mode=args.world_scenario_mode,
        camera_embed=args.camera_embed,
        time_embed=args.time_embed,
        camera_translation_scale=args.camera_translation_scale,
        reference_time_embedding_dim=args.reference_time_embedding_dim,
        reference_time_scale=args.reference_time_scale,
        enforce_conditioning_config=True,
        batch_size=1,
        device=device,
        dtype=dtype,
    )
    model.set_eval()
    model = model.to(device=device, dtype=dtype)

    speed_latency = None
    if args.test_speed and rank == 0:
        first_key, first_entry = assigned_scenes[0][1][0]
        first_prepared = prepare_inference_sample(
            first_key,
            first_entry,
            manifest_path=args.dataset_json,
            num_reference_images=args.num_reference_images,
            size=size,
            device=device,
            dtype=dtype,
            camera_embed=args.camera_embed,
            time_embed=args.time_embed,
            world_scenario=args.world_scenario_mode > 0,
            prediction_history=[],
            temporal=False,
        )
        speed_latency = benchmark_bridge(model, first_prepared, args, device)
        print(f"[GPU {rank}] Speed: {speed_latency:.4f}s/sample ({1.0 / speed_latency:.2f} samples/s)")

    records = []
    for scene_id, scene_samples in assigned_scenes:
        print(f"\n[GPU {rank}] Processing scene {scene_id}: {len(scene_samples)} samples")
        noise_generator = torch.Generator(device=device).manual_seed(stable_scene_seed(args.seed, scene_id))
        temporal_histories: dict[str, list[dict]] = {}
        history_kv_by_sequence = {}
        last_frame_by_sequence = {}

        for sample_key, entry in tqdm(scene_samples, desc=f"GPU {rank}: {scene_id}"):
            sequence_id = temporal_sequence_key(entry, sample_key)
            prediction_history = temporal_histories.setdefault(sequence_id, [])
            frame_index = temporal_frame_index(entry, sample_key)
            if args.resampling_history_kv:
                prior = last_frame_by_sequence.get(sequence_id)
                if frame_index is None or (prior is not None and frame_index != prior + 1):
                    history_kv_by_sequence.pop(sequence_id, None)
                last_frame_by_sequence[sequence_id] = frame_index
            prepared = prepare_inference_sample(
                sample_key,
                entry,
                manifest_path=args.dataset_json,
                num_reference_images=args.num_reference_images,
                size=size,
                device=device,
                dtype=dtype,
                camera_embed=args.camera_embed,
                time_embed=args.time_embed,
                world_scenario=args.world_scenario_mode > 0,
                prediction_history=prediction_history,
                temporal=not (args.nontemporal or args.resampling_history_kv),
            )
            if args.resampling_history_kv:
                cache = history_kv_by_sequence.get(sequence_id, model.empty_history_cache())
                model.set_history_cache(cache)
                try:
                    prediction = run_bridge_sample(model, prepared, args, noise_generator)
                finally:
                    model.clear_history_cache()
                clean_latent = model.vae_encode(prediction.to(dtype=dtype)).detach()
                history_kv_by_sequence[sequence_id] = model.append_clean_history_kv(
                    clean_latent,
                    prepared["references"],
                    prepared["conditioning"],
                    cache,
                )
            else:
                prediction = run_bridge_sample(model, prepared, args, noise_generator)
            output_image = prediction_to_pil(prediction)
            output_path = safe_output_path(output_dir, sample_key, entry)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_image.save(output_path)
            record_prediction(prediction_history, prediction, prepared, dtype)

            records.append(
                {
                    "sample_key": str(sample_key),
                    "scene": str(scene_id),
                    "sequence": str(sequence_id),
                    "frame_index": int(frame_index) if frame_index is not None else None,
                    "prediction_relative_path": str(output_path.relative_to(output_dir)),
                    "prediction_path": str(output_path.resolve()),
                    "target_path": str(Path(entry["target_image"]).resolve()),
                    "input_path": str(Path(entry["image"]).resolve()),
                }
            )

    partial = {
        "rank": rank,
        "records": records,
        "speed_latency": speed_latency,
        "gpu_memory": {
            "rank": rank,
            "peak_allocated_gb": float(torch.cuda.max_memory_allocated(device=device) / (1024**3)),
            "peak_reserved_gb": float(torch.cuda.max_memory_reserved(device=device) / (1024**3)),
        },
    }
    partial_dir.mkdir(parents=True, exist_ok=True)
    with open(partial_dir / f"rank_{rank:03d}.json", "w") as handle:
        json.dump(partial, handle, indent=2)


def write_evaluation_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    world_size: int,
    expected_samples: int,
) -> Path:
    """Merge rank-local records only after every inference worker succeeds."""
    partial_dir = output_dir / "_inference_workers"
    records = []
    gpu_memories = []
    speed_latency = None

    for rank in range(world_size):
        partial_path = partial_dir / f"rank_{rank:03d}.json"
        if not partial_path.is_file():
            raise RuntimeError(f"GPU worker {rank} did not write {partial_path}")
        with open(partial_path) as handle:
            partial = json.load(handle)
        if partial.get("rank") != rank:
            raise RuntimeError(f"Unexpected rank in {partial_path}: {partial.get('rank')}")
        records.extend(partial["records"])
        gpu_memories.append(partial["gpu_memory"])
        if partial["speed_latency"] is not None:
            speed_latency = float(partial["speed_latency"])

    if len(records) != expected_samples:
        raise RuntimeError(f"Inference manifest has {len(records)} records; expected {expected_samples}")
    missing = [record["prediction_path"] for record in records if not Path(record["prediction_path"]).is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} generated predictions; first missing: {missing[0]}")

    manifest = {
        "format_version": 1,
        "complete": True,
        "dataset_json": str(Path(args.dataset_json).resolve()),
        "output_dir": str(output_dir.resolve()),
        "split": args.split,
        "num_samples": len(records),
        "num_gpus": world_size,
        "config": {
            "model_path": str(Path(args.model_path).resolve()),
            "num_reference_images": args.num_reference_images,
            "num_inference_steps": args.num_inference_steps,
            "meanflow": args.meanflow,
            "world_scenario_mode": args.world_scenario_mode,
            "meanflow_use_ema": args.meanflow_use_ema,
            "timestep_shift": args.timestep_shift,
            "camera_embed": args.camera_embed,
            "time_embed": args.time_embed,
            "temporal_inference": not args.nontemporal,
            "resolution": args.resolution,
            "seed": args.seed,
            "start_sample": args.start_sample,
            "sample_stride": args.sample_stride,
            "max_samples": args.max_samples,
        },
        "inference_summary": {
            "speed_latency_per_sample": speed_latency,
            "gpu_memory": {
                "max_peak_allocated_gb": max(memory["peak_allocated_gb"] for memory in gpu_memories),
                "per_gpu": gpu_memories,
            },
        },
        "samples": records,
    }
    manifest_path = output_dir / "evaluation_manifest.json"
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    shutil.rmtree(partial_dir)
    return manifest_path


def main() -> None:
    args = parse_args()
    validate_args(args)

    samples = load_manifest_split(args.dataset_json, args.split)
    samples = samples[args.start_sample :: args.sample_stride]
    if args.max_samples >= 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No samples remain after split selection and slicing")

    grouped_samples: OrderedDict[str, list[tuple[str, dict]]] = OrderedDict()
    for sample_key, entry in samples:
        grouped_samples.setdefault(scene_name(sample_key, entry), []).append((sample_key, entry))
    for scene_samples in grouped_samples.values():
        scene_samples.sort(
            key=lambda item: (
                temporal_sequence_key(item[1], item[0]),
                temporal_frame_index(item[1], item[0]) if temporal_frame_index(item[1], item[0]) is not None else -1,
            )
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir / "_inference_workers"
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation inference")
    visible_gpus = torch.cuda.device_count()
    requested_gpus = args.num_gpus or visible_gpus
    if requested_gpus > visible_gpus:
        raise ValueError(f"Requested {requested_gpus} GPUs, but only {visible_gpus} are visible")
    world_size = min(requested_gpus, len(grouped_samples))

    partitions = partition_scenes(grouped_samples, world_size)
    print(f"Using {world_size} GPU process(es) for {len(grouped_samples)} scenes")
    for rank, partition in enumerate(partitions):
        scene_labels = ", ".join(scene_id for scene_id, _ in partition)
        sample_count = sum(len(scene_samples) for _, scene_samples in partition)
        print(f"  GPU {rank}: {len(partition)} scenes, {sample_count} samples [{scene_labels}]")

    worker_args = (world_size, args, grouped_samples, str(output_dir))
    if world_size == 1:
        inference_worker(0, *worker_args)
    else:
        mp.spawn(inference_worker, args=worker_args, nprocs=world_size, join=True)

    manifest_path = write_evaluation_manifest(
        args,
        output_dir,
        world_size,
        expected_samples=len(samples),
    )
    print(f"\nGenerated {len(samples)} predictions with {world_size} GPU process(es)")
    print(f"Predictions saved to {output_dir}")
    print(f"Evaluation manifest saved to {manifest_path}")
    print("Run calculate_test_metrics.py to calculate metrics.")


if __name__ == "__main__":
    main()
