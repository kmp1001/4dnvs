# SPDX-License-Identifier: Apache-2.0
"""Training previews and stage-specific source-endpoint validation."""

from contextlib import nullcontext
import json
import random
from pathlib import Path

import torch
from torch.utils.data import default_collate
from torchvision.utils import save_image

from meanflow_curriculum import temporary_history_weights
from utils.batch_preparation import prepare_training_batch
from utils.temporal import TEMPORAL_HISTORY_FRAMES, select_consecutive_prediction_history


def save_previews(source, references, target, prediction, opacity, directory):
    """Save batch-one images; context contains history only, newest first."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    images = {
        "current_input": source[0, :, 0],
        "current_target": target[0, :, 0],
        "current_prediction": prediction[0, :, 0],
        "references": torch.cat(list(references[0]), dim=-1),
    }
    if source.shape[2] > 1:
        images["context"] = torch.cat(list(source[0, :, 1:].unbind(dim=1)), dim=-1)
    for name, tensor in images.items():
        save_image((tensor.detach().float().cpu() * 0.5 + 0.5).clamp(0, 1), directory / f"{name}.png")
    # Coverage is already in [0,1], unlike normalized RGB.
    save_image(opacity[0, :, 0].detach().float().cpu().clamp(0, 1), directory / "coverage.png")


def build_validation_dataset(cfg, *, autoregressive=False):
    from utils.training_utils import DrivenVSDataset

    teacher_forcing = cfg["stage"] == 2 and not autoregressive
    dataset = DrivenVSDataset(
        dataset_folder=cfg["validation_dataset"] or cfg["dataset"],
        split=cfg["validation_split"],
        image_prep=cfg["image_prep"],
        num_reference_images=cfg["references"],
        camera_embed=cfg["camera"],
        time_embed=cfg["time"],
        training_stage="stage2" if teacher_forcing else "stage1",
        temporal_sample_probability=1.0 if teacher_forcing else 0.0,
        world_scenario=cfg.get("world_scenario_mode", 0) > 0,
    )
    if teacher_forcing:
        dataset.img_names = list(dataset.temporal_eligible_names)
    dataset.img_names.sort(
        key=lambda name: (
            dataset.data[name]["_source_index"],
            dataset.data[name]["_sequence_key"],
            dataset.data[name]["_frame_index"] is None,
            dataset.data[name]["_frame_index"] or 0,
            name,
        )
    )
    if not len(dataset):
        raise ValueError("No validation samples for this stage/split")
    return dataset


def append_prediction_history(source, opacity, condition, history, *, frame_index, camera):
    """Use the same four-consecutive-predictions rule as inference."""
    recent = select_consecutive_prediction_history(history, frame_index)
    if not recent:
        return source, opacity, condition
    source = torch.cat([source, *[item["prediction"] for item in recent]], dim=2)
    opacity = torch.cat([opacity, *[torch.ones_like(opacity) for _ in recent]], dim=2)
    condition = dict(condition)
    if "world_scenario" in condition:
        condition["world_scenario"] = torch.cat(
            [condition["world_scenario"], *[item["world_scenario"] for item in recent]], dim=2
        )
    if camera:
        for key in ("target_c2w", "target_K"):
            condition[key] = torch.cat([condition[key], *[item[key] for item in recent]], dim=1)
    return source, opacity, condition


@torch.no_grad()
def run_validation(model, dataset, lpips_model, ema, cfg, *, device, dtype, directory, autoregressive=False):
    """Generate from source endpoints; GT is used only as the target or Stage-II context."""
    was_training = model.training
    model.eval()
    # Preserve training RNG streams, including the dataset's Python sampling.
    python_rng = random.getstate()
    cuda_devices = [device.index] if device.type == "cuda" else []
    metrics = []
    temporal_count = 0
    history = []
    sequence = None
    rollout = cfg["stage"] == 3 or autoregressive
    weights = temporary_history_weights(model.unet, ema.state_dict()) if ema else nullcontext()
    try:
        with torch.random.fork_rng(devices=cuda_devices), weights:
            generator = torch.Generator(device=device).manual_seed(cfg["seed"])
            for index in range(min(cfg["num_samples_eval"], len(dataset))):
                entry = dataset.data[dataset.img_names[index]]
                batch = default_collate([dataset[index]])
                source, references, target, opacity, _, condition = prepare_training_batch(
                    batch,
                    device=device,
                    dtype=dtype,
                    expected_reference_views=cfg["references"],
                    camera_embed=cfg["camera"],
                    time_embed=cfg["time"],
                )
                current_condition = condition
                if rollout:
                    current_sequence = (entry["_source_index"], entry["_sequence_key"])
                    if current_sequence != sequence:
                        history = []
                        sequence = current_sequence
                    source, opacity, condition = append_prediction_history(
                        source,
                        opacity,
                        condition,
                        history,
                        frame_index=entry["_frame_index"],
                        camera=cfg["camera"],
                    )
                prediction = model.sample_bridge(
                    source,
                    reference_images=references,
                    opacity=opacity,
                    num_steps=cfg["validation_steps"],
                    generator=generator,
                    **condition,
                )[:, :, :1]
                current_gt = target[:, :, 0].float()
                current_prediction = prediction[:, :, 0].float()
                metrics.append(
                    (
                        torch.nn.functional.mse_loss(current_prediction, current_gt).item(),
                        lpips_model(current_prediction, current_gt).mean().item(),
                    )
                )
                temporal_count += int(source.shape[2] > 1)
                sample_directory = Path(directory) / f"sample_{index:03d}"
                save_previews(source, references, target, prediction, opacity, sample_directory)
                (sample_directory / "sample.json").write_text(
                    json.dumps(
                        {
                            "sample_key": entry["_sample_key"],
                            "sequence": entry["_sequence_key"],
                            "frame_index": entry["_frame_index"],
                            "history": "none" if source.shape[2] == 1 else "generated" if rollout else "gt",
                            "use_ema": ema is not None,
                        },
                        indent=2,
                    )
                )
                if rollout:
                    history.append(
                        {
                            "frame_index": entry["_frame_index"],
                            "prediction": prediction.detach().to(dtype=dtype),
                            **(
                                {"world_scenario": current_condition["world_scenario"]}
                                if "world_scenario" in current_condition
                                else {}
                            ),
                            **{key: current_condition[key] for key in ("target_c2w", "target_K") if cfg["camera"]},
                        }
                    )
                    history = history[-TEMPORAL_HISTORY_FRAMES:]
    finally:
        random.setstate(python_rng)
        model.train(was_training)
        model._inner_vae().eval()
    return {
        "val/l2": sum(item[0] for item in metrics) / len(metrics),
        "val/lpips": sum(item[1] for item in metrics) / len(metrics),
        "val/num_samples": len(metrics),
        "val/temporal_samples": temporal_count,
        "val/num_steps": cfg["validation_steps"],
        "val/use_ema": int(ema is not None),
    }
