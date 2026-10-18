# SPDX-License-Identifier: Apache-2.0
"""Shared mode-B curriculum operations, independent of CUDA-only Cosmos imports."""

from contextlib import contextmanager

import torch

from utils.flow_match import FlowMatchScheduler
from utils.temporal import TEMPORAL_HISTORY_FRAMES


def build_mode_b_paths(current, histories, gt_histories, history_cameras, *, camera_embed):
    """Keep generated conditioning distinct from real RAFT supervision."""
    source, references, target, opacity, _, condition = current
    if len(histories) != TEMPORAL_HISTORY_FRAMES or len(gt_histories) != TEMPORAL_HISTORY_FRAMES:
        raise ValueError("Mode B requires exactly four conditioning and GT histories")
    recent = list(reversed(histories))
    real_recent = list(reversed(gt_histories))
    source_path = torch.cat([source, *recent], dim=2)
    bridge_target = torch.cat([target, *recent], dim=2)
    gt_supervision = torch.cat([target, *real_recent], dim=2)
    opacity_path = torch.cat([opacity, *[torch.ones_like(opacity) for _ in recent]], dim=2)
    condition = dict(condition)
    if "world_scenario" in condition:
        condition["world_scenario"] = torch.cat(
            [condition["world_scenario"], *[item["world_scenario"] for item in reversed(history_cameras)]], dim=2
        )
    if camera_embed:
        if len(history_cameras) != TEMPORAL_HISTORY_FRAMES:
            raise ValueError("Mode B requires four history camera records")
        for key in ("target_c2w", "target_K"):
            condition[key] = torch.cat([condition[key], *[item[key] for item in reversed(history_cameras)]], dim=1)
    return source_path, references, bridge_target, opacity_path, condition, gt_supervision


@torch.no_grad()
def resample_rendered_history(model, prepared, t_s, *, num_steps=1):
    """Denoise one GT/render-opacity bridge; mode B has no temporal cache."""
    rendered, references, gt, opacity, _, condition = prepared
    if rendered.shape[2] != 1:
        raise ValueError("History resampling must receive one frame")
    batch, channels, _, height, width = rendered.shape
    clear = model._set_reference_conditioning(
        references,
        batch_size=batch,
        channels=channels,
        height=height,
        width=width,
        target_c2w=condition.get("target_c2w"),
        target_K=condition.get("target_K"),
        reference_c2ws=condition.get("reference_c2ws"),
        reference_Ks=condition.get("reference_Ks"),
        reference_time_offsets=condition.get("reference_time_offsets"),
        reference_visibility_scores=condition.get("reference_visibility_scores"),
        world_scenario=condition.get("world_scenario"),
        reference_world_scenario=condition.get("reference_world_scenario"),
    )
    try:
        x0 = model.vae_encode(gt).detach()
        rendered_latent = model.vae_encode(rendered).detach()
        coverage = model.prepare_opacity_latent(opacity, rendered_latent)
        endpoint, _ = model.prepare_opacity_source_endpoint(rendered_latent, coverage)
        z = FlowMatchScheduler.add_noise(x0.float(), endpoint.float(), t_s)
        times = torch.linspace(1, 0, num_steps + 1, device=t_s.device)
        for index in range(num_steps):
            t, r = t_s * times[index], t_s * times[index + 1]
            u = model._predict_meanflow(z, r=r, t=t)
            z = z - (t - r).view(batch, 1, 1, 1, 1) * u
        return model.vae_decode(z).detach().to(dtype=rendered.dtype)
    finally:
        model.unet.dit.clear_meanflow_conditioning()
        if clear:
            model.unet.dit.clear_reference_latents()
            model.unet.dit.clear_camera_conditioning()


@torch.no_grad()
def rollout_rendered_history(model, prepared_histories, *, num_steps=1):
    """Generate histories chronologically from inference endpoints, without GT mixing."""
    predictions = []
    for index, prepared in enumerate(prepared_histories):
        rendered, references, _, opacity, _, condition = prepared
        condition = dict(condition)
        if predictions:
            rendered = torch.cat([rendered, *reversed(predictions)], dim=2)
            opacity = torch.cat([opacity, *[torch.ones_like(opacity) for _ in predictions]], dim=2)
            if "world_scenario" in condition:
                condition["world_scenario"] = torch.cat(
                    [
                        condition["world_scenario"],
                        *[item[5]["world_scenario"] for item in reversed(prepared_histories[:index])],
                    ],
                    dim=2,
                )
            for key in ("target_c2w", "target_K"):
                if condition.get(key) is not None:
                    condition[key] = torch.cat(
                        [condition[key], *[item[5][key] for item in reversed(prepared_histories[:index])]], dim=1
                    )
        prediction = model.sample_bridge(
            rendered,
            reference_images=references,
            opacity=opacity,
            num_steps=num_steps,
            **condition,
        )
        predictions.append(prediction[:, :, :1].detach().to(dtype=rendered.dtype))
    return predictions


@contextmanager
def temporary_history_weights(module, weights, *, include_frozen=False):
    """Swap CPU-held generator parameters only during no-grad history generation.

    Restore online parameters before building the training graph. This avoids
    an additional Cosmos/VAE model on GPU. Every training rank owns its EMA.
    """
    parameters = dict(module.named_parameters())
    expected = {name for name, p in parameters.items() if include_frozen or p.requires_grad}
    if set(weights) != expected:
        raise ValueError("History generator parameter names differ from the online model")
    for name, value in weights.items():
        if value.shape != parameters[name].shape:
            raise ValueError(f"History generator shape mismatch: {name}")
    saved = {name: parameters[name].detach().to("cpu").clone() for name in weights}
    was_training = module.training
    try:
        with torch.no_grad():
            for name, value in weights.items():
                parameters[name].copy_(value.to(parameters[name]))
        module.eval()
        with torch.no_grad():
            yield
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                parameters[name].copy_(value.to(parameters[name]))
        module.train(was_training)
