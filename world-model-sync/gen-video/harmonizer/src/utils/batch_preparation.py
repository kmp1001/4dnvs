# SPDX-License-Identifier: Apache-2.0
"""Canonical training batch layouts, with no Cosmos/CUDA imports."""

import torch
from einops import rearrange

from .temporal import TEMPORAL_TARGET_VIEWS


def get_batch_center_index(batch, expected_center):
    center_indices = batch["center_index"]
    if torch.is_tensor(center_indices):
        flat = center_indices.reshape(-1)
        center_index = int(flat[0].item())
        if not torch.all(flat == center_index):
            raise ValueError(f"All samples in a batch must share one center index, got {flat.tolist()}")
    else:
        center_index = int(center_indices)

    if center_index != expected_center:
        raise ValueError(f"Dataset center_index={center_index}, expected {expected_center}")
    return center_index


def prepare_training_batch(
    batch,
    device=None,
    dtype=None,
    expected_reference_views=8,
    camera_embed=False,
    time_embed=False,
):
    """Convert warped target, GS coverage, and references to model layouts."""
    x_src = batch["conditioning_pixel_values"]
    x_ref = batch["reference_pixel_values"]
    x_tgt = batch["output_pixel_values"]
    x_opacity = batch["opacity_pixel_values"]
    target_c2w = batch.get("target_c2w")
    target_K = batch.get("target_K")
    reference_c2ws = batch.get("reference_c2ws")
    reference_Ks = batch.get("reference_Ks")
    reference_time_offsets = batch.get("reference_time_offsets")
    reference_visibility_scores = batch.get("reference_visibility_scores")

    if device is not None:
        x_src = x_src.to(device=device, dtype=dtype)
        x_ref = x_ref.to(device=device, dtype=dtype)
        x_tgt = x_tgt.to(device=device, dtype=dtype)
        x_opacity = x_opacity.to(device=device, dtype=dtype)
        camera_tensors = {
            "target_c2w": target_c2w,
            "target_K": target_K,
            "reference_c2ws": reference_c2ws,
            "reference_Ks": reference_Ks,
        }
        moved_camera_tensors = {
            name: (value.to(device=device, dtype=torch.float32) if value is not None else None)
            for name, value in camera_tensors.items()
        }
        target_c2w = moved_camera_tensors["target_c2w"]
        target_K = moved_camera_tensors["target_K"]
        reference_c2ws = moved_camera_tensors["reference_c2ws"]
        reference_Ks = moved_camera_tensors["reference_Ks"]
        if reference_time_offsets is not None:
            reference_time_offsets = reference_time_offsets.to(device=device, dtype=torch.float32)
        if reference_visibility_scores is not None:
            reference_visibility_scores = reference_visibility_scores.to(device=device, dtype=torch.float32)

    if x_src.ndim != 5:
        raise ValueError(f"Expected conditioning [B,V,C,H,W], got {tuple(x_src.shape)}")
    if x_ref.ndim != 5:
        raise ValueError(f"Expected references [B,R,C,H,W], got {tuple(x_ref.shape)}")
    if x_tgt.ndim != 5:
        raise ValueError(f"Expected target sequence [B,V,C,H,W], got {tuple(x_tgt.shape)}")
    if x_opacity.ndim != 5:
        raise ValueError(f"Expected confident coverage [B,V,1,H,W], got {tuple(x_opacity.shape)}")

    batch_size, num_views, channels, height, width = x_src.shape
    if num_views not in {1, TEMPORAL_TARGET_VIEWS}:
        raise ValueError(f"Expected target-path V=1 or V={TEMPORAL_TARGET_VIEWS}, got V={num_views}")
    if x_ref.shape != (batch_size, expected_reference_views, channels, height, width):
        raise ValueError(
            f"Expected {expected_reference_views} references with shape "
            f"{(batch_size, expected_reference_views, channels, height, width)}, "
            f"got {tuple(x_ref.shape)}"
        )
    if x_tgt.shape != (batch_size, num_views, channels, height, width):
        raise ValueError(
            f"Target shape {tuple(x_tgt.shape)} does not match "
            f"conditioning shape {(batch_size, num_views, channels, height, width)}"
        )
    if x_opacity.shape != (batch_size, num_views, 1, height, width):
        raise ValueError(
            f"Coverage shape {tuple(x_opacity.shape)} does not match {(batch_size, num_views, 1, height, width)}"
        )
    model_conditioning = {}
    if "world_scenario_pixel_values" in batch:
        model_conditioning["world_scenario"] = rearrange(
            batch["world_scenario_pixel_values"].to(device=device, dtype=dtype), "b v c h w -> b c v h w"
        )
        model_conditioning["reference_world_scenario"] = batch["reference_world_scenario_pixel_values"].to(
            device=device, dtype=dtype
        )
    if reference_visibility_scores is None:
        raise KeyError("Dataset must provide reference_visibility_scores")
    if reference_visibility_scores.shape != (batch_size, expected_reference_views):
        raise ValueError(f"Expected reference_visibility_scores [B,R], got {tuple(reference_visibility_scores.shape)}")
    if not torch.isfinite(reference_visibility_scores).all():
        raise ValueError("reference_visibility_scores must be finite")
    model_conditioning["reference_visibility_scores"] = reference_visibility_scores

    if camera_embed:
        camera_values = {
            "target_c2w": target_c2w,
            "target_K": target_K,
            "reference_c2ws": reference_c2ws,
            "reference_Ks": reference_Ks,
        }
        missing = [name for name, value in camera_values.items() if value is None]
        if missing:
            raise KeyError(f"--camera_embed requires dataset values: {missing}")
        if target_c2w.shape != (batch_size, num_views, 4, 4):
            raise ValueError(f"Expected target_c2w [B,V,4,4], got {tuple(target_c2w.shape)}")
        if target_K.shape != (batch_size, num_views, 3, 3):
            raise ValueError(f"Expected target_K [B,V,3,3], got {tuple(target_K.shape)}")
        if reference_c2ws.shape != (batch_size, expected_reference_views, 4, 4):
            raise ValueError(f"Expected reference_c2ws [B,R,4,4], got {tuple(reference_c2ws.shape)}")
        if reference_Ks.shape != (batch_size, expected_reference_views, 3, 3):
            raise ValueError(f"Expected reference_Ks [B,R,3,3], got {tuple(reference_Ks.shape)}")
        model_conditioning.update(camera_values)

    if time_embed:
        if reference_time_offsets is None:
            raise KeyError("--time_embed requires reference_time_offsets")
        if reference_time_offsets.shape != (batch_size, expected_reference_views):
            raise ValueError(f"Expected reference_time_offsets [B,R], got {tuple(reference_time_offsets.shape)}")
        if not torch.isfinite(reference_time_offsets).all():
            raise ValueError("reference_time_offsets must be finite")
        temporal_units = batch.get("reference_time_unit")
        if temporal_units is not None:
            units = {temporal_units} if isinstance(temporal_units, str) else set(temporal_units)
            if len(units) != 1:
                raise ValueError(f"All samples in a batch must use one temporal unit, got {temporal_units}")
        model_conditioning["reference_time_offsets"] = reference_time_offsets
    # Target view 0 is warped(t); optional views 1..4 are clean past targets.
    # Source references remain [B,R,C,H,W].  The model reorders them by
    # visibility: top-1 enters joint self-attention and the remaining views are
    # visual K/V memory.
    center_index = get_batch_center_index(batch, expected_center=0)
    x_src = rearrange(x_src, "b v c h w -> b c v h w")
    x_tgt = rearrange(x_tgt, "b v c h w -> b c v h w")
    x_opacity = rearrange(x_opacity, "b v c h w -> b c v h w")
    return x_src, x_ref, x_tgt, x_opacity, center_index, model_conditioning
