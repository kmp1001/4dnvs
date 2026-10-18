# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target-relative camera normalization and per-view Plücker rays."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .prope import invert_se3


def _build_plucker_rays(
    relative_c2ws: torch.Tensor,
    normalized_Ks: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Build ``[origin x direction, direction]`` rays in the target frame.

    Args:
        relative_c2ws: Camera-to-target-frame transforms ``[B,V,4,4]``.
        normalized_Ks: Centered normalized intrinsics ``[B,V,3,3]``.

    Returns:
        Full-resolution Plücker ray maps ``[B,V,H,W,6]``.
    """
    if relative_c2ws.ndim != 4 or relative_c2ws.shape[-2:] != (4, 4):
        raise ValueError(
            f"Expected relative_c2ws [B,V,4,4], got {tuple(relative_c2ws.shape)}"
        )
    if normalized_Ks.shape != relative_c2ws.shape[:-2] + (3, 3):
        raise ValueError(
            f"Expected normalized_Ks {relative_c2ws.shape[:-2] + (3, 3)}, "
            f"got {tuple(normalized_Ks.shape)}"
        )

    pixel_x = (
        torch.arange(
            image_width,
            device=relative_c2ws.device,
            dtype=torch.float32,
        )
        + 0.5
    ) / image_width - 0.5
    pixel_y = (
        torch.arange(
            image_height,
            device=relative_c2ws.device,
            dtype=torch.float32,
        )
        + 0.5
    ) / image_height - 0.5
    grid_y, grid_x = torch.meshgrid(pixel_y, pixel_x, indexing="ij")
    grid_x = grid_x[None, None]
    grid_y = grid_y[None, None]

    fx = normalized_Ks[..., 0, 0, None, None]
    fy = normalized_Ks[..., 1, 1, None, None]
    cx = normalized_Ks[..., 0, 2, None, None]
    cy = normalized_Ks[..., 1, 2, None, None]
    directions_camera = torch.stack(
        (
            (grid_x - cx) / fx,
            (grid_y - cy) / fy,
            torch.ones_like(grid_x).expand(
                relative_c2ws.shape[0],
                relative_c2ws.shape[1],
                -1,
                -1,
            ),
        ),
        dim=-1,
    )
    directions_camera = F.normalize(directions_camera, dim=-1)
    directions_relative = torch.einsum(
        "bvhwc,bvdc->bvhwd",
        directions_camera,
        relative_c2ws[..., :3, :3],
    )
    origins_relative = relative_c2ws[..., None, None, :3, 3].expand_as(
        directions_relative
    )
    moments = torch.linalg.cross(
        origins_relative,
        directions_relative,
        dim=-1,
    )
    return torch.cat((moments, directions_relative), dim=-1)


def build_relative_camera_conditioning(
    target_c2w: torch.Tensor,
    target_K: torch.Tensor,
    reference_c2ws: torch.Tensor,
    reference_Ks: torch.Tensor,
    image_height: int,
    image_width: int,
    translation_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Express target views and references in the current target-camera frame.

    ``target_c2w``/``target_K`` may contain one target view ``[B,...]`` or a
    temporal target sequence ``[B,V,...]``. View 0 is always the current frame
    and anchors the coordinate system. ``target_K`` and ``reference_Ks`` follow
    ArtiFixer's normalized convention:
    focal lengths are divided by image dimensions and principal points are
    divided by image dimensions then shifted by ``-0.5``.
    """
    if target_c2w.ndim == 3:
        target_c2w = target_c2w[:, None]
    if target_K.ndim == 3:
        target_K = target_K[:, None]
    if target_c2w.ndim != 4 or target_c2w.shape[-2:] != (4, 4):
        raise ValueError(
            f"Expected target_c2w [B,V,4,4] or [B,4,4], got {tuple(target_c2w.shape)}"
        )
    if target_K.shape != target_c2w.shape[:-2] + (3, 3):
        raise ValueError(
            f"Expected target_K {target_c2w.shape[:-2] + (3, 3)}, "
            f"got {tuple(target_K.shape)}"
        )
    if reference_c2ws.ndim != 4 or reference_c2ws.shape[-2:] != (4, 4):
        raise ValueError(f"Expected reference_c2ws [B,R,4,4], got {tuple(reference_c2ws.shape)}")
    if reference_Ks.shape != reference_c2ws.shape[:-2] + (3, 3):
        raise ValueError(
            f"Expected reference_Ks {reference_c2ws.shape[:-2] + (3, 3)}, "
            f"got {tuple(reference_Ks.shape)}"
        )
    if reference_c2ws.shape[0] != target_c2w.shape[0]:
        raise ValueError("Target and reference camera batches must match")
    if reference_c2ws.shape[1] == 0:
        raise ValueError("At least one reference camera is required")
    devices = {target_c2w.device, target_K.device, reference_c2ws.device, reference_Ks.device}
    if len(devices) != 1:
        raise ValueError(f"All camera tensors must share one device, got {devices}")
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {(image_height, image_width)}")
    if not math.isfinite(translation_scale) or translation_scale <= 0:
        raise ValueError(f"translation_scale must be positive, got {translation_scale}")

    # Camera matrices are geometry, so keep the calculations in float32 even
    # when the image/DiT path uses bf16.
    target_c2w = target_c2w.float().clone()
    reference_c2ws = reference_c2ws.float().clone()
    target_K = target_K.float()
    reference_Ks = reference_Ks.float()
    target_c2w[..., :3, 3] *= translation_scale
    reference_c2ws[..., :3, 3] *= translation_scale

    anchor_w2c = invert_se3(target_c2w[:, 0])
    target_relative_c2w = torch.matmul(anchor_w2c[:, None], target_c2w)
    reference_relative_c2ws = torch.matmul(anchor_w2c[:, None], reference_c2ws)
    target_viewmats = invert_se3(target_relative_c2w)
    reference_viewmats = invert_se3(reference_relative_c2ws)

    # Both target and reference ray maps are expressed in the target-camera
    # coordinate frame. Target ray origins are therefore zero, while reference
    # ray origins retain the camera baseline needed by the token conditioner.
    target_camera_rays = _build_plucker_rays(
        target_relative_c2w,
        target_K,
        image_height,
        image_width,
    )
    reference_camera_rays = _build_plucker_rays(
        reference_relative_c2ws,
        reference_Ks,
        image_height,
        image_width,
    )

    return {
        "target_camera_rays": target_camera_rays,
        "reference_camera_rays": reference_camera_rays,
        "target_viewmats": target_viewmats,
        "target_Ks": target_K,
        "reference_viewmats": reference_viewmats,
        "reference_Ks": reference_Ks,
    }
