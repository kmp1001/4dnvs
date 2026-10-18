# MIT License
#
# Copyright (c) Authors of "Cameras as Relative Positional Encoding"
# https://arxiv.org/pdf/2507.10496
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
"""PRoPE camera-relative attention transforms adapted from ArtiFixer."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch import nn


def invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert one or more rigid 4x4 camera transforms."""
    if transforms.shape[-2:] != (4, 4):
        raise ValueError(f"Expected [...,4,4] transforms, got {tuple(transforms.shape)}")
    rotation_inv = transforms[..., :3, :3].transpose(-1, -2)
    output = torch.zeros_like(transforms)
    output[..., :3, :3] = rotation_inv
    output[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation_inv, transforms[..., :3, 3])
    output[..., 3, 3] = 1.0
    return output


class PropeDotProductAttention(nn.Module):
    """Precompute the camera and spatial transforms used around attention."""

    def __init__(
        self,
        head_dim: int,
        patches_x: int = 0,
        patches_y: int = 0,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if head_dim % 4:
            raise ValueError(f"PRoPE requires head_dim divisible by 4, got {head_dim}")
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.freq_base = freq_base
        self.freq_scale = freq_scale
        self.cameras = 0
        self.apply_fn_q = None
        self.apply_fn_kv = None
        self.apply_fn_o = None

    def _set_nonpersistent_buffer(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value, persistent=False)

    def update_coeffs(self, patches_x: int, patches_y: int, device: torch.device) -> None:
        if patches_x <= 0 or patches_y <= 0:
            raise ValueError(f"Patch grid must be positive, got {(patches_y, patches_x)}")
        self.patches_x = patches_x
        self.patches_y = patches_y
        coeffs_x = _rope_precompute_coeffs(
            torch.arange(patches_x, device=device).repeat(patches_y),
            freq_base=self.freq_base,
            freq_scale=self.freq_scale,
            feat_dim=self.head_dim // 4,
        )
        coeffs_y = _rope_precompute_coeffs(
            torch.arange(patches_y, device=device).repeat_interleave(patches_x),
            freq_base=self.freq_base,
            freq_scale=self.freq_scale,
            feat_dim=self.head_dim // 4,
        )
        self._set_nonpersistent_buffer("coeffs_x_0", coeffs_x[0])
        self._set_nonpersistent_buffer("coeffs_x_1", coeffs_x[1])
        self._set_nonpersistent_buffer("coeffs_y_0", coeffs_y[0])
        self._set_nonpersistent_buffer("coeffs_y_1", coeffs_y[1])

    def precompute(self, viewmats: torch.Tensor, normalized_Ks: torch.Tensor) -> None:
        """Cache transforms for cameras expressed in one shared coordinate frame."""
        batch, cameras, _, _ = viewmats.shape
        if viewmats.shape != (batch, cameras, 4, 4):
            raise ValueError(f"Expected viewmats [B,C,4,4], got {tuple(viewmats.shape)}")
        if normalized_Ks.shape != (batch, cameras, 3, 3):
            raise ValueError(f"Expected normalized Ks [B,C,3,3], got {tuple(normalized_Ks.shape)}")
        if not hasattr(self, "coeffs_x_0"):
            raise RuntimeError("Call update_coeffs before precompute")
        self.cameras = cameras
        self.apply_fn_q, self.apply_fn_kv, self.apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            normalized_Ks=normalized_Ks,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
        )

    def _validate(self, tensor: torch.Tensor) -> None:
        batch, heads, sequence, head_dim = tensor.shape
        expected_sequence = self.cameras * self.patches_x * self.patches_y
        if sequence != expected_sequence or head_dim != self.head_dim:
            raise ValueError(
                f"PRoPE expected [B,H,{expected_sequence},{self.head_dim}], "
                f"got {tuple(tensor.shape)}"
            )

    def apply_to_q(self, query: torch.Tensor) -> torch.Tensor:
        self._validate(query)
        if self.apply_fn_q is None:
            raise RuntimeError("PRoPE target transforms are not prepared")
        return self.apply_fn_q(query)

    def apply_to_kv(self, key_or_value: torch.Tensor) -> torch.Tensor:
        self._validate(key_or_value)
        if self.apply_fn_kv is None:
            raise RuntimeError("PRoPE reference transforms are not prepared")
        return self.apply_fn_kv(key_or_value)

    def apply_to_o(self, output: torch.Tensor) -> torch.Tensor:
        self._validate(output)
        if self.apply_fn_o is None:
            raise RuntimeError("PRoPE target output transforms are not prepared")
        return self.apply_fn_o(output)


def _prepare_apply_fns(
    head_dim: int,
    viewmats: torch.Tensor,
    normalized_Ks: torch.Tensor,
    coeffs_x: tuple[torch.Tensor, torch.Tensor],
    coeffs_y: tuple[torch.Tensor, torch.Tensor],
) -> tuple[Callable, Callable, Callable]:
    batch, cameras, _, _ = viewmats.shape
    projection = torch.einsum("...ij,...jk->...ik", _lift_K(normalized_Ks), viewmats)
    projection_transpose = projection.transpose(-1, -2)
    projection_inverse = torch.einsum(
        "...ij,...jk->...ik",
        invert_se3(viewmats),
        _lift_K(_invert_K(normalized_Ks)),
    )
    if projection.shape != projection_inverse.shape or projection.shape != (batch, cameras, 4, 4):
        raise ValueError("Invalid PRoPE camera projection shapes")

    transforms_q = (
        (partial(_apply_tiled_projmat, matrix=projection_transpose), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    )
    transforms_kv = (
        (partial(_apply_tiled_projmat, matrix=projection_inverse), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    )
    transforms_o = (
        (partial(_apply_tiled_projmat, matrix=projection), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 4),
    )
    return (
        partial(_apply_block_diagonal, func_size_pairs=transforms_q),
        partial(_apply_block_diagonal, func_size_pairs=transforms_kv),
        partial(_apply_block_diagonal, func_size_pairs=transforms_o),
    )


def _apply_tiled_projmat(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    batch, heads, sequence, feat_dim = feats.shape
    cameras = matrix.shape[1]
    dimension = matrix.shape[-1]
    if sequence <= cameras or sequence % cameras or feat_dim % dimension:
        raise ValueError("Features cannot be tiled across the supplied cameras")
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape(batch, heads, cameras, -1, feat_dim // dimension, dimension),
    ).reshape(feats.shape)


def _rope_precompute_coeffs(
    positions: torch.Tensor,
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.ndim != 1 or feat_dim % 2:
        raise ValueError("RoPE positions must be 1D and feature dimension even")
    num_freqs = feat_dim // 2
    frequencies = freq_scale * (
        freq_base ** (-torch.arange(num_freqs, device=positions.device)[None, None, None, :] / num_freqs)
    )
    angles = positions[None, None, :, None] * frequencies
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor,
    coeffs: tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    cosine, sine = coeffs
    if cosine.shape[2] != feats.shape[2]:
        if feats.shape[2] % cosine.shape[2]:
            raise ValueError("RoPE coefficients do not tile across the feature sequence")
        repeats = feats.shape[2] // cosine.shape[2]
        cosine = cosine.repeat(1, 1, repeats, 1)
        sine = sine.repeat(1, 1, repeats, 1)
    x_part, y_part = torch.chunk(feats, 2, dim=-1)
    if inverse:
        return torch.cat((cosine * x_part - sine * y_part, sine * x_part + cosine * y_part), dim=-1)
    return torch.cat((cosine * x_part + sine * y_part, -sine * x_part + cosine * y_part), dim=-1)


def _apply_block_diagonal(feats: torch.Tensor, func_size_pairs) -> torch.Tensor:
    functions, block_sizes = zip(*func_size_pairs)
    if feats.shape[-1] != sum(block_sizes):
        raise ValueError("PRoPE block dimensions do not match attention head dimension")
    outputs = [function(block) for function, block in zip(functions, torch.split(feats, block_sizes, dim=-1))]
    return torch.cat(outputs, dim=-1)


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    if Ks.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [...,3,3] intrinsics, got {tuple(Ks.shape)}")
    output = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    output[..., :3, :3] = Ks
    output[..., 3, 3] = 1.0
    return output


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    if Ks.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [...,3,3] intrinsics, got {tuple(Ks.shape)}")
    output = torch.zeros_like(Ks)
    output[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    output[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    output[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    output[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    output[..., 2, 2] = 1.0
    return output
