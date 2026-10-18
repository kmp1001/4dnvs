# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diffusion-Harmonizer-style RAFT temporal consistency loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def flow_to_norm(flow, H, W):
    flow_x = 2.0 * flow[:, 0:1] / max(W - 1, 1)
    flow_y = 2.0 * flow[:, 1:2] / max(H - 1, 1)
    return torch.cat([flow_x, flow_y], dim=1)


def make_identity_grid(N, H, W, device, dtype):
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(N, 1, 1, 1)
    return base_grid


def warp_backward(src, flow_bwd):
    """Warp src at t-1 into t coordinates using backward flow F_{t->t-1}."""
    N, _C, H, W = src.shape
    norm_flow = flow_to_norm(flow_bwd, H, W)
    base_grid = make_identity_grid(N, H, W, src.device, flow_bwd.dtype)
    grid = base_grid + norm_flow.permute(0, 2, 3, 1)
    warped = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid_x = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0)
    valid_y = (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    valid = (valid_x & valid_y).float().unsqueeze(1)
    return warped, valid


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, diff, valid=None):
        penalty = torch.sqrt(diff * diff + self.eps * self.eps)
        if valid is None:
            return penalty.mean()
        # Count valid channel values: an RGB pixel contributes three values.
        # Mask the penalty after Charbonnier so invalid pixels contribute no eps.
        weights = torch.broadcast_to(valid.to(device=diff.device, dtype=diff.dtype), diff.shape)
        return (penalty * weights).sum() / weights.sum().clamp_min(1.0)


class TorchvisionRAFT:
    def __init__(self, device, use_small=False):
        if use_small:
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

            self.model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        else:
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

            self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        self.model.to(device).eval()

    @torch.no_grad()
    def backward(self, curr, prev):
        out = self.model(curr, prev)
        if isinstance(out, (list, tuple)):
            flow = out[-1]
        else:
            flow = out
        return flow


def build_raft_temporal_loss(device, use_small=False):
    raft = TorchvisionRAFT(device=device, use_small=use_small)
    raft.model.requires_grad_(False)
    raft.model.to(device, dtype=torch.float32)

    temp_loss = CharbonnierLoss(eps=1e-3)
    return raft, temp_loss


def build_temporal_loss(
    device,
    *,
    estimator="raft",
    use_small=False,
    waft_repo=None,
    waft_config=None,
    waft_checkpoint=None,
    waft_iterations=None,
):
    """Select the frozen flow estimator; retain the same warp/masked loss."""
    if estimator == "raft":
        return build_raft_temporal_loss(device, use_small=use_small)
    if estimator == "waft":
        from waft_estimator import WAFTOpticalFlow

        return WAFTOpticalFlow(
            device, repo=waft_repo, config=waft_config, checkpoint=waft_checkpoint, iterations=waft_iterations
        ), CharbonnierLoss(eps=1e-3)
    raise ValueError(f"Unknown optical-flow estimator: {estimator}")


def get_dh_temporal_loss(x_tgt_pred, x_tgt, raft, temp_loss):
    """Apply DH consistency between predicted view 0 and clean views 1..4."""
    x_tgt = rearrange(x_tgt, "b c v h w -> b v c h w")
    x_tgt_pred = rearrange(x_tgt_pred, "b c v h w -> b v c h w")

    _, num_views, _, _, _ = x_tgt.shape
    if num_views <= 1:
        return torch.zeros((), device=x_tgt.device, dtype=torch.float32)

    with torch.no_grad():
        flows_bwd = []
        for view_index in range(1, num_views):
            F_bwd_t = raft.backward(
                x_tgt[:, view_index].float(),
                x_tgt[:, 0].float(),
            )
            flows_bwd.append(F_bwd_t)
        flow_bwd = torch.stack(flows_bwd, dim=1)

    tmp_losses = []
    for view_index in range(1, num_views):
        warped, valid = warp_backward(
            x_tgt_pred[:, 0].float(),
            flow_bwd[:, view_index - 1].float(),
        )
        tmp_losses.append(temp_loss(x_tgt[:, view_index].float() - warped, valid=valid))
    return torch.stack(tmp_losses).mean()
