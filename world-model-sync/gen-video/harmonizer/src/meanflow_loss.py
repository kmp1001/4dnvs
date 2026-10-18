# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conditional MeanFlow objective on the existing DrivenVS latent bridge.

Follows official MeanFlow ``loss.py``: z_t=(1-t)x0+t*x1, v=x1-x0,
u_target=v-(t-r)JVP_{(z,t)}u(z,r,t)[v,1], with a stopped target.
Unlike the unconditional paper example, x1 is the existing opacity-guided
render/noise endpoint, so the trained one-step map is conditional restoration.
"""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def sample_meanflow_times(
    batch_size: int,
    device: torch.device,
    *,
    time_sampler: str = "logit_normal",
    time_mu: float = -0.4,
    time_sigma: float = 1.0,
    interval_probability: float = 0.25,
    endpoint_probability: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (r,t) using the local MeanFlow README's ImageNet-256 settings.

    ``interval_probability`` is P(r<t), matching the reference's
    ``ratio_r_not_equal_t`` (not MeanAudio's inverted ``flow_ratio``). The
    optional endpoint override is off by default for exact official sampling.
    """
    if not 0 <= endpoint_probability <= interval_probability <= 1:
        raise ValueError("Require 0 <= endpoint_probability <= interval_probability <= 1")
    if time_sigma <= 0:
        raise ValueError("time_sigma must be positive")
    if time_sampler == "uniform":
        samples = torch.rand(batch_size, 2, device=device)
    elif time_sampler == "logit_normal":
        samples = torch.sigmoid(torch.randn(batch_size, 2, device=device) * time_sigma + time_mu)
    else:
        raise ValueError(f"Unknown MeanFlow time sampler: {time_sampler}")
    sorted_samples, _ = torch.sort(samples, dim=1)
    r, t = sorted_samples[:, 0], sorted_samples[:, 1]
    equal = torch.rand(batch_size, device=device) < (1.0 - interval_probability)
    r = torch.where(equal, t, r)
    if endpoint_probability > 0:
        endpoint = (~equal) & (torch.rand(batch_size, device=device) < endpoint_probability / interval_probability)
        r = torch.where(endpoint, torch.zeros_like(r), r)
        t = torch.where(endpoint, torch.ones_like(t), t)
    return r, t


def meanflow_prediction_and_target(
    predict,
    z_t: torch.Tensor,
    velocity: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    *,
    jvp_backend: str = "func",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return u_theta and the detached MeanFlow identity target.

    The autograd backend is MeanAudio's more compatible double-backward JVP;
    ``func`` selects the official MeanFlow forward-mode JVP. Both differentiate
    the model wrt z and t, holding r and all external conditions fixed.
    """
    if z_t.shape != velocity.shape or r.shape != t.shape or r.ndim != 1:
        raise ValueError("MeanFlow expects matching z/v and [B] r/t tensors")
    if z_t.shape[0] != r.shape[0] or torch.any(r > t):
        raise ValueError("MeanFlow requires 0 <= r <= t with matching batch")

    # At r=t the identity reduces exactly to u_target=v. With the usual
    # batch size of one, avoid a needless and expensive DiT JVP on these
    # instantaneous-velocity examples.
    if bool((r == t).all()):
        return predict(z_t, r, t), velocity.detach()

    primals = (z_t, r, t)
    tangents = (velocity, torch.zeros_like(r), torch.ones_like(t))
    # Cosmos uses SDPA. Fused kernels do not support the forward AD / double
    # backward required here; math SDPA computes the same attention equation.
    # Restrict this selection to the JVP, leaving ordinary inference unchanged.
    # The target is stopped: do not train through the JVP's primal output.
    # no_grad preserves forward AD while avoiding its bf16 backward graph.
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        if jvp_backend == "autograd":
            _, derivative = torch.autograd.functional.jvp(predict, primals, tangents, create_graph=False, strict=False)
        elif jvp_backend == "func":
            _, derivative = torch.func.jvp(predict, primals, tangents)
        else:
            raise ValueError(f"Unknown JVP backend: {jvp_backend}")

    interval = (t - r).view(-1, *([1] * (z_t.ndim - 1)))
    target = (velocity - interval * derivative).detach()
    del derivative, _
    prediction = predict(z_t, r, t)
    return prediction, target


def meanflow_latent_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    adaptive_p: float = 1.0,
    adaptive_c: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official per-sample adaptive L2, plus unweighted MSE for diagnostics."""
    if prediction.shape != target.shape:
        raise ValueError("MeanFlow prediction and target shapes differ")
    if adaptive_p < 0 or adaptive_c <= 0:
        raise ValueError("Adaptive loss requires p >= 0 and c > 0")
    squared = (prediction.float() - target.float()).square()
    per_sample = squared.flatten(1).sum(dim=1)
    weight = (per_sample.detach() + adaptive_c).pow(-adaptive_p)
    return (weight * per_sample).mean(), squared.mean()
