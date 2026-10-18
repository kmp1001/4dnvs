# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rectified-flow scheduler used by DrivenVS training and sampling.

The schedule and weighting follow ArtiFixer's ``FlowMatchScheduler``:

    z_t = (1 - t) * z_data + t * z_source
    v*  = z_source - z_data
    z_next = z_t + (t_next - t) * v_hat

``t`` is the rectified-flow interpolation coordinate in [0, 1]. DrivenVS
passes it directly to Cosmos Predict2's native rectified-flow DiT time embedding;
there is no RF-time-to-sigma conversion in this implementation.
"""

import math

import torch


class FlowMatchScheduler:
    """Small ArtiFixer-compatible RF scheduler with batch-safe helpers."""

    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.0,
        extra_one_step: bool = True,
        device: torch.device | str = "cpu",
    ):
        if num_train_timesteps < 1:
            raise ValueError("num_train_timesteps must be positive")
        if shift <= 0:
            raise ValueError("shift must be positive")
        if not 0.0 <= sigma_min < sigma_max <= 1.0:
            raise ValueError(
                f"Expected 0 <= sigma_min < sigma_max <= 1, got "
                f"{sigma_min}, {sigma_max}"
            )
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.extra_one_step = bool(extra_one_step)
        self.set_timesteps(
            self.num_train_timesteps,
            training=True,
            device=device,
        )

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        *,
        training: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")

        count = num_inference_steps + 1 if self.extra_one_step else num_inference_steps
        sigmas = torch.linspace(
            self.sigma_max,
            self.sigma_min,
            count,
            device=device,
            dtype=torch.float32,
        )
        if self.extra_one_step:
            # Match ArtiFixer: keep an implicit terminal t=0 for the final
            # scheduler.step() rather than evaluating the network at t=0.
            sigmas = sigmas[:-1]

        self.sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        self.training = bool(training)

        if training:
            x = self.timesteps
            n = float(num_inference_steps)
            y = torch.exp(-2.0 * ((x - n / 2.0) / n) ** 2)
            y_shifted = y - y.min()
            denom = y_shifted.sum()
            if denom <= 0:
                self.linear_timesteps_weights = torch.ones_like(y_shifted)
            else:
                self.linear_timesteps_weights = y_shifted * (n / denom)
        else:
            self.linear_timesteps_weights = None

    def sample_training_batch(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ids, ArtiFixer timesteps, flow times, and training weights."""
        if not self.training:
            raise RuntimeError("sample_training_batch requires a training schedule")
        timestep_ids = torch.randint(
            0,
            self.num_train_timesteps,
            (batch_size,),
            device=device,
            generator=generator,
        )
        timesteps = self.timesteps[timestep_ids]
        flow_times = self.sigmas[timestep_ids]
        weights = self.linear_timesteps_weights[timestep_ids]
        return timestep_ids, timesteps, flow_times, weights

    @staticmethod
    def add_noise(
        original_samples: torch.Tensor,
        source_samples: torch.Tensor,
        flow_times: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate from clean data (t=0) to source endpoint (t=1)."""
        if original_samples.shape != source_samples.shape:
            raise ValueError(
                "original_samples and source_samples must have identical shapes, got "
                f"{tuple(original_samples.shape)} and {tuple(source_samples.shape)}"
            )
        t = torch.as_tensor(
            flow_times,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )
        if t.ndim == 0:
            t = t.expand(original_samples.shape[0])
        if t.ndim != 1 or t.shape[0] != original_samples.shape[0]:
            raise ValueError(
                f"flow_times must be scalar or [B], got {tuple(t.shape)} for "
                f"batch {original_samples.shape[0]}"
            )
        t = t.view(original_samples.shape[0], 1, 1, 1, 1)
        return (1.0 - t) * original_samples + t * source_samples

    @staticmethod
    def training_target(
        data_samples: torch.Tensor,
        source_samples: torch.Tensor,
    ) -> torch.Tensor:
        """ArtiFixer RF velocity target dz_t/dt = z_source - z_data."""
        return source_samples - data_samples

    def step_by_index(
        self,
        model_output: torch.Tensor,
        step_index: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Euler RF update using the scheduler's current and next flow times."""
        if self.training:
            raise RuntimeError("step_by_index requires an inference schedule")
        if not 0 <= step_index < len(self.sigmas):
            raise IndexError(
                f"step_index={step_index} is outside [0, {len(self.sigmas)})"
            )
        flow_time = self.sigmas[step_index].to(
            device=sample.device,
            dtype=sample.dtype,
        )
        if step_index + 1 < len(self.sigmas):
            next_flow_time = self.sigmas[step_index + 1].to(
                device=sample.device,
                dtype=sample.dtype,
            )
        else:
            next_flow_time = torch.zeros((), device=sample.device, dtype=sample.dtype)
        return sample + model_output * (next_flow_time - flow_time)
