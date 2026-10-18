# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-II MeanFlow adapter for the pretrained Cosmos DrivenVS bridge."""

import torch

from drivenvs_model import (
    DrivenVSModel,
    _conditioning_config,
    _remap_legacy_unet_state,
    validate_checkpoint_conditioning,
)
from meanflow_loss import meanflow_prediction_and_target
from utils.flow_match import FlowMatchScheduler


class MeanFlowDrivenVSModel(DrivenVSModel):
    """Preserve Cosmos/Option-A weights; add the reference t-r embedder.

    ``pretrained_path`` is the normal 0.6B Stage-I/II bridge checkpoint.
    ``meanflow_checkpoint`` instead loads an exact MeanFlow checkpoint for
    one-step inference. Do not pass both.
    """

    def __init__(self, *args, meanflow_checkpoint=None, meanflow_use_ema=False, **kwargs):
        if meanflow_checkpoint is not None and kwargs.get("pretrained_path") is not None:
            raise ValueError("Use either pretrained_path or meanflow_checkpoint")
        super().__init__(*args, **kwargs)
        self.unet.dit.enable_meanflow_conditioning()
        if meanflow_checkpoint is not None:
            load_meanflow_weights(self, meanflow_checkpoint, use_ema=meanflow_use_ema)

    def to(self, *args, **kwargs):
        # Keep official Cosmos fp32 adapters across bf16 training/inference moves.
        # Snapshot before casting so re-upcasting cannot lose checkpoint precision.
        unet = getattr(self, "unet", None)
        dit = getattr(unet, "dit", None)
        if getattr(dit, "harmonizer_lora_config", None) is None:
            return super().to(*args, **kwargs)
        from meanflow_lora import ADAPTER_PARTS

        saved = {
            name: parameter.detach().clone()
            for name, parameter in unet.named_parameters()
            if any(part in name for part in ADAPTER_PARTS)
        }
        result = super().to(*args, **kwargs)
        with torch.no_grad():
            for name, parameter in unet.named_parameters():
                if name in saved:
                    parameter.data = saved[name].to(device=parameter.device, dtype=torch.float32)
        return result

    def _predict_meanflow(self, latent: torch.Tensor, *, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        r = self._normalize_batch_values(
            r, batch_size=batch_size, device=latent.device, dtype=torch.float32, name="meanflow r"
        )
        t = self._normalize_batch_values(
            t, batch_size=batch_size, device=latent.device, dtype=torch.float32, name="meanflow t"
        )
        # The reference embeds the unscaled interval, not r itself.
        self.unet.dit.meanflow_interval_B_T = (t - r).view(batch_size, 1)
        try:
            # The caller checks the time range before the JVP. Avoid tensor
            # truth-value guards inside torch.func's forward-mode transform.
            return self._predict_velocity(latent, flow_times=t, validate_times=False)
        finally:
            self.unet.dit.clear_meanflow_conditioning()

    def forward(
        self,
        x,
        *,
        r,
        t,
        reference_images=None,
        target_c2w=None,
        target_K=None,
        reference_c2ws=None,
        reference_Ks=None,
        reference_time_offsets=None,
        reference_visibility_scores=None,
        target_images=None,
        opacity=None,
        world_scenario=None,
        reference_world_scenario=None,
        jvp_backend="func",
        compute_clean_output=True,
    ):
        if x.ndim != 5 or target_images is None or opacity is None:
            raise ValueError("MeanFlow training requires [B,C,V,H,W] input, target, and opacity")
        if x.shape != target_images.shape:
            raise ValueError("MeanFlow target must match the warped/history path")
        batch_size, channels, _, height, width = x.shape
        r = self._normalize_batch_values(
            r, batch_size=batch_size, device=x.device, dtype=torch.float32, name="meanflow r"
        )
        t = self._normalize_batch_values(
            t, batch_size=batch_size, device=x.device, dtype=torch.float32, name="meanflow t"
        )
        if (
            not torch.isfinite(r).all()
            or not torch.isfinite(t).all()
            or (r < 0).any()
            or (r > t).any()
            or (t > 1).any()
        ):
            raise ValueError("MeanFlow requires finite 0 <= r <= t <= 1")

        clear_references = self._set_reference_conditioning(
            reference_images,
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            target_c2w=target_c2w,
            target_K=target_K,
            reference_c2ws=reference_c2ws,
            reference_Ks=reference_Ks,
            reference_time_offsets=reference_time_offsets,
            reference_visibility_scores=reference_visibility_scores,
            world_scenario=world_scenario,
            reference_world_scenario=reference_world_scenario,
        )
        try:
            with torch.no_grad():
                target_latent = self.vae_encode(target_images).detach()
                warped_latent = self.vae_encode(x).detach()
                opacity_latent = self.prepare_opacity_latent(opacity, warped_latent)
                source_latent, _ = self.prepare_opacity_source_endpoint(warped_latent, opacity_latent)

            target_latent_f = target_latent.float()
            source_latent_f = source_latent.float()
            z_t = FlowMatchScheduler.add_noise(target_latent_f, source_latent_f, t)
            velocity = source_latent_f - target_latent_f

            def predict(z, current_r, current_t):
                return self._predict_meanflow(z, r=current_r, t=current_t)

            mean_velocity, mean_target = meanflow_prediction_and_target(
                predict, z_t, velocity, r, t, jvp_backend=jvp_backend
            )

            # Auxiliary GT/RAFT losses must supervise a *clean* prediction.
            # For arbitrary r>0, u(z_t,r,t) only predicts z_r, not z_0.
            # A second call with r=0 yields the proper clean prediction.
            predicted_x0 = None
            if compute_clean_output:
                clean_velocity = mean_velocity if bool((r == 0).all()) else predict(z_t, torch.zeros_like(r), t)
                predicted_x0 = z_t - t.view(batch_size, 1, 1, 1, 1) * clean_velocity
        finally:
            self.unet.dit.clear_meanflow_conditioning()
            if clear_references:
                self.unet.dit.clear_reference_latents()
                self.unet.dit.clear_camera_conditioning()

        return {
            "output_image": self.vae_decode(predicted_x0) if predicted_x0 is not None else None,
            "predicted_x0": predicted_x0,
            "target_x0": target_latent,
            "predicted_mean_velocity": mean_velocity,
            "target_mean_velocity": mean_target,
            "instantaneous_velocity": velocity,
            "source_latent": source_latent,
            "model_input": z_t,
            "r": r,
            "t": t,
        }

    @torch.no_grad()
    def sample_bridge(
        self,
        x,
        *,
        reference_images=None,
        target_c2w=None,
        target_K=None,
        reference_c2ws=None,
        reference_Ks=None,
        reference_time_offsets=None,
        reference_visibility_scores=None,
        opacity,
        world_scenario=None,
        reference_world_scenario=None,
        num_steps=1,
        timestep_shift=5.0,
        generator=None,
    ):
        """One DiT call for num_steps=1; exact interval updates otherwise."""
        if num_steps < 1:
            raise ValueError("MeanFlow num_steps must be >= 1")
        batch_size, channels, _, height, width = x.shape
        clear_references = self._set_reference_conditioning(
            reference_images,
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            target_c2w=target_c2w,
            target_K=target_K,
            reference_c2ws=reference_c2ws,
            reference_Ks=reference_Ks,
            reference_time_offsets=reference_time_offsets,
            reference_visibility_scores=reference_visibility_scores,
            world_scenario=world_scenario,
            reference_world_scenario=reference_world_scenario,
        )
        try:
            warped_latent = self.vae_encode(x)
            opacity_latent = self.prepare_opacity_latent(opacity, warped_latent)
            latent, _ = self.prepare_opacity_source_endpoint(warped_latent, opacity_latent, generator=generator)
            # Match MeanFlow/meanflow_sampler.py: evenly spaced intervals.
            # ``timestep_shift`` is retained only for compatibility with the
            # existing evaluation call signature; MeanFlow does not use it.
            times = torch.linspace(1.0, 0.0, num_steps + 1, device=latent.device)
            for index in range(num_steps):
                current_t, next_r = times[index], times[index + 1]
                t = current_t.expand(batch_size)
                r = next_r.expand(batch_size)
                mean_velocity = self._predict_meanflow(latent, r=r, t=t)
                latent = latent.float() - (t - r).view(batch_size, 1, 1, 1, 1) * mean_velocity
        finally:
            self.unet.dit.clear_meanflow_conditioning()
            if clear_references:
                self.unet.dit.clear_reference_latents()
                self.unet.dit.clear_camera_conditioning()
        return self.vae_decode(latent)


def load_meanflow_weights(model: MeanFlowDrivenVSModel, checkpoint_path: str, *, use_ema: bool = False) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("training_objective") not in {
        "meanflow_stage2_v1",
        "self_forcing_opsd_meanflow_v1",
        "resampling_forcing_a_v1",
        "resampling_forcing_b_v1",
        "meanflow_curriculum_v2",
    }:
        raise ValueError(f"Not a MeanFlow-family checkpoint: {checkpoint_path}")
    from meanflow_conditioning import MEANFLOW_TIME_CONDITIONING

    version = checkpoint.get("conditioning_config", {}).get("meanflow_time_conditioning")
    if version != MEANFLOW_TIME_CONDITIONING:
        raise ValueError(
            "Incompatible MeanFlow time conditioning: this model requires the "
            "reference t-r embedder (v2). Older direct-r checkpoints must be "
            "retrained; their weights cannot be silently interpreted as t-r."
        )
    validate_checkpoint_conditioning(model, checkpoint, label="MeanFlow checkpoint")
    # Native PEFT injection preserves the DiT API and runtime conditioning.
    if checkpoint.get("lora_config") is not None:
        from meanflow_lora import configure_lora_trainability, install_lora

        install_lora(model.unet.dit, checkpoint["lora_config"])
        configure_lora_trainability(model.unet)
    model.unet.load_state_dict(_remap_legacy_unet_state(checkpoint["state_dict_unet"]), strict=True)
    model._inner_vae().load_state_dict(checkpoint["state_dict_vae"], strict=True)
    if use_ema:
        shadow = checkpoint.get("ema_state_dict_unet")
        if shadow is None:
            raise ValueError(f"MeanFlow checkpoint has no EMA weights: {checkpoint_path}")
        named_parameters = dict(model.unet.named_parameters())
        if set(shadow) != {name for name, param in named_parameters.items() if param.requires_grad}:
            raise ValueError("EMA parameter names do not match the MeanFlow UNet")
        with torch.no_grad():
            for name, ema_value in shadow.items():
                param = named_parameters[name]
                if ema_value.shape != param.shape:
                    raise ValueError(f"EMA tensor shape mismatch: {name}")
                param.copy_(ema_value.to(device=param.device, dtype=param.dtype))
    return checkpoint


def save_meanflow_checkpoint(model, optimizer, output_path: str, config: dict, ema) -> None:
    from meanflow_lora import get_lora_config

    torch.save(
        {
            "training_objective": "meanflow_stage2_v1",
            "state_dict_unet": model.unet.state_dict(),
            "state_dict_vae": model._inner_vae().state_dict(),
            "conditioning_config": _conditioning_config(model),
            "meanflow_config": config,
            "lora_config": get_lora_config(model.unet.dit),
            "ema_state_dict_unet": ema.state_dict() if ema is not None else None,
            "ema_decay": ema.decay if ema is not None else None,
            "ema_num_updates": ema.num_updates if ema is not None else 0,
            "optimizer": optimizer.state_dict(),
        },
        output_path,
    )
