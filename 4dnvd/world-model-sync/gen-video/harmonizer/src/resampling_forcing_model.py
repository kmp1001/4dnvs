# SPDX-License-Identifier: Apache-2.0
"""History self-resampling for the 0.6B MeanFlow harmonizer.

Mode A uses per-layer causal history K/V distinct from source-reference K/V.
Mode B uses the original V=5 concatenated-history model unchanged.
"""

from __future__ import annotations

import torch

from meanflow_model import MeanFlowDrivenVSModel, load_meanflow_weights


def sample_resampling_time(batch_size: int, device: torch.device, shift: float = 0.6):
    """Resampling Forcing Eq. (6)-(7): LogitNormal(0,1), then shift."""
    if shift <= 0:
        raise ValueError("Resampling timestep shift must be positive")
    sample = torch.sigmoid(torch.randn(batch_size, device=device, dtype=torch.float32))
    return shift * sample / (1.0 + (shift - 1.0) * sample)


class ResamplingMeanFlowModel(MeanFlowDrivenVSModel):
    def __init__(
        self,
        *args,
        mode: str,
        initial_checkpoint: str | None = None,
        resampling_checkpoint: str | None = None,
        use_ema: bool = True,
        max_history: int = 4,
        **kwargs,
    ):
        if mode not in {"a", "b"}:
            raise ValueError("Resampling mode must be 'a' or 'b'")
        if (initial_checkpoint is None) == (resampling_checkpoint is None):
            raise ValueError("Supply exactly one initial or resampling checkpoint")
        self.resampling_mode = mode
        self.max_history = int(max_history)
        # A checkpoint contains extra history branch parameters. Install the
        # branch before strict loading; for Stage-I initialization, load the
        # original MeanFlow weights first and initialize only the new gate.
        super().__init__(
            *args,
            meanflow_checkpoint=(initial_checkpoint if initial_checkpoint else None),
            meanflow_use_ema=use_ema if initial_checkpoint else False,
            **kwargs,
        )
        if mode == "a":
            self.unet.dit.enable_history_kv(self.max_history)
        if resampling_checkpoint is not None:
            load_meanflow_weights(self, resampling_checkpoint, use_ema=use_ema)

    def empty_history_cache(self):
        if self.resampling_mode != "a":
            raise ValueError("Only mode A has a per-layer K/V cache")
        return tuple(() for _ in self.unet.dit.blocks)

    def set_history_cache(self, cache, *, capture: bool = False):
        if self.resampling_mode != "a":
            raise ValueError("Only mode A accepts a history K/V cache")
        self.unet.dit.set_history_kv(cache, capture=capture)

    def clear_history_cache(self):
        if self.resampling_mode == "a":
            self.unet.dit.clear_history_kv()

    def _set_frame_references(self, image, references, condition):
        batch, channels, _, height, width = image.shape
        return self._set_reference_conditioning(
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
        )

    def _clear_frame_references(self, clear):
        self.unet.dit.clear_meanflow_conditioning()
        if clear:
            self.unet.dit.clear_reference_latents()
            self.unet.dit.clear_camera_conditioning()

    @torch.no_grad()
    def resample_history_frame(self, clean_image, references, condition, t_s, cache=None, *, rendered_image, opacity):
        """One Euler step from perturbed GT latent to imperfect clean history.

        The entire operation uses the current online model and stops gradients.
        Mode A conditions on already resampled older frames; B does not.
        """
        if clean_image.ndim != 5 or clean_image.shape[2] != 1:
            raise ValueError("History resampling expects one RGB frame [B,C,1,H,W]")
        batch = clean_image.shape[0]
        t_s = self._normalize_batch_values(
            t_s,
            batch_size=batch,
            device=clean_image.device,
            dtype=torch.float32,
            name="resampling time",
        )
        if bool(((t_s <= 0) | (t_s >= 1)).any()):
            raise ValueError("Resampling time must be strictly between zero and one")
        clear = self._set_frame_references(clean_image, references, condition)
        if self.resampling_mode == "a":
            self.set_history_cache(cache if cache is not None else self.empty_history_cache())
        try:
            x0 = self.vae_encode(clean_image).detach()
            rendered = self.vae_encode(rendered_image).detach()
            opacity_latent = self.prepare_opacity_latent(opacity, rendered)
            source, _ = self.prepare_opacity_source_endpoint(rendered, opacity_latent)
            z = (1 - t_s.view(batch, 1, 1, 1, 1)) * x0 + t_s.view(batch, 1, 1, 1, 1) * source
            flow = self._predict_meanflow(z, r=torch.zeros_like(t_s), t=t_s)
            resampled_latent = (z.float() - t_s.view(batch, 1, 1, 1, 1) * flow).detach()
            return self.vae_decode(resampled_latent).detach(), resampled_latent
        finally:
            self._clear_frame_references(clear)
            self.clear_history_cache()

    @torch.no_grad()
    def append_clean_history_kv(self, clean_latent, references, condition, cache):
        """Build K/V from each block's clean, time-zero hidden state.

        Existing source-reference conditioning remains separate. This pass can
        attend only to the older history cache; no target/future frame exists.
        """
        if self.resampling_mode != "a":
            raise ValueError("Only mode A updates history K/V")
        if clean_latent.ndim != 5 or clean_latent.shape[2] != 1:
            raise ValueError("Clean-history latent must be [B,C,1,H,W]")
        # Reference setup validates RGB spatial dimensions. The latent is
        # already encoded, so use the frame size implied by the VAE factor.
        image_height = clean_latent.shape[-2] * self._vae_spatial_compression_factor()
        image_width = clean_latent.shape[-1] * self._vae_spatial_compression_factor()
        dummy_shape = (clean_latent.shape[0], 3, 1, image_height, image_width)
        reference_shape = references.shape
        if reference_shape[-2:] != dummy_shape[-2:]:
            raise ValueError("History reference resolution does not match latent")
        clear = self._set_reference_conditioning(
            references,
            batch_size=dummy_shape[0],
            channels=3,
            height=image_height,
            width=image_width,
            target_c2w=condition.get("target_c2w"),
            target_K=condition.get("target_K"),
            reference_c2ws=condition.get("reference_c2ws"),
            reference_Ks=condition.get("reference_Ks"),
            reference_time_offsets=condition.get("reference_time_offsets"),
            reference_visibility_scores=condition.get("reference_visibility_scores"),
        )
        self.set_history_cache(cache, capture=True)
        try:
            zero = torch.zeros(clean_latent.shape[0], device=clean_latent.device)
            self._predict_meanflow(clean_latent, r=zero, t=zero)
            captured = self.unet.dit.get_captured_history_kv()
        finally:
            self._clear_frame_references(clear)
            self.clear_history_cache()
        return tuple(tuple((*cache[layer], captured[layer])[-self.max_history :]) for layer in range(len(captured)))
