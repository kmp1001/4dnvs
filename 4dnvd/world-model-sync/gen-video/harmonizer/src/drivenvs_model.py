# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DrivenVS bridge model with Option-A hybrid visual-reference conditioning.

The target path contains ``warped(t)`` and, for temporal samples, four clean
history frames. Clean source references are independently encoded by the frozen
shared VAE and patchified with the target DiT's input embedder. The
highest-visibility reference is carried through the main self-attention/MLP
stream together with the target; the remaining references are supplied through
additional K/V projections on the existing text cross-attention, and only the
target queries that memory. Independently enabled target-relative camera rays,
signed reference-frame offsets, and PRoPE align the streams spatially and
temporally. Opacity constructs the Gaussian-filled source endpoint rather than
being injected as a separate learned token.

The restoration bridge uses GS confident coverage to retain valid
DH/warped latents and replace fully uncovered cells with Gaussian noise:

    z_source = A_z * z_dh + (1 - A_z) * epsilon
    z_t      = (1 - t) * z_target + t * z_source.

Training follows the ArtiFixer rectified-flow path.  The flow target is

    v* = z_source - z_target.

Cosmos Predict2's ``RectifiedFlowScaling`` already trains the raw DiT output as
the RF field.  Its public ``denoise()`` wrapper converts that field to x0 via

    x0_hat = z_t - t * v_hat.

DrivenVS therefore calls the DiT directly with the already-interpolated RF
state ``z_t`` and RF time ``t``.  This avoids applying ``c_in = 1 - t`` a
second time and avoids recovering velocity through division by ``t``.
Inference integrates the directly predicted field with ArtiFixer-style Euler
steps.
"""

import math

import torch
import torch.nn.functional as F
from cosmos_predict2.conditioner import DataType
from cosmos_predict2.configs.base.config_text2image import (
    Text2ImagePipelineConfig,
    get_cosmos_predict2_text2image_pipeline,
)
from cosmos_predict2.pipelines.text2image import Text2ImagePipeline
from cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling
from cosmos_predict2.tokenizers.tokenizer import CausalConv3d, ResidualBlock
from imaginaire.lazy_config import LazyDict

from model import MiniTrainDIT
from model.camera_conditioning import build_relative_camera_conditioning
from model.text2image_dit import CheckpointMode, SACConfig
from utils.flow_match import FlowMatchScheduler
import os

class PipelineCompatibleMiniTrainDIT(MiniTrainDIT):
    """Adapt the local cached DiT to the installed Text2ImagePipeline API."""

    def enable_meanflow_conditioning(self) -> None:
        """Install MeanFlow's independent learned embedding of t-r."""
        from meanflow_conditioning import (
            MEANFLOW_TIME_CONDITIONING,
            MeanFlowIntervalEmbedder,
        )

        if hasattr(self, "meanflow_interval_embedder"):
            return
        self.meanflow_interval_embedder = MeanFlowIntervalEmbedder(self.t_embedder[1].out_dim).to(
            device=self.t_embedder[1].linear_1.weight.device,
            dtype=self.t_embedder[1].linear_1.weight.dtype,
        )
        self.meanflow_time_conditioning = MEANFLOW_TIME_CONDITIONING
        self.meanflow_interval_B_T = None
        if getattr(self, "world_scenario_mode", 0) == 2:
            self.world_scenario_control.enable_meanflow(self.meanflow_interval_embedder)

    def clear_meanflow_conditioning(self) -> None:
        self.meanflow_interval_B_T = None

    @staticmethod
    def _unwrap_checkpoint_module(module):
        for _ in range(3):
            if hasattr(module, "_checkpoint_wrapped_module"):
                module = module._checkpoint_wrapped_module
                continue
            if hasattr(module, "module"):
                module = module.module
                continue
            break
        return module

    def _cache_pipeline_condition(
        self,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
    ) -> None:
        if self.crossattn_proj is not None:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        if timesteps_B_T.ndim != 2:
            raise ValueError(f"Expected timesteps_B_T with shape [B,T], got {tuple(timesteps_B_T.shape)}")

        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        if getattr(self, "meanflow_interval_B_T", None) is not None:
            interval = self.meanflow_interval_B_T.to(device=timesteps_B_T.device)
            if interval.shape != timesteps_B_T.shape:
                raise ValueError("MeanFlow interval shape must match the t conditioning")
            t_embedding_B_T_D = t_embedding_B_T_D + self.meanflow_interval_embedder(interval)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        for block in self.blocks:
            block = self._unwrap_checkpoint_module(block)
            block.compute_caching(
                t_embedding_B_T_D,
                crossattn_emb,
                adaln_lora_B_T_3D,
            )

        final_layer = self._unwrap_checkpoint_module(self.final_layer)
        final_layer.compute_caching(
            t_embedding_B_T_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
        )

        self.control_timesteps_B_T = timesteps_B_T
        self.control_interval_B_T = getattr(self, "meanflow_interval_B_T", None)
        self.t_embedding_B_T_D = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb
        self.adaln_lora_B_T_3D = adaln_lora_B_T_3D

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor | None = None,
        crossattn_emb: torch.Tensor | None = None,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: DataType | None = DataType.VIDEO,
        use_cuda_graphs: bool = False,
    ):
        if (timesteps_B_T is None) != (crossattn_emb is None):
            raise ValueError("timesteps_B_T and crossattn_emb must either both be provided or both be pre-cached")

        if timesteps_B_T is not None:
            self._cache_pipeline_condition(
                timesteps_B_T=timesteps_B_T,
                crossattn_emb=crossattn_emb,
            )
        elif not hasattr(self, "t_embedding_B_T_D"):
            raise RuntimeError(
                "DiT conditioning is not cached. Pass timesteps_B_T and crossattn_emb, or cache them before forward()."
            )

        return super().forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            use_cuda_graphs=use_cuda_graphs,
        )


def get_pipeline_config() -> LazyDict[Text2ImagePipelineConfig]:
    # this avoid hardcoding the model size and checkpoint path in the code
    model_size = os.environ.get("COSMOS_MODEL_SIZE", "2B")
    checkpoint_dir = os.environ.get(
        "COSMOS_CHECKPOINT_DIR",
        "/fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/"
        "weights/checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image",
    )

    config = get_cosmos_predict2_text2image_pipeline(
        model_size=model_size,
        fast_tokenizer=False,
    )
    config.tokenizer["vae_pth"] = (
        f"{checkpoint_dir}/tokenizer/tokenizer.pth"
    )
    config.dit_path = f"{checkpoint_dir}/model.pt"

    config.guardrail_config.enabled = False
    config.net["_target_"] = PipelineCompatibleMiniTrainDIT
    config.net.atten_backend = "torch"
    config.net.sac_config = SACConfig(
        mode=CheckpointMode.NONE,
        every_n_blocks=1,
    )
    return config
    # config = get_cosmos_predict2_text2image_pipeline(model_size="2B", fast_tokenizer=False)
    # config.tokenizer["vae_pth"] = (
    #     "/fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/weights/checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/tokenizer/tokenizer.pth"
    # )
    # config.dit_path = "/fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/weights/checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt"
    # config.guardrail_config.enabled = False
    # config.net["_target_"] = PipelineCompatibleMiniTrainDIT
    # config.net.atten_backend = "torch"
    # config.net.sac_config = SACConfig(
    #     mode=CheckpointMode.NONE,
    #     every_n_blocks=1,
    # )
    # return config


def _remap_legacy_unet_state(state_dict):
    """Map legacy ``net``/``net_ema`` DiT prefixes to pipeline prefixes."""
    unet_sd = {}
    for k, v in state_dict.items():
        if k.startswith("net."):
            unet_sd["dit." + k[len("net.") :]] = v
        elif k.startswith("net_ema."):
            unet_sd["dit_ema." + k[len("net_ema.") :]] = v
        else:
            unet_sd[k] = v
    return unet_sd


def _conditioning_config(model):
    """Return the runtime choices that must be reproduced at inference."""
    return {
        "num_reference_images": int(model.num_reference_images),
        "reference_fusion_mode": "hybrid_top1_joint_option_a",
        "joint_reference_count": 1 if model.num_reference_images else 0,
        "camera_embed": bool(model.camera_embed),
        "time_embed": bool(model.time_embed),
        "camera_translation_scale": float(model.camera_translation_scale),
        "reference_time_embedding_dim": int(model.reference_time_embedding_dim),
        "reference_time_scale": float(model.reference_time_scale),
        "meanflow_time_conditioning": getattr(model.unet.dit, "meanflow_time_conditioning", None),
        **(
            {
                "world_scenario_mode": model.world_scenario_mode,
                "world_scenario_control_blocks": 3 if model.world_scenario_mode == 2 else 0,
            }
            if getattr(model, "world_scenario_mode", 0)
            else {}
        ),
    }


def _infer_conditioning_architecture(unet_state):
    """Infer camera/time modules from older checkpoints without metadata."""
    camera_weights = [value for name, value in unet_state.items() if name.endswith(".camera_embedding.weight")]
    projection_weights = [
        value for name, value in unet_state.items() if name.endswith("reference_condition_projection.weight")
    ]
    camera_dim = int(camera_weights[0].shape[1]) if camera_weights else 0
    if any(int(value.shape[1]) != camera_dim for value in camera_weights):
        raise ValueError("Checkpoint contains inconsistent camera projection widths")
    if len(projection_weights) > 1:
        raise ValueError("Checkpoint contains multiple reference-condition projections")
    condition_dim = int(projection_weights[0].shape[1]) if projection_weights else 0
    if condition_dim < camera_dim:
        raise ValueError("Checkpoint reference-condition width is smaller than its camera width")
    return {
        "camera_embed": bool(camera_weights),
        "time_embed": condition_dim > camera_dim,
        "reference_time_embedding_dim": condition_dim - camera_dim,
    }


def validate_checkpoint_conditioning(model, checkpoint, *, label):
    """Reject inference/resume configurations that cannot load faithfully."""
    expected = _conditioning_config(model)
    stored = checkpoint.get("conditioning_config")
    if stored is None:
        if expected["num_reference_images"]:
            raise ValueError(
                f"{label} predates Option-A reference-fusion metadata and cannot "
                "be used as an exact resume/inference checkpoint. Load it with "
                "--pretrained_path for architecture migration and retrain instead."
            )
        inferred = _infer_conditioning_architecture(_remap_legacy_unet_state(checkpoint["state_dict_unet"]))
        checked_fields = ("camera_embed", "time_embed")
        if inferred["time_embed"]:
            checked_fields += ("reference_time_embedding_dim",)
        mismatches = {
            field: (inferred[field], expected[field]) for field in checked_fields if inferred[field] != expected[field]
        }
        if mismatches:
            raise ValueError(f"{label} conditioning architecture does not match the requested flags: {mismatches}")
        print(
            f"  {label}: legacy checkpoint has no conditioning_config; "
            "camera/time architecture was inferred, but scales and reference "
            "count cannot be verified"
        )
        return

    if stored.get("world_scenario_mode", 0) != expected.get("world_scenario_mode", 0):
        raise ValueError(f"{label} World Scenario mode does not match the model")
    fields = [
        "num_reference_images",
        "reference_fusion_mode",
        "joint_reference_count",
        "camera_embed",
        "time_embed",
    ]
    if expected["camera_embed"]:
        fields.append("camera_translation_scale")
    if expected["time_embed"]:
        fields.extend(("reference_time_embedding_dim", "reference_time_scale"))
    missing = [field for field in fields if field not in stored]
    if missing:
        raise ValueError(f"{label} conditioning_config is missing fields: {missing}")
    mismatches = {}
    for field in fields:
        old_value = stored[field]
        new_value = expected[field]
        if isinstance(new_value, float):
            matches = math.isclose(
                float(old_value),
                new_value,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        else:
            matches = old_value == new_value
        if not matches:
            mismatches[field] = (old_value, new_value)
    if mismatches:
        raise ValueError(f"{label} conditioning_config does not match the requested model: {mismatches}")


def load_training_checkpoint(model, optimizer, checkpoint_path):
    sd = torch.load(checkpoint_path, map_location="cpu")
    validate_checkpoint_conditioning(model, sd, label="resume checkpoint")
    unet_sd = _remap_legacy_unet_state(sd["state_dict_unet"])
    try:
        model.unet.load_state_dict(unet_sd, strict=True)
        model._inner_vae().load_state_dict(sd["state_dict_vae"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "--resume requires an exact model architecture. Use --pretrained_path "
            "and start a new optimizer run when migrating checkpoint parameters."
        ) from exc
    try:
        optimizer.load_state_dict(sd["optimizer"])
    except ValueError as exc:
        raise ValueError(
            "The optimizer state is incompatible with the current visual-reference, "
            "camera, or temporal parameters. To migrate a checkpoint from a different architecture, "
            "load it with --pretrained_path and start a new optimizer run "
            "instead of using --resume."
        ) from exc
    print(f"\nLoaded training state from {checkpoint_path}\n")
    return model, optimizer


def save_training_checkpoint(model, optimizer, output_path):
    sd = {
        "state_dict_unet": model.unet.state_dict(),
        "state_dict_vae": model._inner_vae().state_dict(),
        "conditioning_config": _conditioning_config(model),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(sd, output_path)


CACHE_T = 2


def my_vae_encoder_fwd(self, x, feat_cache=None, feat_idx=[0]):
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    l_blocks = []
    for layer in self.downsamples:
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)
        l_blocks.append(x)

    for layer in self.middle:
        if isinstance(layer, ResidualBlock) and feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    for layer in self.head:
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)

    self.current_down_blocks = l_blocks
    return x


def my_vae_decoder_fwd(self, x, feat_cache=None, feat_idx=[0]):
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    skip_convs = [
        self.skip_conv_0,
        self.skip_conv_1,
        self.skip_conv_2,
        self.skip_conv_3,
        self.skip_conv_4,
        self.skip_conv_5,
        self.skip_conv_6,
        self.skip_conv_7,
        self.skip_conv_8,
    ]
    skip_acts = self.incoming_skip_acts

    for layer in self.middle:
        if isinstance(layer, ResidualBlock) and feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    enc_dec_mapping = {0: 0, 1: 1, 2: 2, 5: 3, 6: 4, 8: 6, 10: 7, 12: 9, 14: 10}

    for dec_idx, layer in enumerate(self.upsamples):
        if dec_idx in enc_dec_mapping:
            enc_indx = enc_dec_mapping[dec_idx]
            layer_index = list(enc_dec_mapping.keys()).index(dec_idx)
            skip_input = skip_acts[::-1][enc_indx]
            skip_in = skip_convs[layer_index](skip_input)
            x = x + skip_in

        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    for layer in self.head:
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x


def load_compatible_module_state(module, incoming_state, *, label):
    """Load matching checkpoint tensors and report architecture-specific skips.

    PyTorch's ``strict=False`` still raises for a known key whose shape changed.
    Camera-only, time-only, and combined conditioning intentionally use
    different projection widths, so migration through ``--pretrained_path``
    must filter those tensors before calling ``load_state_dict``.
    """
    current_state = module.state_dict()
    compatible = {}
    mismatched = []
    unexpected_input = []
    for name, value in incoming_state.items():
        current_value = current_state.get(name)
        if current_value is None:
            unexpected_input.append(name)
        elif tuple(current_value.shape) != tuple(value.shape):
            mismatched.append((name, tuple(value.shape), tuple(current_value.shape)))
        else:
            compatible[name] = value

    incompatible = module.load_state_dict(compatible, strict=False)
    print(
        f"  {label} load: {len(incompatible.missing_keys)} missing, "
        f"{len(unexpected_input)} unexpected, {len(mismatched)} shape-mismatched"
    )
    if mismatched:
        preview = ", ".join(f"{name}: {old_shape}->{new_shape}" for name, old_shape, new_shape in mismatched[:5])
        print(f"    skipped incompatible {label} tensors: {preview}")
    if unexpected_input:
        print(f"    first unexpected {label} keys: {unexpected_input[:5]}")
    return incompatible, mismatched, unexpected_input


class DrivenVSModel(torch.nn.Module):
    """Cosmos restoration model with Option-A hybrid references."""

    def __init__(
        self,
        pretrained_path=None,
        freeze_vae=False,
        train_full_unet=True,
        vae_skip_connection=False,
        batch_size=1,
        num_reference_images=0,
        camera_embed=False,
        time_embed=False,
        camera_translation_scale=1.0,
        reference_time_embedding_dim=32,
        reference_time_scale=1.0,
        enforce_conditioning_config=False,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        world_scenario_mode=0,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_reference_images = num_reference_images
        self.camera_embed = bool(camera_embed)
        self.time_embed = bool(time_embed)
        self.camera_translation_scale = float(camera_translation_scale)
        self.reference_time_embedding_dim = int(reference_time_embedding_dim)
        self.reference_time_scale = float(reference_time_scale)
        self.enforce_conditioning_config = bool(enforce_conditioning_config)
        if self.num_reference_images < 0:
            raise ValueError(f"num_reference_images must be non-negative, got {self.num_reference_images}")
        if self.camera_embed and (
            not math.isfinite(self.camera_translation_scale) or self.camera_translation_scale <= 0
        ):
            raise ValueError(f"camera_translation_scale must be positive, got {self.camera_translation_scale}")
        if self.time_embed and (self.reference_time_embedding_dim <= 0 or self.reference_time_embedding_dim % 2):
            raise ValueError(
                f"reference_time_embedding_dim must be a positive even integer, got {self.reference_time_embedding_dim}"
            )
        if self.time_embed and (not math.isfinite(self.reference_time_scale) or self.reference_time_scale <= 0):
            raise ValueError(f"reference_time_scale must be positive, got {self.reference_time_scale}")
        if self.num_reference_images and not freeze_vae:
            raise ValueError(
                "Hybrid reference conditioning requires --freeze_vae so reference features come from a fixed VAE"
            )

        self.train_full_unet = train_full_unet
        self.freeze_vae = freeze_vae
        self.world_scenario_mode = world_scenario_mode
        self.vae_skip_connection = vae_skip_connection

        self.initialize_cosmos_model(device=device, dtype=dtype)
        if self.world_scenario_mode:
            if not freeze_vae:
                raise ValueError("World Scenario requires the frozen Cosmos VAE")
            # Keep original reference initialization and sampler RNG identical
            # across the two new architectures at the same experiment seed.
            rng_devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
            with torch.random.fork_rng(devices=rng_devices):
                self.unet.dit.enable_world_scenario(world_scenario_mode)
        if self.num_reference_images:
            self._initialize_reference_conditioning()
        self.set_train()

        if pretrained_path is not None:
            print("loading from", pretrained_path)
            sd = torch.load(pretrained_path, map_location="cpu")
            if self.enforce_conditioning_config:
                validate_checkpoint_conditioning(
                    self,
                    sd,
                    label="pretrained checkpoint",
                )
            # Legacy harmonizer checkpoints stored DiT weights under `net.` / `net_ema.`
            # prefixes. Current Text2ImagePipeline exposes them as `dit` / `dit_ema`,
            # so remap before loading. With strict=False, a prefix mismatch would
            # otherwise silently drop every UNet weight.
            unet_sd = _remap_legacy_unet_state(sd["state_dict_unet"])
            if self.enforce_conditioning_config:
                # Stage-2 initialization and inference ask for an exact Option-A
                # architecture. Metadata agreement alone is not enough: a partial
                # load could silently leave a missing joint/KV/camera weight at its
                # fresh initialization and still produce outputs. Require every
                # tensor to match in exact mode. Stage-1 migration intentionally
                # keeps the permissive compatible-load path below.
                try:
                    self.unet.load_state_dict(unet_sd, strict=True)
                    self._inner_vae().load_state_dict(sd["state_dict_vae"], strict=True)
                except RuntimeError as exc:
                    raise ValueError(
                        "Exact Option-A checkpoint loading failed after conditioning "
                        "metadata validation. The checkpoint is missing, has extra, "
                        "or has shape-mismatched model tensors. Use an exact Option-A "
                        "checkpoint for Stage 2/inference; use Stage-1 --pretrained_path "
                        "only when intentionally migrating an older architecture."
                    ) from exc
            else:
                load_compatible_module_state(self.unet, unet_sd, label="unet")
                load_compatible_module_state(
                    self._inner_vae(),
                    sd["state_dict_vae"],
                    label="vae",
                )

        print("=" * 50)
        print(
            f"Number of trainable parameters in UNet: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad) / 1e6:.2f}M"
        )
        print(
            f"Number of trainable parameters in VAE: {sum(p.numel() for p in self._inner_vae().parameters() if p.requires_grad) / 1e6:.2f}M"
        )
        print("=" * 50)

    def initialize_cosmos_model(self, device: str | torch.device, dtype: torch.dtype):
        config = get_pipeline_config()
        model = Text2ImagePipeline.from_config(config, use_text_encoder=False, dit_path=config.dit_path)

        # Build an unconditional condition matching the legacy training setup
        def sample_batch_image(h: int = 544, w: int = 960):
            return {
                "dataset_name": "image_data",
                "images": torch.zeros(self.batch_size, 3, h, w).to(device, dtype=dtype),
                "t5_text_embeddings": torch.zeros(self.batch_size, 512, 1024).to(device, dtype=dtype),
                "fps": torch.ones((self.batch_size,)).to(device, dtype=dtype) * 24,
                "padding_mask": torch.zeros(self.batch_size, 1, h, w).to(device, dtype=dtype),
            }

        conditioner = model.conditioner
        condition, uncondition = conditioner.get_condition_uncondition(sample_batch_image())
        del condition
        uncondition = uncondition.edit_data_type(DataType.IMAGE)
        self.condition = uncondition

        self.unet = model
        self.sigma_data = model.sigma_data
        self.vae = model.tokenizer

        if self.vae_skip_connection:
            inner_vae = self._inner_vae()
            print("------ adding skip connection convs to VAE decoder")
            for ch, idx_list in [(384, [0, 1, 2, 3, 4]), (192, [5, 6]), (96, [7, 8])]:
                for i in idx_list:
                    conv = torch.nn.Conv3d(ch, ch, kernel_size=1, stride=1, bias=False).to(device)
                    torch.nn.init.constant_(conv.weight, 1e-5)
                    setattr(inner_vae.decoder, f"skip_conv_{i}", conv)
            inner_vae.encoder.forward = my_vae_encoder_fwd.__get__(inner_vae.encoder, inner_vae.encoder.__class__)
            inner_vae.decoder.forward = my_vae_decoder_fwd.__get__(inner_vae.decoder, inner_vae.decoder.__class__)

        print("=" * 50)
        print("SUCCESS in initializing Cosmos model")
        print(f"Number of parameters in UNet: {sum(p.numel() for p in self.unet.parameters()) / 1e6:.2f}M")
        print(f"Number of parameters in VAE: {sum(p.numel() for p in self._inner_vae().parameters()) / 1e6:.2f}M")
        print("=" * 50)
        self.unet.to(device)
        self._inner_vae().to(device)

    def _initialize_reference_conditioning(self):
        """Install Option-A hybrid reference fusion and optional geometry."""
        if not hasattr(self.unet, "dit") or not hasattr(self.unet.dit, "enable_hybrid_reference_fusion"):
            raise AttributeError("The active Cosmos DiT does not expose Option-A hybrid reference fusion")
        self.unet.dit.enable_hybrid_reference_fusion()
        spatial_compression = self._vae_spatial_compression_factor()
        camera_patch_dim = None
        if self.camera_embed:
            token_pixel_size = spatial_compression * self.unet.dit.patch_spatial
            camera_patch_dim = 6 * token_pixel_size * token_pixel_size
        if self.camera_embed or self.time_embed:
            self.unet.dit.enable_camera_conditioning(
                camera_patch_dim=camera_patch_dim,
                reference_time_embedding_dim=(self.reference_time_embedding_dim if self.time_embed else None),
                reference_time_scale=(self.reference_time_scale if self.time_embed else 1.0),
            )
        enabled = [
            "top-1 joint self-attention",
            f"remaining {max(0, self.num_reference_images - 1)} reference K/V",
        ]
        if self.camera_embed:
            enabled.append("camera rays + PRoPE")
        if self.time_embed:
            enabled.append("signed time Fourier embedding")
        print(
            "Option-A hybrid reference conditioning enabled: "
            f"R={self.num_reference_images}, frozen VAE, shared patch embedder, "
            f"{', '.join(enabled)} (VAE scale={spatial_compression}, "
            f"ray patch dim={camera_patch_dim})"
        )

    def _vae_spatial_compression_factor(self) -> int:
        """Resolve Cosmos's fixed spatial compression from tokenizer wrappers."""
        queue = [self.vae, self._inner_vae()]
        visited = set()
        for _ in range(12):
            if not queue:
                break
            module = queue.pop(0)
            if id(module) in visited:
                continue
            visited.add(id(module))
            value = getattr(module, "spatial_compression_factor", None)
            if value is not None:
                if isinstance(value, (tuple, list)):
                    if len(value) != 2 or value[0] != value[1]:
                        raise ValueError(f"Expected isotropic VAE compression, got {value}")
                    value = value[0]
                factor = int(value)
                if factor <= 0:
                    raise ValueError(f"Invalid VAE spatial compression factor {factor}")
                return factor
            for attr in ("model", "tokenizer", "module", "vae"):
                child = getattr(module, attr, None)
                if child is not None:
                    queue.append(child)
        # Cosmos-Predict2's non-fast tokenizer uses 16x spatial compression.
        # The DiT validates this against the actual ray/token grids.
        return 16

    def _inner_vae(self):
        # Resolve to the underlying VAE module that owns encoder/decoder. Different
        # cosmos_predict2 builds wrap the tokenizer at different depths; walk down
        # until we hit something with .encoder and .decoder.
        cur = self.vae
        for _ in range(4):
            if hasattr(cur, "encoder") and hasattr(cur, "decoder"):
                return cur
            for attr in ("model", "tokenizer", "module"):
                if hasattr(cur, attr):
                    cur = getattr(cur, attr)
                    break
            else:
                break
        return cur

    def set_eval(self):
        inner = self._inner_vae()
        self.unet.eval()
        inner.eval()
        self.unet.requires_grad_(False)
        inner.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        if self.train_full_unet:
            self.unet.requires_grad_(True)
        else:
            raise ValueError("!! train partial Unet not implemented")

        inner = self._inner_vae()
        inner.train()
        inner.requires_grad_(True)
        for name, param in inner.named_parameters():
            if "time_conv" in name:
                param.requires_grad = False

        if self.freeze_vae:
            inner.requires_grad_(False)
            inner.eval()

    def vae_encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(state) * self.sigma_data

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.vae_skip_connection:
            inner = self._inner_vae()
            inner.decoder.incoming_skip_acts = inner.encoder.current_down_blocks
        return self.vae.decode(latent / self.sigma_data)

    @staticmethod
    def _normalize_batch_values(
        values: torch.Tensor | float | int,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> torch.Tensor:
        values = torch.as_tensor(values, device=device, dtype=dtype)
        if values.ndim == 0:
            values = values.expand(batch_size)
        elif values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        elif values.ndim != 1:
            raise ValueError(f"{name} must be scalar, [B], or [B,1], got {tuple(values.shape)}")

        if values.shape[0] == 1 and batch_size != 1:
            values = values.expand(batch_size)
        if values.shape[0] != batch_size:
            raise ValueError(f"{name} batch {values.shape[0]} does not match input batch {batch_size}")
        return values

    def _predict_velocity(
        self,
        latent: torch.Tensor,
        *,
        flow_times: torch.Tensor | float,
        validate_times: bool = True,
    ) -> torch.Tensor:
        """Predict the native Cosmos rectified-flow field directly.

        Cosmos Predict2's ``RectifiedFlowScaling`` feeds the DiT

            z_t = (1 - t) * x0 + t * x1

        and converts the raw DiT output ``v_hat`` back to a clean prediction as

            x0_hat = z_t - t * v_hat.

        ``latent`` here is already ``z_t``, so calling ``unet.denoise`` would
        incorrectly multiply it by ``c_in = 1 - t`` again.  Call the DiT
        directly and provide Cosmos's native RF time embedding instead.
        """
        if latent.ndim != 5:
            raise ValueError(f"Expected latent [B,C,T,H,W], got {tuple(latent.shape)}")

        flow_time_B = self._normalize_batch_values(
            flow_times,
            batch_size=latent.shape[0],
            device=latent.device,
            dtype=torch.float32,
            name="flow_times",
        )
        if validate_times:
            if not torch.isfinite(flow_time_B).all():
                raise ValueError("flow_times must be finite")
            if (flow_time_B < 0).any() or (flow_time_B > 1).any():
                raise ValueError(
                    f"flow_times must be in [0, 1], got [{flow_time_B.min().item()}, {flow_time_B.max().item()}]"
                )

        scaling = getattr(self.unet, "scaling", None)
        if not isinstance(scaling, RectifiedFlowScaling):
            raise RuntimeError(
                "Direct velocity prediction requires Cosmos RectifiedFlowScaling; "
                f"got {type(scaling).__name__ if scaling is not None else None}"
            )
        time_scale = float(getattr(scaling, "t_scaling_factor", 1.0))
        timesteps_B_T = (flow_time_B * time_scale).view(latent.shape[0], 1)

        tensor_kwargs = getattr(
            self.unet,
            "tensor_kwargs",
            {"device": latent.device, "dtype": latent.dtype},
        )
        condition_kwargs = self.condition.to_dict()
        return self.unet.dit(
            x_B_C_T_H_W=latent.to(**tensor_kwargs),
            timesteps_B_T=timesteps_B_T.to(**tensor_kwargs),
            **condition_kwargs,
        ).float()

    @staticmethod
    def prepare_opacity_latent(
        opacity: torch.Tensor,
        source_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Downsample soft GS opacity/coverage to the VAE latent grid.

        As in ArtiFixer, max pooling preserves the strongest coverage in each
        latent bin.  Continuous opacity is retained, so partially covered cells
        interpolate between the rendered latent and Gaussian source noise.
        """
        if opacity.ndim == 4:
            opacity = opacity.unsqueeze(2)
        if opacity.ndim != 5:
            raise ValueError(f"Expected opacity [B,1,H,W] or [B,1,V,H,W], got {tuple(opacity.shape)}")
        if opacity.shape[0] != source_latent.shape[0] or opacity.shape[1] != 1:
            raise ValueError(
                "Opacity must have shape [B,1,T,H,W] with the same batch as "
                f"the source latent; got opacity={tuple(opacity.shape)}, "
                f"source={tuple(source_latent.shape)}"
            )

        opacity = opacity.to(
            device=source_latent.device,
            dtype=torch.float32,
        ).clamp_(0.0, 1.0)
        # Pool spatially per RGB view first. Temporal pooling must keep view 0
        # separate: the causal VAE maps V=5 RGB frames to two latent time cells,
        # and mixing the four all-valid history masks into cell 0 would erase the
        # current frame's missing regions.
        batch_size, _, opacity_views, opacity_height, opacity_width = opacity.shape
        latent_views, latent_height, latent_width = source_latent.shape[-3:]
        opacity_per_view = opacity.permute(0, 2, 1, 3, 4).reshape(
            batch_size * opacity_views, 1, opacity_height, opacity_width
        )
        opacity_per_view = F.adaptive_max_pool2d(
            opacity_per_view,
            output_size=(latent_height, latent_width),
        )
        opacity_latent = opacity_per_view.reshape(batch_size, opacity_views, 1, latent_height, latent_width).permute(
            0, 2, 1, 3, 4
        )
        if opacity_views != latent_views:
            if opacity_views < latent_views:
                raise ValueError(
                    f"The VAE expanded V={opacity_views} opacity views to "
                    f"V={latent_views} latent views; this unsupported mapping "
                    "cannot preserve the current-frame mask"
                )

            current_opacity = opacity_latent[:, :, :1]
            if latent_views == 1:
                opacity_latent = current_opacity
            else:
                history_opacity = opacity_latent[:, :, 1:]
                if history_opacity.shape[2] < latent_views - 1:
                    raise ValueError(f"There are not enough history opacity views for the V={latent_views} latent grid")
                history_opacity = F.adaptive_max_pool3d(
                    history_opacity,
                    output_size=(
                        latent_views - 1,
                        latent_height,
                        latent_width,
                    ),
                )
                opacity_latent = torch.cat(
                    (current_opacity, history_opacity),
                    dim=2,
                )
        if opacity_latent.shape != (
            batch_size,
            1,
            latent_views,
            latent_height,
            latent_width,
        ):
            raise RuntimeError(f"Opacity-to-latent conversion produced an invalid shape: {tuple(opacity_latent.shape)}")
        return opacity_latent.to(dtype=source_latent.dtype)

    @staticmethod
    def prepare_opacity_source_endpoint(
        source_latent: torch.Tensor,
        opacity_latent: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep confident GS latents and replace uncovered cells with N(0, I)."""
        source_noise = torch.randn(
            source_latent.shape,
            device=source_latent.device,
            dtype=source_latent.dtype,
            generator=generator,
        )
        source_endpoint = opacity_latent * source_latent + (1.0 - opacity_latent) * source_noise
        return source_endpoint, source_noise

    @staticmethod
    def _visibility_reference_order(
        reference_visibility_scores: torch.Tensor | None,
        *,
        batch_size: int,
        num_references: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Put the highest-visibility reference first for Option A.

        The manifest may provide one visibility score per reference.  If no
        scores are supplied, reference list order is the backward-compatible
        contract: index 0 is treated as the best reference.  Remaining views
        preserve their original relative order so the seven-reference K/V
        memory stays deterministic.
        """
        if num_references < 1:
            raise ValueError("Option A requires at least one reference")
        if reference_visibility_scores is None:
            best = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            scores = torch.as_tensor(
                reference_visibility_scores,
                device=device,
                dtype=torch.float32,
            )
            if scores.ndim == 1 and batch_size == 1:
                scores = scores.unsqueeze(0)
            if tuple(scores.shape) != (batch_size, num_references):
                raise ValueError(
                    "reference_visibility_scores must have shape [B,R] = "
                    f"{(batch_size, num_references)}, got {tuple(scores.shape)}"
                )
            if not torch.isfinite(scores).all():
                raise ValueError("reference_visibility_scores must be finite")
            best = scores.argmax(dim=1)

        base = torch.arange(num_references, device=device, dtype=torch.long)
        orders = []
        for batch_index in range(batch_size):
            best_index = best[batch_index]
            remaining = base[base != best_index]
            orders.append(torch.cat((best_index.view(1), remaining), dim=0))
        return torch.stack(orders, dim=0), best

    @staticmethod
    def _gather_reference_axis(values: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        """Gather a ``[B,R,...]`` tensor with a per-batch reference order."""
        if values.ndim < 2:
            raise ValueError(f"Reference tensor must be at least 2D, got {tuple(values.shape)}")
        if tuple(values.shape[:2]) != tuple(order.shape):
            raise ValueError(
                f"Reference gather shape mismatch: values[:2]={tuple(values.shape[:2])}, order={tuple(order.shape)}"
            )
        index = order.view(*order.shape, *([1] * (values.ndim - 2))).expand_as(values)
        return torch.gather(values, 1, index)

    def _set_reference_conditioning(
        self,
        reference_images: torch.Tensor | None,
        *,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        target_c2w: torch.Tensor | None,
        target_K: torch.Tensor | None,
        reference_c2ws: torch.Tensor | None,
        reference_Ks: torch.Tensor | None,
        reference_time_offsets: torch.Tensor | None,
        reference_visibility_scores: torch.Tensor | None,
        world_scenario=None,
        reference_world_scenario=None,
    ) -> bool:
        """Validate and install image, camera, and temporal reference state.

        Runtime state is installed once around a training forward or complete
        multi-step inference pass and must be cleared by the caller.
        """
        if self.num_reference_images:
            if reference_images is None:
                raise ValueError(f"Expected {self.num_reference_images} reference images, got none")
            expected_shape = (
                batch_size,
                self.num_reference_images,
                channels,
                height,
                width,
            )
            if tuple(reference_images.shape) != expected_shape:
                raise ValueError(
                    f"Expected reference images with shape {expected_shape}, got {tuple(reference_images.shape)}"
                )

            geometry_device = reference_images.device
            reference_order, best_reference_indices = self._visibility_reference_order(
                reference_visibility_scores,
                batch_size=batch_size,
                num_references=self.num_reference_images,
                device=geometry_device,
            )
            reference_images = self._gather_reference_axis(
                reference_images,
                reference_order,
            )
            if self.world_scenario_mode:
                if world_scenario is None or reference_world_scenario is None:
                    raise ValueError("World Scenario mode requires current/history and reference conditions")
                with torch.no_grad():
                    self.unet.dit._runtime_world_latent = self.vae_encode(world_scenario).detach()
                    ordered_world = self._gather_reference_axis(reference_world_scenario, reference_order)
                    self.unet.dit._runtime_reference_world_latent = self.encode_reference_images(ordered_world)

            # Exposed only for diagnostics/logging; it is not part of the
            # learned state and is overwritten for each sample/batch.
            self._last_joint_reference_indices = best_reference_indices.detach().cpu()

            runtime_conditioning = {}
            if self.camera_embed:
                camera_inputs = {
                    "target_c2w": target_c2w,
                    "target_K": target_K,
                    "reference_c2ws": reference_c2ws,
                    "reference_Ks": reference_Ks,
                }
                missing_camera = [name for name, value in camera_inputs.items() if value is None]
                if missing_camera:
                    raise ValueError(f"Missing inputs required by --camera_embed: {missing_camera}")
                target_c2w = torch.as_tensor(
                    target_c2w,
                    device=geometry_device,
                    dtype=torch.float32,
                )
                target_K = torch.as_tensor(
                    target_K,
                    device=geometry_device,
                    dtype=torch.float32,
                )
                reference_c2ws = torch.as_tensor(
                    reference_c2ws,
                    device=geometry_device,
                    dtype=torch.float32,
                )
                reference_Ks = torch.as_tensor(
                    reference_Ks,
                    device=geometry_device,
                    dtype=torch.float32,
                )
                reference_c2ws = self._gather_reference_axis(reference_c2ws, reference_order)
                reference_Ks = self._gather_reference_axis(reference_Ks, reference_order)
                runtime_conditioning.update(
                    build_relative_camera_conditioning(
                        target_c2w=target_c2w,
                        target_K=target_K,
                        reference_c2ws=reference_c2ws,
                        reference_Ks=reference_Ks,
                        image_height=height,
                        image_width=width,
                        translation_scale=self.camera_translation_scale,
                    )
                )
            if self.time_embed:
                if reference_time_offsets is None:
                    raise ValueError("Missing reference_time_offsets required by --time_embed")
                runtime_conditioning["reference_time_offsets"] = torch.as_tensor(
                    reference_time_offsets,
                    device=geometry_device,
                    dtype=torch.float32,
                )
                runtime_conditioning["reference_time_offsets"] = self._gather_reference_axis(
                    runtime_conditioning["reference_time_offsets"],
                    reference_order,
                )

            reference_latents = self.encode_reference_images(reference_images)
            self.unet.dit.set_reference_latents(reference_latents)
            if runtime_conditioning:
                try:
                    self.unet.dit.set_camera_conditioning(**runtime_conditioning)
                except Exception:
                    self.unet.dit.clear_reference_latents()
                    self.unet.dit.clear_camera_conditioning()
                    raise
            return True

        unexpected = {
            "reference_images": reference_images,
            "target_c2w": target_c2w,
            "target_K": target_K,
            "reference_c2ws": reference_c2ws,
            "reference_Ks": reference_Ks,
            "reference_time_offsets": reference_time_offsets,
            "reference_visibility_scores": reference_visibility_scores,
        }
        provided = [name for name, value in unexpected.items() if value is not None]
        if provided:
            raise ValueError(f"Reference/camera inputs were provided, but num_reference_images=0: {provided}")
        return False

    def vae_encode_views(self, state: torch.Tensor) -> torch.Tensor:
        """Encode each view independently, then restore the view dimension.

        ``(B, C, V, H, W) -> (B*V, C, 1, H, W) ->
        (B, C_latent, V, H_latent, W_latent)``

        The VAE always sees a one-frame sample, so its causal temporal kernels
        never mix the degraded view with its clean reference view.
        """
        if state.ndim != 5:
            raise ValueError(f"Expected RGB views [B,C,V,H,W], got {tuple(state.shape)}")

        batch_size, channels, num_views, height, width = state.shape
        if num_views < 1:
            raise ValueError("At least one input view is required")

        state_flat = (
            state.permute(0, 2, 1, 3, 4).reshape(batch_size * num_views, channels, 1, height, width).contiguous()
        )
        latent_flat = self.vae_encode(state_flat)
        if latent_flat.ndim != 5:
            raise ValueError(f"Expected the VAE encoder to return [B*V,C,T,H,W], got {tuple(latent_flat.shape)}")

        flat_batch, latent_channels, latent_time, latent_height, latent_width = latent_flat.shape
        if flat_batch != batch_size * num_views:
            raise ValueError(f"VAE encoder changed the flattened batch from {batch_size * num_views} to {flat_batch}")
        if latent_time != 1:
            raise ValueError(
                f"Independent view encoding requires one latent frame per view; the VAE returned T={latent_time}"
            )

        return (
            latent_flat.reshape(
                batch_size,
                num_views,
                latent_channels,
                latent_height,
                latent_width,
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    @torch.no_grad()
    def encode_reference_images(self, reference_images: torch.Tensor) -> torch.Tensor:
        """Independently encode clean source references with the frozen VAE."""
        if reference_images.ndim != 5:
            raise ValueError(f"Expected reference images [B,R,C,H,W], got {tuple(reference_images.shape)}")
        reference_views = reference_images.permute(0, 2, 1, 3, 4).contiguous()
        return self.vae_encode_views(reference_views).detach()

    def forward(
        self,
        x,
        timesteps=None,
        reference_images=None,
        target_c2w=None,
        target_K=None,
        reference_c2ws=None,
        reference_Ks=None,
        reference_time_offsets=None,
        reference_visibility_scores=None,
        target_images=None,
        opacity=None,
        return_training_latents=False,
        world_scenario=None,
        reference_world_scenario=None,
    ):
        """Train one ArtiFixer-style rectified-flow state with reference memory.

        Args:
            x: Target path ``[B,C,V,H,W]``. View 0 is the current warped
                image; optional views 1..4 are clean past targets in Stage 2.
            timesteps: Per-sample rectified-flow times ``t`` in ``(0, 1]``.
            reference_images: Source references ``[B,R,C,H,W]``.
            target_c2w: Target camera-to-world matrices ``[B,V,4,4]``.
            target_K: Normalized centered target intrinsics ``[B,V,3,3]``.
            reference_c2ws: Reference camera-to-world matrices ``[B,R,4,4]``.
            reference_Ks: Normalized centered reference intrinsics ``[B,R,3,3]``.
            reference_time_offsets: Signed reference-minus-target times
                ``[B,R]`` in the consistent unit selected by the dataset.
            reference_visibility_scores: Per-reference visibility ranking
                scores ``[B,R]``. The highest score is promoted to the joint
                self-attention stream; if omitted, reference index 0 is used.
            target_images: Clean targets ``[B,C,V,H,W]``. View 0 is
                current GT and temporal history views equal those in ``x``.
            opacity: White-valid GS coverage ``[B,1,H,W]`` or
                ``[B,1,1,H,W]``. Required during bridge training.
            return_training_latents: Return latent targets and bridge states
                used by the training losses and diagnostics.

        Returns:
            Enhanced target path ``[B,C,V,H,W]`` or a training dictionary.
        """
        if x.ndim != 5:
            raise ValueError(f"Expected target path [B,C,V,H,W], got {tuple(x.shape)}")
        if target_images is None:
            raise ValueError("Training forward requires target_images")
        if timesteps is None:
            raise ValueError("Training forward requires bridge timesteps")
        if opacity is None:
            raise ValueError("Training forward requires confident-coverage opacity")

        batch_size, channels, num_target_views, height, width = x.shape
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
            if tuple(target_images.shape) != tuple(x.shape):
                raise ValueError(f"target_images shape {tuple(target_images.shape)} must match input {tuple(x.shape)}")
            flow_times = self._normalize_batch_values(
                timesteps,
                batch_size=batch_size,
                device=x.device,
                dtype=torch.float32,
                name="flow timesteps",
            )
            if not torch.isfinite(flow_times).all():
                raise ValueError("flow timesteps must be finite")
            if (flow_times <= 0).any() or (flow_times > 1.0).any():
                raise ValueError(
                    f"flow timesteps must be in (0, 1], got [{flow_times.min().item()}, {flow_times.max().item()}]"
                )

            # The VAE is frozen. Decode remains differentiable with respect to
            # the predicted latent while encoding retains no activation graph.
            with torch.no_grad():
                target_latent = self.vae_encode(target_images).detach()
                warped_latent = self.vae_encode(x).detach()
                opacity_latent = self.prepare_opacity_latent(opacity, warped_latent)
                source_latent, source_noise = self.prepare_opacity_source_endpoint(
                    warped_latent,
                    opacity_latent,
                )

            # ArtiFixer rectified-flow path:
            #   z_t = (1 - t) z_target + t z_source
            #   v*  = z_source - z_target
            model_input = FlowMatchScheduler.add_noise(
                target_latent,
                source_latent,
                flow_times,
            )
            flow_times_latent = flow_times.float().view(batch_size, 1, 1, 1, 1)

            # Native Cosmos Predict2 RF parameterization:
            #   DiT(z_t, t) -> v_hat
            #   x0_hat = z_t - t * v_hat
            # Calling unet.denoise() here would be wrong because model_input is
            # already z_t and denoise() would multiply it by c_in=(1-t) again.
            predicted_velocity = self._predict_velocity(
                model_input,
                flow_times=flow_times,
            )
            target_velocity = FlowMatchScheduler.training_target(
                target_latent,
                source_latent,
            ).float()
            predicted_x0 = model_input.float() - flow_times_latent * predicted_velocity
        finally:
            if clear_references:
                self.unet.dit.clear_reference_latents()
                self.unet.dit.clear_camera_conditioning()

        output_image = self.vae_decode(predicted_x0)

        if return_training_latents:
            return {
                "output_image": output_image,
                "predicted_x0": predicted_x0,
                "target_x0": target_latent,
                "predicted_velocity": predicted_velocity,
                "target_velocity": target_velocity,
                "source_latent": source_latent,
                "warped_latent": warped_latent,
                "source_noise": source_noise,
                "opacity_latent": opacity_latent,
                "model_input": model_input,
                "flow_times": flow_times,
            }
        return output_image

    @torch.no_grad()
    def sample_bridge(
        self,
        x: torch.Tensor,
        *,
        reference_images: torch.Tensor | None = None,
        target_c2w: torch.Tensor | None = None,
        target_K: torch.Tensor | None = None,
        reference_c2ws: torch.Tensor | None = None,
        reference_Ks: torch.Tensor | None = None,
        reference_time_offsets: torch.Tensor | None = None,
        reference_visibility_scores: torch.Tensor | None = None,
        opacity: torch.Tensor,
        world_scenario=None,
        reference_world_scenario=None,
        num_steps: int = 4,
        timestep_shift: float = 5.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Run ArtiFixer-style rectified-flow inference for V=1 or V=5.

        The opacity-guided Gaussian source endpoint is created once.  Each of
        ``num_steps`` Cosmos DiT evaluations predicts the RF velocity directly
        and advances the current latent with an Euler step.  With
        ``num_steps=4`` this remains exactly four DiT evaluations.
        """
        if x.ndim != 5:
            raise ValueError(f"Expected target path [B,C,V,H,W], got {tuple(x.shape)}")
        batch_size, channels, num_views, height, width = x.shape
        if num_views < 1:
            raise ValueError("Bridge sampling requires at least one target view")

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
            opacity_latent = self.prepare_opacity_latent(
                opacity,
                warped_latent,
            )
            source_latent, _ = self.prepare_opacity_source_endpoint(
                warped_latent,
                opacity_latent,
                generator=generator,
            )
            scheduler = FlowMatchScheduler(
                num_inference_steps=num_steps,
                num_train_timesteps=1000,
                shift=timestep_shift,
                sigma_min=0.0,
                extra_one_step=True,
                device=source_latent.device,
            )
            scheduler.set_timesteps(
                num_steps,
                training=False,
                device=source_latent.device,
            )

            latent = source_latent
            for step_index, flow_time in enumerate(scheduler.sigmas):
                flow_time_B = flow_time.expand(batch_size)
                predicted_velocity = self._predict_velocity(
                    latent,
                    flow_times=flow_time_B,
                )
                latent = scheduler.step_by_index(
                    predicted_velocity,
                    step_index,
                    latent,
                )
        finally:
            if clear_references:
                self.unet.dit.clear_reference_latents()
                self.unet.dit.clear_camera_conditioning()

        output_image = self.vae_decode(latent)
        return output_image
