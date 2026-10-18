# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-stage DH-style temporal training with visual-reference conditioning.

Stage 1 uses one target-path input/target pair:

    input  = [warped(t)]
    target = [GT(t)]

Stage 2 mixes those V=1 samples with temporal V=5 samples:

    input  = [warped(t), GT(t-1), GT(t-2), GT(t-3), GT(t-4)]
    target = [GT(t),     GT(t-1), GT(t-2), GT(t-3), GT(t-4)]

Source images are independently encoded by the frozen VAE and patchified into
reference tokens. Option A promotes the highest-visibility reference into the
main self-attention stream with the RF target; the remaining references stay
in the added visual K/V branch, queried only by target tokens. Optional
target-relative ray patches and signed frame-time offsets condition all
reference tokens, while optional PRoPE aligns target Q with the remaining
reference K/V. Only output view 0 contributes to the latent and image losses;
the history views provide temporal context.

Unlike the original fixed-timestep one-step trainer, this script uses the
white-valid GS confident-coverage mask to form an ArtiFixer-style source
endpoint:

    z_source = A_z * z_dh + (1 - A_z) * epsilon
    z_t      = (1 - t) * z_target + t * z_source.

Both stages now use ArtiFixer-style rectified-flow supervision.  A 1000-level
shift=5 scheduler samples the bridge state, the primary latent target is

    v* = z_source - z_target,

and Cosmos Predict2's raw DiT output is supervised directly as ``v_hat`` with a
timestep-weighted velocity MSE. Image L2/LPIPS and the Stage-2 RAFT temporal
loss remain optional auxiliary clean-image supervision. Stage 2 starts from a
Stage-1 checkpoint; by default 25% of its samples are V=1 and 75% are V=5.
Adversarial loss is disabled.
"""

import gc
import json
import logging
import math
import os
import random
from datetime import datetime
from glob import glob

import diffusers
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from tqdm.auto import tqdm

from drivenvs_model import (
    DrivenVSModel,
    load_training_checkpoint,
    save_training_checkpoint,
)
from meanflow_ema import CPUParameterEMA
from meanflow_loss import meanflow_latent_loss, sample_meanflow_times
from meanflow_model import (
    MeanFlowDrivenVSModel,
    load_meanflow_weights,
    save_meanflow_checkpoint,
)
from temporal_loss import build_temporal_loss, get_dh_temporal_loss
from utils.batch_preparation import get_batch_center_index as get_batch_center_index
from utils.batch_preparation import prepare_training_batch
from utils.flow_match import FlowMatchScheduler
from utils.temporal import TEMPORAL_HISTORY_FRAMES
from utils.temporal import TEMPORAL_TARGET_VIEWS as TEMPORAL_TARGET_VIEWS
from utils.training_utils import DrivenVSDataset, parse_training_args
from utils.training_stop import TrainingStop


def setup_file_logger(args, accelerator):
    logger = logging.getLogger("drivenvs_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    if accelerator.is_main_process:
        log_dir = os.path.join(args.output_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))
        file_handler = logging.FileHandler(log_filename, mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        logger.info("Writing logs to %s", log_filename)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


def save_visual_tensor(tensor, out_path):
    image = tensor.float().detach().cpu()
    if image.min().item() < 0:
        image = image * 0.5 + 0.5
    image = image.clamp(0, 1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchvision.utils.save_image(image, out_path)


def save_visualizations(image_dict, output_dir, global_step, logger):
    viz_dir = os.path.join(output_dir, "visualizations", f"step_{global_step:07d}")
    for tag, tensors in image_dict.items():
        tag_dir = os.path.join(viz_dir, tag.replace("/", "_"))
        for idx, tensor in enumerate(tensors):
            save_visual_tensor(tensor, os.path.join(tag_dir, f"idx_{idx:03d}.png"))
    logger.info("Saved visualization images for global_step=%d to %s", global_step, viz_dir)


def build_train_val_datasets(args):
    """Build one canonical train/validation dataset per JSON manifest."""
    sources = [source.strip() for source in args.dataset_folder.split(",") if source.strip()]
    if not sources:
        raise ValueError("--dataset_folder does not contain any dataset source")

    train_list = []
    val_list = []
    for source in sources:
        if not source.endswith(".json"):
            raise ValueError(
                f"train_drivenvs.py requires JSON manifests compatible with DrivenVSDataset; got: {source}"
            )

        print(f"------- Processing source {source}")
        ds_train = DrivenVSDataset(
            dataset_folder=source,
            split="train",
            image_prep=args.train_image_prep,
            num_reference_images=args.num_reference_images,
            camera_embed=args.camera_embed,
            time_embed=args.time_embed,
            training_stage=args.training_stage,
            temporal_sample_probability=(args.temporal_sample_probability if args.training_stage == "stage2" else 0.0),
            world_scenario=args.world_scenario_mode > 0,
        )
        ds_val = DrivenVSDataset(
            dataset_folder=args.validation_dataset or source,
            split="test",
            image_prep=args.test_image_prep,
            num_reference_images=args.num_reference_images,
            camera_embed=args.camera_embed,
            time_embed=args.time_embed,
            training_stage="stage1",
            temporal_sample_probability=0.0,
            world_scenario=args.world_scenario_mode > 0,
        )
        random.Random(42).shuffle(ds_val.img_names)
        train_list.append(ds_train)
        val_list.append(ds_val)

    return train_list, val_list


def select_current_prediction(prediction_all, center_index, target):
    if prediction_all.ndim != 5:
        raise ValueError(f"Expected model output [B,C,V,H,W], got {tuple(prediction_all.shape)}")
    if not 0 <= center_index < prediction_all.shape[2]:
        raise IndexError(f"center_index={center_index} is invalid for V={prediction_all.shape[2]}")

    prediction = prediction_all[:, :, center_index]
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}")
    return prediction


def validate_training_args(args):
    if args.training_objective == "meanflow":
        if os.environ.get("COSMOS_MODEL_SIZE") != "0.6B":
            raise ValueError("Stage-II MeanFlow requires COSMOS_MODEL_SIZE=0.6B")
        if not 0 <= args.meanflow_endpoint_probability <= args.meanflow_interval_probability <= 1:
            raise ValueError("Require 0 <= endpoint probability <= interval probability <= 1")
        if args.meanflow_adaptive_p < 0:
            raise ValueError("--meanflow_adaptive_p must be nonnegative")
        if (
            not math.isfinite(args.meanflow_time_mu)
            or not math.isfinite(args.meanflow_time_sigma)
            or args.meanflow_time_sigma <= 0
        ):
            raise ValueError("MeanFlow logit-normal parameters must be finite, with sigma > 0")
        if not 0 <= args.meanflow_ema_decay < 1:
            raise ValueError("--meanflow_ema_decay must be in [0, 1)")
    if args.bridge_timestep_shift <= 0:
        raise ValueError("--bridge_timestep_shift must be positive")
    if args.rf_num_train_timesteps < 1:
        raise ValueError("--rf_num_train_timesteps must be positive")
    if args.validation_steps < 1:
        raise ValueError("--validation_steps must be positive")
    for name in ("lambda_latent", "lambda_l2", "lambda_lpips"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be non-negative")
    if args.training_stage == "stage2":
        if args.lambda_temporal < 0:
            raise ValueError("--lambda_temporal must be non-negative")
        if args.lambda_temporal == 0 and args.training_objective != "meanflow":
            raise ValueError("Rectified-flow Stage 2 requires a positive --lambda_temporal")
        if args.temporal_loss_warmup_steps < 0:
            raise ValueError("--temporal_loss_warmup_steps must be non-negative")
    if args.camera_embed and (not math.isfinite(args.camera_translation_scale) or args.camera_translation_scale <= 0):
        raise ValueError("--camera_translation_scale must be positive")
    if args.time_embed and (args.reference_time_embedding_dim <= 0 or args.reference_time_embedding_dim % 2):
        raise ValueError("--reference_time_embedding_dim must be a positive even integer")
    if args.time_embed and (not math.isfinite(args.reference_time_scale) or args.reference_time_scale <= 0):
        raise ValueError("--reference_time_scale must be positive")
    if args.num_reference_images < 1:
        raise ValueError("--num_reference_images must be positive")
    if args.training_stage == "stage2":
        if not 0.0 <= args.temporal_sample_probability <= 1.0:
            raise ValueError("--temporal_sample_probability must be in [0, 1]")
        if args.training_objective != "meanflow" and args.pretrained_path is None and args.resume is None:
            raise ValueError(
                "Stage 2 must start from a Stage-1 checkpoint through "
                "--pretrained_path, or continue an existing Stage-2 run with --resume"
            )


def sample_rectified_flow_batch(scheduler, batch_size, device):
    """Sample ArtiFixer's discrete RF timestep distribution and weight."""
    _, timesteps, flow_times, weights = scheduler.sample_training_batch(
        batch_size,
        device=device,
    )
    return timesteps, flow_times, weights


def make_train_visuals(
    x_src,
    x_ref,
    x_tgt,
    x_tgt_pred,
    x_opacity,
    center_index,
):
    batch_size = x_src.shape[0]
    return {
        "train/context_window": [
            rearrange(x_src[idx], "c v h w -> c h (v w)").float().detach().cpu() for idx in range(batch_size)
        ],
        "train/current_input": [x_src[idx, :, center_index].float().detach().cpu() for idx in range(batch_size)],
        "train/current_target": [x_tgt[idx, :, center_index].float().detach().cpu() for idx in range(batch_size)],
        "train/current_prediction": [x_tgt_pred[idx].float().detach().cpu() for idx in range(batch_size)],
        "train/confident_coverage": [
            x_opacity[idx, :, center_index].float().detach().cpu() for idx in range(batch_size)
        ],
        "train/reference_window": [
            rearrange(x_ref[idx], "r c h w -> c h (r w)").float().detach().cpu() for idx in range(batch_size)
        ],
    }


def run_validation(
    args,
    accelerator,
    model,
    net_lpips,
    dl_val,
    weight_dtype,
    global_step,
    logger,
):
    l_l2 = []
    l_lpips = []
    image_dict = {
        "sample/context_window": [],
        "sample/reference_window": [],
        "sample/current_input": [],
        "sample/current_target": [],
        "sample/current_prediction": [],
        "sample/confident_coverage": [],
    }

    unwrapped_model = accelerator.unwrap_model(model)
    was_training = unwrapped_model.training
    unwrapped_model.eval()
    validation_generator = torch.Generator(
        device=accelerator.device,
    ).manual_seed(args.seed if args.seed is not None else 0)

    try:
        with torch.no_grad():
            for val_step, batch_val in enumerate(dl_val):
                if val_step >= args.num_samples_eval:
                    break

                (
                    x_src,
                    x_ref,
                    x_tgt,
                    x_opacity,
                    center_index,
                    camera_conditioning,
                ) = prepare_training_batch(
                    batch_val,
                    device=accelerator.device,
                    dtype=weight_dtype,
                    expected_reference_views=args.num_reference_images,
                    camera_embed=args.camera_embed,
                    time_embed=args.time_embed,
                )
                x_tgt_pred_all = unwrapped_model.sample_bridge(
                    x_src,
                    reference_images=x_ref,
                    opacity=x_opacity,
                    num_steps=args.validation_steps,
                    timestep_shift=args.bridge_timestep_shift,
                    generator=validation_generator,
                    **camera_conditioning,
                )
                x_tgt_current = x_tgt[:, :, center_index]
                x_tgt_pred = select_current_prediction(x_tgt_pred_all, center_index, x_tgt_current)

                l_l2.append(F.mse_loss(x_tgt_pred.float(), x_tgt_current.float()).item())
                l_lpips.append(net_lpips(x_tgt_pred.float(), x_tgt_current.float()).mean().item())

                if val_step % 10 == 0:
                    image_dict["sample/context_window"].append(
                        rearrange(x_src[0], "c v h w -> c h (v w)").float().detach().cpu()
                    )
                    image_dict["sample/reference_window"].append(
                        rearrange(x_ref[0], "r c h w -> c h (r w)").float().detach().cpu()
                    )
                    image_dict["sample/current_input"].append(x_src[0, :, center_index].float().detach().cpu())
                    image_dict["sample/current_target"].append(x_tgt_current[0].float().detach().cpu())
                    image_dict["sample/current_prediction"].append(x_tgt_pred[0].float().detach().cpu())
                    image_dict["sample/confident_coverage"].append(x_opacity[0, :, center_index].float().detach().cpu())

    finally:
        if was_training:
            unwrapped_model.train()
            if args.freeze_vae:
                unwrapped_model._inner_vae().eval()

    logs = {
        "val/l2": float(np.mean(l_l2)) if l_l2 else 0.0,
        "val/lpips": float(np.mean(l_lpips)) if l_lpips else 0.0,
        "val/num_steps": args.validation_steps,
    }
    logger.info(
        "validation global_step=%d steps=%d val/l2=%.6f val/lpips=%.6f",
        global_step,
        args.validation_steps,
        logs["val/l2"],
        logs["val/lpips"],
    )
    if accelerator.is_main_process:
        save_visualizations(image_dict, args.output_dir, global_step, logger)
    gc.collect()
    torch.cuda.empty_cache()
    return logs


def main(args):
    validate_training_args(args)
    if not args.freeze_vae:
        raise ValueError(
            "Bridge training requires --freeze_vae so source, target, "
            "and reference latents share a fixed coordinate system"
        )
    if args.train_batch_size != 1:
        raise ValueError(
            "DrivenVSModel caches an unconditional condition for a fixed "
            "batch size, while validation uses batch size 1. Use --train_batch_size 1."
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    logger = setup_file_logger(args, accelerator)
    if accelerator.is_main_process:
        logger.info("Args:\n%s", json.dumps(vars(args), indent=2, sort_keys=True, default=str))

    model_class = MeanFlowDrivenVSModel if args.training_objective == "meanflow" else DrivenVSModel
    model = model_class(
        freeze_vae=args.freeze_vae,
        world_scenario_mode=args.world_scenario_mode,
        train_full_unet=args.train_full_unet,
        vae_skip_connection=args.vae_skip_connection,
        pretrained_path=args.pretrained_path,
        batch_size=args.train_batch_size,
        num_reference_images=args.num_reference_images,
        camera_embed=args.camera_embed,
        time_embed=args.time_embed,
        camera_translation_scale=args.camera_translation_scale,
        reference_time_embedding_dim=args.reference_time_embedding_dim,
        reference_time_scale=args.reference_time_scale,
        enforce_conditioning_config=args.training_stage == "stage2",
        device=accelerator.device,
        dtype=weight_dtype,
    )
    model.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError("xformers is not available")
        model.unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net="vgg")
    net_lpips.requires_grad_(False)

    layers_to_opt = []
    if args.train_full_unet:
        print("=" * 50, "\nadding unet parameters\n", "=" * 50)
        layers_to_opt.extend(model.unet.parameters())
    if not layers_to_opt:
        raise ValueError("No trainable parameters were selected")

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    dataset_train_list, dataset_val_list = build_train_val_datasets(args)
    dataset_train = torch.utils.data.ConcatDataset(dataset_train_list)
    dataset_val = torch.utils.data.ConcatDataset(dataset_val_list)
    if len(dataset_train) == 0:
        raise ValueError("Training dataset is empty")

    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=args.train_batch_size > 1,
    )

    dl_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    global_step = 0
    resume_checkpoint = None
    if args.resume is not None:
        if os.path.isdir(args.resume):
            checkpoint_files = glob(os.path.join(args.resume, "*.pkl"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found: {args.resume}")
            checkpoint_files = sorted(
                checkpoint_files,
                key=lambda path: int(os.path.basename(path).replace("model_", "").replace(".pkl", "")),
            )
            resume_path = checkpoint_files[-1]
        elif args.resume.endswith(".pkl"):
            resume_path = args.resume
        else:
            raise ValueError(f"Invalid resume path: {args.resume}")

        global_step = int(os.path.basename(resume_path).replace("model_", "").replace(".pkl", ""))
        print("=" * 50, f"\nLoading checkpoint from {resume_path}\n", "=" * 50)
        logger.info("Loading checkpoint from %s", resume_path)
        if args.training_objective == "meanflow":
            checkpoint = load_meanflow_weights(model, resume_path)
            optimizer.load_state_dict(checkpoint["optimizer"])
            resume_checkpoint = checkpoint
        else:
            model, optimizer = load_training_checkpoint(model, optimizer, resume_path)
    else:
        print("=" * 50, "\nStarting a new optimizer run from pretrained weights\n", "=" * 50)
        logger.info("Starting a new optimizer run from pretrained weights")

    model.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    model, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        dl_train,
        lr_scheduler,
    )
    net_lpips = accelerator.prepare(net_lpips)

    meanflow_ema = None
    if args.training_objective == "meanflow" and args.meanflow_ema and accelerator.is_main_process:
        meanflow_ema = CPUParameterEMA(
            accelerator.unwrap_model(model).unet,
            decay=args.meanflow_ema_decay,
        )
        if resume_checkpoint is not None:
            if resume_checkpoint.get("ema_state_dict_unet") is None:
                raise ValueError("MeanFlow resume checkpoint has no EMA state")
            if not math.isclose(
                float(resume_checkpoint["ema_decay"]), args.meanflow_ema_decay, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("EMA decay differs from the resume checkpoint")
            meanflow_ema.load_state_dict(
                resume_checkpoint["ema_state_dict_unet"],
                num_updates=resume_checkpoint["ema_num_updates"],
            )

    # Match ArtiFixer training: 1000 discrete RF levels, shift=5 by default,
    # sigma_min=0, and an implicit extra terminal t=0 step.
    train_flow_scheduler = None
    if args.training_objective == "rectified_flow":
        train_flow_scheduler = FlowMatchScheduler(
            num_train_timesteps=args.rf_num_train_timesteps,
            shift=args.bridge_timestep_shift,
            sigma_min=0.0,
            extra_one_step=True,
            device=accelerator.device,
        )
        logger.info(
            "ArtiFixer RF scheduler: train_timesteps=%d shift=%.4f t_range=[%.6f, %.6f]",
            args.rf_num_train_timesteps,
            args.bridge_timestep_shift,
            train_flow_scheduler.sigmas.min().item(),
            train_flow_scheduler.sigmas.max().item(),
        )
    else:
        logger.info(
            "Official MeanFlow time sampling: %s mu=%.3f sigma=%.3f ratio_r_not_equal_t=%.3f endpoint_override=%.3f",
            args.meanflow_time_sampler,
            args.meanflow_time_mu,
            args.meanflow_time_sigma,
            args.meanflow_interval_probability,
            args.meanflow_endpoint_probability,
        )

    raft = None
    temporal_charbonnier = None
    if args.training_stage == "stage2" and args.lambda_temporal > 0:
        raft, temporal_charbonnier = build_temporal_loss(
            accelerator.device,
            estimator=args.flow_estimator,
            use_small=args.temporal_loss_use_small_raft,
            waft_repo=args.waft_repo,
            waft_config=args.waft_config,
            waft_checkpoint=args.waft_checkpoint,
            waft_iterations=args.waft_iterations,
        )
        logger.info(
            "Enabled DH target-path temporal loss: lambda=%.6f history=%d temporal_probability=%.3f estimator=%s",
            args.lambda_temporal,
            TEMPORAL_HISTORY_FRAMES,
            args.temporal_sample_probability,
            args.flow_estimator + ("-small" if args.temporal_loss_use_small_raft else "-large")
            if args.flow_estimator == "raft"
            else "waft",
        )

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.report_to in {"wandb", "all"}:
            init_kwargs["wandb"] = {"name": args.tracker_run_name}
        accelerator.init_trackers(
            args.tracker_project_name,
            config=dict(vars(args)),
            init_kwargs=init_kwargs,
        )

    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=min(global_step, args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    epoch = 0
    stop = TrainingStop(accelerator.device)
    stop_signal = 0
    last_saved_step = global_step if args.resume else None
    while global_step < args.max_train_steps and not stop_signal:
        for batch in dl_train:
            stop_signal = stop.requested()
            if stop_signal:
                break
            with accelerator.accumulate(model):
                (
                    x_src,
                    x_ref,
                    x_tgt,
                    x_opacity,
                    center_index,
                    camera_conditioning,
                ) = prepare_training_batch(
                    batch,
                    expected_reference_views=args.num_reference_images,
                    camera_embed=args.camera_embed,
                    time_embed=args.time_embed,
                )
                if args.training_objective == "meanflow":
                    meanflow_r, flow_times = sample_meanflow_times(
                        batch_size=x_src.shape[0],
                        device=x_src.device,
                        time_sampler=args.meanflow_time_sampler,
                        time_mu=args.meanflow_time_mu,
                        time_sigma=args.meanflow_time_sigma,
                        interval_probability=args.meanflow_interval_probability,
                        endpoint_probability=args.meanflow_endpoint_probability,
                    )
                    rf_timesteps = flow_times * args.rf_num_train_timesteps
                    flow_weights = torch.ones_like(flow_times)
                    training_result = model(
                        x_src,
                        r=meanflow_r,
                        t=flow_times,
                        reference_images=x_ref,
                        target_images=x_tgt,
                        opacity=x_opacity,
                        jvp_backend=args.meanflow_jvp_backend,
                        **camera_conditioning,
                    )
                else:
                    rf_timesteps, flow_times, flow_weights = sample_rectified_flow_batch(
                        train_flow_scheduler,
                        batch_size=x_src.shape[0],
                        device=x_src.device,
                    )
                    training_result = model(
                        x_src,
                        timesteps=flow_times,
                        reference_images=x_ref,
                        target_images=x_tgt,
                        opacity=x_opacity,
                        return_training_latents=True,
                        **camera_conditioning,
                    )
                    flow_times = training_result["flow_times"]
                x_tgt_pred_all = training_result["output_image"]
                x_tgt_current = x_tgt[:, :, center_index]
                x_tgt_pred = select_current_prediction(x_tgt_pred_all, center_index, x_tgt_current)

                # Primary ArtiFixer RF objective.  History latent cells remain
                # context-only; supervise the current output cell exactly as in
                # the previous DrivenVS trainer.
                if args.training_objective == "meanflow":
                    loss_latent_raw, latent_mse = meanflow_latent_loss(
                        training_result["predicted_mean_velocity"][:, :, center_index],
                        training_result["target_mean_velocity"][:, :, center_index],
                        adaptive_p=args.meanflow_adaptive_p,
                    )
                    loss_latent = loss_latent_raw * args.lambda_latent
                else:
                    predicted_velocity = training_result["predicted_velocity"][:, :, center_index].float()
                    target_velocity = training_result["target_velocity"][:, :, center_index].float()
                    velocity_mse = (
                        F.mse_loss(
                            predicted_velocity,
                            target_velocity,
                            reduction="none",
                        )
                        .flatten(1)
                        .mean(dim=1)
                    )
                    latent_mse = velocity_mse.mean()
                    loss_latent = (
                        velocity_mse * flow_weights.to(velocity_mse, non_blocking=True)
                    ).mean() * args.lambda_latent
                if args.lambda_l2 > 0:
                    loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt_current.float(), reduction="mean") * args.lambda_l2
                else:
                    loss_l2 = torch.zeros((), device=x_tgt_pred.device, dtype=torch.float32)

                # Keep LPIPS available as an ablation/auxiliary loss, but do not
                # run the expensive VGG network when its weight is zero.
                if args.lambda_lpips > 0:
                    loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt_current.float()).mean() * args.lambda_lpips
                else:
                    loss_lpips = torch.zeros((), device=x_tgt_pred.device, dtype=torch.float32)
                loss_temporal = torch.zeros(
                    (),
                    device=x_tgt_pred.device,
                    dtype=torch.float32,
                )
                temporal_warmup = 0.0
                if raft is not None and x_src.shape[2] > 1:
                    temporal_warmup = (
                        1.0
                        if args.temporal_loss_warmup_steps == 0
                        else min(1.0, (global_step + 1) / args.temporal_loss_warmup_steps)
                    )
                    loss_temporal = get_dh_temporal_loss(
                        x_tgt_pred=x_tgt_pred_all.float(),
                        x_tgt=x_tgt.float(),
                        raft=raft,
                        temp_loss=temporal_charbonnier,
                    ) * (args.lambda_temporal * temporal_warmup)

                loss = loss_latent + loss_l2 + loss_lpips + loss_temporal

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            progress_bar.update(1)
            if meanflow_ema is not None:
                meanflow_ema.update(accelerator.unwrap_model(model).unet)

            stop_signal = stop.requested()
            checkpoint_due = (
                (args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0)
                or global_step == args.max_train_steps
                or stop_signal
            )
            if checkpoint_due:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint_path = os.path.join(
                        args.output_dir,
                        "checkpoints",
                        f"model_{global_step}.pkl",
                    )
                    if args.training_objective == "meanflow":
                        save_meanflow_checkpoint(
                            accelerator.unwrap_model(model), optimizer, checkpoint_path, vars(args), meanflow_ema
                        )
                    else:
                        save_training_checkpoint(accelerator.unwrap_model(model), optimizer, checkpoint_path)
                    logger.info("Saved checkpoint: %s", checkpoint_path)
                accelerator.wait_for_everyone()
                last_saved_step = global_step
            if stop_signal:
                break

            logs = {
                "loss": loss.detach().item(),
                "loss_latent": loss_latent.detach().item(),
                "latent_mse": latent_mse.detach().item(),
                "loss_l2": loss_l2.detach().item(),
                "loss_lpips": loss_lpips.detach().item(),
                "loss_temporal": loss_temporal.detach().item(),
                "temporal_warmup": temporal_warmup,
                "flow_time_mean": flow_times.detach().float().mean().item(),
                "flow_time_min": flow_times.detach().float().min().item(),
                "flow_time_max": flow_times.detach().float().max().item(),
                "valid_coverage": x_opacity[:, :, center_index].detach().float().mean().item(),
                "target_views": x_src.shape[2],
                "temporal_sample": float(x_src.shape[2] > 1),
                "rf_flow_time": flow_times.detach().float().mean().item(),
                "rf_timestep": rf_timesteps.detach().float().mean().item(),
                "rf_weight": flow_weights.detach().float().mean().item(),
                "meanflow_r": (
                    meanflow_r.detach().float().mean().item() if args.training_objective == "meanflow" else 0.0
                ),
                "meanflow_endpoint": (
                    float(((meanflow_r == 0) & (flow_times == 1)).float().mean().item())
                    if args.training_objective == "meanflow"
                    else 0.0
                ),
                "lr": optimizer.param_groups[0]["lr"],
            }
            progress_bar.set_postfix(
                loss_latent=logs["loss_latent"],
                loss_l2=logs["loss_l2"],
                loss_lpips=logs["loss_lpips"],
                loss_temporal=logs["loss_temporal"],
                flow_time=logs["flow_time_mean"],
                target_views=logs["target_views"],
            )

            if accelerator.is_main_process:
                logger.info(
                    "stage=%s target_views=%d epoch=%d global_step=%d lr=%.6g loss=%.6f "
                    "loss_latent=%.6f loss_l2=%.6f loss_lpips=%.6f "
                    "loss_temporal=%.6f temporal_warmup=%.4f "
                    "flow_time=%.6f",
                    args.training_stage,
                    logs["target_views"],
                    epoch,
                    global_step,
                    logs["lr"],
                    logs["loss"],
                    logs["loss_latent"],
                    logs["loss_l2"],
                    logs["loss_lpips"],
                    logs["loss_temporal"],
                    logs["temporal_warmup"],
                    logs["flow_time_mean"],
                )

                if args.viz_freq > 0 and global_step % args.viz_freq == 0:
                    save_visualizations(
                        make_train_visuals(
                            x_src,
                            x_ref,
                            x_tgt,
                            x_tgt_pred,
                            x_opacity,
                            center_index,
                        ),
                        args.output_dir,
                        global_step,
                        logger,
                    )

            if args.eval_freq > 0 and global_step % args.eval_freq == 0:
                accelerator.wait_for_everyone()
                logs.update(
                    run_validation(
                        args=args,
                        accelerator=accelerator,
                        model=model,
                        net_lpips=net_lpips,
                        dl_val=dl_val,
                        weight_dtype=weight_dtype,
                        global_step=global_step,
                        logger=logger,
                    )
                )
                accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        epoch += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_checkpoint = os.path.join(
            args.output_dir,
            "checkpoints",
            f"model_{global_step}.pkl",
        )
        if global_step > 0 and last_saved_step != global_step:
            if args.training_objective == "meanflow":
                save_meanflow_checkpoint(
                    accelerator.unwrap_model(model), optimizer, final_checkpoint, vars(args), meanflow_ema
                )
            else:
                save_training_checkpoint(accelerator.unwrap_model(model), optimizer, final_checkpoint)
            logger.info("Saved final checkpoint: %s", final_checkpoint)

    accelerator.wait_for_everyone()
    stop.close()
    progress_bar.close()
    accelerator.end_training()
    if hasattr(raft, "close"):
        raft.close()
    if stop_signal:
        accelerator.print(f"Stop requested; completed optimizer step {global_step}.")
        raise SystemExit(128 + stop_signal)


if __name__ == "__main__":
    main(parse_training_args())
