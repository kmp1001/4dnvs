# SPDX-License-Identifier: Apache-2.0
"""Three-stage MeanFlow: V=1, GT history warm-up, mode-B rendered resampling."""

from __future__ import annotations

import gc
import json
import math
import os
import random
from pathlib import Path

import click
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset

from meanflow_curriculum import (
    build_mode_b_paths,
    resample_rendered_history,
    rollout_rendered_history,
    temporary_history_weights,
)
from meanflow_ema import CPUParameterEMA
from meanflow_lora import (
    configure_lora_trainability,
    frozen_history_parameters,
    get_lora_config,
    install_lora,
    make_lora_config,
    merge_lora_for_full_training,
)
from meanflow_loss import meanflow_latent_loss, sample_meanflow_times
from meanflow_monitoring import build_validation_dataset, run_validation, save_previews
from utils.temporal import TEMPORAL_HISTORY_FRAMES
from utils.training_stop import TrainingStop


class CurriculumDataset(Dataset):
    """Original 25% V=1 / 75% V=5 mixture; retain all V=1 samples."""

    def __init__(self, *, manifest, image_prep, references, camera, time, stage, probability, world_scenario=False):
        from utils.training_utils import DrivenVSDataset

        if stage == 1:
            self.base = DrivenVSDataset(
                dataset_folder=manifest,
                split="train",
                image_prep=image_prep,
                num_reference_images=references,
                camera_embed=camera,
                time_embed=time,
                training_stage="stage1",
                temporal_sample_probability=0.0,
                world_scenario=world_scenario,
            )
            self.windows = {}
        else:
            from self_forcing_harmonizer import SelfForcingWindowDataset

            windowed = SelfForcingWindowDataset(
                manifest=manifest,
                split="train",
                image_prep=image_prep,
                num_reference_images=references,
                camera_embed=camera,
                time_embed=time,
                window_length=TEMPORAL_HISTORY_FRAMES + 1,
                world_scenario=world_scenario,
            )
            self.base = windowed.base
            self.windows = {window[-1]: window for window in windowed.windows}
        self.eligible = list(self.windows.values())
        self.probability = 0.0 if stage == 1 else probability

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        if random.random() < self.probability:
            indices = self.windows.get(index)
            if indices is None:
                indices = random.choice(self.eligible)
            return tuple(self.base[i] for i in indices)
        return (self.base[index],)


def validate_options(cfg):
    if cfg.get("flow_estimator", "raft") == "waft" and cfg["stage"] > 1 and cfg["lambda_temporal"] > 0:
        from waft_estimator import waft_paths

        waft_paths(cfg.get("waft_repo"), cfg.get("waft_config"), cfg.get("waft_checkpoint"))
    if cfg.get("lora", False):
        if cfg["stage"] == 1:
            raise ValueError("LoRA is supported only in Stages II and III")
        make_lora_config(cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"])
    if cfg["max_steps"] < 1 or cfg["checkpointing_steps"] < 0:
        raise ValueError(
            "max-steps must be positive; checkpointing-steps must be nonnegative (0 disables periodic saves)"
        )
    if cfg["stage"] > 1 and not cfg.get("check_data", False) and not (cfg["initial_checkpoint"] or cfg["resume"]):
        raise ValueError("Stages II and III require the preceding stage's checkpoint")
    if cfg["initial_checkpoint"] and cfg["resume"]:
        raise ValueError("Choose initialization or resume, not both")
    if not 0 <= cfg["temporal_probability"] <= 1:
        raise ValueError("Temporal probability must be in [0,1]")
    if cfg["stage"] > 1 and cfg["temporal_probability"] == 0:
        raise ValueError("Stages II and III need temporal samples")
    if not 0 <= cfg["endpoint_probability"] <= cfg["interval_probability"] <= 1:
        raise ValueError("Require 0 <= endpoint probability <= interval probability <= 1")
    for name in ("lambda_latent", "lambda_temporal", "adaptive_p"):
        if not math.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if cfg["stage"] == 1 and cfg["lambda_temporal"] != 0:
        raise ValueError("Stage I has no optical-flow loss")
    if cfg["references"] < 1 or cfg["learning_rate"] <= 0:
        raise ValueError("Reference count and learning rate must be positive")
    if not 0 <= cfg["ema_decay"] < 1:
        raise ValueError("EMA decay must be in [0,1)")
    if cfg["stage"] == 3 and cfg["history_generator"] == "stage3_ema" and not cfg["ema"]:
        raise ValueError("stage3_ema history generation requires Stage-III EMA enabled")
    if cfg["time_sigma"] <= 0 or cfg["resampling_shift"] <= 0:
        raise ValueError("Time sigma and resampling shift must be positive")
    if (
        cfg["stage"] == 3
        and cfg["history_generator"] == "frozen_stage2"
        and cfg["resume"]
        and not cfg["history_checkpoint"]
    ):
        raise ValueError("Resuming frozen_stage2 requires --history-checkpoint")


def prepare(frame, cfg, device, dtype):
    from utils.batch_preparation import prepare_training_batch

    return prepare_training_batch(
        frame,
        device=device,
        dtype=dtype,
        expected_reference_views=cfg["references"],
        camera_embed=cfg["camera"],
        time_embed=cfg["time"],
    )


def sample_resampling_time(device, shift):
    value = torch.sigmoid(torch.randn(1, device=device, dtype=torch.float32))
    return shift * value / (1 + (shift - 1) * value)


def checkpoint_generator_weights(model, path, *, use_ema):
    from drivenvs_model import validate_checkpoint_conditioning
    from meanflow_conditioning import MEANFLOW_TIME_CONDITIONING

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != 2 or checkpoint.get("training_objective") != "meanflow_curriculum_v2":
        raise ValueError("Frozen history generator must be a new Stage-II curriculum checkpoint")
    if checkpoint["conditioning_config"].get("meanflow_time_conditioning") != MEANFLOW_TIME_CONDITIONING:
        raise ValueError("Frozen generator has incompatible time conditioning")
    validate_checkpoint_conditioning(model, checkpoint, label="Frozen history generator")
    return frozen_history_parameters(model.unet, checkpoint, use_ema=use_ema)


def save_checkpoint(model, optimizer, scheduler, ema, path, cfg, step):
    from drivenvs_model import _conditioning_config

    torch.save(
        {
            "training_objective": "meanflow_curriculum_v2",
            "stage": cfg["stage"],
            "global_step": step,
            "state_dict_unet": model.unet.state_dict(),
            "state_dict_vae": model._inner_vae().state_dict(),
            "conditioning_config": _conditioning_config(model),
            "curriculum_config": cfg,
            "lora_config": get_lora_config(model.unet.dit),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "ema_state_dict_unet": ema.state_dict() if ema else None,
            "ema_decay": ema.decay if ema else None,
            "ema_num_updates": ema.num_updates if ema else 0,
        },
        path,
    )


@click.command()
@click.option("--stage", type=click.IntRange(1, 3), required=True)
@click.option("--dataset", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), required=True)
@click.option("--initial-checkpoint", type=click.Path(exists=True), default=None)
@click.option("--resume", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--history-checkpoint", type=click.Path(exists=True), default=None)
@click.option("--history-generator", type=click.Choice(["frozen_stage2", "stage3_ema"]), default="frozen_stage2")
@click.option("--history-mode", type=click.Choice(["independent", "autoregressive"]), default="independent")
@click.option("--history-generation-steps", type=click.IntRange(min=1), default=1)
@click.option("--history-checkpoint-ema/--no-history-checkpoint-ema", default=False)
@click.option("--lora/--no-lora", default=False)
@click.option("--lora-rank", type=int, default=16)
@click.option("--lora-alpha", type=float, default=16.0)
@click.option("--lora-dropout", type=float, default=0.0)
@click.option("--ema/--no-ema", default=True)
@click.option("--ema-decay", type=float, default=0.9999)
@click.option("--max-steps", type=int, required=True)
@click.option("--learning-rate", type=float, default=5e-6)
@click.option("--checkpointing-steps", type=int, default=0)
@click.option("--viz-freq", type=click.IntRange(min=0), default=1000)
@click.option("--eval-freq", type=click.IntRange(min=0), default=1000)
@click.option("--num-samples-eval", type=click.IntRange(min=1), default=20)
@click.option("--validation-steps", type=click.IntRange(min=1), default=1)
@click.option("--validation-dataset", type=click.Path(exists=True), default=None)
@click.option("--validation-split", default="test")
@click.option("--image-prep", default="resize_288x512")
@click.option("--references", type=int, default=8)
@click.option("--camera/--no-camera", default=True)
@click.option("--time/--no-time", default=True)
@click.option("--temporal-probability", type=float, default=0.75)
@click.option("--lambda-latent", type=float, default=1.0)
@click.option("--lambda-temporal", type=float, default=0.5)
@click.option("--small-raft/--large-raft", default=True)
@click.option("--flow-estimator", type=click.Choice(["raft", "waft"]), default="raft")
@click.option("--waft-repo", type=click.Path(), default=None)
@click.option("--waft-config", type=click.Path(), default=None)
@click.option("--waft-checkpoint", type=click.Path(), default=None)
@click.option("--waft-iterations", type=click.IntRange(min=1), default=None)
@click.option("--mixed-precision", type=click.Choice(["no", "bf16"]), default="bf16")
@click.option("--workers", type=int, default=2)
@click.option("--seed", type=int, default=42)
@click.option("--time-sampler", type=click.Choice(["uniform", "logit_normal"]), default="logit_normal")
@click.option("--time-mu", type=float, default=-0.4)
@click.option("--time-sigma", type=float, default=1.0)
@click.option("--interval-probability", type=float, default=0.25)
@click.option("--endpoint-probability", type=float, default=0.0)
@click.option("--adaptive-p", type=float, default=1.0)
@click.option("--jvp-backend", type=click.Choice(["func", "autograd"]), default="func")
@click.option("--resampling-shift", type=float, default=0.6)
@click.option("--report-to", default="wandb")
@click.option("--run-name", default=None)
@click.option("--world-scenario-mode", type=click.IntRange(0, 2), default=0)
@click.option("--check-data", is_flag=True)
def main(**cfg):
    validate_options(cfg)
    os.environ["COSMOS_MODEL_SIZE"] = "0.6B"
    set_seed(cfg["seed"])
    dataset = CurriculumDataset(
        manifest=cfg["dataset"],
        image_prep=cfg["image_prep"],
        references=cfg["references"],
        camera=cfg["camera"],
        time=cfg["time"],
        stage=cfg["stage"],
        probability=cfg["temporal_probability"],
        world_scenario=cfg["world_scenario_mode"] > 0,
    )
    print(f"Stage {cfg['stage']}: {len(dataset)} samples, {len(dataset.windows)} temporal windows", flush=True)
    if cfg["check_data"]:
        # Force both shapes through image loading/collation on a login node.
        from torch.utils.data import default_collate

        single = default_collate([(dataset.base[0],)])
        prepared = prepare(single[0], cfg, torch.device("cpu"), torch.float32)
        print(f"V=1 source={tuple(prepared[0].shape)} references={tuple(prepared[1].shape)}")
        if dataset.eligible:
            window = default_collate([tuple(dataset.base[i] for i in dataset.eligible[0])])
            frames = [prepare(frame, cfg, torch.device("cpu"), torch.float32) for frame in window]
            gt = [frame[2] for frame in frames[:-1]]
            paths = build_mode_b_paths(
                frames[-1], gt, gt, [frame[5] for frame in frames[:-1]], camera_embed=cfg["camera"]
            )
            print(f"V=5 source={tuple(paths[0].shape)} RAFT_GT={tuple(paths[-1].shape)}")
        return

    from meanflow_model import MeanFlowDrivenVSModel, load_meanflow_weights
    from temporal_loss import build_temporal_loss, get_dh_temporal_loss

    accelerator = Accelerator(
        mixed_precision=cfg["mixed_precision"],
        log_with=None if cfg["report_to"] == "none" else cfg["report_to"],
    )
    if accelerator.device.type != "cuda":
        raise RuntimeError("Training requires an existing GPU allocation")
    dtype = torch.bfloat16 if cfg["mixed_precision"] == "bf16" else torch.float32
    model = MeanFlowDrivenVSModel(
        train_full_unet=True,
        freeze_vae=True,
        world_scenario_mode=cfg["world_scenario_mode"],
        batch_size=1,
        num_reference_images=cfg["references"],
        camera_embed=cfg["camera"],
        time_embed=cfg["time"],
        camera_translation_scale=1.0,
        reference_time_embedding_dim=32,
        reference_time_scale=1.0,
        enforce_conditioning_config=True,
        device=accelerator.device,
        dtype=dtype,
    )
    checkpoint = None
    initial = cfg["resume"] or cfg["initial_checkpoint"]
    if initial:
        checkpoint = load_meanflow_weights(model, initial)
        expected_stage = cfg["stage"] if cfg["resume"] else cfg["stage"] - 1
        if checkpoint.get("stage") != expected_stage:
            raise ValueError(f"Expected Stage {expected_stage} checkpoint for this handoff")
    model.to(device=accelerator.device, dtype=dtype)
    if not cfg["lora"] and get_lora_config(model.unet.dit) is not None:
        if cfg["resume"]:
            raise ValueError("Cannot resume a LoRA run as full training; use --initial-checkpoint")
        merge_lora_for_full_training(model.unet.dit)
    model.set_train()
    if cfg["lora"]:
        install_lora(model.unet.dit, make_lora_config(cfg["lora_rank"], cfg["lora_alpha"], cfg["lora_dropout"]))
        configure_lora_trainability(model.unet)
    trainable_parameters = [p for p in model.unet.parameters() if p.requires_grad]
    accelerator.print(
        f"Training mode={'LoRA' if cfg['lora'] else 'full'}: "
        f"{sum(p.numel() for p in trainable_parameters):,} trainable parameters"
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=cfg["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    step = 0
    if cfg["resume"]:
        old_cfg = checkpoint["curriculum_config"]
        for key in ("flow_estimator", "waft_repo", "waft_config", "waft_checkpoint", "waft_iterations"):
            default = "raft" if key == "flow_estimator" else None
            if old_cfg.get(key, default) != cfg[key]:
                raise ValueError(f"Resume changes {key}; use --initial-checkpoint for a new run")
        if old_cfg.get("lora", False) != cfg["lora"]:
            raise ValueError("Resume cannot change full/LoRA mode; use --initial-checkpoint")
        for key in ("ema", "history_generator", "history_checkpoint_ema"):
            if old_cfg[key] != cfg[key]:
                raise ValueError(f"Resume changes {key}; initialize a new run instead")
        for key, default in (("history_mode", "independent"), ("history_generation_steps", 1)):
            if old_cfg.get(key, default) != cfg[key]:
                raise ValueError(f"Resume changes {key}; initialize a new run instead")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        step = int(checkpoint["global_step"])
        if step >= cfg["max_steps"]:
            raise ValueError("Resume checkpoint already reached max-steps")
    ema = CPUParameterEMA(model.unet, decay=cfg["ema_decay"]) if cfg["ema"] else None
    # Preserve EMA continuity across stage handoffs when the previous stage used it.
    if ema and checkpoint and checkpoint.get("ema_state_dict_unet") is not None:
        previous_shadow = checkpoint["ema_state_dict_unet"]
        if set(previous_shadow) == set(ema.state_dict()):
            if not math.isclose(checkpoint["ema_decay"], cfg["ema_decay"], abs_tol=1e-12, rel_tol=0):
                raise ValueError("EMA decay differs from preceding checkpoint")
            ema.load_state_dict(previous_shadow, num_updates=checkpoint["ema_num_updates"])
        elif cfg["resume"]:
            raise ValueError("EMA parameter topology differs on resume")
        else:
            accelerator.print("EMA initialized from the new full/LoRA training topology")
    elif cfg["resume"] and ema:
        raise ValueError("EMA-enabled resume has no EMA state")
    frozen_weights = None
    if cfg["stage"] == 3 and cfg["history_generator"] == "frozen_stage2":
        frozen_weights = checkpoint_generator_weights(
            model,
            cfg["history_checkpoint"] or cfg["initial_checkpoint"],
            use_ema=cfg["history_checkpoint_ema"],
        )
    del checkpoint
    gc.collect()
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=cfg["workers"], pin_memory=True)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    unwrapped = accelerator.unwrap_model(model)
    # Each rank maintains EMA for its local history generation; only rank 0 saves.
    raft, charbonnier = (
        build_temporal_loss(
            accelerator.device,
            estimator=cfg["flow_estimator"],
            use_small=cfg["small_raft"],
            waft_repo=cfg["waft_repo"],
            waft_config=cfg["waft_config"],
            waft_checkpoint=cfg["waft_checkpoint"],
            waft_iterations=cfg["waft_iterations"],
        )
        if cfg["stage"] > 1 and cfg["lambda_temporal"] > 0
        else (None, None)
    )
    output = Path(cfg["output_dir"])
    if accelerator.is_main_process:
        (output / "checkpoints").mkdir(parents=True, exist_ok=True)
        (output / "training_config.json").write_text(json.dumps(cfg, indent=2))
    accelerator.init_trackers(
        "cosmos_harmonizer",
        config=cfg,
        init_kwargs={"wandb": {"name": cfg["run_name"] or f"meanflow_stage{cfg['stage']}"}}
        if cfg["report_to"] == "wandb"
        else {},
    )
    accelerator.wait_for_everyone()
    validation_dataset = validation_rollout_dataset = net_lpips = None
    if cfg["eval_freq"] and accelerator.is_main_process:
        import lpips

        validation_dataset = build_validation_dataset(cfg)
        if cfg["stage"] == 2:
            validation_rollout_dataset = build_validation_dataset(cfg, autoregressive=True)
        with torch.random.fork_rng(devices=[accelerator.device.index]):
            net_lpips = lpips.LPIPS(net="vgg").to(device=accelerator.device, dtype=torch.float32).eval()
        net_lpips.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    stop = TrainingStop(accelerator.device)
    stop_signal = 0
    last_saved_step = step if cfg["resume"] else None
    while step < cfg["max_steps"] and not stop_signal:
        for window in loader:
            stop_signal = stop.requested()
            if stop_signal:
                break
            viz_due = cfg["viz_freq"] > 0 and (step + 1) % cfg["viz_freq"] == 0
            current = prepare(window[-1], cfg, accelerator.device, dtype)
            source, refs, target, opacity, _, condition = current
            gt_supervision = target
            temporal = len(window) == TEMPORAL_HISTORY_FRAMES + 1
            t_s = None
            if temporal:
                histories = [prepare(frame, cfg, accelerator.device, dtype) for frame in window[:-1]]
                gt_histories = [item[2].detach() for item in histories]
                cameras = [item[5] for item in histories]
                if cfg["stage"] == 2:
                    predictions = gt_histories
                else:
                    weights = frozen_weights if frozen_weights is not None else ema.state_dict()
                    with temporary_history_weights(unwrapped.unet, weights, include_frozen=frozen_weights is not None):
                        if cfg["history_mode"] == "autoregressive":
                            t_s = torch.ones(1, device=accelerator.device)
                            predictions = rollout_rendered_history(
                                unwrapped, histories, num_steps=cfg["history_generation_steps"]
                            )
                        else:
                            t_s = sample_resampling_time(accelerator.device, cfg["resampling_shift"])
                            predictions = [
                                resample_rendered_history(
                                    unwrapped, item, t_s, num_steps=cfg["history_generation_steps"]
                                )
                                for item in histories
                            ]
                source, refs, target, opacity, condition, gt_supervision = build_mode_b_paths(
                    current,
                    predictions,
                    gt_histories,
                    cameras,
                    camera_embed=cfg["camera"],
                )
            r, t = sample_meanflow_times(
                1,
                accelerator.device,
                time_sampler=cfg["time_sampler"],
                time_mu=cfg["time_mu"],
                time_sigma=cfg["time_sigma"],
                interval_probability=cfg["interval_probability"],
                endpoint_probability=cfg["endpoint_probability"],
            )
            with accelerator.autocast():
                result = model(
                    source,
                    r=r,
                    t=t,
                    reference_images=refs,
                    target_images=target,
                    opacity=opacity,
                    jvp_backend=cfg["jvp_backend"],
                    compute_clean_output=(temporal and cfg["lambda_temporal"] > 0) or viz_due,
                    **condition,
                )
                latent, mse = meanflow_latent_loss(
                    result["predicted_mean_velocity"][:, :, 0],
                    result["target_mean_velocity"][:, :, 0],
                    adaptive_p=cfg["adaptive_p"],
                )
                latent = cfg["lambda_latent"] * latent
                temporal_loss = (
                    get_dh_temporal_loss(
                        result["output_image"].float(),
                        gt_supervision.float(),
                        raft,
                        charbonnier,
                    )
                    * cfg["lambda_temporal"]
                    if temporal and cfg["lambda_temporal"] > 0
                    else latent.new_zeros(())
                )
                loss = latent + temporal_loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if ema:
                ema.update(unwrapped.unet)
            stop_signal = stop.requested()
            if (
                (cfg["checkpointing_steps"] > 0 and step % cfg["checkpointing_steps"] == 0)
                or step == cfg["max_steps"]
                or stop_signal
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_checkpoint(
                        unwrapped, optimizer, scheduler, ema, output / "checkpoints" / f"model_{step}.pkl", cfg, step
                    )
                accelerator.wait_for_everyone()
                last_saved_step = step
            if stop_signal:
                break
            if viz_due and accelerator.is_main_process:
                save_previews(
                    source,
                    refs,
                    gt_supervision,
                    result["output_image"][:, :, :1],
                    opacity,
                    output / "visualizations" / f"step_{step:07d}" / "train",
                )
            logs = {
                "loss": loss.detach().item(),
                "loss_latent": latent.detach().item(),
                "latent_mse": mse.detach().item(),
                "loss_temporal": temporal_loss.detach().item(),
                "temporal_sample": float(temporal),
                "stage": cfg["stage"],
                "meanflow_r": r.item(),
                "meanflow_t": t.item(),
                "resample_t": t_s.item() if t_s is not None else 0.0,
            }
            # Release the training graph before validation allocates sampling activations.
            del result, loss, latent, mse, temporal_loss
            accelerator.log(logs, step=step)
            if accelerator.is_main_process:
                with (output / "losses.jsonl").open("a") as handle:
                    handle.write(json.dumps({"step": step, **logs}) + "\n")
            if step == 1 or step % 10 == 0:
                accelerator.print(
                    f"stage={cfg['stage']} step={step}/{cfg['max_steps']} "
                    f"loss={logs['loss']:.6f} loss_latent={logs['loss_latent']:.6f} "
                    f"latent_mse={logs['latent_mse']:.6f} loss_temporal={logs['loss_temporal']:.6f}"
                )
            if cfg["eval_freq"] > 0 and step % cfg["eval_freq"] == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    validation_modes = [(validation_dataset, "val", "validation", False)]
                    if cfg["stage"] == 2:
                        validation_modes.append(
                            (validation_rollout_dataset, "val_autoregressive", "validation_autoregressive", True)
                        )
                    validation_logs = {}
                    for val_dataset, prefix, folder, autoregressive in validation_modes:
                        metrics = run_validation(
                            unwrapped,
                            val_dataset,
                            net_lpips,
                            ema,
                            cfg,
                            device=accelerator.device,
                            dtype=dtype,
                            directory=output / "visualizations" / f"step_{step:07d}" / folder,
                            autoregressive=autoregressive,
                        )
                        validation_logs.update(
                            {key.replace("val/", f"{prefix}/", 1): value for key, value in metrics.items()}
                        )
                        accelerator.print(
                            f"validation mode={prefix} step={step} EMA={bool(ema)} "
                            f"L2={metrics['val/l2']:.6f} LPIPS={metrics['val/lpips']:.6f} "
                            f"temporal={metrics['val/temporal_samples']}/{metrics['val/num_samples']}"
                        )
                    accelerator.log(validation_logs, step=step)
                    with (output / "validation.jsonl").open("a") as handle:
                        handle.write(json.dumps({"step": step, **validation_logs}) + "\n")
                accelerator.wait_for_everyone()
            if step >= cfg["max_steps"]:
                break
    accelerator.wait_for_everyone()
    if step > 0 and last_saved_step != step:
        if accelerator.is_main_process:
            save_checkpoint(
                unwrapped, optimizer, scheduler, ema, output / "checkpoints" / f"model_{step}.pkl", cfg, step
            )
        accelerator.wait_for_everyone()
    stop.close()
    accelerator.end_training()
    if hasattr(raft, "close"):
        raft.close()
    if stop_signal:
        accelerator.print(f"Stop requested; completed optimizer step {step}.")
        raise SystemExit(128 + stop_signal)


if __name__ == "__main__":
    main()
