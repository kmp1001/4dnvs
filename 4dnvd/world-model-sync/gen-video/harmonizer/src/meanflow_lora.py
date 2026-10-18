# SPDX-License-Identifier: Apache-2.0
"""PEFT LoRA for native Cosmos, using SimpleTuner's Cosmos attention targets.

SimpleTuner to_q/to_k/to_v/to_out.0 correspond to the native names below.
Injection and fp32 trainable parameters follow NVIDIA's text2image_model.py.
PEFT is imported only when an adapter is requested or an adapter checkpoint loads.
"""

import math

import torch
from torch import nn

LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "output_proj")
ADAPTER_PARTS = (".lora_A.", ".lora_B.")


def make_lora_config(rank=16, alpha=16.0, dropout=0.0):
    if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0,1)")
    return {
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "target_modules": list(LORA_TARGET_MODULES),
        "init_lora_weights": True,
        "bias": "none",
        "adapter_name": "default",
    }


def get_lora_config(dit):
    return getattr(dit, "harmonizer_lora_config", None)


def configure_lora_trainability(unet):
    unet.requires_grad_(False)
    count = 0
    for name, parameter in unet.named_parameters():
        if any(part in name for part in ADAPTER_PARTS):
            parameter.requires_grad_(True)
            parameter.data = parameter.data.to(dtype=torch.float32)
            count += parameter.numel()
    if count == 0:
        raise ValueError("No PEFT LoRA parameters were installed")
    return count


def install_lora(dit, config):
    from peft import LoraConfig, inject_adapter_in_model

    expected = make_lora_config(config["rank"], config["alpha"], config["dropout"])
    if config != expected:
        raise ValueError("Unsupported LoRA configuration; expected SimpleTuner Cosmos attention targets")
    existing = get_lora_config(dit)
    if existing is not None:
        if existing != config:
            raise ValueError("LoRA checkpoint rank/alpha/dropout differ; keep its configuration for continuation")
        return
    for target in LORA_TARGET_MODULES:
        if not any(
            name.endswith("." + target) and isinstance(module, nn.Linear) for name, module in dit.named_modules()
        ):
            raise ValueError(f"Cosmos attention LoRA target not found: {target}")
    dit.requires_grad_(False)
    peft_config = LoraConfig(
        r=config["rank"],
        lora_alpha=config["alpha"],
        lora_dropout=config["dropout"],
        init_lora_weights=True,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
    )
    inject_adapter_in_model(peft_config, dit, adapter_name="default")
    dit.harmonizer_lora_config = dict(config)
    configure_lora_trainability(dit)


def merge_lora_for_full_training(dit):
    """Explicit LoRA Stage II -> full Stage III handoff, retaining learned deltas."""
    if get_lora_config(dit) is None:
        return
    from peft.tuners.lora.layer import LoraLayer

    for name, module in list(dit.named_modules()):
        if isinstance(module, LoraLayer):
            module.merge(safe_merge=True)
            parent_name, _, child_name = name.rpartition(".")
            parent = dit.get_submodule(parent_name) if parent_name else dit
            setattr(parent, child_name, module.get_base_layer())
    del dit.harmonizer_lora_config
    if hasattr(dit, "peft_config"):
        del dit.peft_config


def merged_checkpoint_state(state, lora_config):
    """Canonical plain-linear state for a frozen generator, without mutating it."""
    if lora_config is None:
        return dict(state)
    expected = make_lora_config(lora_config["rank"], lora_config["alpha"], lora_config["dropout"])
    if lora_config != expected:
        raise ValueError("Unsupported frozen-generator LoRA config")
    plain = {}
    for name, value in state.items():
        if any(part in name for part in ADAPTER_PARTS):
            continue
        if ".base_layer." in name:
            plain[name.replace(".base_layer.", ".")] = value
        else:
            plain[name] = value
    for name, a in state.items():
        suffix = ".lora_A.default.weight"
        if not name.endswith(suffix):
            continue
        prefix = name[: -len(suffix)]
        b = state[prefix + ".lora_B.default.weight"]
        base = state[prefix + ".base_layer.weight"]
        delta = b.float() @ a.float() * (lora_config["alpha"] / lora_config["rank"])
        plain[prefix + ".weight"] = (base.float() + delta).to(base.dtype)
    return plain


def frozen_history_parameters(unet, checkpoint, *, use_ema):
    """Represent Stage-II online/EMA weights in either full or LoRA Stage III.

    For a LoRA student, install the fixed generator's effective weights into
    its base layers and zero its adapter residuals during history generation.
    Frozen base parameters are included because Stage-II EMA may differ from
    the online base, and the two stages may use different training modes.
    """
    state = dict(checkpoint["state_dict_unet"])
    if use_ema:
        shadow = checkpoint.get("ema_state_dict_unet")
        if shadow is None:
            raise ValueError("Frozen Stage-II checkpoint has no EMA; use --no-history-checkpoint-ema")
        unknown = set(shadow) - set(state)
        if unknown:
            raise ValueError("Frozen EMA contains unknown parameter names")
        state.update(shadow)
    if checkpoint.get("lora_config") == get_lora_config(unet.dit):
        # Preserve exact adapter computation when the architectures agree.
        return {name: state[name] for name, _ in unet.named_parameters()}
    plain = merged_checkpoint_state(state, checkpoint.get("lora_config"))
    weights = {}
    for name, parameter in unet.named_parameters():
        if any(part in name for part in ADAPTER_PARTS):
            weights[name] = torch.zeros_like(parameter, device="cpu")
        else:
            canonical = name.replace(".base_layer.", ".")
            value = plain[canonical]
            if value.shape != parameter.shape:
                raise ValueError(f"Frozen generator shape mismatch: {name}")
            weights[name] = value
    return weights
