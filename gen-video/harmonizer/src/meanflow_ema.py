# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-resident parameter EMA for 0.6B MeanFlow fine-tuning.

Official MeanFlow updates an EMA copy after every optimizer step with decay
0.9999. Keeping float32 shadow parameters on rank 0 CPU avoids doubling the
model's GPU footprint during the already expensive DiT JVP.
"""

import torch


class CPUParameterEMA:
    def __init__(self, module: torch.nn.Module, decay: float = 0.9999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: param.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, param in module.named_parameters()
            if param.requires_grad
        }
        if not self.shadow:
            raise ValueError("EMA found no trainable UNet parameters")

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        parameters = dict(module.named_parameters())
        if set(self.shadow) != {name for name, param in parameters.items() if param.requires_grad}:
            raise ValueError("Trainable parameter set changed during EMA training")
        for name, old in self.shadow.items():
            current = parameters[name].detach().to(device="cpu", dtype=torch.float32)
            old.lerp_(current, 1.0 - self.decay)
        self.num_updates += 1

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, shadow: dict[str, torch.Tensor], *, num_updates: int) -> None:
        if set(shadow) != set(self.shadow):
            raise ValueError("EMA checkpoint parameter names do not match the model")
        for name, value in shadow.items():
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"EMA checkpoint tensor shape mismatch: {name}")
            self.shadow[name].copy_(value.to(device="cpu", dtype=torch.float32))
        self.num_updates = int(num_updates)
