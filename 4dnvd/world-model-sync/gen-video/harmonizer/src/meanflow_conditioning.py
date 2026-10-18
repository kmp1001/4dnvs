# SPDX-License-Identifier: Apache-2.0
"""MeanFlow/sit.py's TimestepEmbedder for the interval t-r.

Keep Cosmos's pretrained t path. Only the additional interval embedder uses
this reference SiT architecture and initialization.
"""

import math

import torch
from torch import nn

MEANFLOW_TIME_CONDITIONING = "cosmos_t_plus_meanflow_t_minus_r_v2"


class MeanFlowIntervalEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # MeanFlow/SiT.initialize_weights(): normal(0,.02), zero biases.
        for layer in (self.mlp[0], self.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, interval):
        shape = interval.shape
        frequency = self.positional_embedding(interval.reshape(-1), self.frequency_embedding_size).to(
            dtype=self.mlp[0].weight.dtype
        )
        return self.mlp(frequency).reshape(*shape, self.mlp[2].out_features)
