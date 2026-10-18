# SPDX-License-Identifier: Apache-2.0
"""Three-block Cosmos ControlNet, adapted from Cosmos-Transfer1's AV encoder.

Reference: cosmos-transfer1 b25a3ac8, GeneralDITEncoder / ctrl_7b_tp_sample_av.
The native Cosmos blocks/patch/time/position components are copied; there is no
second output head. One encoder is shared across targets and independent refs.
"""

from copy import deepcopy
from itertools import pairwise

import torch
from torch import nn

from .text2image_dit import PatchEmbed


def zero_linear(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class WorldScenarioControl(nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.x_embedder = deepcopy(dit.x_embedder)
        self.t_embedder = deepcopy(dit.t_embedder)
        self.t_embedding_norm = deepcopy(dit.t_embedding_norm)
        self.pos_embedder = deepcopy(dit.pos_embedder)
        self.extra_pos_embedder = deepcopy(dit.extra_pos_embedder) if dit.extra_per_block_abs_pos_emb else None
        self.blocks = nn.ModuleList([deepcopy(dit._unwrap_block(block)) for block in dit.blocks[:3]])
        self.concat_padding_mask = dit.concat_padding_mask
        self.rope = "rope" in dit.pos_emb_cls.lower()
        width = dit.model_channels
        self.hint_patch = PatchEmbed(dit.patch_spatial, dit.patch_temporal, dit.out_channels, width)
        # Exact AV hint-encoder widths, with the output adapted to Cosmos 0.6B.
        widths = [width, 16, 16, 32, 32, 96, 96, 256]
        layers = []
        for before, after in pairwise(widths):
            layers.extend([nn.Linear(before, after), nn.SiLU()])
        layers.append(zero_linear(256, width))
        self.hint_encoder = nn.Sequential(*layers)
        self.zero_projections = nn.ModuleList([zero_linear(width, width) for _ in self.blocks])
        self.interval_embedder = None

    def enable_meanflow(self, embedder):
        self.interval_embedder = deepcopy(embedder)

    def forward(self, rgb, condition, times, context, *, interval=None, padded=False, fps=None):
        if self.concat_padding_mask and not padded:
            rgb = torch.cat([rgb, torch.zeros_like(rgb[:, :1])], dim=1)
        tokens = self.x_embedder(rgb)
        guidance = self.hint_encoder(self.hint_patch(condition))
        extra = self.extra_pos_embedder(tokens, fps=fps) if self.extra_pos_embedder is not None else None
        rope = self.pos_embedder(tokens, fps=fps) if self.rope else None
        if not self.rope:
            tokens = tokens + self.pos_embedder(tokens)
        time_embedding, adaln = self.t_embedder(times)
        if self.interval_embedder is not None:
            time_embedding = time_embedding + self.interval_embedder(interval)
        time_embedding = self.t_embedding_norm(time_embedding)
        outputs = []
        for index, (block, projection) in enumerate(zip(self.blocks, self.zero_projections, strict=True)):
            block.compute_caching(time_embedding, context, adaln)
            tokens = block(tokens, rope_emb_L_1_1_D=rope, extra_per_block_pos_emb=extra)
            # Match the official AV encoder: inject hint after its first block.
            if index == 0:
                tokens = tokens + guidance
            outputs.append(projection(tokens))
        return outputs
