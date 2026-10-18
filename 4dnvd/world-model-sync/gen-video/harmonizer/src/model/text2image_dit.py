# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import collections
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
from cosmos_predict2.conditioner import DataType
from cosmos_predict2.module.a2a_cp import MinimalA2AAttnOp, NattenA2AAttnOp
from cosmos_predict2.module.neighborhood_attn import NeighborhoodAttention
from cosmos_predict2.networks.model_weights_stats import WeightTrainingStat
from cosmos_predict2.networks.selective_activation_checkpoint import SACConfig as _SACConfig
from cosmos_predict2.utils.context_parallel import split_inputs_cp
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from imaginaire.constants import TEXT_ENCODER_CLASS, TextEncoderClass
from imaginaire.utils import log
from imaginaire.utils.graph import create_cuda_graph
from torch import nn
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
from torchvision import transforms
from transformer_engine.pytorch.attention import DotProductAttention

from .prope import PropeDotProductAttention
from utils import NVTXRangeDecorator

# NOTE: changes to the original code:
# 0. cache time and text embedding as they are constant during inference
# 1. import NVTXRangeDecorator from utils.py
# 2. define apply_rotary_pos_emb function
# 3. use nn.RMSNorm instead of te.pytorch.RMSNorm

# if Version(te.__version__) >= Version("2.4.0"):
#     from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
# else:
#     from transformer_engine.pytorch.attention import apply_rotary_pos_emb


def apply_rotary_pos_emb(t, freqs, tensor_format="sbhd", cu_seqlens=None, fused=True):
    assert tensor_format == "bshd", "only support bshd format"

    # Original implementation:
    #
    # seq_len = t.shape[0] if tensor_format == "sbhd" else t.shape[1]
    # if freqs.dim() == 4:
    #     freqs = freqs[:seq_len]
    #     if tensor_format == "bshd":
    #         freqs = freqs.transpose(0, 1)
    # elif freqs.dim() == 2:
    #     freqs = freqs[:seq_len]
    #     freqs = freqs.view(seq_len, 1, 1, rot_dim) if tensor_format == "sbhd" else freqs.view(1, seq_len, 1, rot_dim)
    #

    rot_dim = freqs.shape[-1]
    seq_len = t.shape[1]
    freqs = freqs[:seq_len]
    freqs = freqs.transpose(0, 1)

    cos = freqs.cos().to(t.dtype)
    sin = freqs.sin().to(t.dtype)

    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    t_embed = t_rot * cos + rotate_half(t_rot) * sin

    # Original implementation:
    #
    # return torch.cat([t_embed, t_pass], dim=-1) if t_pass.shape[-1] > 0 else t_embed
    #
    assert t_pass.shape[-1] == 0, "t_pass should have no dimension"
    return t_embed


# selective activation checkpoint; only apply to the minimal v4 model. if there are change in the networks, some policy will not work as we expect.
def predict2_2B_720_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            op_count_key = f"{mode}_mm_count"
            # from imaginaire.utils import log
            # log.info(f"op_count_key: {op_count_key}, op_count[op_count_key]: {op_count[op_count_key]}, {args[0].shape}, {args[1].shape}")
            # there are totally 6 + 4 + 4 + 2 = 16 block
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 16
            if op_count[op_count_key] > 8:  # recompute self attn first 3 linear layers
                return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def predict2_14B_720_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            op_count_key = f"{mode}_mm_count"
            # from imaginaire.utils import log
            # log.info(f"op_count_key: {op_count_key}, op_count[op_count_key]: {op_count[op_count_key]}, {args[0].shape}, {args[1].shape}")
            # there are totally 6 + 4 + 4 + 2 = 16 block
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 16
            if op_count[op_count_key] > 8:  # recompute self attn first 1 linear layers
                return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def linear_selfattn_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


class CheckpointMode(str, Enum):
    NONE = "none"
    MM_ONLY = "mm_only"
    BLOCK_WISE = "block_wise"
    LINEAR_SELFATTN = "linear_selfattn"
    PREDICT2_2B_720 = "predict2_2b_720"
    PREDICT2_14B_720 = "predict2_14b_720"

    def __str__(self) -> str:
        return self.value


@dataclass
class SACConfig(_SACConfig):
    def get_context_fn(self):
        if self.mode == CheckpointMode.LINEAR_SELFATTN:
            return linear_selfattn_context_fn
        elif self.mode == CheckpointMode.PREDICT2_2B_720:
            return predict2_2B_720_context_fn
        elif self.mode == CheckpointMode.PREDICT2_14B_720:
            return predict2_14B_720_context_fn
        else:
            # Reuse parent class implementation for other modes
            return super().get_context_fn()


from collections import namedtuple

VideoSize = namedtuple("VideoSize", ["T", "H", "W"])


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Keep this implementation strictly in terms of basic ops so it exports
        # cleanly to ONNX (avoid aten::rms_norm).
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    @NVTXRangeDecorator("RMSNorm:forward")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Avoid dtype casts here; keep the graph minimal for ONNX export.
        output = self._norm(x)
        return output * self.weight


# ---------------------- Feed Forward Network -----------------------
class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)

        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    @NVTXRangeDecorator("GPT2FeedForward:forward")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)

        x = self.activation(x)
        x = self.layer2(x)
        return x


@NVTXRangeDecorator("torch_attention_op")
def torch_attention_op(q_B_S_H_D: torch.Tensor, k_B_S_H_D: torch.Tensor, v_B_S_H_D: torch.Tensor) -> torch.Tensor:
    """Computes multi-head attention using PyTorch's native implementation.

    This function provides a PyTorch backend alternative to Transformer Engine's attention operation.
    It rearranges the input tensors to match PyTorch's expected format, computes scaled dot-product
    attention, and rearranges the output back to the original format.

    The input tensor names use the following dimension conventions:

    - B: batch size
    - S: sequence length
    - H: number of attention heads
    - D: head dimension

    Args:
        q_B_S_H_D: Query tensor with shape (batch, seq_len, n_heads, head_dim)
        k_B_S_H_D: Key tensor with shape (batch, seq_len, n_heads, head_dim)
        v_B_S_H_D: Value tensor with shape (batch, seq_len, n_heads, head_dim)

    Returns:
        Attention output tensor with shape (batch, seq_len, n_heads * head_dim)
    """
    in_q_shape = q_B_S_H_D.shape
    in_k_shape = k_B_S_H_D.shape
    q_B_H_S_D = rearrange(q_B_S_H_D, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
    k_B_H_S_D = rearrange(k_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    v_B_H_S_D = rearrange(v_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    result_B_S_HD = rearrange(
        torch.nn.functional.scaled_dot_product_attention(q_B_H_S_D, k_B_H_S_D, v_B_H_S_D), "b h ... l -> b ... (h l)"
    )

    return result_B_S_HD


class Attention(nn.Module):
    """
    A flexible attention module supporting both self-attention and cross-attention mechanisms.

    This module implements a multi-head attention layer that can operate in either self-attention
    or cross-attention mode. The mode is determined by whether a context dimension is provided.
    The implementation uses scaled dot-product attention and supports optional bias terms and
    dropout regularization.

    Args:
        query_dim (int): The dimensionality of the query vectors.
        context_dim (int, optional): The dimensionality of the context (key/value) vectors.
            If None, the module operates in self-attention mode using query_dim. Default: None
        n_heads (int, optional): Number of attention heads for multi-head attention. Default: 8
        head_dim (int, optional): The dimension of each attention head. Default: 64
        dropout (float, optional): Dropout probability applied to the output. Default: 0.0
        qkv_format (str, optional): Format specification for QKV tensors. Default: "bshd"
        backend (str, optional): Backend to use for the attention operation. Default: "transformer_engine"

    Examples:
        >>> # Self-attention with 512 dimensions and 8 heads
        >>> self_attn = Attention(query_dim=512)
        >>> x = torch.randn(32, 16, 512)  # (batch_size, seq_len, dim)
        >>> out = self_attn(x)  # (32, 16, 512)

        >>> # Cross-attention
        >>> cross_attn = Attention(query_dim=512, context_dim=256)
        >>> query = torch.randn(32, 16, 512)
        >>> context = torch.randn(32, 8, 256)
        >>> out = cross_attn(query, context)  # (32, 16, 512)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        qkv_format: str = "bshd",
        backend: str = "transformer_engine",
        natten_params: Mapping | None = None,
    ) -> None:
        super().__init__()
        log.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{n_heads} heads with a dimension of {head_dim}."
        )
        self.is_selfattn = context_dim is None  # self attention

        if backend not in ["transformer_engine", "torch", "minimal_a2a", "natten"]:
            raise NotImplementedError(f"Unrecognized {backend=}.")

        self.backend = backend

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        # Optional ArtiFixer-style visual-reference projections. They are
        # installed only after the pretrained Cosmos checkpoint is loaded.
        self.add_k_proj = None
        self.add_v_proj = None
        self.add_k_norm = None

        if self.backend == "transformer_engine":
            self.attn_op = DotProductAttention(
                self.n_heads,
                self.head_dim,
                num_gqa_groups=self.n_heads,
                attention_dropout=0,
                qkv_format=qkv_format,
                attn_mask_type="no_mask",
            )
        elif self.backend == "minimal_a2a":
            self.attn_op = MinimalA2AAttnOp()
        elif self.backend == "natten":
            if natten_params is None:
                raise ValueError("You must specify `natten_params` when using the NATTEN backend, got None.")
            self.attn_op = NattenA2AAttnOp(natten_parameters=natten_params)
        elif self.backend == "torch":
            self.attn_op = torch_attention_op
        else:
            raise NotImplementedError()

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)

        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        def apply_norm_and_rotary_pos_emb(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rope_emb: torch.Tensor | None
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = self.v_norm(v)
            if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
                q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
                k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
            return q, k, v

        q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)

        return q, k, v

    def compute_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, video_size: VideoSize | None = None
    ) -> torch.Tensor:
        result = self.compute_attention_result(q, k, v, video_size=video_size)
        return self.output_dropout(self.output_proj(result))

    def compute_attention_result(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, video_size: VideoSize | None = None
    ) -> torch.Tensor:
        """Return attention values before the shared output projection."""
        additional_args = {}
        if isinstance(self.attn_op, (NattenA2AAttnOp, NeighborhoodAttention)):
            additional_args["video_size"] = video_size

        result = self.attn_op(q, k, v, **additional_args)  # [B, S, H, D]
        return result

    def enable_added_reference_kv(self) -> None:
        """Add ArtiFixer-style K/V projections for visual reference tokens.

        Reference tokens have ``query_dim`` channels because they come from
        this DiT's patch embedder, whereas text context has ``context_dim``
        channels. As in ArtiFixer, the added V projection is initialized to zero
        so enabling this branch exactly preserves the pretrained function.
        """
        if self.add_k_proj is not None:
            return

        weight = self.q_proj.weight
        self.add_k_proj = nn.Linear(
            self.query_dim,
            self._inner_dim,
            bias=True,
            device=weight.device,
            dtype=weight.dtype,
        )
        self.add_v_proj = nn.Linear(
            self.query_dim,
            self._inner_dim,
            bias=True,
            device=weight.device,
            dtype=weight.dtype,
        )
        self.add_k_norm = RMSNorm(self.head_dim, eps=1e-6).to(
            device=weight.device,
            dtype=weight.dtype,
        )

        # Keep PyTorch's normal Linear initialization for K, matching ArtiFixer.
        # A zero V weight and bias make the complete reference contribution zero
        # at initialization, regardless of K.
        torch.nn.init.zeros_(self.add_v_proj.weight)
        torch.nn.init.zeros_(self.add_v_proj.bias)
        self.add_k_norm.reset_parameters()

    def compute_added_reference_kv(
        self,
        reference_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.add_k_proj is None or self.add_v_proj is None or self.add_k_norm is None:
            raise RuntimeError("Visual-reference K/V projections are not enabled")
        if reference_tokens.ndim != 3 or reference_tokens.shape[-1] != self.query_dim:
            raise ValueError(
                f"Expected reference tokens [B,S,{self.query_dim}], "
                f"got {tuple(reference_tokens.shape)}"
            )

        key = rearrange(
            self.add_k_proj(reference_tokens),
            "b s (h d) -> b s h d",
            h=self.n_heads,
            d=self.head_dim,
        )
        value = rearrange(
            self.add_v_proj(reference_tokens),
            "b s (h d) -> b s h d",
            h=self.n_heads,
            d=self.head_dim,
        )
        return self.add_k_norm(key), value

    def forward_with_added_reference_kv(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        reference_tokens: torch.Tensor,
        target_prope: PropeDotProductAttention | None = None,
        reference_prope: PropeDotProductAttention | None = None,
    ) -> torch.Tensor:
        """Sum text and reference attention before one output projection.

        This follows ``ArtifixerCrossAttnProcessor``: target and reference
        attention share Q and the output projection, while reference tokens use
        additional K/V projections. PRoPE is applied only to target Q and visual
        reference K/V/output; the pretrained text cross-attention is unchanged.
        """
        if (target_prope is None) != (reference_prope is None):
            raise ValueError("Target and reference PRoPE must either both be provided or both be absent")

        q, k, v = self.compute_qkv(x, context, rope_emb=None)
        reference_k, reference_v = self.compute_added_reference_kv(reference_tokens)
        text_result = self.compute_attention_result(q, k, v)
        if target_prope is None:
            reference_result = self.compute_attention_result(q, reference_k, reference_v)
        else:
            # ArtiFixer applies camera-relative transforms only to the visual
            # reference attention. Text cross-attention remains unchanged.
            query_dtype = q.dtype
            prope_q = target_prope.apply_to_q(rearrange(q, "b s h d -> b h s d").float())
            prope_k = reference_prope.apply_to_kv(
                rearrange(reference_k, "b s h d -> b h s d").float()
            )
            prope_v = reference_prope.apply_to_kv(
                rearrange(reference_v, "b s h d -> b h s d").float()
            )
            reference_result = self.compute_attention_result(
                rearrange(prope_q, "b h s d -> b s h d").to(query_dtype),
                rearrange(prope_k, "b h s d -> b s h d").to(query_dtype),
                rearrange(prope_v, "b h s d -> b s h d").to(query_dtype),
            )
            reference_result = rearrange(
                target_prope.apply_to_o(
                    rearrange(
                        reference_result,
                        "b s (h d) -> b h s d",
                        h=self.n_heads,
                        d=self.head_dim,
                    ).float()
                ).to(query_dtype),
                "b h s d -> b s (h d)",
            )
        return self.output_dropout(self.output_proj(text_result + reference_result))

    @NVTXRangeDecorator("Attention:forward")
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
        video_size: VideoSize | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
            rope_emb (Optional[Tensor]): RoPE embedding tensor, or no RoPE embeddings (i.e. in cross attention)
            video_size(VideoSize): Shape [T, H, W]
        """
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)

        return self.compute_attention(q, k, v, video_size=video_size)

    def set_context_parallel_group(
        self, process_group: ProcessGroup, ranks: list[int], stream: torch.cuda.Stream
    ) -> None:
        self.attn_op.set_context_parallel_group(process_group, ranks, stream)


class VideoPositionEmb(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._cp_group = None

    def enable_context_parallel(self, process_group: ProcessGroup) -> None:
        self._cp_group = process_group

    def disable_context_parallel(self) -> None:
        self._cp_group = None

    @property
    def seq_dim(self) -> int:
        return 1

    @NVTXRangeDecorator("VideoPositionEmb:forward")
    def forward(self, x_B_T_H_W_C: torch.Tensor, fps: torch.Tensor | None) -> torch.Tensor:
        """
        With CP, the function assume that the input tensor is already split.
        It delegates the embedding generation to generate_embeddings function.
        """
        B_T_H_W_C = x_B_T_H_W_C.shape
        if self._cp_group is not None:
            cp_ranks = get_process_group_ranks(self._cp_group)
            cp_size = len(cp_ranks)
            B, T, H, W, C = B_T_H_W_C
            B_T_H_W_C = torch.Size((B, T * cp_size, H, W, C))
        embeddings = self.generate_embeddings(B_T_H_W_C, fps=fps)

        return self._split_for_context_parallel(embeddings)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps: torch.Tensor | None) -> Any:
        raise NotImplementedError

    def _split_for_context_parallel(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self._cp_group is not None:
            embeddings = split_inputs_cp(x=embeddings, seq_dim=self.seq_dim, cp_group=self._cp_group)
        return embeddings


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().to(self.dim_spatial_range.device)
        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.dim_spatial_range.device) / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.dim_spatial_range.device) / dim_t
        )

    @NVTXRangeDecorator("VideoRopePosition3DEmb:generate_embeddings")
    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: torch.Tensor | None = None,
        h_ntk_factor: float | None = None,
        w_ntk_factor: float | None = None,
        t_ntk_factor: float | None = None,
    ) -> torch.Tensor:
        """
        Generate embeddings for the given input size.

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time, Height, Width, Channels).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor.

        Returns:
            Not specified in the original code snippet.
        """
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor  # type: ignore
        w_theta = 10000.0 * w_ntk_factor  # type: ignore
        t_theta = 10000.0 * t_ntk_factor  # type: ignore

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        assert H <= self.max_h and W <= self.max_w, (
            f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        )
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        if self.enable_fps_modulation:
            uniform_fps = (fps is None) or (fps.min() == fps.max())
            assert uniform_fps or B == 1 or T == 1, (
                "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"
            )

            # apply sequence scaling in temporal dimension
            if fps is None:  # image case
                assert T == 1, "T should be 1 for image batch."
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self) -> int:
        return 0


class LearnablePosEmbAxis(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        **kwargs,
    ):
        """
        Args:
            interpolation (str): we curretly only support "crop", ideally when we need extrapolation capacity, we should adjust frequency or other more advanced methods. they are not implemented yet.
        """
        del kwargs  # unused
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"
        self.model_channels = model_channels

        self.pos_emb_h = nn.Parameter(torch.zeros(len_h, model_channels))
        self.pos_emb_w = nn.Parameter(torch.zeros(len_w, model_channels))
        self.pos_emb_t = nn.Parameter(torch.zeros(len_t, model_channels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.model_channels)
        torch.nn.init.trunc_normal_(self.pos_emb_h, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_w, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_t, std=std, a=-3 * std, b=3 * std)

    @NVTXRangeDecorator("LearnablePosEmbAxis:generate_embeddings")
    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps: torch.Tensor | None) -> torch.Tensor:
        B, T, H, W, _ = B_T_H_W_C
        if self.interpolation == "crop":
            emb_h_H = self.pos_emb_h[:H]
            emb_w_W = self.pos_emb_w[:W]
            emb_t_T = self.pos_emb_t[:T]
            emb = (
                repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W)
                + repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W)
                + repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H)
            )
            assert list(emb.shape)[:4] == [B, T, H, W], f"bad shape: {list(emb.shape)[:4]} != {B, T, H, W}"
        else:
            raise ValueError(f"Unknown interpolation method {self.interpolation}")

        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(1e-6, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return emb / norm.to(emb.dtype)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    @NVTXRangeDecorator("Timesteps:forward")
    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        in_dype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return rearrange(emb.to(dtype=in_dype), "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        log.debug(
            f"Using AdaLN LoRA Flag:  {use_adaln_lora}. We enable bias if no AdaLN LoRA for backward compatibility."
        )
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)

    @NVTXRangeDecorator("TimestepEmbedding:forward")
    def forward(self, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            adaln_lora_B_T_3D = None
            emb_B_T_D = emb

        return emb_B_T_D, adaln_lora_B_T_3D


class FourierFeatures(nn.Module):
    """
    Implements a layer that generates Fourier features from input tensors, based on randomly sampled
    frequencies and phases. This can help in learning high-frequency functions in low-dimensional problems.

    [B] -> [B, D]

    Parameters:
        num_channels (int): The number of Fourier features to generate.
        bandwidth (float, optional): The scaling factor for the frequency of the Fourier features. Defaults to 1.
        normalize (bool, optional): If set to True, the outputs are scaled by sqrt(2), usually to normalize
                                    the variance of the features. Defaults to False.

    Example:
        >>> layer = FourierFeatures(num_channels=256, bandwidth=0.5, normalize=True)
        >>> x = torch.randn(10, 256)  # Example input tensor
        >>> output = layer(x)
        >>> print(output.shape)  # Expected shape: (10, 256)
    """

    def __init__(self, num_channels: int, bandwidth: int = 1, normalize: bool = False):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
        self.gain = np.sqrt(2) if normalize else 1
        self.bandwidth = bandwidth
        self.num_channels = num_channels

        self.reset_parameters()

    def reset_parameters(self) -> None:
        generator = torch.Generator()
        generator.manual_seed(0)
        self.freqs = (
            2 * np.pi * self.bandwidth * torch.randn(self.num_channels, generator=generator).to(self.freqs.device)
        )
        self.phases = 2 * np.pi * torch.rand(self.num_channels, generator=generator).to(self.freqs.device)

    @NVTXRangeDecorator("FourierFeatures:forward")
    def forward(self, x: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """
        Apply the Fourier feature transformation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            gain (float, optional): An additional gain factor applied during the forward pass. Defaults to 1.

        Returns:
            torch.Tensor: The transformed tensor, with Fourier features applied.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
        x = x.cos().mul(self.gain * gain).to(in_dtype)
        return x


class PatchEmbed(nn.Module):
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the . This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches
    and embedding each patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int): The size of each spatial patch.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    """

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False
            ),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    @NVTXRangeDecorator("PatchEmbed:forward")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0, (
            f"H,W {(H, W)} should be divisible by spatial_patch_size {self.spatial_patch_size}"
        )
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)

        self.layer_norm.reset_parameters()

    def compute_caching(
        self,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: torch.Tensor | None = None,
    ) -> None:
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_B_T_D, scale_B_T_D = (
                self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        self.shift_B_T_1_1_D, self.scale_B_T_1_1_D = (
            rearrange(shift_B_T_D, "b t d -> b t 1 1 d"),
            rearrange(scale_B_T_D, "b t d -> b t 1 1 d"),
        )

    @NVTXRangeDecorator("FinalLayer:forward")
    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
    ):
        def _fn(
            _x_B_T_H_W_D: torch.Tensor,
            _norm_layer: nn.Module,
            _scale_B_T_1_1_D: torch.Tensor,
            _shift_B_T_1_1_D: torch.Tensor,
        ) -> torch.Tensor:
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        x_B_T_H_W_D = _fn(x_B_T_H_W_D, self.layer_norm, self.scale_B_T_1_1_D, self.shift_B_T_1_1_D)
        x_B_T_H_W_O = self.linear(x_B_T_H_W_D)
        return x_B_T_H_W_O


class Block(nn.Module):
    """
    A transformer block that combines self-attention, cross-attention and MLP layers with AdaLN modulation.
    Each component (self-attention, cross-attention, MLP) has its own layer normalization and AdaLN modulation.

    Parameters:
        x_dim (int): Dimension of input features
        context_dim (int): Dimension of context features for cross-attention
        num_heads (int): Number of attention heads
        mlp_ratio (float): Multiplier for MLP hidden dimension. Default: 4.0
        use_adaln_lora (bool): Whether to use AdaLN-LoRA modulation. Default: False
        adaln_lora_dim (int): Hidden dimension for AdaLN-LoRA layers. Default: 256

    The block applies the following sequence:
    1. Self-attention with AdaLN modulation
    2. Cross-attention with AdaLN modulation
    3. MLP with AdaLN modulation

    Each component uses skip connections and layer normalization.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        self_attention_backend: str = "transformer_engine",
        cross_attention_backend: str = "transformer_engine",
        natten_params: Mapping | None = None,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            x_dim,
            None,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
            backend=self_attention_backend,
            natten_params=natten_params,
        )

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(
            x_dim, context_dim, num_heads, x_dim // num_heads, qkv_format="bshd", backend=cross_attention_backend
        )
        # Installed after the pretrained checkpoint is loaded. The input is a
        # flattened Plücker-ray patch matching one target DiT token.
        self.camera_embedding = None

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        self.cp_size = None
        # Optional Resampling-Forcing history memory, installed only for mode A
        # after loading the original MeanFlow checkpoint.
        self.history_kv_gate = None
        self.history_age_embedding = None
        self._runtime_history_kv = None
        self._capture_history_kv = False
        self._captured_history_kv = None

    def enable_history_kv(self, max_history: int = 4) -> None:
        """Use this block's own Q/K/V projections for causal history memory."""
        if self.history_kv_gate is not None:
            if self.history_age_embedding.shape[0] != max_history:
                raise ValueError("History capacity cannot change after initialization")
            return
        if max_history < 1:
            raise ValueError("max_history must be positive")
        weight = self.self_attn.q_proj.weight
        self.history_kv_gate = nn.Parameter(torch.zeros((), device=weight.device, dtype=weight.dtype))
        self.history_age_embedding = nn.Parameter(torch.zeros(
            max_history, self.self_attn.n_heads, self.self_attn.head_dim,
            device=weight.device, dtype=weight.dtype,
        ))

    def set_history_kv(self, entries, *, capture: bool = False) -> None:
        if self.history_kv_gate is None:
            raise RuntimeError("History K/V is not enabled")
        if len(entries) > self.history_age_embedding.shape[0]:
            raise ValueError("History cache exceeds configured capacity")
        self._runtime_history_kv = tuple(entries)
        self._capture_history_kv = bool(capture)
        self._captured_history_kv = None

    def clear_history_kv(self) -> None:
        self._runtime_history_kv = None
        self._capture_history_kv = False
        self._captured_history_kv = None

    def enable_reference_kv(self) -> None:
        """Install the added visual-reference K/V projections."""
        self.cross_attn.enable_added_reference_kv()

    def enable_camera_conditioning(self, camera_patch_dim: int) -> None:
        """Install ArtiFixer's zero-initialized target ray-map projection."""
        if self.camera_embedding is not None:
            if self.camera_embedding.in_features != camera_patch_dim:
                raise ValueError(
                    f"Camera embedding already expects {self.camera_embedding.in_features} features, "
                    f"not {camera_patch_dim}"
                )
            return
        if camera_patch_dim <= 0:
            raise ValueError(f"camera_patch_dim must be positive, got {camera_patch_dim}")
        weight = self.self_attn.q_proj.weight
        self.camera_embedding = nn.Linear(
            camera_patch_dim,
            self.x_dim,
            bias=True,
            device=weight.device,
            dtype=weight.dtype,
        )
        torch.nn.init.zeros_(self.camera_embedding.weight)
        torch.nn.init.zeros_(self.camera_embedding.bias)

    def set_context_parallel_group(self, process_group, ranks, stream):
        self.cp_size = None if ranks is None else len(ranks)
        self.self_attn.set_context_parallel_group(
            process_group=process_group,
            ranks=ranks,
            stream=stream,
        )

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def compute_caching(
        self,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        adaln_lora_B_T_3D: torch.Tensor | None = None,
    ) -> None:
        if self.use_adaln_lora:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape tensors from (B, T, D) to (B, T, 1, 1, D) for broadcasting
        self.shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
        self.scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
        self.gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

        self.shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        self.scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        self.gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

        self.shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
        self.scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
        self.gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")

        # cache crossattn_emb
        self.crossattn_emb = crossattn_emb

    @NVTXRangeDecorator("Text2ImageDIT:forward")
    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        rope_emb_L_1_1_D: torch.Tensor | None = None,
        extra_per_block_pos_emb: torch.Tensor | None = None,
        reference_tokens: torch.Tensor | None = None,
        camera_tokens: torch.Tensor | None = None,
        target_prope: PropeDotProductAttention | None = None,
        reference_prope: PropeDotProductAttention | None = None,
        joint_reference_B_R_H_W_D: torch.Tensor | None = None,
        joint_reference_rope_L_1_1_D: torch.Tensor | None = None,
        joint_reference_extra_pos_emb: torch.Tensor | None = None,
        joint_reference_camera_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run one transformer block.

        Option A carries one clean reference stream through the main
        self-attention/MLP path.  Target and joint-reference tokens therefore
        share self-attention, but **only target tokens** execute cross-attention
        against text plus the remaining reference K/V memory.  This keeps the
        selected reference as a strong joint reasoning partner without letting
        it recursively query the other seven references.
        """
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        B, T, H, W, D = x_B_T_H_W_D.shape
        joint_views = 0
        if joint_reference_B_R_H_W_D is not None:
            if joint_reference_B_R_H_W_D.ndim != 5:
                raise ValueError(
                    "Expected joint reference [B,R,H,W,D], got "
                    f"{tuple(joint_reference_B_R_H_W_D.shape)}"
                )
            if joint_reference_B_R_H_W_D.shape[0] != B:
                raise ValueError("Joint-reference batch does not match target batch")
            joint_views = joint_reference_B_R_H_W_D.shape[1]
            if joint_views != 1:
                raise ValueError(
                    f"Option A requires exactly one joint reference view, got {joint_views}"
                )
            if tuple(joint_reference_B_R_H_W_D.shape[2:]) != (H, W, D):
                raise ValueError(
                    "Joint-reference token grid must match target spatial/model dimensions: "
                    f"target={(H, W, D)}, joint={tuple(joint_reference_B_R_H_W_D.shape[2:])}"
                )
            if joint_reference_extra_pos_emb is not None:
                if tuple(joint_reference_extra_pos_emb.shape) != tuple(joint_reference_B_R_H_W_D.shape):
                    raise ValueError(
                        "Joint-reference absolute-position grid mismatch: "
                        f"pos={tuple(joint_reference_extra_pos_emb.shape)}, "
                        f"joint={tuple(joint_reference_B_R_H_W_D.shape)}"
                    )
                joint_reference_B_R_H_W_D = (
                    joint_reference_B_R_H_W_D + joint_reference_extra_pos_emb
                )

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            self.scale_self_attn_B_T_1_1_D,
            self.shift_self_attn_B_T_1_1_D,
        )
        if camera_tokens is not None:
            if self.camera_embedding is None:
                raise RuntimeError("Camera tokens were provided before camera conditioning was enabled")
            if camera_tokens.shape[:-1] != normalized_x_B_T_H_W_D.shape[:-1]:
                raise ValueError(
                    "Camera-token grid does not match target token grid: "
                    f"{tuple(camera_tokens.shape)} vs {tuple(normalized_x_B_T_H_W_D.shape)}"
                )
            normalized_x_B_T_H_W_D = normalized_x_B_T_H_W_D + self.camera_embedding(
                camera_tokens.to(dtype=normalized_x_B_T_H_W_D.dtype)
            )

        if self.history_kv_gate is not None and self._capture_history_kv:
            if T != 1:
                raise ValueError("Clean history capture requires one latent time cell")
            clean_tokens = rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d")
            _, history_k, history_v = self.self_attn.compute_qkv(clean_tokens, None, rope_emb=None)
            self._captured_history_kv = (history_k.detach(), history_v.detach())

        normalized_joint_reference = None
        joint_self_scale = None
        joint_self_shift = None
        joint_self_gate = None
        joint_mlp_scale = None
        joint_mlp_shift = None
        joint_mlp_gate = None
        if joint_reference_B_R_H_W_D is not None:
            # The target cache is usually [B,1,D] and broadcasts over the V=1/V=5
            # latent grid. Keep the joint clean reference explicitly one-view even
            # if a future caller supplies per-target-latent timestep modulation
            # [B,T,D]. Using [:,:1] prevents Stage-2 T>1 from duplicating the
            # single joint-reference stream by broadcasting.
            joint_self_scale = self.scale_self_attn_B_T_1_1_D[:, :1]
            joint_self_shift = self.shift_self_attn_B_T_1_1_D[:, :1]
            joint_self_gate = self.gate_self_attn_B_T_1_1_D[:, :1]
            joint_mlp_scale = self.scale_mlp_B_T_1_1_D[:, :1]
            joint_mlp_shift = self.shift_mlp_B_T_1_1_D[:, :1]
            joint_mlp_gate = self.gate_mlp_B_T_1_1_D[:, :1]
            normalized_joint_reference = _fn(
                joint_reference_B_R_H_W_D,
                self.layer_norm_self_attn,
                joint_self_scale,
                joint_self_shift,
            )
            if joint_reference_camera_tokens is not None:
                if self.camera_embedding is None:
                    raise RuntimeError(
                        "Joint-reference camera tokens were provided before camera conditioning was enabled"
                    )
                if joint_reference_camera_tokens.shape[:-1] != normalized_joint_reference.shape[:-1]:
                    raise ValueError(
                        "Joint-reference camera-token grid mismatch: "
                        f"{tuple(joint_reference_camera_tokens.shape)} vs "
                        f"{tuple(normalized_joint_reference.shape)}"
                    )
                normalized_joint_reference = normalized_joint_reference + self.camera_embedding(
                    joint_reference_camera_tokens.to(dtype=normalized_joint_reference.dtype)
                )

        video_size = VideoSize(T=T + joint_views, H=H, W=W)

        if self.cp_size is not None and self.cp_size > 1:
            if joint_reference_B_R_H_W_D is not None:
                raise ValueError(
                    "Context parallel is not implemented for Option-A joint-reference self-attention"
                )
            video_size = VideoSize(T=T * self.cp_size, H=H, W=W)

        self_attn_input = normalized_x_B_T_H_W_D
        self_attn_rope = rope_emb_L_1_1_D
        if normalized_joint_reference is not None:
            self_attn_input = torch.cat(
                (normalized_x_B_T_H_W_D, normalized_joint_reference), dim=1
            )
            if (rope_emb_L_1_1_D is None) != (joint_reference_rope_L_1_1_D is None):
                raise ValueError(
                    "Target and joint-reference RoPE must either both be present or both be absent"
                )
            if rope_emb_L_1_1_D is not None:
                expected_target_rope = T * H * W
                expected_joint_rope = joint_views * H * W
                if rope_emb_L_1_1_D.shape[0] != expected_target_rope:
                    raise ValueError(
                        "Target RoPE length does not match target token grid: "
                        f"rope={rope_emb_L_1_1_D.shape[0]}, expected={expected_target_rope}"
                    )
                if joint_reference_rope_L_1_1_D.shape[0] != expected_joint_rope:
                    raise ValueError(
                        "Joint-reference RoPE length does not match its token grid: "
                        f"rope={joint_reference_rope_L_1_1_D.shape[0]}, expected={expected_joint_rope}"
                    )
                self_attn_rope = torch.cat(
                    (rope_emb_L_1_1_D, joint_reference_rope_L_1_1_D), dim=0
                )

        self_attn_result = rearrange(
            self.self_attn(
                rearrange(self_attn_input, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=self_attn_rope,
                video_size=video_size,
            ),
            "b (t h w) d -> b t h w d",
            t=T + joint_views,
            h=H,
            w=W,
        )
        result_B_T_H_W_D = self_attn_result[:, :T]
        x_B_T_H_W_D = (
            x_B_T_H_W_D
            + self.gate_self_attn_B_T_1_1_D * result_B_T_H_W_D
        )
        if self._runtime_history_kv:
            if T != 1 or self.history_kv_gate is None:
                raise ValueError("Historical cross-attention requires enabled V=1 target")
            query_tokens = rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d")
            history_q, _, _ = self.self_attn.compute_qkv(query_tokens, None, rope_emb=None)
            history_keys = []
            history_values = []
            age_start = self.history_age_embedding.shape[0] - len(self._runtime_history_kv)
            for index, (key, value) in enumerate(self._runtime_history_kv):
                if key.shape != value.shape or key.shape[0] != B:
                    raise ValueError("History K/V batch or token dimensions do not match")
                history_keys.append(key + self.history_age_embedding[age_start + index].view(1, 1, *key.shape[-2:]))
                history_values.append(value)
            history_k = torch.cat(history_keys, dim=1)
            history_v = torch.cat(history_values, dim=1)
            # Use explicit attention here: MeanFlow's autograd JVP can require
            # double backward, which some fused SDPA kernels do not support.
            q = history_q.transpose(1, 2)
            k = history_k.transpose(1, 2)
            v = history_v.transpose(1, 2)
            logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            attention = torch.softmax(logits.float(), dim=-1).to(dtype=v.dtype)
            history_values = torch.matmul(attention, v).transpose(1, 2)
            history_result = self.self_attn.output_proj(rearrange(
                history_values, "b s h d -> b s (h d)"
            ))
            history_result = rearrange(history_result, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
            x_B_T_H_W_D = x_B_T_H_W_D + (
                self.gate_self_attn_B_T_1_1_D * self.history_kv_gate * history_result
            )
        if joint_reference_B_R_H_W_D is not None:
            joint_result = self_attn_result[:, T:]
            joint_reference_B_R_H_W_D = (
                joint_reference_B_R_H_W_D
                + joint_self_gate * joint_result
            )

        def _x_fn(
            _x_B_T_H_W_D: torch.Tensor,
            layer_norm_cross_attn: Callable,
            _scale_cross_attn_B_T_1_1_D: torch.Tensor,
            _shift_cross_attn_B_T_1_1_D: torch.Tensor,
        ) -> torch.Tensor:
            _normalized_x_B_T_H_W_D = _fn(
                _x_B_T_H_W_D, layer_norm_cross_attn, _scale_cross_attn_B_T_1_1_D, _shift_cross_attn_B_T_1_1_D
            )
            query_tokens = rearrange(_normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d")
            if reference_tokens is None:
                cross_attn_result = self.cross_attn(
                    query_tokens,
                    self.crossattn_emb,
                    rope_emb=rope_emb_L_1_1_D,
                )
            else:
                cross_attn_result = self.cross_attn.forward_with_added_reference_kv(
                    query_tokens,
                    self.crossattn_emb,
                    reference_tokens,
                    target_prope=target_prope,
                    reference_prope=reference_prope,
                )
            _result_B_T_H_W_D = rearrange(
                cross_attn_result,
                "b (t h w) d -> b t h w d",
                t=T,
                h=H,
                w=W,
            )
            return _result_B_T_H_W_D

        result_B_T_H_W_D = _x_fn(
            x_B_T_H_W_D,
            self.layer_norm_cross_attn,
            self.scale_cross_attn_B_T_1_1_D,
            self.shift_cross_attn_B_T_1_1_D,
        )
        x_B_T_H_W_D = result_B_T_H_W_D * self.gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_mlp,
            self.scale_mlp_B_T_1_1_D,
            self.shift_mlp_B_T_1_1_D,
        )
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
        x_B_T_H_W_D = x_B_T_H_W_D + self.gate_mlp_B_T_1_1_D * result_B_T_H_W_D
        if joint_reference_B_R_H_W_D is not None:
            # Option A deliberately skips cross-attention for the joint
            # reference.  It is nevertheless updated by the shared MLP so it
            # remains a contextualized main-stream token bank for the next
            # block.
            normalized_joint_reference = _fn(
                joint_reference_B_R_H_W_D,
                self.layer_norm_mlp,
                joint_mlp_scale,
                joint_mlp_shift,
            )
            joint_reference_B_R_H_W_D = (
                joint_reference_B_R_H_W_D
                + joint_mlp_gate * self.mlp(normalized_joint_reference)
            )
            return x_B_T_H_W_D, joint_reference_B_R_H_W_D
        return x_B_T_H_W_D


class MiniTrainDIT(WeightTrainingStat):
    """
    A clean impl of DIT that can load and  reproduce the training results of the original DIT model in~(cosmos 1)
    A general implementation of adaln-modulated VIT-like~(DiT) transformer for video processing.

    Args:
        max_img_h (int): Maximum height of the input images.
        max_img_w (int): Maximum width of the input images.
        max_frames (int): Maximum number of frames in the video sequence.
        in_channels (int): Number of input channels (e.g., RGB channels for color images).
        out_channels (int): Number of output channels.
        patch_spatial (tuple): Spatial resolution of patches for input processing.
        patch_temporal (int): Temporal resolution of patches for input processing.
        concat_padding_mask (bool): If True, includes a mask channel in the input to handle padding.
        model_channels (int): Base number of channels used throughout the model.
        num_blocks (int): Number of transformer blocks.
        num_heads (int): Number of heads in the multi-head attention layers.
        mlp_ratio (float): Expansion ratio for MLP blocks.
        crossattn_emb_channels (int): Number of embedding channels for cross-attention.
        pos_emb_cls (str): Type of positional embeddings.
        pos_emb_learnable (bool): Whether positional embeddings are learnable.
        pos_emb_interpolation (str): Method for interpolating positional embeddings.
        min_fps (int): Minimum frames per second.
        max_fps (int): Maximum frames per second.
        use_adaln_lora (bool): Whether to use AdaLN-LoRA.
        adaln_lora_dim (int): Dimension for AdaLN-LoRA.
        rope_h_extrapolation_ratio (float): Height extrapolation ratio for RoPE.
        rope_w_extrapolation_ratio (float): Width extrapolation ratio for RoPE.
        rope_t_extrapolation_ratio (float): Temporal extrapolation ratio for RoPE.
        extra_per_block_abs_pos_emb (bool): Whether to use extra per-block absolute positional embeddings.
        extra_h_extrapolation_ratio (float): Height extrapolation ratio for extra embeddings.
        extra_w_extrapolation_ratio (float): Width extrapolation ratio for extra embeddings.
        extra_t_extrapolation_ratio (float): Temporal extrapolation ratio for extra embeddings.
        natten_parameters (Optional[List[dict]]): list of parameters for NATTEN attention backend.
            Should include at least one key: `window_size`. Other keys such as `stride`, `dilation`,
            and `is_causal` are optional, and fall back to NATTEN defaults.
            Key `base_shape` may also be passed so that all other parameters (i.e. `window_size`)
            are scaled according to input video resolution.
    """

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,  # tuple,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        # attention settings
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        atten_backend: str = "transformer_engine",
        # cross attention settings
        crossattn_emb_channels: int = 1024,
        use_crossattn_projection: bool = TEXT_ENCODER_CLASS is TextEncoderClass.COSMOS_REASON1,
        crossattn_proj_in_channels: int = 100352,
        # positional embedding settings
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        sac_config: SACConfig = SACConfig(),  # noqa: B008
        natten_parameters: dict | list = None,  # noqa: RUF013
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.atten_backend = atten_backend
        # positional embedding settings
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.cuda_graphs = {}

        self.build_patch_embed()
        self.build_pos_embed()
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )

        if natten_parameters is not None and not isinstance(natten_parameters, Sequence):
            raise ValueError(
                "`natten_parameters` must either be None, or a list of the same length as the number of blocks, "
                f"got {type(natten_parameters)=}."
            )

        if natten_parameters is not None and len(natten_parameters) != num_blocks:
            raise ValueError(
                "`natten_parameters` must either be None, or a list of the same length as the number of blocks, "
                f"got {len(natten_parameters)=} != {num_blocks=}."
            )

        if natten_parameters is not None and any(
            x is not None and not isinstance(x, Mapping) for x in natten_parameters
        ):
            raise ValueError(
                "`natten_parameters` must be a list of Nones (self attention, no sparsity) or dicts "
                f"that indicate at least a `window_size`, got {natten_parameters=}."
            )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    self_attention_backend=atten_backend
                    if natten_parameters is None or natten_parameters[i] is None
                    else "natten",
                    cross_attention_backend=atten_backend,
                    natten_params=None if natten_parameters is None else natten_parameters[i],
                )
                for i in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
        )

        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )
        else:
            self.crossattn_proj = None

        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)
        self.init_weights()
        self.enable_selective_checkpoint(sac_config)
        self._is_context_parallel_enabled = False
        self._runtime_reference_latents = None
        self._runtime_camera_conditioning = None
        head_dim = self.model_channels // self.num_heads
        self.target_camera_prope = PropeDotProductAttention(head_dim=head_dim)
        self.reference_camera_prope = PropeDotProductAttention(head_dim=head_dim)
        self.camera_patch_dim = None
        self.reference_time_embedding_dim = None
        self.reference_time_scale = None
        self.reference_condition_projection = None
        self.camera_embed_enabled = False
        self.time_embed_enabled = False
        # Installed only after the pretrained Cosmos checkpoint is loaded.
        # Keeping this as None here avoids introducing a missing checkpoint key
        # during Text2ImagePipeline construction.
        self.joint_reference_stream_embedding = None
        self.world_scenario_mode = 0
        self._runtime_world_latent = None
        self._runtime_reference_world_latent = None

    @staticmethod
    def _unwrap_block(block):
        """Resolve a Block through optional activation-checkpoint wrappers."""
        for _ in range(3):
            if hasattr(block, "enable_reference_kv"):
                return block
            if hasattr(block, "_checkpoint_wrapped_module"):
                block = block._checkpoint_wrapped_module
                continue
            if hasattr(block, "module"):
                block = block.module
                continue
            break
        raise AttributeError("Could not resolve a DiT block")

    def enable_reference_kv(self) -> None:
        """Install ArtiFixer-style reference K/V projections in every block."""
        for block in self.blocks:
            self._unwrap_block(block).enable_reference_kv()

    def enable_history_kv(self, max_history: int = 4) -> None:
        """Install a separate, bounded causal history-memory branch in each block."""
        for block in self.blocks:
            self._unwrap_block(block).enable_history_kv(max_history)

    def set_history_kv(self, per_layer_entries, *, capture: bool = False) -> None:
        if len(per_layer_entries) != len(self.blocks):
            raise ValueError("Expected one history cache list per DiT block")
        for block, entries in zip(self.blocks, per_layer_entries):
            self._unwrap_block(block).set_history_kv(entries, capture=capture)

    def get_captured_history_kv(self):
        captured = []
        for block in self.blocks:
            pair = self._unwrap_block(block)._captured_history_kv
            if pair is None:
                raise RuntimeError("A DiT block did not capture clean-history K/V")
            captured.append(pair)
        return tuple(captured)

    def clear_history_kv(self) -> None:
        for block in self.blocks:
            self._unwrap_block(block).clear_history_kv()

    def enable_world_scenario(self, mode):
        from model.world_scenario import WorldScenarioControl

        self.world_scenario_mode = mode
        if mode == 1:
            self.world_scenario_patch = PatchEmbed(
                self.patch_spatial, self.patch_temporal, self.out_channels, self.model_channels
            ).to(self.x_embedder.proj[1].weight)
            nn.init.zeros_(self.world_scenario_patch.proj[1].weight)
        elif mode == 2:
            self.world_scenario_control = WorldScenarioControl(self).to(self.x_embedder.proj[1].weight)
        else:
            raise ValueError("World Scenario mode must be 1 or 2")

    def enable_hybrid_reference_fusion(self) -> None:
        """Enable Option-A hybrid reference fusion.

        One selected reference view is carried through the *main* transformer
        blocks together with the target tokens.  The remaining references stay
        in the added visual K/V cross-attention branch.  A small learnable
        stream/type embedding lets the transformer distinguish the joint clean
        reference from the RF target stream without pretending that it is an
        additional temporal target frame.

        The parameter is created only after the pretrained Cosmos checkpoint is
        loaded, exactly like the added reference K/V projections.
        """
        self.enable_reference_kv()
        if self.joint_reference_stream_embedding is not None:
            return
        weight = self.x_embedder.proj[1].weight
        self.joint_reference_stream_embedding = nn.Parameter(
            torch.zeros(
                self.model_channels,
                device=weight.device,
                dtype=weight.dtype,
            )
        )

    def enable_camera_conditioning(
        self,
        camera_patch_dim: int | None = None,
        reference_time_embedding_dim: int | None = None,
        reference_time_scale: float = 1.0,
    ) -> None:
        """Install independently switchable camera and temporal conditioning.

        A non-None ``camera_patch_dim`` enables target/reference Plucker rays
        and PRoPE. A non-None ``reference_time_embedding_dim`` enables signed
        Fourier time features. Active features are concatenated, projected
        once, and added to the shared reference patch embeddings. Zero
        initialization preserves the pretrained reference-only behavior.
        """
        if camera_patch_dim is None and reference_time_embedding_dim is None:
            raise ValueError("At least one of camera or time embedding must be enabled")
        if camera_patch_dim is not None:
            if camera_patch_dim <= 0:
                raise ValueError(f"camera_patch_dim must be positive, got {camera_patch_dim}")
            if self.camera_patch_dim is not None and self.camera_patch_dim != camera_patch_dim:
                raise ValueError(
                    f"Camera conditioning already uses patch dim {self.camera_patch_dim}, "
                    f"not {camera_patch_dim}"
                )
        if reference_time_embedding_dim is not None:
            if reference_time_embedding_dim <= 0 or reference_time_embedding_dim % 2:
                raise ValueError(
                    "reference_time_embedding_dim must be a positive even integer, "
                    f"got {reference_time_embedding_dim}"
                )
        if reference_time_embedding_dim is not None and (
            not math.isfinite(reference_time_scale)
            or reference_time_scale <= 0
        ):
            raise ValueError(
                f"reference_time_scale must be positive, got {reference_time_scale}"
            )
        if (
            reference_time_embedding_dim is not None
            and
            self.reference_time_embedding_dim is not None
            and self.reference_time_embedding_dim != reference_time_embedding_dim
        ):
            raise ValueError(
                "Temporal conditioning already uses embedding dim "
                f"{self.reference_time_embedding_dim}, not {reference_time_embedding_dim}"
            )
        self.camera_patch_dim = camera_patch_dim
        self.reference_time_embedding_dim = reference_time_embedding_dim
        self.reference_time_scale = (
            float(reference_time_scale)
            if reference_time_embedding_dim is not None
            else 1.0
        )
        self.camera_embed_enabled = camera_patch_dim is not None
        self.time_embed_enabled = reference_time_embedding_dim is not None
        if self.camera_embed_enabled:
            for block in self.blocks:
                self._unwrap_block(block).enable_camera_conditioning(camera_patch_dim)

        if self.reference_condition_projection is None:
            weight = self.x_embedder.proj[1].weight
            condition_dim = (camera_patch_dim or 0) + (reference_time_embedding_dim or 0)
            self.reference_condition_projection = nn.Linear(
                condition_dim,
                self.model_channels,
                bias=True,
                device=weight.device,
                dtype=weight.dtype,
            )
            torch.nn.init.zeros_(self.reference_condition_projection.weight)
            torch.nn.init.zeros_(self.reference_condition_projection.bias)

            if self.time_embed_enabled:
                half_dim = reference_time_embedding_dim // 2
                frequencies = torch.exp(
                    -math.log(10000.0)
                    * torch.arange(
                        half_dim,
                        device=weight.device,
                        dtype=torch.float32,
                    )
                    / max(half_dim - 1, 1)
                )
                self.register_buffer(
                    "reference_time_frequencies",
                    frequencies,
                    persistent=False,
                )

    def set_reference_latents(self, reference_latents: torch.Tensor) -> None:
        """Set independently VAE-encoded references for the next DiT call."""
        if reference_latents.ndim != 5:
            raise ValueError(
                "Expected reference latents [B,C,R,H,W], got "
                f"{tuple(reference_latents.shape)}"
            )
        if reference_latents.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} reference latent channels, "
                f"got {reference_latents.shape[1]}"
            )
        self._runtime_reference_latents = reference_latents

    def clear_reference_latents(self) -> None:
        self._runtime_world_latent = None
        self._runtime_reference_world_latent = None
        self._runtime_reference_latents = None

    def set_camera_conditioning(
        self,
        target_camera_rays: torch.Tensor | None = None,
        reference_camera_rays: torch.Tensor | None = None,
        reference_time_offsets: torch.Tensor | None = None,
        target_viewmats: torch.Tensor | None = None,
        target_Ks: torch.Tensor | None = None,
        reference_viewmats: torch.Tensor | None = None,
        reference_Ks: torch.Tensor | None = None,
    ) -> None:
        """Set only the runtime inputs required by the enabled embeddings."""
        if not (self.camera_embed_enabled or self.time_embed_enabled):
            raise RuntimeError("Camera/time conditioning modules are not enabled")
        if self._runtime_reference_latents is None:
            raise RuntimeError("Set reference latents before camera/time conditioning")

        batch = self._runtime_reference_latents.shape[0]
        references = self._runtime_reference_latents.shape[2]
        runtime = {}

        if self.camera_embed_enabled:
            camera_values = {
                "target_camera_rays": target_camera_rays,
                "reference_camera_rays": reference_camera_rays,
                "target_viewmats": target_viewmats,
                "target_Ks": target_Ks,
                "reference_viewmats": reference_viewmats,
                "reference_Ks": reference_Ks,
            }
            missing = [name for name, value in camera_values.items() if value is None]
            if missing:
                raise ValueError(f"Missing enabled camera inputs: {missing}")
            if target_camera_rays.ndim != 5 or target_camera_rays.shape[-1] != 6:
                raise ValueError(
                    "Expected target_camera_rays [B,T,H,W,6], got "
                    f"{tuple(target_camera_rays.shape)}"
                )
            target_views = target_camera_rays.shape[1]
            if target_camera_rays.shape[0] != batch:
                raise ValueError("Target ray batch does not match reference latents")
            if reference_camera_rays.ndim != 5 or reference_camera_rays.shape[-1] != 6:
                raise ValueError(
                    "Expected reference_camera_rays [B,R,H,W,6], got "
                    f"{tuple(reference_camera_rays.shape)}"
                )
            if reference_camera_rays.shape[:2] != (batch, references):
                raise ValueError("Reference ray count does not match reference latents")
            if target_viewmats.shape != (batch, target_views, 4, 4):
                raise ValueError(f"Invalid target_viewmats shape {tuple(target_viewmats.shape)}")
            if target_Ks.shape != (batch, target_views, 3, 3):
                raise ValueError(f"Invalid target_Ks shape {tuple(target_Ks.shape)}")
            if reference_viewmats.shape != (batch, references, 4, 4):
                raise ValueError(f"Invalid reference_viewmats shape {tuple(reference_viewmats.shape)}")
            if reference_Ks.shape != (batch, references, 3, 3):
                raise ValueError(f"Invalid reference_Ks shape {tuple(reference_Ks.shape)}")
            runtime.update(camera_values)

        if self.time_embed_enabled:
            if reference_time_offsets is None:
                raise ValueError("Missing enabled time input: reference_time_offsets")
            if reference_time_offsets.shape != (batch, references):
                raise ValueError(
                    f"Expected reference_time_offsets {(batch, references)}, got "
                    f"{tuple(reference_time_offsets.shape)}"
                )
            if not torch.isfinite(reference_time_offsets).all():
                raise ValueError("reference_time_offsets must be finite")
            runtime["reference_time_offsets"] = reference_time_offsets

        self._runtime_camera_conditioning = runtime

    def clear_camera_conditioning(self) -> None:
        self._runtime_camera_conditioning = None

    def prepare_reference_embeddings(
        self, reference_latents: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Patchify references using the target DiT's shared patch embedder.

        References were VAE-encoded independently, so their R dimension is a
        view axis rather than a temporally compressed video. Cosmos T2I uses a
        temporal patch size of one, preserving that per-image layout.

        The returned tensor intentionally keeps the explicit reference-view
        axis ``[B,R,H,W,D]``.  Option A needs that axis so view 0 can become the
        joint self-attention stream while views 1..R-1 remain cross-attention
        memory.
        """
        if self.patch_temporal != 1:
            raise ValueError(
                "Independent ArtiFixer-style references require patch_temporal=1, "
                f"got {self.patch_temporal}"
            )
        if self.concat_padding_mask:
            reference_padding = torch.zeros(
                reference_latents.shape[0],
                1,
                reference_latents.shape[2],
                reference_latents.shape[3],
                reference_latents.shape[4],
                device=reference_latents.device,
                dtype=reference_latents.dtype,
            )
            reference_latents = torch.cat((reference_latents, reference_padding), dim=1)

        reference_embeddings = self.x_embedder(reference_latents)
        if self.world_scenario_mode == 1:
            reference_embeddings = reference_embeddings + self.world_scenario_patch(
                self._runtime_reference_world_latent
            )
        elif self.world_scenario_mode == 2:
            batch, _, views, height, width = reference_latents.shape
            rgb = reference_latents.permute(0, 2, 1, 3, 4).reshape(batch * views, -1, 1, height, width)
            condition = self._runtime_reference_world_latent.permute(0, 2, 1, 3, 4).reshape(
                batch * views, -1, 1, height, width
            )
            interval = self.control_interval_B_T
            residual = self.world_scenario_control(
                rgb,
                condition,
                self.control_timesteps_B_T.repeat_interleave(views, dim=0),
                self.crossattn_emb.repeat_interleave(views, dim=0),
                interval=None if interval is None else interval.repeat_interleave(views, dim=0),
                padded=True,
            )[-1]
            reference_embeddings = reference_embeddings + residual.reshape(batch, views, *residual.shape[2:])
        _, references, height, width, _ = reference_embeddings.shape
        if self.camera_embed_enabled or self.time_embed_enabled:
            if self._runtime_camera_conditioning is None:
                raise RuntimeError("Enabled camera/time embeddings require runtime inputs")
            if self.reference_condition_projection is None:
                raise RuntimeError("Reference camera/time fusion is not enabled")

            condition_parts = []
            if self.camera_embed_enabled:
                reference_camera_condition = self.prepare_camera_tokens(
                    self._runtime_camera_conditioning["reference_camera_rays"],
                    (references, height, width),
                ).float()
                if self.joint_reference_stream_embedding is not None and references > 0:
                    # Cross-attention references keep the original one-time
                    # reference-camera projection. The joint reference instead
                    # receives target-style per-block camera injection below;
                    # zero this component here to avoid conditioning it twice.
                    reference_camera_condition = reference_camera_condition.clone()
                    reference_camera_condition[:, :1] = 0.0
                condition_parts.append(
                    reference_camera_condition
                )
            if self.time_embed_enabled:
                time_offsets = self._runtime_camera_conditioning[
                    "reference_time_offsets"
                ].float() / self.reference_time_scale
                time_arguments = (
                    time_offsets[..., None]
                    * self.reference_time_frequencies[None, None]
                )
                time_features = torch.cat(
                    (torch.cos(time_arguments), torch.sin(time_arguments)),
                    dim=-1,
                )
                condition_parts.append(
                    time_features[:, :, None, None].expand(
                        -1,
                        -1,
                        height,
                        width,
                        -1,
                    )
                )
            reference_condition = torch.cat(condition_parts, dim=-1).to(
                dtype=reference_embeddings.dtype
            )
            reference_embeddings = reference_embeddings + self.reference_condition_projection(
                reference_condition
            )
        return reference_embeddings, (references, height, width)

    def prepare_reference_tokens(
        self, reference_latents: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Backward-compatible flattened reference token helper."""
        reference_embeddings, grid = self.prepare_reference_embeddings(reference_latents)
        return rearrange(reference_embeddings, "b r h w d -> b (r h w) d"), grid

    @staticmethod
    def _match_target_camera_views(
        values: torch.Tensor,
        target_views: int,
    ) -> torch.Tensor:
        """Match RGB-view camera data to the VAE's latent temporal grid.

        A causal video VAE may temporally compress V=5 RGB inputs. Nearest
        left-aligned sampling preserves current view 0 and selects the camera
        associated with each latent temporal cell.
        """
        source_views = values.shape[1]
        if source_views == target_views:
            return values
        if source_views < 1 or target_views < 1:
            raise ValueError(
                f"Camera view counts must be positive, got {source_views}/{target_views}"
            )
        indices = torch.div(
            torch.arange(target_views, device=values.device) * source_views,
            target_views,
            rounding_mode="floor",
        ).clamp_max(source_views - 1)
        return values.index_select(1, indices)

    def prepare_camera_tokens(
        self,
        camera_rays: torch.Tensor,
        target_grid: tuple[int, int, int],
    ) -> torch.Tensor:
        """Flatten each view token's full-resolution Plücker-ray patch."""
        views, token_height, token_width = target_grid
        camera_rays = self._match_target_camera_views(camera_rays, views)
        image_height, image_width = camera_rays.shape[2:4]
        if image_height % token_height or image_width % token_width:
            raise ValueError(
                f"Camera-ray resolution {(image_height, image_width)} is not divisible by "
                f"target token grid {(token_height, token_width)}"
            )
        patch_height = image_height // token_height
        patch_width = image_width // token_width
        camera_tokens = rearrange(
            camera_rays,
            "b t (h ph) (w pw) c -> b t h w (c ph pw)",
            h=token_height,
            w=token_width,
            ph=patch_height,
            pw=patch_width,
        )
        if camera_tokens.shape[-1] != self.camera_patch_dim:
            raise ValueError(
                f"Camera patch has {camera_tokens.shape[-1]} features, but projections "
                f"expect {self.camera_patch_dim}. Check the VAE compression factor."
            )
        return camera_tokens

    def prepare_camera_prope(
        self,
        target_grid: tuple[int, int, int],
        reference_grid: tuple[int, int, int],
        *,
        reference_start: int = 0,
    ) -> tuple[PropeDotProductAttention, PropeDotProductAttention]:
        """Precompute target-Q and reference-K/V PRoPE transforms.

        ``reference_start`` lets Option A exclude the joint reference (view 0)
        from the added K/V memory.  Cross-attention then uses only the remaining
        references while the selected view participates in main self-attention.
        """
        if self._runtime_camera_conditioning is None:
            raise RuntimeError("Camera matrices are required for reference PRoPE")
        target_views, target_height, target_width = target_grid
        references, reference_height, reference_width = reference_grid
        camera = self._runtime_camera_conditioning
        total_references = camera["reference_viewmats"].shape[1]
        if reference_start < 0 or reference_start + references > total_references:
            raise ValueError(
                "Reference PRoPE slice is out of bounds: "
                f"start={reference_start}, count={references}, total={total_references}"
            )
        reference_viewmats = camera["reference_viewmats"][:, reference_start : reference_start + references]
        reference_Ks = camera["reference_Ks"][:, reference_start : reference_start + references]

        target_viewmats = self._match_target_camera_views(
            camera["target_viewmats"], target_views
        )
        target_Ks = self._match_target_camera_views(
            camera["target_Ks"], target_views
        )

        self.target_camera_prope.update_coeffs(target_width, target_height, target_viewmats.device)
        self.reference_camera_prope.update_coeffs(
            reference_width,
            reference_height,
            reference_viewmats.device,
        )
        self.target_camera_prope.precompute(target_viewmats, target_Ks)
        self.reference_camera_prope.precompute(
            reference_viewmats,
            reference_Ks,
        )
        return self.target_camera_prope, self.reference_camera_prope

    def init_weights(self) -> None:
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.reset_parameters()

        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

    def build_patch_embed(self) -> None:
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
        )

    def build_pos_embed(self) -> None:
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        log.debug(f"Building positional embedding with {self.pos_emb_cls} class, impl {cls_type}")
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )
        self.pos_embedder = cls_type(
            **kwargs,  # type: ignore
        )

        if self.extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            self.extra_pos_embedder = LearnablePosEmbAxis(
                **kwargs,  # type: ignore
            )

    @NVTXRangeDecorator("Text2ImageDIT:prepare_embedded_sequence")
    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Prepares an embedded sequence tensor by applying positional embeddings and handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                    If None, a default value (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the positional embedding class
                (`self.pos_emb_cls`) includes 'rope'. Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are generated using the
            `self.pos_embedder` with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering fps.
        """
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]

        return x_B_T_H_W_D, None, extra_pos_emb

    @NVTXRangeDecorator("Text2ImageDIT:unpatchify")
    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        x_B_C_Tt_Hp_Wp = rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        return x_B_C_Tt_Hp_Wp

    def compute_caching(
        self,
        timesteps_B_T: torch.Tensor,  # cache
        crossattn_emb: torch.Tensor,  # cache
    ) -> None:
        if self.crossattn_proj is not None:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        for block in self.blocks:
            block.compute_caching(t_embedding_B_T_D, crossattn_emb, adaln_lora_B_T_3D)
        self.final_layer.compute_caching(t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)

        self.t_embedding_B_T_D = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb
        self.adaln_lora_B_T_3D = adaln_lora_B_T_3D

    @NVTXRangeDecorator("Text2ImageDIT:forward")
    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: DataType | None = DataType.VIDEO,
        use_cuda_graphs: bool = False,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B, ) tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
        """
        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )
        assert not (self.training and use_cuda_graphs), "CUDA Graphs are supported only for inference"
        reference_tokens = None
        reference_grid = None
        joint_reference = None
        joint_reference_rope = None
        joint_reference_extra_pos = None
        joint_reference_camera_tokens = None
        if self._runtime_reference_latents is not None:
            if use_cuda_graphs:
                raise ValueError("CUDA graphs are not implemented with visual-reference conditioning")
            if self._runtime_reference_latents.shape[0] != x_B_C_T_H_W.shape[0]:
                raise ValueError(
                    "Reference latent batch does not match target batch: "
                    f"{self._runtime_reference_latents.shape[0]} != {x_B_C_T_H_W.shape[0]}"
                )
            reference_embeddings, full_reference_grid = self.prepare_reference_embeddings(
                self._runtime_reference_latents
            )
            total_references, reference_height, reference_width = full_reference_grid
            if total_references < 1:
                raise ValueError("Option A requires at least one reference image")
            if self.joint_reference_stream_embedding is None:
                raise RuntimeError(
                    "Reference latents were installed but Option-A hybrid fusion is not enabled"
                )

            # _set_reference_conditioning() reorders views so index 0 is the
            # highest-visibility reference.  Keep it as a persistent main
            # self-attention stream; only views 1..R-1 become added K/V memory.
            joint_reference = reference_embeddings[:, :1]
            joint_reference = joint_reference + self.joint_reference_stream_embedding.view(
                1, 1, 1, 1, -1
            ).to(device=joint_reference.device, dtype=joint_reference.dtype)
            if total_references > 1:
                reference_tokens = rearrange(
                    reference_embeddings[:, 1:],
                    "b r h w d -> b (r h w) d",
                )
                reference_grid = (
                    total_references - 1,
                    reference_height,
                    reference_width,
                )

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )
        target_grid = tuple(x_B_T_H_W_D.shape[1:4])
        if joint_reference is not None:
            _, joint_height, joint_width, joint_dim = joint_reference.shape[1:]
            if (joint_height, joint_width, joint_dim) != (
                target_grid[1],
                target_grid[2],
                x_B_T_H_W_D.shape[-1],
            ):
                raise ValueError(
                    "Joint-reference token grid must match the target spatial grid: "
                    f"target={tuple(x_B_T_H_W_D.shape)}, "
                    f"joint={tuple(joint_reference.shape)}"
                )
            if self._is_context_parallel_enabled:
                raise ValueError(
                    "Context parallel is not implemented for Option-A hybrid reference fusion"
                )
            if rope_emb_L_1_1_D is not None:
                # The joint reference is a separate view, not target history.
                # Give it its own one-view spatial RoPE (temporal coordinate 0)
                # rather than pretending it is frame T+1.  Camera/time and the
                # stream embedding disambiguate it from target view 0.
                joint_reference_rope = self.pos_embedder.generate_embeddings(
                    torch.Size(
                        (
                            joint_reference.shape[0],
                            1,
                            joint_reference.shape[2],
                            joint_reference.shape[3],
                            joint_reference.shape[4],
                        )
                    ),
                    fps=None,
                ).to(device=joint_reference.device)
            if self.extra_per_block_abs_pos_emb:
                joint_reference_extra_pos = self.extra_pos_embedder.generate_embeddings(
                    torch.Size(
                        (
                            joint_reference.shape[0],
                            1,
                            joint_reference.shape[2],
                            joint_reference.shape[3],
                            joint_reference.shape[4],
                        )
                    ),
                    fps=None,
                ).to(device=joint_reference.device, dtype=joint_reference.dtype)

        camera_tokens = None
        target_prope = None
        reference_prope = None
        if joint_reference is not None:
            if self.camera_embed_enabled:
                if self._runtime_camera_conditioning is None:
                    raise RuntimeError("Camera embedding requires target/reference calibration")
                camera_tokens = self.prepare_camera_tokens(
                    self._runtime_camera_conditioning["target_camera_rays"],
                    target_grid,
                )
                joint_reference_camera_tokens = self.prepare_camera_tokens(
                    self._runtime_camera_conditioning["reference_camera_rays"][:, :1],
                    (1, joint_reference.shape[2], joint_reference.shape[3]),
                )
                if reference_tokens is not None:
                    target_prope, reference_prope = self.prepare_camera_prope(
                        target_grid,
                        reference_grid,
                        reference_start=1,
                    )
        elif self._runtime_camera_conditioning is not None:
            raise RuntimeError("Camera/time conditioning was supplied without visual references")

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        if use_cuda_graphs:
            shapes_key = create_cuda_graph(
                self.cuda_graphs,
                self.blocks,
                x_B_T_H_W_D,
                rope_emb_L_1_1_D,
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            blocks = self.cuda_graphs[shapes_key]
        else:
            blocks = self.blocks

        control_residuals = ()
        if self.world_scenario_mode == 1:
            x_B_T_H_W_D = x_B_T_H_W_D + self.world_scenario_patch(self._runtime_world_latent)
        elif self.world_scenario_mode == 2:
            control_residuals = self.world_scenario_control(
                x_B_C_T_H_W,
                self._runtime_world_latent,
                self.control_timesteps_B_T,
                self.crossattn_emb,
                interval=self.control_interval_B_T,
                fps=fps,
            )
        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        }
        if joint_reference is not None:
            block_kwargs["reference_tokens"] = reference_tokens
            block_kwargs["camera_tokens"] = camera_tokens
            block_kwargs["target_prope"] = target_prope
            block_kwargs["reference_prope"] = reference_prope
            block_kwargs["joint_reference_rope_L_1_1_D"] = joint_reference_rope
            block_kwargs["joint_reference_extra_pos_emb"] = joint_reference_extra_pos
            block_kwargs["joint_reference_camera_tokens"] = joint_reference_camera_tokens
        for block_index, block in enumerate(blocks):
            if joint_reference is None:
                x_B_T_H_W_D = block(
                    x_B_T_H_W_D,
                    **block_kwargs,
                )
            else:
                x_B_T_H_W_D, joint_reference = block(
                    x_B_T_H_W_D,
                    joint_reference_B_R_H_W_D=joint_reference,
                    **block_kwargs,
                )

            if block_index < len(control_residuals):
                x_B_T_H_W_D = x_B_T_H_W_D + control_residuals[block_index]

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        return x_B_C_Tt_Hp_Wp

    def enable_selective_checkpoint(self, sac_config: SACConfig):
        if sac_config.mode == CheckpointMode.NONE:
            pass
        else:
            log.debug(
                f"Enable selective checkpoint with {sac_config.mode}, for every {sac_config.every_n_blocks} blocks. Total blocks: {len(self.blocks)}"
            )
            _context_fn = sac_config.get_context_fn()
            for block_id, block in self.blocks.named_children():
                if int(block_id) % sac_config.every_n_blocks == 0:
                    log.debug(f"Enable selective checkpoint for block {block_id}")
                    block = ptd_checkpoint_wrapper(
                        block,
                        context_fn=_context_fn,
                        preserve_rng_state=False,
                    )
                    self.blocks.register_module(block_id, block)
            self.register_module(
                "final_layer",
                ptd_checkpoint_wrapper(
                    self.final_layer,
                    context_fn=_context_fn,
                    preserve_rng_state=False,
                ),
            )

        return self

    def fully_shard(self, mesh: DeviceMesh) -> None:
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward)

        fully_shard(self.final_layer, mesh=mesh, reshard_after_forward=True)
        if self.extra_per_block_abs_pos_emb:
            fully_shard(self.extra_pos_embedder, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=True)

    def disable_context_parallel(self) -> None:
        # pos_embedder
        self.pos_embedder.disable_context_parallel()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.disable_context_parallel()

        # attention
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = False

    def enable_context_parallel(self, process_group: ProcessGroup | None = None) -> None:
        # pos_embedder
        self.pos_embedder.enable_context_parallel(process_group=process_group)
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.enable_context_parallel(process_group=process_group)

        # attention
        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=process_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = True

    @property
    def is_context_parallel_enabled(self) -> bool:
        return self._is_context_parallel_enabled
