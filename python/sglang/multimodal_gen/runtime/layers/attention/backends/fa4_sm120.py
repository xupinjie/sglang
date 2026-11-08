# SPDX-License-Identifier: Apache-2.0
"""Dense attention using SGLang's SM120 kernel and automatic tile/SplitKV policy."""

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    flash_attn_varlen_func_op,
    flash_attn_varlen_func_op_lse,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class FlashAttentionSM120Backend(AttentionBackend):
    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.FA4_SM120

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionSM120Impl"]:
        return FlashAttentionSM120Impl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        raise NotImplementedError

    @classmethod
    def supports_ring_rotation(cls) -> bool:
        return True


class FlashAttentionSM120Impl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        *,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Reuse the FA4 custom-op boundary for tracing. Its architecture
        # dispatch reaches the owned SM120 kernel with the full BSHD batch.
        op = (
            flash_attn_varlen_func_op_lse
            if return_softmax_lse
            else flash_attn_varlen_func_op
        )
        return op(
            q=query,
            k=key,
            v=value,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            num_splits=0,
            pack_gqa=True,
            return_softmax_lse=return_softmax_lse,
            ver=4,
        )
