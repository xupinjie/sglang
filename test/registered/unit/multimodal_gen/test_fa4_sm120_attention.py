# SPDX-License-Identifier: Apache-2.0
"""Check the diffusion adapter's batched GQA, scale and LSE contracts on SM120."""

import unittest

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.runtime.layers.attention.backends.fa4_sm120 import (
    FlashAttentionSM120Impl,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "SM120 attention requires CUDA SM120",
)
class TestSM120Attention(CustomTestCase):
    def test_strided_rectangular_gqa_and_batch_isolation(self):
        for batch, nq, nk, dtype, causal in (
            (1, 224, 663, torch.bfloat16, False),
            (1, 448, 14999, torch.bfloat16, False),
            (1, 224, 14775, torch.bfloat16, False),
            (2, 33, 101, torch.bfloat16, False),
            (2, 33, 101, torch.float16, False),
            (1, 17, 53, torch.bfloat16, True),
        ):
            with self.subTest(batch=batch, nq=nq, nk=nk, dtype=dtype, causal=causal):
                torch.manual_seed(42)
                q = torch.randn(batch, nq, 48, 128, device="cuda", dtype=dtype)[
                    :, :, :32
                ]
                kv = torch.randn(batch, nk, 16, 128, device="cuda", dtype=dtype)
                k, v = kv[:, :, :8], kv[:, :, 8:]
                scale = 0.07
                impl = FlashAttentionSM120Impl(32, 128, causal, scale, num_kv_heads=8)
                actual = impl.forward(q, k, v)
                mask = None
                if causal:
                    mask = torch.arange(nk, device="cuda")[None, :] <= (
                        torch.arange(nq, device="cuda")[:, None] + nk - nq
                    )
                with sdpa_kernel(SDPBackend.MATH):
                    expected = torch.nn.functional.scaled_dot_product_attention(
                        q.float().transpose(1, 2),
                        k.float().transpose(1, 2),
                        v.float().transpose(1, 2),
                        attn_mask=mask,
                        scale=scale,
                        enable_gqa=True,
                    ).transpose(1, 2)
                self.assertEqual(actual.shape, q.shape)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(
                    actual.float(), expected, atol=2e-3, rtol=2e-2
                )

                if batch == 2 and dtype == torch.bfloat16:
                    out, lse = impl.forward(q, k, v, return_softmax_lse=True)
                    torch.testing.assert_close(out, actual)
                    logits = (
                        torch.einsum(
                            "bqhd,bkhd->bhqk",
                            q.float(),
                            k.float().repeat_interleave(4, dim=2),
                        )
                        * scale
                    )
                    torch.testing.assert_close(
                        lse, logits.logsumexp(-1), atol=2e-3, rtol=2e-3
                    )


if __name__ == "__main__":
    unittest.main()
