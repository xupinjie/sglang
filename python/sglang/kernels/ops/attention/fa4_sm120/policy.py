# Copyright (c) 2026, SGLang Team.
"""Pure workload qualification shared by the SM120 FA4 host and kernel."""

from functools import lru_cache
from typing import Optional

LOW_HD_DECODE_SHAPES = frozenset({(64, 64), (128, 128)})
LOW_HD_DECODE_TILE_N = 64
LOW_HD_DECODE_MIN_VISIBLE_K = 256
LOW_HD_DECODE_SHORT_VISIBLE_K = 512
LOW_HD_DECODE_MAX_SPLITS = 8
LOW_HD_DECODE_SHORT_MIN_TILES_PER_CTA = 2
LOW_HD_DECODE_LONG_MIN_TILES_PER_CTA = 8


def visible_decode_seqlen_k(
    max_seqlen_k: int,
    *,
    is_local: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
) -> int:
    """Return the exact KV span visible to a single query position."""
    if not is_local:
        return max(0, max_seqlen_k)
    left = max_seqlen_k if window_size_left is None else max(0, window_size_left)
    right = max_seqlen_k if window_size_right is None else max(0, window_size_right)
    return max(0, min(max_seqlen_k, left + 1 + right))


def low_hd_paged_decode_tile_m(
    *,
    head_dim: int,
    head_dim_v: int,
    paged_kv: bool,
    seqlen_q: Optional[int],
    visible_seqlen_k: Optional[int],
    qhead_per_kvhead: Optional[int],
    num_sms: Optional[int] = None,
    total_mblocks: Optional[int] = None,
) -> Optional[int]:
    """Return the qualified low-HD decode M tile, or ``None`` for fallback."""
    if (
        not paged_kv
        or seqlen_q != 1
        or visible_seqlen_k is None
        or visible_seqlen_k <= LOW_HD_DECODE_MIN_VISIBLE_K
        or (head_dim, head_dim_v) not in LOW_HD_DECODE_SHAPES
    ):
        return None
    qhead_ratio = 1 if qhead_per_kvhead is None else qhead_per_kvhead
    if (
        (head_dim, head_dim_v) == (64, 64)
        and qhead_ratio >= 8
        and visible_seqlen_k <= LOW_HD_DECODE_SHORT_VISIBLE_K
    ):
        if num_sms is not None and total_mblocks is not None:
            num_n_blocks = (
                visible_seqlen_k + LOW_HD_DECODE_TILE_N - 1
            ) // LOW_HD_DECODE_TILE_N
            max_short_splits = min(
                LOW_HD_DECODE_MAX_SPLITS,
                num_n_blocks // LOW_HD_DECODE_SHORT_MIN_TILES_PER_CTA,
            )
            # Use the one-warp M16 CTA only when bounded SplitKV can fill an
            # SM wave without reducing each partition below two KV tiles.
            if total_mblocks * max_short_splits >= num_sms:
                return 16
        return None
    if qhead_ratio >= 8 and visible_seqlen_k <= LOW_HD_DECODE_SHORT_VISIBLE_K:
        return 32
    return 16


def is_low_hd_paged_decode_tile(
    *,
    head_dim: int,
    head_dim_v: int,
    paged_kv: bool,
    seqlen_q: int,
    tile_m: int,
    tile_n: int,
) -> bool:
    """Return whether a selected tile belongs to the qualified low-HD path."""
    return (
        paged_kv
        and seqlen_q == 1
        and (head_dim, head_dim_v) in LOW_HD_DECODE_SHAPES
        and tile_m in (16, 32)
        and tile_n == LOW_HD_DECODE_TILE_N
    )


@lru_cache(maxsize=256)
def use_rectangular_gqa_tile(
    *,
    head_dim: int,
    head_dim_v: int,
    batch_size: int,
    num_head_kv: int,
    qhead_per_kvhead: int,
    seqlen_q: Optional[int],
    seqlen_k: Optional[int],
    num_sms: Optional[int],
    pack_gqa: bool,
    paged_kv: bool,
    is_causal: bool,
    is_local: bool,
) -> bool:
    """Select M128 only when its measured per-CTA cost wins after splitting.

    The HD128 sweep qualifies packed rows and CTA waves, not a particular
    batch/head topology. M128 processes twice the rows with eight compute
    warps; its per-KV-tile cost is about 1.7 times M64's four-warp cost.
    Require a 5% predicted margin and keep the measured one-wave regime.
    """
    if (
        (head_dim, head_dim_v) != (128, 128)
        or batch_size is None
        or num_head_kv is None
        or qhead_per_kvhead is None
        or batch_size <= 0
        or num_head_kv <= 0
        or qhead_per_kvhead <= 0
        or seqlen_q is None
        or seqlen_k is None
        or seqlen_k < 4096
        or num_sms is None
        or num_sms < 128
        or not pack_gqa
        or paged_kv
        or is_causal
        or is_local
    ):
        return False
    packed_rows = seqlen_q * qhead_per_kvhead
    if not 512 <= packed_rows <= 2048:
        return False
    blocks_128 = batch_size * num_head_kv * ((packed_rows + 127) // 128)
    if blocks_128 > num_sms:
        return False
    blocks_64 = batch_size * num_head_kv * ((packed_rows + 63) // 64)
    kv_tiles = (seqlen_k + 63) // 64

    def cost(blocks, tile_cost):
        splits = rectangular_gqa_num_splits(blocks, kv_tiles, num_sms)
        waves = (blocks * splits + num_sms - 1) // num_sms
        tiles = (kv_tiles + splits - 1) // splits
        # Costs are in tenths of an M64 KV iteration. Preserve the merge
        # penalty used by split selection and compare complete operations.
        return tile_cost * waves * tiles + 30 * (splits - 1)

    return 105 * cost(blocks_128, 17) < 100 * cost(blocks_64, 10)


def rectangular_gqa_num_splits(
    total_mblocks: int, num_n_blocks: int, num_sms: int
) -> int:
    """Balance full CTA waves against FP32 partial-output/merge traffic.

    Keep at least 16 N64 tiles per split. The merge penalty is calibrated in
    KV-tile units for the qualified HD128, eight-warp kernel; it prevents
    adding near-empty scheduling waves just to increase parallelism.
    """
    max_splits = max(1, min(16, num_n_blocks // 16))

    def cost(splits):
        waves = (total_mblocks * splits + num_sms - 1) // num_sms
        tiles = (num_n_blocks + splits - 1) // splits
        return waves * tiles + 3 * splits

    return min(range(1, max_splits + 1), key=cost)
