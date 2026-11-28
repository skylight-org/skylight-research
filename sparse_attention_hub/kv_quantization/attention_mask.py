"""Align native HF visibility with a longer independently stored KV history."""

from typing import Optional

import torch


def align_attention_mask(
    mask: Optional[torch.Tensor],
    *,
    source_key_length: int,
    target_key_length: int,
    query_length: int,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Map HF's contiguous suffix mask onto our history, without exposing evicted KV.

    Storage retention is independent of attention visibility. Leading tokens that
    HF did not supply remain masked even though our store retains them. Boolean
    masks use True for visible entries; floating masks preserve additive biases.
    """
    if not 0 < query_length <= source_key_length <= target_key_length:
        raise ValueError("Expected query length <= HF KV length <= stored KV length")
    if sliding_window is not None and sliding_window < 1:
        raise ValueError("sliding_window must be positive")
    offset = target_key_length - source_key_length
    if mask is not None:
        if mask.dtype != torch.bool and not mask.is_floating_point():
            raise ValueError("HF mask must be boolean or floating point")
        if mask.ndim != 4 or mask.shape[-2] not in (1, query_length):
            raise ValueError("Expected a 4D HF mask with one row or one row per query")
        if mask.shape[-1] not in (source_key_length, target_key_length):
            raise ValueError(
                "HF mask width does not match its KV suffix or our history"
            )
        suffix = mask[..., -source_key_length:]
        if offset:
            leading = suffix.new_full(
                (*suffix.shape[:-1], offset),
                False if suffix.dtype == torch.bool else float("-inf"),
            )
            mask = torch.cat([leading, suffix], dim=-1)
        else:
            mask = suffix
    elif query_length == 1 and offset == 0 and sliding_window is None:
        return None
    else:
        mask = torch.zeros(
            1, 1, query_length, target_key_length, dtype=dtype, device=device
        )

    keys = torch.arange(target_key_length, device=device)
    queries = torch.arange(
        target_key_length - query_length, target_key_length, device=device
    )
    visible = (keys[None, :] <= queries[:, None]) & (keys[None, :] >= offset)
    if sliding_window is not None:
        visible &= keys[None, :] > queries[:, None] - sliding_window
    visible = visible[None, None]
    if mask.dtype == torch.bool:
        return mask & visible
    return mask.masked_fill(~visible, float("-inf"))
