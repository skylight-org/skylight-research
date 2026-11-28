"""Finite reconstruction rules shared by codecs, transforms and cache input."""

from typing import Optional

import torch


def require_finite(x: torch.Tensor, *, name: str) -> None:
    """Reject invalid floating data without replacing NaN or infinity."""
    if not x.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or infinity")


def finite_cast(
    x: torch.Tensor,
    dtype: torch.dtype,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Reject nonfinite reconstruction, saturate finite overflow, then convert.

    Narrow inputs first widen to FP32 so, for example, BF16's rounded 65536
    can be clamped to FP16's exact 65504 limit. FP64 inputs stay FP64 until
    clamping so a large but finite value cannot overflow in an intermediate cast.
    """
    if not dtype.is_floating_point:
        raise ValueError("Reconstruction output dtype must be floating point")
    require_finite(x, name="Reconstructed values")
    working = x if x.dtype == torch.float64 else x.float()
    limits = torch.finfo(dtype)
    if limits.max < torch.finfo(working.dtype).max:
        working = working.clamp(min=limits.min, max=limits.max)
    return working.to(device=device, dtype=dtype).contiguous()
