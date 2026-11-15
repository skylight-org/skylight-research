"""Kitty's group-wise fake-quantization arithmetic, with per-channel bit widths.

Adapted from Summer-Summer/Kitty, revision dfd2c07b407d6b407179359207c612ab631f3ed1,
src/kitty_sim/utils_quant.py (fake_quant_groupwise_lastdim). Copyright (c) 2025
Haojun Xia, MIT; see ../../LICENSE-KITTY. This is the arithmetic of the authors'
accuracy simulation; their CUDA kernels (two 2-bit planes) are a storage backend.
"""

from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Tuple

import torch

from ...cache.block import QuantizedTensor
from ...context import AttentionContext
from ...numerics import finite_cast, require_finite
from ...policies.precision.base import PrecisionPlan
from ...registry import KVQuantizerRegistry
from ..base import KVQuantizer, KVQuantizerConfig, ParamDType, QuantGranularity
from ..common import (
    check_input,
    group_for_quantization,
    layout_from_metadata,
    layout_metadata,
    pack_bitstream,
    ungroup_values,
    unpack_bitstream,
)

KITTY_REFERENCE_REVISION = "dfd2c07b407d6b407179359207c612ab631f3ed1"
NATIVE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
FORMAT_ID = "kitty_int_v1"


def kitty_range_eps(dtype: torch.dtype) -> float:
    """The reference's smallest group range: 1e-4 for 16-bit floats, else 1e-6."""
    return 1e-4 if dtype in (torch.float16, torch.bfloat16) else 1e-6


@dataclass(frozen=True)
class KittyQuantizerConfig(KVQuantizerConfig):
    """Asymmetric grouped INT in Kitty's operation order.

    Minimum, maximum, scale, division, clamping and rounding all use the input
    dtype, exactly like the authors' simulation:
    ``scale = max(range, eps) / (2**b - 1)``,
    ``q = round(clamp((x - min) / scale, 0, 2**b - 1))``, ``x~ = q * scale + min``.

    Attributes:
        bits: Bits of every group unless a precision plan gives per-channel bits
            (Kitty: 2, with boosted key channels at 4).
        group_size: Values per group (Kitty: 128 tokens per key channel, i.e. one
            page, and 128 channels per value token, i.e. the whole head).
        range_eps: Smallest group range. ``None`` uses the reference rule
            (1e-4 for FP16/BF16 inputs, 1e-6 otherwise).

    Groups must be complete. ``symmetric``, ``scale_dtype`` and ``offset_dtype``
    keep their defaults: parameters are stored in the input dtype.
    """

    bits: int = 2
    group_size: int = 128
    range_eps: Optional[float] = None
    parameter_dtype: str = field(default="input", init=False)
    reference_revision: str = field(default=KITTY_REFERENCE_REVISION, init=False)

    # ChannelBoost may give this quantizer one bit width per channel.
    supports_channel_bits: ClassVar[bool] = True


def _check_bits(bits: object) -> int:
    if type(bits) is not int or not 1 <= bits <= 8:
        raise ValueError("Kitty quantization supports 1 to 8 bits")
    return bits


@KVQuantizerRegistry.register(KittyQuantizerConfig)
class KittyQuantizer(KVQuantizer):
    """Kitty's numerics; codes are packed per bit width with Skylight's bitstream."""

    config: KittyQuantizerConfig

    def __init__(self, config: KittyQuantizerConfig) -> None:
        super().__init__(config)
        _check_bits(config.bits)
        if type(config.group_size) is not int or config.group_size < 1:
            raise ValueError("Kitty group_size must be a positive integer")
        if config.symmetric:
            raise ValueError("Kitty quantization is asymmetric")
        if (
            config.scale_dtype != ParamDType.FP16
            or config.offset_dtype != ParamDType.FP16
        ):
            raise ValueError(
                "Kitty parameters follow the input dtype; leave scale_dtype and "
                "offset_dtype at defaults"
            )
        if config.range_eps is not None and (
            isinstance(config.range_eps, bool)
            or not isinstance(config.range_eps, (int, float))
            or not config.range_eps > 0
        ):
            raise ValueError("Kitty range_eps must be a positive number or None")
        self.granularity = QuantGranularity(config.granularity)
        if self.granularity == QuantGranularity.PER_BLOCK:
            raise ValueError("Kitty groups keys PER_CHANNEL and values PER_TOKEN")

    def _widths(
        self, x: torch.Tensor, plan: Optional[PrecisionPlan]
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """Default bits and, when the plan has them, per-channel bits."""
        bits = _check_bits(self.config.bits if plan is None else plan.bits)
        if plan is None or plan.channel_bits is None:
            return bits, None
        if self.granularity != QuantGranularity.PER_CHANNEL:
            raise ValueError(
                "Per-channel bits need PER_CHANNEL granularity, so each group has "
                "one bit width"
            )
        channel_bits = plan.channel_bits
        if tuple(channel_bits.shape) != (x.shape[0], x.shape[1], x.shape[3]):
            raise ValueError("channel_bits must have shape [batch, heads, head_dim]")
        for width in channel_bits.unique().tolist():
            _check_bits(int(width))
        return bits, channel_bits.to(device=x.device, dtype=torch.int64)

    def quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> QuantizedTensor:
        check_input(x, plan, allow_channel_bits=True)
        if x.dtype not in NATIVE_DTYPES:
            raise ValueError("Kitty inputs must be FP16, BF16 or FP32")
        bits, channel_bits = self._widths(x, plan)
        axis = 2 if self.granularity == QuantGranularity.PER_CHANNEL else 3
        if x.shape[axis] % self.config.group_size:
            raise ValueError(
                "Kitty requires complete groups; group_size must divide the grouped "
                "dimension (tokens for keys, head_dim for values)"
            )
        groups, layout, _ = group_for_quantization(
            x, self.granularity, self.config.group_size, None
        )
        # Grouping is lossless for these dtypes; return to the input dtype BEFORE
        # every reference operation, as the authors' code never leaves it.
        groups = groups.to(x.dtype)
        mn = groups.min(dim=-1, keepdim=True).values
        mx = groups.max(dim=-1, keepdim=True).values
        eps = (
            kitty_range_eps(x.dtype)
            if self.config.range_eps is None
            else float(self.config.range_eps)
        )
        span = (mx - mn).clamp(min=eps)

        # Groups: [batch, heads, rows, n_groups, 1]; rows are channels for keys.
        if channel_bits is None:
            widths: List[int] = [bits]
            group_bits = None
            scale = span / (2**bits - 1)
            max_val = torch.full_like(scale, 2**bits - 1)
        else:
            widths = sorted(int(w) for w in channel_bits.unique().tolist())
            group_bits = channel_bits[..., None, None].expand(mn.shape)
            scale = torch.empty_like(span)
            max_val = torch.empty_like(span)
            for width in widths:
                chosen = group_bits == width
                scale = torch.where(chosen, span / (2**width - 1), scale)
                max_val = torch.where(
                    chosen, torch.full_like(span, 2**width - 1), max_val
                )
        require_finite(scale, name="Kitty scale")
        q = ((groups - mn) / scale).clamp(torch.zeros_like(max_val), max_val).round()
        require_finite(q, name="Kitty codes")

        rows = q.reshape(-1, q.shape[-1])
        if group_bits is None:
            packed, count = pack_bitstream(rows, bits)
            parts, counts, byte_counts = [packed], [count], [packed.numel()]
            flat_bits = None
        else:
            flat_bits = group_bits.reshape(-1)
            parts, counts, byte_counts = [], [], []
            for width in widths:
                packed, count = pack_bitstream(rows[flat_bits == width], width)
                parts.append(packed)
                counts.append(count)
                byte_counts.append(packed.numel())
        return QuantizedTensor(
            codes=torch.cat(parts),
            scales=scale.contiguous(),
            zero_points=mn.contiguous(),
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=layout.axis,
            bits=bits,
            format_id=FORMAT_ID,
            group_size=self.config.group_size,
            metadata={
                **layout_metadata(layout, rows.numel()),
                "widths": widths,
                "width_code_counts": counts,
                "width_byte_counts": byte_counts,
                "range_eps": eps,
                "reference_revision": KITTY_REFERENCE_REVISION,
            },
            group_bits=None
            if flat_bits is None
            else flat_bits.to(torch.uint8).contiguous(),
        )

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        if q.format_id != FORMAT_ID:
            raise ValueError(f"Unsupported Kitty encoding: {q.format_id}")
        if q.zero_points is None:
            raise ValueError("Kitty encoding requires group minima")
        if q.scales.dtype != q.dtype or q.zero_points.dtype != q.dtype:
            raise ValueError("Kitty parameters must use the input dtype")
        require_finite(q.scales, name="Kitty stored scales")
        require_finite(q.zero_points, name="Kitty stored minima")
        layout = layout_from_metadata(q.metadata, q.axis)
        widths = q.metadata["widths"]
        counts = q.metadata["width_code_counts"]
        byte_counts = q.metadata["width_byte_counts"]
        size = q.group_size
        if q.group_bits is None:
            if len(widths) != 1:
                raise ValueError("Mixed-width Kitty records need group_bits")
            flat = unpack_bitstream(q.codes, widths[0], counts[0])
        else:
            flat_bits = q.group_bits.long()
            rows = torch.zeros(
                flat_bits.numel(), size, dtype=torch.int16, device=q.codes.device
            )
            start = 0
            for width, count, nbytes in zip(widths, counts, byte_counts):
                part = unpack_bitstream(q.codes[start : start + nbytes], width, count)
                rows[flat_bits == width] = part.reshape(-1, size)
                start += nbytes
            flat = rows
        # The reference keeps codes in the input dtype and reconstructs there.
        codes = flat.to(q.dtype).reshape(q.scales.shape[:-1] + (size,))
        restored = codes * q.scales + q.zero_points
        return finite_cast(
            ungroup_values(restored, layout, q.shape), q.dtype, device=q.device
        )
