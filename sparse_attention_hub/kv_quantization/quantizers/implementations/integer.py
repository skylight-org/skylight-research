"""IntQuantizer component and its configuration.

One uniform integer quantizer for every grid-based method. The default settings
are Skylight's generic arithmetic (FP32 statistics, conservatively rounded FP16
parameters). The other settings reproduce published reference code operation
by operation. Each recipe sets them with its own helper, exactly as the
authors' code does: ``kivi_int_quantizer()`` (implementations/kivi.py),
``kitty_int_quantizer()`` (implementations/kitty.py) and
``quarot_int_quantizer()`` (implementations/quarot_kv.py). The references:

- KIVI: jy-yuan/KIVI @ 876b4d2, quant/new_pack.py (quant_and_pack_kcache/vcache,
  unpack_and_dequant_kcache/vcache).
- Kitty: Summer-Summer/Kitty @ dfd2c07, src/kitty_sim (fake_quant_groupwise_lastdim).
- QuaRot: spcl/QuaRot @ 5008669, fake_quant/quant_utils.py (ActQuantizer).
"""

import math
from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional, Tuple

import torch

from ...cache.block import QuantizedTensor
from ...context import AttentionContext
from ...numerics import finite_cast, require_finite
from ...policies.precision.base import PrecisionPlan
from ...registry import KVQuantizerRegistry
from ..base import KVQuantizer, KVQuantizerConfig, QuantGranularity
from ..common import (
    check_input,
    group_for_quantization,
    group_min_max,
    layout_from_metadata,
    layout_metadata,
    pack_bitstream,
    pack_codes,
    store_param,
    ungroup_values,
    unpack_bitstream,
    unpack_codes,
)

GENERIC_FORMAT = "uniform_int_v1"
NATIVE_FORMAT = "uniform_int_v2"
NATIVE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

COMPUTE_DTYPES = ("fp32", "input")
PARAMETER_STORAGE = ("rounded", "compute")
ZERO_POINTS = ("float", "integer")
DEGENERATE_RANGES = ("unit_scale", "zero_scale", "eps", "pm_one")
ROUNDINGS = ("round_clamp", "clamp_round")
DECODE_DTYPES = ("fp32", "fp16", "compute")
SYMMETRIC_RANGES = ("balanced", "full")


def default_range_eps(dtype: torch.dtype) -> float:
    """Kitty's smallest group range: 1e-4 for 16-bit floats, else 1e-6."""
    return 1e-4 if dtype in (torch.float16, torch.bfloat16) else 1e-6


@dataclass(frozen=True)
class IntQuantizerConfig(KVQuantizerConfig):
    """Uniform integer grid; the numerical settings choose the operation order.

    Attributes (defaults = Skylight's generic arithmetic):
        compute_dtype: ``"fp32"``: statistics and codes in FP32. ``"input"``: in
            the K/V dtype, as KIVI, Kitty and QuaRot's value path do.
        parameter_storage: ``"rounded"``: scales/offsets are rounded to
            ``scale_dtype``/``offset_dtype`` (conservatively) before codes are
            computed. ``"compute"``: kept exactly as computed, in the compute dtype.
        zero_point: ``"float"``: ``code = (x - min) / scale``, decoded as
            ``code * scale + min`` (KIVI, Kitty). ``"integer"``:
            ``zero = round(-min / scale)``, ``code = round(x / scale) + zero``,
            decoded as ``scale * (code - zero)`` (QuaRot).
        clip_ratio: Multiplies the group minimum and maximum (QuaRot: 0.95).
        range_includes_zero: Extend each group's range to contain 0 (QuaRot keys).
        degenerate_range: Groups whose range is 0. ``"unit_scale"``: scale 1.
            ``"zero_scale"``: scale 0, code 0 (KIVI). ``"eps"``: the range is at
            least ``range_eps`` (Kitty). ``"pm_one"``: an all-zero group uses
            [-1, 1], symmetric: scale 1 (QuaRot).
        range_eps: Smallest range for ``"eps"``; ``None`` = 1e-4 for FP16/BF16
            inputs, 1e-6 otherwise (Kitty's rule).
        rounding: ``"round_clamp"`` or ``"clamp_round"`` (KIVI, Kitty).
        decode_dtype: Dtype of codes when decoding: ``"fp32"``, ``"fp16"`` (KIVI's
            unpack, then PyTorch type promotion) or ``"compute"``.
        symmetric_range: Symmetric codes in ``[-q, q]`` (``"balanced"``) or
            ``[-(q+1), q]`` (``"full"``, QuaRot), with ``q = 2**(bits-1) - 1``.
        complete_groups: Reject tensors whose grouped length is not a multiple of
            ``group_size`` instead of padding (all three reference codes).

    Non-default settings require ``parameter_storage="compute"``; the generic
    path (``"rounded"``) is unchanged and keeps its own record format.
    """

    compute_dtype: str = "fp32"
    parameter_storage: str = "rounded"
    zero_point: str = "float"
    clip_ratio: float = 1.0
    range_includes_zero: bool = False
    degenerate_range: str = "unit_scale"
    range_eps: Optional[float] = None
    rounding: str = "round_clamp"
    decode_dtype: str = "fp32"
    symmetric_range: str = "balanced"
    complete_groups: bool = False

    # ChannelBoost may give this quantizer one bit width per channel.
    supports_channel_bits: ClassVar[bool] = True

    @property
    def generic(self) -> bool:
        """True for Skylight's generic arithmetic (the ``"rounded"`` path)."""
        return self.parameter_storage == "rounded"


def _check_choice(name: str, value: object, choices: Tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def validate_int_config(config: IntQuantizerConfig) -> None:
    """Reject settings that are unknown or that do not combine."""
    _check_choice("compute_dtype", config.compute_dtype, COMPUTE_DTYPES)
    _check_choice("parameter_storage", config.parameter_storage, PARAMETER_STORAGE)
    _check_choice("zero_point", config.zero_point, ZERO_POINTS)
    _check_choice("degenerate_range", config.degenerate_range, DEGENERATE_RANGES)
    _check_choice("rounding", config.rounding, ROUNDINGS)
    _check_choice("decode_dtype", config.decode_dtype, DECODE_DTYPES)
    _check_choice("symmetric_range", config.symmetric_range, SYMMETRIC_RANGES)
    for name in ("symmetric", "range_includes_zero", "complete_groups"):
        if type(getattr(config, name)) is not bool:
            raise ValueError(f"{name} must be a boolean")
    clip = config.clip_ratio
    if (
        isinstance(clip, bool)
        or not isinstance(clip, (int, float))
        or not math.isfinite(clip)
        or not 0 < clip <= 1
    ):
        raise ValueError("clip_ratio must be finite and in (0, 1]")
    if config.range_eps is not None:
        eps = config.range_eps
        if (
            isinstance(eps, bool)
            or not isinstance(eps, (int, float))
            or not math.isfinite(eps)
            or not eps > 0
        ):
            raise ValueError("range_eps must be a positive number or None")
        if config.degenerate_range != "eps":
            raise ValueError("range_eps is only used with degenerate_range='eps'")
    if type(config.group_size) is not int or config.group_size < 1:
        raise ValueError("group_size must be a positive integer")
    if config.generic:
        defaults = IntQuantizerConfig()
        changed = [
            name
            for name in (
                "compute_dtype",
                "zero_point",
                "clip_ratio",
                "range_includes_zero",
                "degenerate_range",
                "rounding",
                "decode_dtype",
                "symmetric_range",
                "complete_groups",
            )
            if getattr(config, name) != getattr(defaults, name)
        ]
        if changed:
            raise ValueError(
                f"{', '.join(changed)} need parameter_storage='compute' (the "
                "generic 'rounded' path has fixed arithmetic)"
            )
        if config.bits not in (2, 4, 8):
            raise ValueError("IntQuantizer supports 2, 4 or 8 bits")
        return
    if type(config.bits) is not int or not 1 <= config.bits <= 8:
        raise ValueError("IntQuantizer supports 1 to 8 bits")
    if config.symmetric:
        if config.bits < 2:
            raise ValueError("Symmetric integer codes need at least 2 bits")
        if config.zero_point != "float":
            raise ValueError("Symmetric codes have no zero point; keep zero_point='float'")
        if config.degenerate_range not in ("unit_scale", "pm_one"):
            raise ValueError(
                "Symmetric codes handle an all-zero group with scale 1 "
                "(degenerate_range 'unit_scale' or 'pm_one')"
            )
    elif config.symmetric_range != "balanced":
        raise ValueError("symmetric_range only applies to symmetric codes")
    if config.zero_point == "integer" and config.degenerate_range == "zero_scale":
        raise ValueError("An integer zero point needs a nonzero scale")


@KVQuantizerRegistry.register(IntQuantizerConfig)
class IntQuantizer(KVQuantizer):
    """FP32 group statistics and uint8-packed codes, on CPU or CUDA.

    PER_TOKEN groups channels, PER_CHANNEL groups tokens, and PER_BLOCK groups
    consecutive channels of one token (``group_size`` must divide head_dim, so a
    block never spans two tokens). Padding repeats the last entry so it cannot
    change a group's min/max.

    A plan's ``channel_bits`` (mixed precision, e.g. ChannelBoost) needs PER_CHANNEL
    groups, so every group has one bit width. Codes of each width are packed
    separately; one byte per group records its width.

    With ``parameter_storage="compute"`` the reference-order path runs instead
    (``_quantize_native``); see ``IntQuantizerConfig``.
    """

    config: IntQuantizerConfig

    def __init__(self, config: IntQuantizerConfig) -> None:
        super().__init__(config)
        validate_int_config(config)
        self.granularity = QuantGranularity(config.granularity)

    def quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> QuantizedTensor:
        if not self.config.generic:
            return self._quantize_native(x, plan)
        check_input(x, plan, allow_channel_bits=True, allow_keep_tokens=True)
        bits = self.config.bits if plan is None else plan.bits
        if bits not in (2, 4, 8):
            raise ValueError("INT precision plan must use 2, 4 or 8 bits")
        if plan is not None and plan.channel_bits is not None:
            return self._quantize_mixed(x, plan.channel_bits, bits, plan)
        groups, layout, valid = group_for_quantization(
            x, self.granularity, self.config.group_size, plan
        )
        lo, hi = group_min_max(groups, valid)

        # Statistics are FP32. Offsets and scales are rounded to their storage
        # dtypes *before* computing codes, so decoding uses exactly what is stored.
        offset: Optional[torch.Tensor]
        if self.config.symmetric:
            # Signed symmetric [-qmax, qmax], stored as unsigned codes; no offset.
            qmax = 2 ** (bits - 1) - 1
            scale = store_param(
                torch.maximum(lo.abs(), hi.abs()) / qmax, self.config.scale_dtype
            )
            offset = None
            codes = (groups / scale).round().clamp(-qmax, qmax) + qmax
        else:
            offset = store_param(lo, self.config.offset_dtype, kind="offset")
            scale = store_param(
                (hi - offset) / (2**bits - 1), self.config.scale_dtype
            )
            # A constant group decodes to its offset (exact when the offset dtype
            # represents the value).
            codes = ((groups - offset) / scale).round().clamp(0, 2**bits - 1)

        packed, code_count = pack_codes(codes, bits)
        return QuantizedTensor(
            codes=packed,
            scales=scale.to(self.config.scale_dtype.torch_dtype).contiguous(),
            # Floating offset (the group minimum), not an integer zero code.
            zero_points=(
                None
                if offset is None
                else offset.to(self.config.offset_dtype.torch_dtype).contiguous()
            ),
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=layout.axis,
            bits=bits,
            format_id="uniform_int_v1",
            group_size=self.config.group_size,
            metadata={
                **layout_metadata(layout, code_count),
                "symmetric": self.config.symmetric,
            },
        )

    def _quantize_mixed(
        self,
        x: torch.Tensor,
        channel_bits: torch.Tensor,
        default_bits: int,
        plan: PrecisionPlan,
    ) -> QuantizedTensor:
        """Per-channel bit widths: every PER_CHANNEL group has its channel's width."""
        if self.granularity != QuantGranularity.PER_CHANNEL:
            raise ValueError(
                "Per-channel bits need PER_CHANNEL granularity, so each group has "
                "one bit width"
            )
        if tuple(channel_bits.shape) != (x.shape[0], x.shape[1], x.shape[3]):
            raise ValueError("channel_bits must have shape [batch, heads, head_dim]")
        widths = sorted(int(w) for w in channel_bits.unique())
        if any(w not in (2, 4, 8) for w in widths):
            raise ValueError("Per-channel bits must be 2, 4 or 8")
        groups, layout, valid = group_for_quantization(
            x, self.granularity, self.config.group_size, plan
        )
        # groups: [batch, heads, channels, n_groups, size]; one width per channel.
        group_bits = channel_bits.to(groups.device)[..., None].expand(groups.shape[:-1])
        bits = group_bits[..., None].float()
        lo, hi = group_min_max(groups, valid)
        offset: Optional[torch.Tensor]
        if self.config.symmetric:
            qmax = 2 ** (bits - 1) - 1
            scale = store_param(
                torch.maximum(lo.abs(), hi.abs()) / qmax, self.config.scale_dtype
            )
            offset = None
            codes = torch.minimum(torch.maximum((groups / scale).round(), -qmax), qmax)
            codes = codes + qmax
        else:
            levels = 2**bits - 1
            offset = store_param(lo, self.config.offset_dtype, kind="offset")
            scale = store_param((hi - offset) / levels, self.config.scale_dtype)
            codes = torch.minimum(
                ((groups - offset) / scale).round().clamp(min=0), levels
            )

        flat_bits = group_bits.flatten()
        flat_codes = codes.reshape(-1, groups.shape[-1])
        parts, counts, byte_counts = [], [], []
        for width in widths:
            packed, count = pack_codes(flat_codes[flat_bits == width], width)
            parts.append(packed)
            counts.append(count)
            byte_counts.append(packed.numel())
        return QuantizedTensor(
            codes=torch.cat(parts),
            scales=scale.to(self.config.scale_dtype.torch_dtype).contiguous(),
            zero_points=(
                None
                if offset is None
                else offset.to(self.config.offset_dtype.torch_dtype).contiguous()
            ),
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=layout.axis,
            bits=default_bits,
            format_id="uniform_int_v1",
            group_size=self.config.group_size,
            metadata={
                **layout_metadata(layout, flat_codes.numel()),
                "symmetric": self.config.symmetric,
                "widths": widths,
                "width_code_counts": counts,
                "width_byte_counts": byte_counts,
            },
            group_bits=flat_bits.to(torch.uint8).contiguous(),
        )

    def _unpack_mixed(self, q: QuantizedTensor) -> torch.Tensor:
        assert q.group_bits is not None
        flat_bits = q.group_bits.long()
        codes = torch.zeros(
            flat_bits.numel(), q.group_size, dtype=torch.float32, device=q.codes.device
        )
        start = 0
        for width, count, nbytes in zip(
            q.metadata["widths"],
            q.metadata["width_code_counts"],
            q.metadata["width_byte_counts"],
        ):
            part = unpack_codes(q.codes[start : start + nbytes], width, count)
            codes[flat_bits == width] = part.float().reshape(-1, q.group_size)
            start += nbytes
        return codes

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        if q.format_id == NATIVE_FORMAT:
            return self._dequantize_native(q)
        if q.format_id != GENERIC_FORMAT:
            raise ValueError(f"Unsupported integer encoding: {q.format_id}")
        layout = layout_from_metadata(q.metadata, q.axis)
        if q.group_bits is not None:
            flat = self._unpack_mixed(q)
            group_bits = q.group_bits.float().reshape(q.scales.shape)
        else:
            flat = unpack_codes(q.codes, q.bits, q.metadata["code_count"]).float()
            group_bits = None
        groups = flat.reshape(
            *layout.layout_shape[:-1],
            layout.padded_length // q.group_size,
            q.group_size,
        )
        scales = q.scales.float()
        if q.metadata["symmetric"]:
            bits = q.bits if group_bits is None else group_bits
            restored = (groups - (2 ** (bits - 1) - 1)) * scales
        else:
            assert q.zero_points is not None
            restored = groups * scales + q.zero_points.float()
        restored = ungroup_values(restored, layout, q.shape)
        return finite_cast(restored, q.dtype, device=q.device)

    # ------------------------------------------------------------------
    # Reference-order path (parameter_storage="compute")
    #
    # Each step below is the reference code's own operation, in its order and
    # dtype; the comments name the reference line it reproduces. Equal formulas
    # in another order can round differently in BF16/FP16, so the order matters.
    # ------------------------------------------------------------------

    def _native_widths(
        self, x: torch.Tensor, plan: Optional[PrecisionPlan]
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """Default bits and, when the plan has them, per-channel bits."""
        smallest = 2 if self.config.symmetric else 1
        bits = self.config.bits if plan is None else plan.bits
        if type(bits) is not int or not smallest <= bits <= 8:
            raise ValueError(f"INT precision plan must use {smallest} to 8 bits")
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
            if not smallest <= int(width) <= 8:
                raise ValueError(f"Per-channel bits must be {smallest} to 8")
        return bits, channel_bits.to(device=x.device, dtype=torch.int64)

    @staticmethod
    def _per_width(
        widths: List[int],
        group_bits: Optional[torch.Tensor],
        like: torch.Tensor,
        value: Any,
    ) -> torch.Tensor:
        """``value(width)`` for every group; with mixed widths, chosen per group
        (Kitty's ``torch.where`` loop)."""
        if group_bits is None:
            return value(widths[0])
        result = torch.empty_like(like)
        for width in widths:
            result = torch.where(group_bits == width, value(width), result)
        return result

    def _quantize_native(
        self, x: torch.Tensor, plan: Optional[PrecisionPlan]
    ) -> QuantizedTensor:
        config = self.config
        check_input(x, plan, allow_channel_bits=True, allow_keep_tokens=True)
        if x.dtype not in NATIVE_DTYPES:
            raise ValueError("Integer inputs must be FP16, BF16 or FP32")
        bits, channel_bits = self._native_widths(x, plan)
        if config.complete_groups:
            if self.granularity == QuantGranularity.PER_CHANNEL:
                length = x.shape[2]
            else:
                length = x.shape[3]
            if length % config.group_size:
                raise ValueError(
                    "complete_groups: group_size must divide the grouped dimension "
                    "(tokens for PER_CHANNEL, head_dim otherwise)"
                )
        groups, layout, valid = group_for_quantization(
            x, self.granularity, config.group_size, plan
        )
        # Grouping is lossless for FP16/BF16/FP32. Return to the compute dtype
        # BEFORE every reference operation.
        work = x.dtype if config.compute_dtype == "input" else torch.float32
        groups = groups.to(work)
        lo, hi = group_min_max(groups, valid)
        if config.range_includes_zero:
            # QuaRot find_params: torch.minimum(x.min(1)[0], 0), torch.maximum(...).
            zero = torch.zeros_like(lo)
            lo, hi = torch.minimum(lo, zero), torch.maximum(hi, zero)
        # QuaRot: xmin * clip_ratio, xmax * clip_ratio (exact when the ratio is 1).
        lo = lo * config.clip_ratio
        hi = hi * config.clip_ratio

        widths: List[int]
        group_bits: Optional[torch.Tensor]
        if channel_bits is None:
            widths, group_bits = [bits], None
        else:
            widths = sorted(int(w) for w in channel_bits.unique().tolist())
            group_bits = channel_bits[..., None, None].expand(lo.shape)

        def bounds(low_for: Any, high_for: Any) -> Tuple[Any, Any]:
            """Clamp bounds: Python numbers for one width, tensors when mixed."""
            if group_bits is None:
                return low_for(widths[0]), high_for(widths[0])
            low = self._per_width(
                widths, group_bits, lo, lambda w: torch.full_like(lo, low_for(w))
            )
            high = self._per_width(
                widths, group_bits, lo, lambda w: torch.full_like(lo, high_for(w))
            )
            return low, high

        offset: Optional[torch.Tensor]
        if config.symmetric:
            # QuaRot sym: xmax = maximum(|xmin|, xmax); scale = xmax / maxq;
            # scale[xmax == 0] = 1; q = clamp(round(x / scale), -(maxq+1), maxq).
            amax = torch.maximum(torch.abs(lo), hi)
            scale = self._per_width(
                widths,
                group_bits,
                amax,
                lambda w: amax / torch.tensor(2 ** (w - 1) - 1, device=amax.device),
            )
            scale = torch.where(amax == 0, torch.ones_like(scale), scale)
            full = config.symmetric_range == "full"
            low, high = bounds(
                lambda w: -(2 ** (w - 1)) if full else -(2 ** (w - 1) - 1),
                lambda w: 2 ** (w - 1) - 1,
            )
            quotient = groups / scale
            codes = self._round(quotient, low, high)
            bias, _ = bounds(
                lambda w: 2 ** (w - 1) if full else 2 ** (w - 1) - 1, lambda w: 0
            )
            codes = codes + bias  # stored unsigned
            offset = None
        elif config.zero_point == "integer":
            # QuaRot asym: all-zero groups use [-1, 1]; scale = (xmax - xmin) / maxq;
            # zero = round(-xmin / scale); q = clamp(round(x / scale) + zero, 0, maxq).
            if config.degenerate_range == "pm_one":
                both = (lo == 0) & (hi == 0)
                lo = torch.where(both, torch.full_like(lo, -1), lo)
                hi = torch.where(both, torch.full_like(hi, 1), hi)
            span = hi - lo
            if config.degenerate_range == "eps":
                span = span.clamp(min=self._range_eps(x.dtype))
            scale = self._per_width(
                widths,
                group_bits,
                span,
                lambda w: span / torch.tensor(2**w - 1, device=span.device),
            )
            if config.degenerate_range == "unit_scale":
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            require_finite(scale, name="Integer scales")
            if (scale <= 0).any():
                raise ValueError(
                    "An integer zero point needs a positive scale; a constant "
                    "nonzero group gives a nonpositive scale"
                )
            offset = torch.round(-lo / scale)
            require_finite(offset, name="Integer zero points")
            low, high = bounds(lambda w: 0, lambda w: 2**w - 1)
            quotient = groups / scale
            if config.rounding == "round_clamp":
                codes = torch.clamp(torch.round(quotient) + offset, low, high)
            else:
                codes = torch.clamp(quotient + offset, low, high).round()
        else:
            # KIVI: scale = (mx - mn) / max_int; data = (data - mn) / scale;
            #       data.clamp_(0, max_int).round_().
            # Kitty: scale = (mx - mn).clamp(min=eps) / (2**b - 1);
            #       q = ((x - mn) / scale).clamp(0, 2**b - 1).round().
            if config.degenerate_range == "pm_one":
                both = (lo == 0) & (hi == 0)
                lo = torch.where(both, torch.full_like(lo, -1), lo)
                hi = torch.where(both, torch.full_like(hi, 1), hi)
            span = hi - lo
            if config.degenerate_range == "eps":
                span = span.clamp(min=self._range_eps(x.dtype))
            scale = self._per_width(widths, group_bits, span, lambda w: span / (2**w - 1))
            require_finite(scale, name="Integer scales")
            if config.degenerate_range == "zero_scale":
                # KIVI's reference divides 0 by 0 here (undefined); scale 0 with
                # code 0 decodes a constant group exactly.
                if ((scale == 0) & (hi != lo)).any():
                    raise ValueError(
                        "Scale underflow; use a higher input precision or the "
                        "generic path"
                    )
                denominator = torch.where(scale == 0, torch.ones_like(scale), scale)
            else:
                if config.degenerate_range == "unit_scale":
                    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                if (scale == 0).any():
                    raise ValueError("A group's scale is zero; use another degenerate_range")
                denominator = scale
            if valid is not None:
                # Kept/padded slots must not leave the group's range.
                groups = torch.where(valid, groups, lo)
            low, high = bounds(lambda w: 0, lambda w: 2**w - 1)
            codes = self._round((groups - lo) / denominator, low, high)
            offset = lo
        require_finite(codes, name="Integer codes")

        rows = codes.reshape(-1, codes.shape[-1])
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
            zero_points=None if offset is None else offset.contiguous(),
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=layout.axis,
            bits=bits,
            format_id=NATIVE_FORMAT,
            group_size=config.group_size,
            metadata={
                **layout_metadata(layout, rows.numel()),
                "widths": widths,
                "width_code_counts": counts,
                "width_byte_counts": byte_counts,
                "symmetric": config.symmetric,
                "symmetric_range": config.symmetric_range,
                "zero_point": config.zero_point,
                "decode_dtype": config.decode_dtype,
            },
            group_bits=None
            if flat_bits is None
            else flat_bits.to(torch.uint8).contiguous(),
        )

    def _range_eps(self, dtype: torch.dtype) -> float:
        if self.config.range_eps is None:
            return default_range_eps(dtype)
        return float(self.config.range_eps)

    def _round(self, quotient: torch.Tensor, low: Any, high: Any) -> torch.Tensor:
        if self.config.rounding == "clamp_round":
            return quotient.clamp(low, high).round()
        return torch.clamp(torch.round(quotient), low, high)

    def _dequantize_native(self, q: QuantizedTensor) -> torch.Tensor:
        meta = q.metadata
        symmetric = meta["symmetric"]
        if symmetric != (q.zero_points is None):
            raise ValueError("Integer record: zero points must exist iff asymmetric")
        require_finite(q.scales, name="Stored integer scales")
        if q.zero_points is not None:
            require_finite(q.zero_points, name="Stored integer offsets")
        layout = layout_from_metadata(meta, q.axis)
        widths = meta["widths"]
        counts = meta["width_code_counts"]
        byte_counts = meta["width_byte_counts"]
        size = q.group_size
        group_bits: Optional[torch.Tensor] = None
        if q.group_bits is None:
            if len(widths) != 1:
                raise ValueError("Mixed-width integer records need group_bits")
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
            group_bits = flat_bits.reshape(q.scales.shape)
        codes = flat.reshape(q.scales.shape[:-1] + (size,))
        scales, offsets = q.scales, q.zero_points
        decode = meta["decode_dtype"]
        if decode == "fp16":
            codes = codes.to(torch.float16)  # KIVI: data.to(torch.float16)
        elif decode == "compute":
            codes = codes.to(scales.dtype)  # Kitty/QuaRot: codes in their dtype
        else:
            codes = codes.float()
            scales = scales.float()
            offsets = None if offsets is None else offsets.float()
        if symmetric:
            full = meta["symmetric_range"] == "full"
            if group_bits is None:
                bias: Any = 2 ** (widths[0] - 1) - (0 if full else 1)
            else:
                bias = (2 ** (group_bits - 1) - (0 if full else 1)).to(codes.dtype)
            restored = scales * (codes - bias)  # QuaRot sym_dequant: scale * q
        elif meta["zero_point"] == "integer":
            assert offsets is not None
            restored = scales * (codes - offsets)  # QuaRot asym_dequant
        else:
            assert offsets is not None
            restored = codes * scales + offsets  # KIVI/Kitty: data * scale + mn
            if decode == "fp16" and not torch.isfinite(restored).all():
                # A half intermediate may overflow where the wide result is
                # finite; keep the wide value (KIVI's rule).
                wide = codes.float() * scales.float() + offsets.float()
                restored = torch.where(torch.isfinite(restored), restored.float(), wide)
        return finite_cast(
            ungroup_values(restored, layout, q.shape), q.dtype, device=q.device
        )
