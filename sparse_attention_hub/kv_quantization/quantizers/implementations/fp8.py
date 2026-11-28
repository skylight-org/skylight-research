"""FP8Quantizer component and its configuration."""

from dataclasses import dataclass
from typing import Optional

import torch

from ...cache.block import QuantizedTensor
from ...context import AttentionContext
from ...numerics import finite_cast
from ...policies.precision.base import PrecisionPlan
from ...registry import KVQuantizerRegistry
from ..base import KVQuantizer, KVQuantizerConfig, QuantGranularity
from ..common import (
    check_input,
    group_for_quantization,
    layout_from_metadata,
    layout_metadata,
    store_param,
    ungroup_values,
)

FP8_FORMATS = {"E4M3": torch.float8_e4m3fn, "E5M2": torch.float8_e5m2}


@dataclass(frozen=True)
class FP8QuantizerConfig(KVQuantizerConfig):
    """8-bit floating point values with one scale per group.

    Attributes:
        format: ``"E4M3"`` (max 448, more precision) or ``"E5M2"`` (max 57344,
            more range).
        symmetric: Must stay True; floating point formats are signed, no offset.

    ``scale_dtype`` stores each group's scale; ``offset_dtype`` is unused.
    """

    bits: int = 8
    symmetric: bool = True
    format: str = "E4M3"


@KVQuantizerRegistry.register(FP8QuantizerConfig)
class FP8Quantizer(KVQuantizer):
    """Scale each group so its largest magnitude maps to the format's maximum,
    then round to FP8 (round to nearest even, saturating)."""

    config: FP8QuantizerConfig

    def __init__(self, config: FP8QuantizerConfig) -> None:
        super().__init__(config)
        if config.format not in FP8_FORMATS:
            raise ValueError(f"FP8 format must be one of {sorted(FP8_FORMATS)}")
        if config.bits != 8:
            raise ValueError("FP8Quantizer uses 8 bits")
        if not config.symmetric:
            raise ValueError("FP8 is symmetric (signed, no offset); set symmetric=True")
        if config.group_size < 1:
            raise ValueError("group_size must be positive")
        self.granularity = QuantGranularity(config.granularity)
        self.dtype = FP8_FORMATS[config.format]

    def quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> QuantizedTensor:
        check_input(x, plan, allow_keep_tokens=True)
        if plan is not None and plan.bits != 8:
            raise ValueError("FP8 precision plan must use 8 bits")
        groups, layout, _ = group_for_quantization(
            x, self.granularity, self.config.group_size, plan
        )
        largest = torch.finfo(self.dtype).max
        amax = groups.abs().amax(dim=-1, keepdim=True)
        scale = store_param(amax / largest, self.config.scale_dtype)
        values = (groups / scale).clamp(-largest, largest).to(self.dtype)
        return QuantizedTensor(
            codes=values.view(torch.uint8).flatten(),
            scales=scale.to(self.config.scale_dtype.torch_dtype).contiguous(),
            zero_points=None,
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=layout.axis,
            bits=8,
            format_id=f"fp8_{self.config.format.lower()}_v1",
            group_size=self.config.group_size,
            metadata=layout_metadata(layout, values.numel()),
        )

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        if q.format_id != f"fp8_{self.config.format.lower()}_v1":
            raise ValueError(f"Unsupported FP8 encoding: {q.format_id}")
        layout = layout_from_metadata(q.metadata, q.axis)
        values = (
            q.codes.view(self.dtype)
            .float()
            .reshape(
                *layout.layout_shape[:-1],
                layout.padded_length // q.group_size,
                q.group_size,
            )
        )
        restored = ungroup_values(values * q.scales.float(), layout, q.shape)
        return finite_cast(restored, q.dtype, device=q.device)
