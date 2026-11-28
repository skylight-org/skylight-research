"""MXFPQuantizer (OCP microscaling MXFP4 / MXFP8) component and its configuration."""

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
    pack_codes,
    ungroup_values,
    unpack_codes,
)
from .fake_nvfp4 import E2M1_MAX, decode_e2m1, encode_e2m1

# element format -> (bits, largest value, exponent of the largest normal value)
MX_FORMATS = {
    "E2M1": (4, E2M1_MAX, 2),  # MXFP4
    "E4M3": (8, 448.0, 8),  # MXFP8
    "E5M2": (8, 57344.0, 15),  # MXFP8
}
FP8_TYPES = {"E4M3": torch.float8_e4m3fn, "E5M2": torch.float8_e5m2}


@dataclass(frozen=True)
class MXFPQuantizerConfig(KVQuantizerConfig):
    """OCP MX: blocks of 32 values share one power-of-two (E8M0) scale.

    Attributes:
        format: Element format, ``"E2M1"`` (MXFP4, ``bits=4``), ``"E4M3"`` or
            ``"E5M2"`` (MXFP8, ``bits=8``).

    The format fixes the scale type: ``scale_dtype`` and ``offset_dtype`` are
    unused, ``symmetric`` must be True and ``group_size`` 32. ``granularity``
    chooses the block direction (PER_TOKEN / PER_BLOCK along channels, PER_CHANNEL
    along tokens).
    """

    bits: int = 4
    group_size: int = 32
    symmetric: bool = True
    format: str = "E2M1"


@KVQuantizerRegistry.register(MXFPQuantizerConfig)
class MXFPQuantizer(KVQuantizer):
    """Shared scale ``2 ** (floor(log2(block_amax)) - emax)``, ``emax`` being the
    exponent of the element format's largest value; scaled values above the
    largest value saturate. The scale is stored as one byte (E8M0, bias 127)."""

    config: MXFPQuantizerConfig

    def __init__(self, config: MXFPQuantizerConfig) -> None:
        super().__init__(config)
        if config.format not in MX_FORMATS:
            raise ValueError(
                f"MX element format must be one of {sorted(MX_FORMATS)} "
                "(MXFP6 is not supported yet)"
            )
        bits, self.largest, self.emax = MX_FORMATS[config.format]
        if config.bits != bits:
            raise ValueError(
                f"MX {config.format} uses {bits}-bit values; set bits={bits}"
            )
        if config.group_size != 32:
            raise ValueError("MX formats use blocks of 32")
        if not config.symmetric:
            raise ValueError(
                "MX formats are symmetric (signed, no offset); set symmetric=True"
            )
        self.granularity = QuantGranularity(config.granularity)

    def quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> QuantizedTensor:
        check_input(x, plan, allow_keep_tokens=True)
        if plan is not None and plan.bits != self.config.bits:
            raise ValueError(f"MX precision plan must use {self.config.bits} bits")
        if self.granularity != QuantGranularity.PER_CHANNEL and x.shape[-1] % 32:
            raise ValueError(
                f"MX needs head_dim to be a multiple of 32, got {x.shape[-1]}"
            )
        groups, layout, _ = group_for_quantization(x, self.granularity, 32, plan)
        amax = groups.abs().amax(dim=-1, keepdim=True)
        exponent = torch.where(
            amax > 0,
            torch.floor(torch.log2(amax.clamp(min=torch.finfo(torch.float32).tiny)))
            - self.emax,
            torch.full_like(amax, -127.0),
        ).clamp(-127, 127)
        scaled = (groups / torch.exp2(exponent)).clamp(-self.largest, self.largest)
        if self.config.format == "E2M1":
            codes, code_count = pack_codes(encode_e2m1(scaled), 4)
        else:
            values = scaled.to(FP8_TYPES[self.config.format])
            codes, code_count = values.view(torch.uint8).flatten(), values.numel()
        return QuantizedTensor(
            codes=codes,
            scales=(exponent + 127).to(torch.uint8).contiguous(),
            zero_points=None,
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=layout.axis,
            bits=self.config.bits,
            format_id=f"mx_{self.config.format.lower()}_v1",
            group_size=32,
            metadata=layout_metadata(layout, code_count),
        )

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        if q.format_id != f"mx_{self.config.format.lower()}_v1":
            raise ValueError(f"Unsupported MX encoding: {q.format_id}")
        layout = layout_from_metadata(q.metadata, q.axis)
        if self.config.format == "E2M1":
            values = decode_e2m1(unpack_codes(q.codes, 4, q.metadata["code_count"]))
        else:
            values = q.codes.view(FP8_TYPES[self.config.format]).float()
        values = values.reshape(
            *layout.layout_shape[:-1], layout.padded_length // 32, 32
        )
        scale = torch.exp2(q.scales.float() - 127)
        restored = ungroup_values(values * scale, layout, q.shape)
        return finite_cast(restored, q.dtype, device=q.device)
