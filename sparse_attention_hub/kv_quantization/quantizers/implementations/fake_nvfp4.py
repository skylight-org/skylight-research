"""FakeNVFP4Quantizer: NVFP4's numbers computed in plain PyTorch, on any device."""

from dataclasses import dataclass
from typing import Optional

import torch

from ...cache.block import QuantizedTensor
from ...context import AttentionContext
from ...numerics import finite_cast
from ...policies.precision.base import PrecisionPlan
from ...registry import KVQuantizerRegistry
from ..base import KVQuantizer, QuantGranularity
from ..common import (
    check_input,
    fp_grid,
    group_for_quantization,
    layout_from_metadata,
    layout_metadata,
    pack_codes,
    round_to_grid,
    ungroup_values,
    unpack_codes,
)
from .nvfp4 import NVFP4QuantizerConfig

E2M1 = fp_grid(2, 1)  # 0, 0.5, 1, 1.5, 2, 3, 4, 6
E2M1_MAX = 6.0
E4M3_MAX = 448.0


@dataclass(frozen=True)
class FakeNVFP4QuantizerConfig(NVFP4QuantizerConfig):
    """Same settings as NVFP4; see ``NVFP4QuantizerConfig``."""


def encode_e2m1(scaled: torch.Tensor) -> torch.Tensor:
    """4-bit E2M1 codes (sign bit, then the 3-bit magnitude index) for values in
    [-6, 6], rounded to nearest even. Zero is always +0."""
    index = round_to_grid(scaled.abs().clamp(max=E2M1_MAX), E2M1)
    sign = (scaled < 0) & (index > 0)
    return index + sign.to(index.dtype) * 8


def decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    magnitude = E2M1.to(codes.device)[(codes & 7).long()]
    return torch.where(codes >= 8, -magnitude, magnitude)


@KVQuantizerRegistry.register(FakeNVFP4QuantizerConfig)
class FakeNVFP4Quantizer(KVQuantizer):
    """NVFP4 in plain PyTorch.

    Per quantized chunk, ``global = amax / (6 * 448)`` (FP32). Per block of 16,
    ``block_scale = E4M3(block_amax / (6 * global))``, the effective scale is
    ``block_scale * global``, and each value is rounded to the nearest E2M1 value
    (ties to even) after dividing by it. Codes pack two per byte; storage is
    4 bits per value plus one byte per 16 values (4.5 bits) plus 4 bytes per chunk.
    """

    config: FakeNVFP4QuantizerConfig

    def __init__(self, config: NVFP4QuantizerConfig) -> None:
        super().__init__(config)
        if config.bits != 4 or config.group_size != 16:
            raise ValueError("NVFP4 uses 4-bit values in blocks of 16")
        if not config.symmetric:
            raise ValueError(
                "NVFP4 is symmetric (signed, no offset); set symmetric=True"
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
        if plan is not None and plan.bits != 4:
            raise ValueError("NVFP4 precision plan must use 4 bits")
        if self.granularity != QuantGranularity.PER_CHANNEL and x.shape[-1] % 16:
            raise ValueError(
                f"NVFP4 needs head_dim to be a multiple of 16, got {x.shape[-1]}"
            )
        groups, layout, _ = group_for_quantization(x, self.granularity, 16, plan)
        amax = groups.abs().amax()
        global_scale = (amax / (E2M1_MAX * E4M3_MAX)).float()
        global_scale = torch.where(
            amax > 0, global_scale, torch.ones_like(global_scale)
        )
        block_amax = groups.abs().amax(dim=-1, keepdim=True)
        block_scale = (
            (block_amax / (E2M1_MAX * global_scale))
            .clamp(max=E4M3_MAX)
            .to(torch.float8_e4m3fn)
        )
        scale = block_scale.float() * global_scale
        scaled = groups / torch.where(scale > 0, scale, torch.ones_like(scale))
        packed, code_count = pack_codes(encode_e2m1(scaled), 4)
        return QuantizedTensor(
            codes=packed,
            scales=block_scale.view(torch.uint8).contiguous(),
            zero_points=None,
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            axis=layout.axis,
            bits=4,
            format_id="nvfp4_v1",
            group_size=16,
            metadata=layout_metadata(layout, code_count),
            global_scale=global_scale.reshape(1),
        )

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        if q.format_id != "nvfp4_v1":
            raise ValueError(f"Unsupported NVFP4 encoding: {q.format_id}")
        assert q.global_scale is not None
        layout = layout_from_metadata(q.metadata, q.axis)
        values = decode_e2m1(unpack_codes(q.codes, 4, q.metadata["code_count"]))
        values = values.reshape(
            *layout.layout_shape[:-1], layout.padded_length // 16, 16
        )
        scale = q.scales.view(torch.float8_e4m3fn).float() * q.global_scale.float()
        restored = ungroup_values(values * scale, layout, q.shape)
        return finite_cast(restored, q.dtype, device=q.device)
