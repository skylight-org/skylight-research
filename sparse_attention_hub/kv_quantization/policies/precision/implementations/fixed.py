"""FixedPrecision component and its configuration."""

from dataclasses import dataclass

import torch

from ....registry import PrecisionPolicyRegistry
from ..base import PrecisionPlan, PrecisionPolicy, PrecisionPolicyConfig


@dataclass(frozen=True)
class FixedPrecisionConfig(PrecisionPolicyConfig):
    """Use each K/V quantizer's configured bit width."""


@PrecisionPolicyRegistry.register(FixedPrecisionConfig)
class FixedPrecision(PrecisionPolicy):
    def bits_for(
        self,
        layer_idx: int,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        tensor_kind: str,
        default_bits: int,
    ) -> PrecisionPlan:
        return PrecisionPlan(bits=default_bits)
