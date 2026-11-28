"""OutlierTokenPolicy: keep abnormally large tokens in their original precision."""

from dataclasses import dataclass
from typing import Tuple

import torch

from ....registry import PrecisionPolicyRegistry
from ..base import PrecisionPlan, PrecisionPolicy, PrecisionPolicyConfig

KINDS = ("key", "value")


@dataclass(frozen=True)
class OutlierTokenPolicyConfig(PrecisionPolicyConfig):
    """A token is an outlier when its largest |value| exceeds ``threshold`` times the
    median of that statistic over the chunk (per head). Outliers stay in full
    precision and are excluded from the codec's encoding statistics; every other
    token uses the quantizer's bits. This is an optional generic experiment, not
    an outlier rule automatically applied to named paper implementations.

    Attributes:
        threshold: How many times the median a token must exceed.
        max_fraction: At most this share of a chunk's tokens is kept (the largest
            ones), which bounds the extra memory.
        kinds: Which tensors are checked.
    """

    threshold: float = 6.0
    max_fraction: float = 0.05
    kinds: Tuple[str, ...] = ("key", "value")


@PrecisionPolicyRegistry.register(OutlierTokenPolicyConfig)
class OutlierTokenPolicy(PrecisionPolicy):
    config: OutlierTokenPolicyConfig

    def __init__(self, config: OutlierTokenPolicyConfig) -> None:
        super().__init__(config)
        if config.threshold <= 0:
            raise ValueError("threshold must be positive")
        if not 0 <= config.max_fraction <= 1:
            raise ValueError("max_fraction must be in [0, 1]")
        if any(kind not in KINDS for kind in config.kinds):
            raise ValueError(f"kinds must be drawn from {KINDS}")

    def bits_for(
        self,
        layer_idx: int,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        tensor_kind: str,
        default_bits: int,
    ) -> PrecisionPlan:
        tokens = x.shape[-2]
        limit = int(self.config.max_fraction * tokens)
        if tensor_kind not in self.config.kinds or limit == 0:
            return PrecisionPlan(bits=default_bits)
        score = x.abs().float().amax(dim=-1)  # [batch, heads, tokens]
        median = score.median(dim=-1, keepdim=True).values
        outlier = score > self.config.threshold * median
        # Keep only the `limit` largest outliers per head.
        rank = score.argsort(dim=-1, descending=True).argsort(dim=-1)
        keep = outlier & (rank < limit)
        if not keep.any():
            return PrecisionPlan(bits=default_bits)
        return PrecisionPlan(bits=default_bits, keep_tokens=keep)
