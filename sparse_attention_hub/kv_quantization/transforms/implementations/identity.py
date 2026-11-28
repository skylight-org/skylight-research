"""IdentityTransform component and its configuration."""

from dataclasses import dataclass
from typing import Any, Tuple

import torch

from ...context import AttentionContext
from ...numerics import require_finite
from ...registry import KVTransformRegistry
from ..base import KVTransform, KVTransformConfig


@dataclass(frozen=True)
class IdentityTransformConfig(KVTransformConfig):
    pass


@KVTransformRegistry.register(IdentityTransformConfig)
class IdentityTransform(KVTransform):
    def forward(
        self, x: torch.Tensor, context: AttentionContext
    ) -> Tuple[torch.Tensor, Any]:
        return x, None

    def inverse(
        self, x: torch.Tensor, context: AttentionContext, state: Any
    ) -> torch.Tensor:
        require_finite(x, name="Reconstructed values")
        return x
