"""RandomRotation: a seeded random orthogonal rotation per layer and tensor kind."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from ...context import AttentionContext
from ...numerics import finite_cast
from ...registry import KVTransformRegistry
from ..base import KVTransform, KVTransformConfig, apply_rotation


@dataclass(frozen=True)
class RandomRotationConfig(KVTransformConfig):
    """Rotate each key/value vector by a random orthogonal matrix (TurboQuant).

    One matrix per (layer, keys/values), all derived from ``seed``, so keys and
    values of the same layer never share a rotation, and runs are reproducible.
    """

    seed: int = 0


def random_orthogonal(size: int, seed: int) -> torch.Tensor:
    """Uniformly random orthogonal matrix: QR of a Gaussian with a sign fix."""
    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(size, size, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    return (q * torch.sign(torch.diagonal(r))).float()


@KVTransformRegistry.register(RandomRotationConfig)
class RandomRotation(KVTransform):
    config: RandomRotationConfig

    def __init__(self, config: RandomRotationConfig) -> None:
        super().__init__(config)
        self._matrices: Dict[Tuple[int, str, int], torch.Tensor] = {}

    def matrix(self, layer_idx: int, tensor_kind: str, size: int) -> torch.Tensor:
        key = (layer_idx, tensor_kind, size)
        if key not in self._matrices:
            kind_offset = 0 if tensor_kind == "key" else 1
            seed = (self.config.seed * 1_000_003 + layer_idx) * 2 + kind_offset
            self._matrices[key] = random_orthogonal(size, seed)
        return self._matrices[key]

    def forward(
        self, x: torch.Tensor, context: AttentionContext
    ) -> Tuple[torch.Tensor, Any]:
        matrix = self.matrix(context.layer_idx, context.tensor_kind, x.shape[-1])
        return apply_rotation(x, matrix), x.dtype

    def inverse(
        self, x: torch.Tensor, context: AttentionContext, state: Any
    ) -> torch.Tensor:
        matrix = self.matrix(context.layer_idx, context.tensor_kind, x.shape[-1])
        return finite_cast(apply_rotation(x, matrix.T), state)
