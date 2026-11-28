"""RandomHadamardTransform: random +-1 signs followed by a normalized Hadamard."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from ...context import AttentionContext
from ...numerics import finite_cast
from ...registry import KVTransformRegistry
from ..base import KVTransform, KVTransformConfig, apply_rotation
from .hadamard import hadamard_matrix


@dataclass(frozen=True)
class RandomHadamardTransformConfig(KVTransformConfig):
    """Rotate by ``diag(s) @ H`` with random signs ``s`` (QuaRot's randomized Hadamard).

    One sign vector per (layer, keys/values), all derived from ``seed``, so runs
    are reproducible and keys and values never share a rotation.
    """

    seed: int = 0


@KVTransformRegistry.register(RandomHadamardTransformConfig)
class RandomHadamardTransform(KVTransform):
    """As fast as a plain Hadamard, but the random signs keep an adversarial input
    from lining up with H; still orthogonal, so the inverse is the transpose."""

    config: RandomHadamardTransformConfig

    def __init__(self, config: RandomHadamardTransformConfig) -> None:
        super().__init__(config)
        self._matrices: Dict[Tuple[int, str, int], torch.Tensor] = {}

    def matrix(self, layer_idx: int, tensor_kind: str, size: int) -> torch.Tensor:
        key = (layer_idx, tensor_kind, size)
        if key not in self._matrices:
            kind_offset = 0 if tensor_kind == "key" else 1
            seed = (self.config.seed * 1_000_003 + layer_idx) * 2 + kind_offset
            generator = torch.Generator().manual_seed(seed)
            signs = torch.randint(0, 2, (size,), generator=generator) * 2 - 1
            self._matrices[key] = signs[:, None].float() * hadamard_matrix(size)
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
