"""HadamardTransform: a normalized Hadamard rotation."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from ...context import AttentionContext
from ...numerics import finite_cast
from ...registry import KVTransformRegistry
from ..base import KVTransform, KVTransformConfig, apply_rotation


@dataclass(frozen=True)
class HadamardTransformConfig(KVTransformConfig):
    """Rotate each key/value vector by the normalized Hadamard matrix H."""

    preserve_input_dtype: bool = False


def hadamard_matrix(size: int) -> torch.Tensor:
    """Normalized Sylvester Hadamard matrix (``H @ H.T == I``); ``size`` a power of 2."""
    if size < 1 or size & (size - 1):
        raise ValueError(f"Hadamard needs head_dim to be a power of 2, got {size}")
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)], 0
        )
    return matrix / size**0.5


@KVTransformRegistry.register(HadamardTransformConfig)
class HadamardTransform(KVTransform):
    """Spreads outlier channels across all channels before quantization.

    H is orthogonal, so ``inverse`` (multiplying by its transpose) restores the
    original coordinates exactly up to FP32 rounding.
    """

    config: HadamardTransformConfig

    def __init__(self, config: HadamardTransformConfig) -> None:
        super().__init__(config)
        if type(config.preserve_input_dtype) is not bool:
            raise ValueError("preserve_input_dtype must be boolean")
        self._matrices: Dict[int, torch.Tensor] = {}

    def matrix(self, size: int) -> torch.Tensor:
        if size not in self._matrices:
            self._matrices[size] = hadamard_matrix(size)
        return self._matrices[size]

    def forward(
        self, x: torch.Tensor, context: AttentionContext
    ) -> Tuple[torch.Tensor, Any]:
        rotated = apply_rotation(x, self.matrix(x.shape[-1]))
        if self.config.preserve_input_dtype:
            rotated = finite_cast(rotated, x.dtype)
        return rotated, x.dtype

    def inverse(
        self, x: torch.Tensor, context: AttentionContext, state: Any
    ) -> torch.Tensor:
        return finite_cast(apply_rotation(x, self.matrix(x.shape[-1]).T), state)
