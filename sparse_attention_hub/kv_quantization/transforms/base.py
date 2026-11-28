"""Transform contract, base configuration and pending implementation helper."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, cast

import torch

from ..context import AttentionContext
from ..registry import KVTransformRegistry


@dataclass(frozen=True)
class KVTransformConfig:
    pass


class KVTransform(ABC):
    def __init__(self, config: KVTransformConfig) -> None:
        self.config = config

    @abstractmethod
    def forward(
        self, x: torch.Tensor, context: AttentionContext
    ) -> Tuple[torch.Tensor, Any]:
        """Return transformed data and all chunk-specific state needed by inverse."""

    @abstractmethod
    def inverse(
        self, x: torch.Tensor, context: AttentionContext, state: Any
    ) -> torch.Tensor:
        """Restore coordinates, rejecting nonfinite values and saturating dtype overflow."""

    def calibrate(self, samples: Any) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has no calibration implementation"
        )

    @classmethod
    def create_from_config(cls, config: KVTransformConfig) -> "KVTransform":
        return cast(KVTransform, KVTransformRegistry.create(config))


class PendingTransform(KVTransform):
    """Base for transforms that are not implemented yet: selecting one raises."""

    def __init__(self, config: KVTransformConfig) -> None:
        super().__init__(config)
        raise NotImplementedError(
            f"{type(self).__name__} is a scaffold; implement before selecting it"
        )

    def forward(
        self, x: torch.Tensor, context: AttentionContext
    ) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError

    def inverse(
        self, x: torch.Tensor, context: AttentionContext, state: Any
    ) -> torch.Tensor:
        raise NotImplementedError


def apply_rotation(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """``x @ matrix`` over the last (head_dim) axis, computed in FP32."""
    if x.shape[-1] != matrix.shape[0]:
        raise ValueError(
            f"head_dim {x.shape[-1]} does not match the {matrix.shape[0]}-dim rotation"
        )
    return x.float() @ matrix.to(device=x.device, dtype=torch.float32)
