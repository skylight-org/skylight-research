"""Choose numerical precision independently of cache buffering."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from ...quantizers.base import KVQuantizerConfig


@dataclass(frozen=True)
class PrecisionPlan:
    """How one chunk of keys or values (``[batch, heads, tokens, head_dim]``) is stored.

    Attributes:
        bits: Bits for every value, unless overridden below.
        channel_bits: Optional ``[batch, heads, head_dim]`` integer bits per channel
            (mixed precision across channels, e.g. Kitty's boosted key channels).
        keep_tokens: Optional ``[batch, heads, tokens]`` boolean mask of tokens
            kept in their original precision instead of being quantized. Opt-in
            codecs exclude them from encoding statistics; the coordinator owns
            saving/restoring originals. This generic contract does not prescribe
            any named paper's outlier treatment.
    """

    bits: int
    channel_bits: Optional[torch.Tensor] = None
    keep_tokens: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class PrecisionPolicyConfig:
    pass


class PrecisionPolicy(ABC):
    def __init__(self, config: PrecisionPolicyConfig) -> None:
        self.config = config

    @abstractmethod
    def bits_for(
        self,
        layer_idx: int,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        tensor_kind: str,
        default_bits: int,
    ) -> PrecisionPlan:
        """Return the plan for one chunk ``x`` (already transformed) of the middle region."""

    def validate_quantizers(
        self,
        key_quantizer: "KVQuantizerConfig",
        value_quantizer: "KVQuantizerConfig",
    ) -> None:
        """Reject quantizers that cannot carry out this policy's plans."""
