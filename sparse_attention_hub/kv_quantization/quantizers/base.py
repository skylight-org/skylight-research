"""Quantizer contracts, grouping and base configuration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, cast

import torch

from ..cache.block import QuantizedTensor
from ..context import AttentionContext
from ..policies.precision.base import PrecisionPlan
from ..registry import KVQuantizerRegistry


class QuantGranularity(str, Enum):
    PER_TOKEN = "per_token"
    PER_CHANNEL = "per_channel"
    PER_BLOCK = "per_block"


class ParamDType(str, Enum):
    """Data type used to store per-group quantization parameters."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            ParamDType.FP32: torch.float32,
            ParamDType.FP16: torch.float16,
            ParamDType.BF16: torch.bfloat16,
        }[self]


@dataclass(frozen=True)
class KVQuantizerConfig:
    """Settings shared by every quantizer.

    Attributes:
        bits: Bits per stored code.
        granularity: Which values share one scale (and offset).
        group_size: How many values share one scale (and offset).
        symmetric: Signed range around zero with no offset, or an asymmetric
            range with an offset (the group minimum).
        scale_dtype: How each group's scale is stored. 16-bit types halve the
            overhead of fp32; codes are computed with the stored (rounded) value.
        offset_dtype: How each group's offset is stored (asymmetric only). bf16
            has few mantissa bits, so offsets far from zero lose precision.
    """

    bits: int = 4
    granularity: QuantGranularity = QuantGranularity.PER_TOKEN
    group_size: int = 32
    symmetric: bool = False
    scale_dtype: ParamDType = ParamDType.FP16
    offset_dtype: ParamDType = ParamDType.FP16


class KVQuantizer(ABC):
    def __init__(self, config: KVQuantizerConfig) -> None:
        self.config = config

    @abstractmethod
    def quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> QuantizedTensor:
        """Encode with the supplied precision plan, including layout metadata.

        ``context`` says which layer and tensor kind ``x`` belongs to; quantizers with
        per-layer state (calibrated tables) need it, the others ignore it.

        Codecs that support ``plan.keep_tokens`` exclude those positions from
        encoding statistics and emit disposable placeholders in the same slots.
        ``KVQuantization`` saves/restores their exact originals; raw dequantization
        alone does not restore kept tokens. Other codecs must reject such plans.
        """

    @abstractmethod
    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        """Restore shape/dtype/device, rejecting NaN/inf and saturating finite overflow.

        This is the generic reconstruction boundary rule. Finite values beyond
        the output dtype's limits clamp to those limits before conversion.
        """

    def fake_quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> torch.Tensor:
        return self.dequantize(self.quantize(x, plan, context=context))

    def calibrate(self, samples: Any) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} has no calibration implementation"
        )

    @classmethod
    def create_from_config(cls, config: KVQuantizerConfig) -> "KVQuantizer":
        return cast(KVQuantizer, KVQuantizerRegistry.create(config))


class PendingQuantizer(KVQuantizer):
    """Base for formats that are not implemented yet: selecting one raises."""

    def __init__(self, config: KVQuantizerConfig) -> None:
        super().__init__(config)
        raise NotImplementedError(
            f"{type(self).__name__} is a scaffold; implement before selecting it"
        )

    def quantize(
        self,
        x: torch.Tensor,
        plan: Optional[PrecisionPlan] = None,
        *,
        context: Optional[AttentionContext] = None,
    ) -> QuantizedTensor:
        raise NotImplementedError

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        raise NotImplementedError
