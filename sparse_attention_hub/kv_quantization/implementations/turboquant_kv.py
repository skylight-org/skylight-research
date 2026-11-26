"""TurboQuant MSE/prod KV recipes using the shared immediate-reconstruction flow."""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ..base import KVQuantization, KVQuantizationConfig
from ..policies.cache.implementations.immediate import FullQuantizationCachePolicyConfig
from ..policies.cache.implementations.sink_and_local import SinkAndLocalCachePolicyConfig
from ..policies.precision.implementations.channel_boost import ChannelBoostConfig
from ..policies.precision.implementations.fixed import FixedPrecisionConfig
from ..quantizers.implementations.turboquant import TurboQuantQuantizerConfig
from ..quantizers.implementations.turboquant_codebook import (
    TurboQuantCodebookQuantizerConfig,
)
from ..registry import KVQuantizationRegistry
from ..transforms.implementations.identity import IdentityTransformConfig
from ..transforms.implementations.random_rotation import RandomRotationConfig


@dataclass(frozen=True)
class TurboQuantKVConfig(KVQuantizationConfig):
    """Uniform whole-head profile: 3-bit MSE K/V, no exact residual regions.

    The paper quantizes every token (the default FullQuantizationCachePolicy).
    ``SinkAndLocalCachePolicyConfig`` instead keeps KIVI's exact recent tokens
    (a key buffer and a value window), which is not part of the paper
    (diagnostic/ablation only).

    Select variant="prod" in a K/V codec to spend one of its nominal bits on
    QJL correction. ``outlier_split(2.5)`` / ``outlier_split(3.5)`` build the
    paper's fractional-bit profile: ChannelBoost (frozen at prefill) picks the
    outlier channels and each codec splits every vector into two TurboQuant
    instances. Normalization and rotation live in the codec; outer transforms
    are identity.
    """

    key_transform: IdentityTransformConfig = field(
        default_factory=IdentityTransformConfig
    )
    value_transform: IdentityTransformConfig = field(
        default_factory=IdentityTransformConfig
    )
    key_quantizer: TurboQuantQuantizerConfig = field(
        default_factory=TurboQuantQuantizerConfig
    )
    value_quantizer: TurboQuantQuantizerConfig = field(
        default_factory=TurboQuantQuantizerConfig
    )
    cache_policy: Union[
        FullQuantizationCachePolicyConfig, SinkAndLocalCachePolicyConfig
    ] = field(default_factory=FullQuantizationCachePolicyConfig)
    store_codes: bool = True
    quantize_prefill: bool = True

    @classmethod
    def outlier_split(
        cls,
        average_bits: float,
        variant: str = "mse",
        center_keys: bool = False,
        **settings: Any,
    ) -> "TurboQuantKVConfig":
        """Fractional-bit profile for K and V (head_dim 128).

        2.5: 64 outlier channels (50%) at 3 bits + 64 at 2 bits.
        3.5: 64 outlier channels (50%) at 4 bits + 64 at 3 bits.
        The paper's 2.5-bit example (32 channels at 3 bits + 96 at 2 bits) works
        out to 288 / 128 = 2.25 bits, and it gives no 3.5-bit ratio, so both
        splits here are our choice of a true 2.5 / 3.5 average. Outliers are the
        channels with the largest mean |x| over the request's prefill, chosen per
        layer, K/V and head, and reused for every decoded token.

        ``center_keys=True`` (not in the paper) removes each key channel's
        prefill mean before quantizing (see ``TurboQuantQuantizerConfig.center``)
        and picks outliers by variance over the prefill instead of mean |x|, so
        the boosted channels are the ones whose values differ between tokens
        rather than the ones with the largest shared offset.
        """
        if type(center_keys) is not bool:
            raise ValueError("center_keys must be a bool")
        if average_bits not in OUTLIER_SPLITS:
            raise ValueError(f"average_bits must be one of {tuple(OUTLIER_SPLITS)}")
        bits, boosted_bits, fraction = OUTLIER_SPLITS[average_bits]
        quantizer = TurboQuantQuantizerConfig(bits=bits, variant=variant)
        config = cls(
            key_quantizer=replace(quantizer, center=("key",) if center_keys else ()),
            value_quantizer=quantizer,
            precision_policy=ChannelBoostConfig(
                fraction=fraction,
                boosted_bits=boosted_bits,
                kinds=("key", "value"),
                selection="variance" if center_keys else "magnitude",
                freeze="prefill",
            ),
        )
        return replace(config, **settings)


# average bits -> (normal bits, outlier bits, outlier fraction of the channels)
OUTLIER_SPLITS: Dict[float, Tuple[int, int, float]] = {
    2.5: (2, 3, 0.5),
    3.5: (3, 4, 0.5),
}


def validate_turboquant_config(config: TurboQuantKVConfig) -> None:
    """Keep the named profile's codec, coordinate and retention contracts explicit."""
    for kind in ("key", "value"):
        if type(getattr(config, f"{kind}_transform")) is not IdentityTransformConfig:
            raise ValueError(
                "TurboQuant owns its rotation; outer transforms must be identity"
            )
        if type(getattr(config, f"{kind}_quantizer")) is not TurboQuantQuantizerConfig:
            raise ValueError("TurboQuant requires TurboQuantQuantizerConfig")
    if type(config.cache_policy) not in (
        FullQuantizationCachePolicyConfig,
        SinkAndLocalCachePolicyConfig,
    ):
        raise ValueError(
            "TurboQuant requires FullQuantizationCachePolicyConfig (the paper) or "
            "SinkAndLocalCachePolicyConfig (KIVI's exact recent tokens)"
        )
    precision = config.precision_policy
    if type(precision) is ChannelBoostConfig:
        if precision.freeze != "prefill":
            raise ValueError(
                "TurboQuant's outlier split needs ChannelBoostConfig(freeze="
                "'prefill'): one channel choice per request, not one per token"
            )
    elif type(precision) is not FixedPrecisionConfig:
        raise ValueError(
            "TurboQuant requires FixedPrecisionConfig or ChannelBoostConfig"
        )
    if (
        type(config.store_codes) is not bool
        or type(config.quantize_prefill) is not bool
    ):
        raise ValueError("TurboQuant storage/prefill flags must be boolean")


def validate_turboquant_layout(keys: torch.Tensor, values: torch.Tensor) -> None:
    """Validate the head schema before the shared helper creates request state."""
    if keys.ndim != 4 or values.ndim != 4 or keys.shape[:3] != values.shape[:3]:
        raise ValueError(
            "TurboQuant requires matching [batch, kv_heads, tokens, channels] K/V"
        )
    for tensor in (keys, values):
        if tensor.shape[-1] < 2 or tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(
                "TurboQuant requires FP16/BF16/FP32 heads of dimension at least 2"
            )


@KVQuantizationRegistry.register(TurboQuantKVConfig)
class TurboQuantKV(KVQuantization):
    """Reconstruct previous and arriving K/V before ordinary attention."""

    config: TurboQuantKVConfig

    def __init__(self, config: TurboQuantKVConfig) -> None:
        validate_turboquant_config(config)
        super().__init__(config)

    def process(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        meta_data: Dict[str, Any],
        *,
        num_new_tokens: int,
        phase: str = "decode",
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        validate_turboquant_layout(keys, values)
        return self._process_kv_history(
            layer_idx,
            keys,
            values,
            meta_data,
            num_new_tokens=num_new_tokens,
            phase=phase,
            position_ids=position_ids,
            reconstruct_current=True,
        )


@dataclass(frozen=True)
class TurboQuantSplitKVConfig(KVQuantizationConfig):
    """TurboQuant with its two stages as separate components.

    The random rotation is ``RandomRotation`` in the transform slot and the
    codebook stage is ``TurboQuantCodebookQuantizer``; every other setting is the
    same as ``TurboQuantKVConfig``. The same rotation seed gives the same
    rotation matrices as ``TurboQuantQuantizerConfig.rotation``.
    """

    key_transform: RandomRotationConfig = field(default_factory=RandomRotationConfig)
    value_transform: RandomRotationConfig = field(
        default_factory=RandomRotationConfig
    )
    key_quantizer: TurboQuantCodebookQuantizerConfig = field(
        default_factory=TurboQuantCodebookQuantizerConfig
    )
    value_quantizer: TurboQuantCodebookQuantizerConfig = field(
        default_factory=TurboQuantCodebookQuantizerConfig
    )
    cache_policy: FullQuantizationCachePolicyConfig = field(
        default_factory=FullQuantizationCachePolicyConfig
    )
    store_codes: bool = True
    quantize_prefill: bool = True


def validate_turboquant_split_config(config: TurboQuantSplitKVConfig) -> None:
    """The codebook is only correct for randomly rotated vectors: require the
    rotation in the transform slot."""
    for kind in ("key", "value"):
        if type(getattr(config, f"{kind}_transform")) is not RandomRotationConfig:
            raise ValueError(
                "TurboQuant's codebook assumes randomly rotated vectors; use "
                "RandomRotationConfig as the transform"
            )
        quantizer = getattr(config, f"{kind}_quantizer")
        if type(quantizer) is not TurboQuantCodebookQuantizerConfig:
            raise ValueError("TurboQuant split requires TurboQuantCodebookQuantizerConfig")
    if type(config.cache_policy) is not FullQuantizationCachePolicyConfig:
        raise ValueError("TurboQuant requires FullQuantizationCachePolicyConfig")
    if type(config.precision_policy) is not FixedPrecisionConfig:
        raise ValueError("TurboQuant requires FixedPrecisionConfig")
    if (
        type(config.store_codes) is not bool
        or type(config.quantize_prefill) is not bool
    ):
        raise ValueError("TurboQuant storage/prefill flags must be boolean")


@KVQuantizationRegistry.register(TurboQuantSplitKVConfig)
class TurboQuantSplitKV(KVQuantization):
    """Rotate (transform), quantize (codebook codec), reconstruct, attend."""

    config: TurboQuantSplitKVConfig

    def __init__(self, config: TurboQuantSplitKVConfig) -> None:
        validate_turboquant_split_config(config)
        super().__init__(config)

    def process(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        meta_data: Dict[str, Any],
        *,
        num_new_tokens: int,
        phase: str = "decode",
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        validate_turboquant_layout(keys, values)
        return self._process_kv_history(
            layer_idx,
            keys,
            values,
            meta_data,
            num_new_tokens=num_new_tokens,
            phase=phase,
            position_ids=position_ids,
            reconstruct_current=True,
        )
