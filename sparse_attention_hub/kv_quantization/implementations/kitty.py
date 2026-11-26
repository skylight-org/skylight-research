"""Kitty (Xia et al., MLSys 2026, arXiv 2511.18643): 2-bit KV with boosted key channels.

The recipe follows the authors' accuracy simulation (Summer-Summer/Kitty @ dfd2c07,
src/kitty_sim): 32 full-precision sink tokens; keys quantized per channel in pages
of 128 tokens, where the ``fraction`` of channels with the largest mean |k| in each
page (per head) get 4 bits and the rest 2 bits; values quantized per token to 2 bits
with the newest 128 in full precision. The current call attends to its own exact
K/V and quantization happens after it (the reference's ``PostQuant``).

Parts: IntQuantizer with ``kitty_int_quantizer()`` (keys and values) +
ChannelBoost (precision policy) + KittyCachePolicy (sinks, page-wise key buffer,
value window).
"""

from dataclasses import dataclass, field
from typing import Any

from ..base import KVQuantization, KVQuantizationConfig
from ..policies.cache.implementations.sink_and_local import KittyCachePolicyConfig
from ..policies.cache.base import RegionConfig
from ..policies.precision.base import PrecisionPolicyConfig
from ..policies.precision.implementations.channel_boost import (
    ChannelBoostConfig,
    supports_channel_bits,
)
from ..policies.precision.implementations.fixed import FixedPrecisionConfig
from ..quantizers.base import KVQuantizerConfig, QuantGranularity
from ..quantizers.implementations.integer import IntQuantizerConfig
from ..registry import KVQuantizationRegistry


def kitty_int_quantizer(
    bits: int = 2,
    granularity: QuantGranularity = QuantGranularity.PER_TOKEN,
    group_size: int = 128,
    **overrides: Any,
) -> IntQuantizerConfig:
    """IntQuantizerConfig with Kitty's simulation arithmetic
    (src/kitty_sim fake_quant_groupwise_lastdim @ dfd2c07), in the input dtype:
    ``scale = max(max - min, eps) / (2**b - 1)``,
    ``code = ((x - min) / scale).clamp(0, 2**b - 1).round()``,
    decoded as ``code * scale + min``. Boosted channels come from ChannelBoost.
    ``overrides`` changes any other IntQuantizerConfig field.
    """
    settings: dict = dict(
        bits=bits,
        granularity=granularity,
        group_size=group_size,
        compute_dtype="input",
        parameter_storage="compute",
        zero_point="float",
        degenerate_range="eps",
        rounding="clamp_round",
        decode_dtype="compute",
        complete_groups=True,
    )
    settings.update(overrides)
    return IntQuantizerConfig(**settings)


def kitty_channel_boost(fraction: float, selection: str = "magnitude") -> ChannelBoostConfig:
    """ChannelBoost exactly as in the reference: keys only, 4 bits, floor count,
    scores computed in the cache dtype."""
    return ChannelBoostConfig(
        fraction=fraction,
        boosted_bits=4,
        kinds=("key",),
        selection=selection,
        count_rounding="floor",
        score_dtype="input",
    )


@dataclass(frozen=True)
class KittyConfig(KVQuantizationConfig):
    """Kitty with 12.5% boosted key channels; ``KittyConfig.pro()`` boosts 25%.

    Every field can be changed for experiments; ``with_settings`` exposes the
    paper's knobs. Use ``with_settings(boost_fraction=0)`` for the paper's KIVI*
    baseline (same sinks and buffers, no boost).
    """

    key_quantizer: KVQuantizerConfig = field(
        default_factory=lambda: kitty_int_quantizer(
            bits=2, granularity=QuantGranularity.PER_CHANNEL, group_size=128
        )
    )
    value_quantizer: KVQuantizerConfig = field(
        default_factory=lambda: kitty_int_quantizer(bits=2, group_size=128)
    )
    precision_policy: PrecisionPolicyConfig = field(
        default_factory=lambda: kitty_channel_boost(0.125)
    )
    cache_policy: KittyCachePolicyConfig = field(
        default_factory=KittyCachePolicyConfig
    )

    @classmethod
    def with_settings(
        cls,
        boost_fraction: float = 0.125,
        bits: int = 2,
        sink_length: int = 32,
        buffer_length: int = 128,
        group_size: int = 128,
        channel_selection: str = "magnitude",
        **overrides: Any,
    ) -> "KittyConfig":
        """Kitty with the reference's knobs (``promote_ratio``, ``kbits``/``vbits``,
        ``sink_length``, ``buffer_length``, ``group_size``, ``channel_selection``).

        ``boost_fraction=0`` turns the boost off. ``overrides`` sets any other
        ``KVQuantizationConfig`` field (``store_codes``, ``error_metric``, ...).
        """
        if isinstance(boost_fraction, bool) or not 0 <= boost_fraction <= 1:
            raise ValueError("boost_fraction must be in [0, 1]")
        precision: PrecisionPolicyConfig = (
            kitty_channel_boost(boost_fraction, channel_selection)
            if boost_fraction > 0
            else FixedPrecisionConfig()
        )
        settings: dict = dict(
            key_quantizer=kitty_int_quantizer(
                bits=bits,
                granularity=QuantGranularity.PER_CHANNEL,
                group_size=group_size,
            ),
            value_quantizer=kitty_int_quantizer(bits=bits, group_size=group_size),
            precision_policy=precision,
            cache_policy=KittyCachePolicyConfig(
                key_sink=RegionConfig(sink_length),
                value_sink=RegionConfig(sink_length),
                key_local=RegionConfig(buffer_length),
                value_local=RegionConfig(buffer_length),
            ),
        )
        settings.update(overrides)
        return cls(**settings)

    @classmethod
    def pro(cls, **settings: Any) -> "KittyConfig":
        """Kitty-Pro: 25% of key channels boosted (the paper's setting)."""
        return cls.with_settings(boost_fraction=0.25, **settings)


@KVQuantizationRegistry.register(KittyConfig)
class Kitty(KVQuantization):
    """Kitty through the shared coordinator; ``validate_kitty`` guards its structure."""

    config: KittyConfig

    def __init__(self, config: KittyConfig) -> None:
        validate_kitty(config)
        super().__init__(config)


def validate_kitty(config: KVQuantizationConfig) -> None:
    """Raise if ``config`` breaks what defines Kitty (other settings are free)."""
    keys, values = config.key_quantizer, config.value_quantizer
    if not isinstance(keys, IntQuantizerConfig) or not isinstance(
        values, IntQuantizerConfig
    ):
        raise ValueError("Kitty uses integer quantizers (IntQuantizerConfig) for K and V")
    if QuantGranularity(keys.granularity) != QuantGranularity.PER_CHANNEL:
        raise ValueError("Kitty quantizes keys per channel (PER_CHANNEL)")
    if QuantGranularity(values.granularity) == QuantGranularity.PER_CHANNEL:
        raise ValueError("Kitty quantizes values per token (PER_TOKEN)")
    if not isinstance(config.cache_policy, KittyCachePolicyConfig):
        raise ValueError(
            "Kitty uses sinks, a page-wise key buffer and a value window; use "
            "KittyCachePolicyConfig"
        )
    precision = config.precision_policy
    if isinstance(precision, ChannelBoostConfig):
        if tuple(precision.kinds) != ("key",):
            raise ValueError("Kitty boosts key channels only (kinds=('key',))")
        if not supports_channel_bits(keys):
            raise ValueError("Kitty's key quantizer must support per-channel bits")
    elif not isinstance(precision, FixedPrecisionConfig):
        raise ValueError("Kitty uses ChannelBoostConfig (or FixedPrecision: no boost)")
    buffer = config.cache_policy.key_local.tokens
    if type(buffer) is not int or buffer < 1:
        raise ValueError("Kitty's key buffer must be a positive integer")
    if type(keys.group_size) is not int or keys.group_size < 1 or buffer % keys.group_size:
        raise ValueError(
            f"Kitty needs the key buffer ({buffer}) to be a multiple of the key "
            f"group size ({keys.group_size}), so every key group is complete"
        )
