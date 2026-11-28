"""SinkAndLocalCachePolicy (KIVI) and KittyCachePolicy: buffered keys, rolling values."""

from dataclasses import dataclass, field

from ....cache.state import TensorCacheState
from ....registry import CachePolicyRegistry
from ..base import (
    CachePolicy,
    CachePolicyConfig,
    RegionConfig,
    move_beyond_window,
    move_full_buffers,
    move_full_pages_past_buffer,
)


def _buffered_keys_rolling_values(
    policy: CachePolicy, tensor_kind: str, state: TensorCacheState, phase: str
) -> int:
    if phase != "before_attention":
        return 0
    size = policy.config.local(tensor_kind).tokens
    if tensor_kind == "value":
        return move_beyond_window(state.recent_tokens, size)
    return move_full_buffers(state.recent_tokens, size)


def _min_move(policy: CachePolicy, tensor_kind: str) -> int:
    if tensor_kind == "value":
        return 1
    return max(1, policy.config.local(tensor_kind).tokens)


@dataclass(frozen=True)
class SinkAndLocalCachePolicyConfig(CachePolicyConfig):
    """Sinks plus a key buffer and a value window; KIVI (Liu et al., 2024) buffering.

    ``key_local.tokens`` (R) is the key buffer: keys wait in it and are quantized
    together once R have collected (in whole multiples of R when many arrive at
    once), so 0 to R-1 keys are in full precision. ``value_local.tokens`` is an
    exact rolling window: the newest values stay in full precision and each value
    is quantized as soon as it falls out of the window.
    """

    key_local: RegionConfig = field(default_factory=lambda: RegionConfig(tokens=128))
    value_local: RegionConfig = field(default_factory=lambda: RegionConfig(tokens=128))


@CachePolicyRegistry.register(SinkAndLocalCachePolicyConfig)
class SinkAndLocalCachePolicy(CachePolicy):
    def next_chunk_tokens(
        self,
        tensor_kind: str,
        state: TensorCacheState,
        *,
        phase: str = "before_attention",
    ) -> int:
        return _buffered_keys_rolling_values(self, tensor_kind, state, phase)

    def min_move_tokens(self, tensor_kind: str) -> int:
        return _min_move(self, tensor_kind)


@dataclass(frozen=True)
class KittyCachePolicyConfig(CachePolicyConfig):
    """Kitty (Xia et al., MLSys 2026) buffering, as in the authors' simulation.

    The first ``key_sink`` / ``value_sink`` tokens (32) stay in full precision.
    Keys then wait in a buffer of ``key_local.tokens`` (R = 128, one page): once
    more than R keys wait, every full page moves to the middle, *page by page*, so
    1 to R keys stay in full precision. Each page is encoded separately, so
    ChannelBoost picks the boosted channels of every page. Values keep an exact
    rolling window of ``value_local.tokens`` (128) and each older value moves as
    soon as it leaves the window (KIVI-style values).

    Kitty's per-channel bit boost is a precision policy (``ChannelBoost``), not part
    of the buffering.
    """

    key_sink: RegionConfig = field(default_factory=lambda: RegionConfig(tokens=32))
    value_sink: RegionConfig = field(default_factory=lambda: RegionConfig(tokens=32))
    key_local: RegionConfig = field(default_factory=lambda: RegionConfig(tokens=128))
    value_local: RegionConfig = field(default_factory=lambda: RegionConfig(tokens=128))


@CachePolicyRegistry.register(KittyCachePolicyConfig)
class KittyCachePolicy(CachePolicy):
    """Kitty's page-wise key buffer and rolling value window."""

    def next_chunk_tokens(
        self,
        tensor_kind: str,
        state: TensorCacheState,
        *,
        phase: str = "before_attention",
    ) -> int:
        if phase != "before_attention":
            return 0
        size = self.config.local(tensor_kind).tokens
        if tensor_kind == "value":
            return move_beyond_window(state.recent_tokens, size)
        return move_full_pages_past_buffer(state.recent_tokens, size)

    def min_move_tokens(self, tensor_kind: str) -> int:
        return _min_move(self, tensor_kind)

    def page_tokens(self, tensor_kind: str) -> int:
        return self.config.local(tensor_kind).tokens if tensor_kind == "key" else 0
