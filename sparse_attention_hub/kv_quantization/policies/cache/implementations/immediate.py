"""Policies that quantize tokens as soon as they arrive (no full-precision tail)."""

from dataclasses import dataclass, field

from ....cache.state import TensorCacheState
from ....registry import CachePolicyRegistry
from ..base import CachePolicy, CachePolicyConfig, RegionConfig, require_empty


@dataclass(frozen=True)
class SinkOnlyCachePolicyConfig(CachePolicyConfig):
    """Only the sink region is special; every other token is quantized on arrival."""

    key_sink: RegionConfig = field(default_factory=lambda: RegionConfig(tokens=4))
    value_sink: RegionConfig = field(default_factory=lambda: RegionConfig(tokens=4))
    key_local: RegionConfig = field(default_factory=RegionConfig)
    value_local: RegionConfig = field(default_factory=RegionConfig)


@CachePolicyRegistry.register(SinkOnlyCachePolicyConfig)
class SinkOnlyCachePolicy(CachePolicy):
    def __init__(self, config: SinkOnlyCachePolicyConfig) -> None:
        super().__init__(config)
        require_empty(self, "local")

    def next_chunk_tokens(
        self,
        tensor_kind: str,
        state: TensorCacheState,
        *,
        phase: str = "before_attention",
    ) -> int:
        return state.recent_tokens if phase == "before_attention" else 0

    def min_move_tokens(self, tensor_kind: str) -> int:
        return 1


@dataclass(frozen=True)
class FullQuantizationCachePolicyConfig(CachePolicyConfig):
    """Nothing stays in full precision: no sinks, no local window."""

    key_local: RegionConfig = field(default_factory=RegionConfig)
    value_local: RegionConfig = field(default_factory=RegionConfig)


@CachePolicyRegistry.register(FullQuantizationCachePolicyConfig)
class FullQuantizationCachePolicy(CachePolicy):
    def __init__(self, config: FullQuantizationCachePolicyConfig) -> None:
        super().__init__(config)
        require_empty(self, "sink")
        require_empty(self, "local")

    def next_chunk_tokens(
        self,
        tensor_kind: str,
        state: TensorCacheState,
        *,
        phase: str = "before_attention",
    ) -> int:
        return state.recent_tokens if phase == "before_attention" else 0

    def min_move_tokens(self, tensor_kind: str) -> int:
        return 1
