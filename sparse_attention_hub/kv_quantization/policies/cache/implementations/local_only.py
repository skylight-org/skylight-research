"""LocalOnlyCachePolicy: an exact full-precision window of the newest tokens."""

from dataclasses import dataclass

from ....cache.state import TensorCacheState
from ....registry import CachePolicyRegistry
from ..base import (
    KINDS,
    CachePolicy,
    CachePolicyConfig,
    block_tokens,
    move_block_beyond_window,
    require_empty,
)


@dataclass(frozen=True)
class LocalOnlyCachePolicyConfig(CachePolicyConfig):
    """No sinks; ``*_local.tokens`` newest tokens stay in the local region.

    Tokens leave one at a time by default, so the window size is exact. Raise
    ``*_block_tokens`` to move them in blocks (needed for per-channel quantizers).
    """

    key_block_tokens: int = 1
    value_block_tokens: int = 1


@CachePolicyRegistry.register(LocalOnlyCachePolicyConfig)
class LocalOnlyCachePolicy(CachePolicy):
    def __init__(self, config: LocalOnlyCachePolicyConfig) -> None:
        super().__init__(config)
        require_empty(self, "sink")
        for kind in KINDS:
            if block_tokens(config, kind) < 1:
                raise ValueError("block token counts must be positive")

    def next_chunk_tokens(
        self,
        tensor_kind: str,
        state: TensorCacheState,
        *,
        phase: str = "before_attention",
    ) -> int:
        if phase != "before_attention":
            return 0
        return move_block_beyond_window(
            state.recent_tokens,
            self.config.local(tensor_kind).tokens,
            block_tokens(self.config, tensor_kind),
        )

    def min_move_tokens(self, tensor_kind: str) -> int:
        return block_tokens(self.config, tensor_kind)
