"""DefaultCachePolicy: keep a minimum local window and move older tokens in blocks."""

from dataclasses import dataclass

from ....cache.state import TensorCacheState
from ....registry import CachePolicyRegistry
from ..base import (
    KINDS,
    CachePolicy,
    CachePolicyConfig,
    block_tokens,
    move_block_beyond_window,
)


@dataclass(frozen=True)
class DefaultCachePolicyConfig(CachePolicyConfig):
    """``*_local.tokens`` is the minimum local window; tokens leave it in blocks.

    With the defaults (no sinks, 128 local tokens, blocks of 64) the full-precision
    tail holds between 128 and 191 tokens.
    """

    key_block_tokens: int = 64
    value_block_tokens: int = 64


@CachePolicyRegistry.register(DefaultCachePolicyConfig)
class DefaultCachePolicy(CachePolicy):
    """Move the oldest block once the local window holds ``local + block`` tokens."""

    def __init__(self, config: DefaultCachePolicyConfig) -> None:
        super().__init__(config)
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
