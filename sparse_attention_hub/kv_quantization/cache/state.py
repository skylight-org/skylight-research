"""Per-layer state with independent key and value histories."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .block import StoredChunk


@dataclass
class TensorCacheState:
    """One tensor kind's history as three ordered regions of stored chunks."""

    sink: List[StoredChunk] = field(default_factory=list)
    chunks: List[StoredChunk] = field(default_factory=list)  # middle region
    recent: List[StoredChunk] = field(default_factory=list)  # local window

    @property
    def sink_tokens(self) -> int:
        return sum(chunk.token_count for chunk in self.sink)

    @property
    def compressed_tokens(self) -> int:
        return sum(chunk.token_count for chunk in self.chunks)

    @property
    def recent_tokens(self) -> int:
        return sum(chunk.token_count for chunk in self.recent)

    def all_chunks(self) -> List[StoredChunk]:
        """Every chunk in token order: sink, middle, local."""
        return [*self.sink, *self.chunks, *self.recent]

    def encoded_ranges(self) -> List[Tuple[int, int]]:
        """Token ranges [start, end) that are stored quantized, in any region."""
        return [
            (chunk.start_token, chunk.start_token + chunk.token_count)
            for chunk in self.all_chunks()
            if chunk.is_encoded
        ]


@dataclass
class LayerKVState:
    # Total tokens captured by our store, independently of HF's retained window.
    seen_tokens: int = 0
    key: TensorCacheState = field(default_factory=TensorCacheState)
    value: TensorCacheState = field(default_factory=TensorCacheState)
    key_schema: Optional[Tuple[Any, ...]] = None
    value_schema: Optional[Tuple[Any, ...]] = None
    method_state: Dict[str, Any] = field(default_factory=dict)
