"""Choose when key and value regions are compressed, and how each region is stored."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from ...cache.state import TensorCacheState
from ...quantizers.base import KVQuantizerConfig, QuantGranularity

if TYPE_CHECKING:
    from ...cache.store import QuantizedKVStore

KINDS = ("key", "value")


@dataclass(frozen=True)
class RegionConfig:
    """One edge region of the cache (sink or local window) for keys or values.

    Attributes:
        tokens: How many tokens the region holds. Each policy documents what the
            local size means for it (a minimum window, a buffer size, ...).
        quantizer: ``None`` keeps the region in its original (full) precision;
            otherwise the region is stored with this quantizer.
    """

    tokens: int = 0
    quantizer: Optional[KVQuantizerConfig] = None


@dataclass(frozen=True)
class CachePolicyConfig:
    """Settings shared by every cache policy.

    The history of keys (and, separately, values) is split into three regions::

        [ sink: first tokens ][ middle: older history ][ local: newest tokens ]

    Sink and local sizes and storage are set here, independently for keys and
    values. The middle region uses ``KVQuantizationConfig.key_quantizer`` /
    ``value_quantizer``. Each policy subclass decides *when* local tokens move to
    the middle. A quantized local token is re-quantized from its stored local copy
    when it moves.
    """

    key_sink: RegionConfig = field(default_factory=RegionConfig)
    value_sink: RegionConfig = field(default_factory=RegionConfig)
    key_local: RegionConfig = field(default_factory=lambda: RegionConfig(tokens=128))
    value_local: RegionConfig = field(default_factory=lambda: RegionConfig(tokens=128))

    def sink(self, tensor_kind: str) -> RegionConfig:
        return self.key_sink if tensor_kind == "key" else self.value_sink

    def local(self, tensor_kind: str) -> RegionConfig:
        return self.key_local if tensor_kind == "key" else self.value_local


class CachePolicy(ABC):
    """Decides when the oldest local tokens move to the quantized middle region.

    To add a policy: subclass the config (new settings, if any) and this class,
    implement ``next_chunk_tokens`` and ``min_move_tokens``, and register it with
    ``@CachePolicyRegistry.register(MyCachePolicyConfig)``.
    """

    def __init__(self, config: CachePolicyConfig) -> None:
        self.config = config
        for kind in KINDS:
            sink, local = config.sink(kind), config.local(kind)
            if sink.tokens < 0 or local.tokens < 0:
                raise ValueError("Sink and local token counts cannot be negative")
            if local.quantizer is not None and _is_per_channel(local.quantizer):
                # Local tokens arrive one at a time; per-channel groups span tokens.
                raise ValueError(
                    f"The {kind} local window needs a per-token or per-block "
                    "quantizer; per-channel groups span several tokens"
                )

    @abstractmethod
    def next_chunk_tokens(
        self,
        tensor_kind: str,
        state: TensorCacheState,
        *,
        phase: str = "before_attention",
    ) -> int:
        """Return how many of the oldest local tokens move to the middle now."""

    @abstractmethod
    def min_move_tokens(self, tensor_kind: str) -> int:
        """Smallest number of tokens this policy moves to the middle at once."""

    def page_tokens(self, tensor_kind: str) -> int:
        """Encode moved tokens in pages of this many tokens (0: one chunk per move).

        Each page is its own stored chunk, so per-chunk decisions (for example
        ChannelBoost's boosted channels) are made page by page.
        """
        return 0

    def validate_quantizers(
        self,
        key_quantizer: KVQuantizerConfig,
        value_quantizer: KVQuantizerConfig,
    ) -> None:
        """Reject a per-channel middle quantizer that would see partial groups.

        Per-channel groups run along tokens. If tokens move in smaller batches than
        ``group_size``, groups are padded; a one-token group reconstructs exactly,
        so those tokens would silently not be quantized.
        """
        for kind, quantizer in zip(KINDS, (key_quantizer, value_quantizer)):
            move = self.min_move_tokens(kind)
            if _is_per_channel(quantizer) and move < quantizer.group_size:
                raise ValueError(
                    f"{type(self).__name__} moves {kind} tokens {move} at a time, "
                    f"fewer than the per-channel group_size {quantizer.group_size}; "
                    "use a per-token quantizer or move larger blocks"
                )

    def after_attention(self, store: "QuantizedKVStore", layer_idx: int) -> None:
        """Extension point for methods that commit compression after attention."""


def _is_per_channel(quantizer: KVQuantizerConfig) -> bool:
    return QuantGranularity(quantizer.granularity) == QuantGranularity.PER_CHANNEL


# Shared rules. Policies are siblings under CachePolicy; they reuse these helpers
# instead of inheriting from each other.


def block_tokens(config: CachePolicyConfig, tensor_kind: str) -> int:
    """``key_block_tokens`` / ``value_block_tokens`` of a config that defines them."""
    return int(getattr(config, f"{tensor_kind}_block_tokens"))


def move_block_beyond_window(local_tokens: int, window: int, block: int) -> int:
    """Move one block once the local region holds ``window + block`` tokens."""
    return block if local_tokens >= window + block else 0


def move_full_buffers(local_tokens: int, buffer: int) -> int:
    """Move whole multiples of ``buffer`` (everything when ``buffer`` is 0)."""
    return local_tokens if buffer == 0 else local_tokens - local_tokens % buffer


def move_full_pages_past_buffer(local_tokens: int, page: int) -> int:
    """Move whole pages once the buffer holds more than one page.

    Kitty's rule: keys wait until a page of ``page`` tokens is full *and* another
    token has arrived, so 1 to ``page`` keys stay in full precision (a prompt that
    fills an exact number of pages beyond the first moves them all).
    """
    if page == 0:
        return local_tokens
    return local_tokens - local_tokens % page if local_tokens > page else 0


def move_beyond_window(local_tokens: int, window: int) -> int:
    """Move every token older than an exact window of ``window`` tokens."""
    return max(0, local_tokens - window)


def require_empty(policy: CachePolicy, region: str) -> None:
    """Reject a non-zero ``sink`` or ``local`` size for a policy without that region."""
    for kind in KINDS:
        size = getattr(policy.config, region)(kind).tokens
        if size != 0:
            raise ValueError(
                f"{type(policy).__name__} has no {region} region; set "
                f"{kind}_{region} tokens to 0"
            )
