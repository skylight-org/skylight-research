"""KV cache that fake-quantizes to NVFP4.

Same accuracy behaviour as :mod:`nvfp4_cache`, same residual-buffer semantics,
same axis choices — but the cache holds ordinary dtype tensors that have been
rounded to the NVFP4 grid rather than packed 4-bit codes. So it measures what
the format costs a model without needing nvidia-modelopt or a Blackwell GPU,
and saves no memory while doing it.
"""

from typing import Any

import torch

from .constants import (
    AXIS_PER_CHANNEL,
    AXIS_PER_TOKEN,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_RESIDUAL_LENGTH,
    VALID_AXES,
)
from .layers import QuantizePrefillMixin, build_cache_layers
from .nvfp4 import fake_quantize_nvfp4

try:
    from transformers.cache_utils import Cache, QuantizedLayer
except ImportError as exc:  # pragma: no cover - depends on the installed transformers
    raise ImportError(
        "KV cache quantization needs the layer-based cache API from "
        "transformers>=5.0 (transformers.cache_utils.QuantizedLayer). Upgrade "
        "transformers, or construct the adapter with quantize_kv_cache=False."
    ) from exc


class FakeNVFP4QuantizedLayer(QuantizePrefillMixin, QuantizedLayer):
    """One layer's KV cache, rounded to the NVFP4 grid but stored unpacked.

    Keys and values get independent axis choices for the same reason as in the
    real backend: keys carry persistent large-magnitude channels, so a scale
    shared across channels is set by an outlier and costs every well-behaved
    channel in the block its precision.
    """

    def __init__(
        self,
        block_size: int = DEFAULT_BLOCK_SIZE,
        residual_length: int = DEFAULT_RESIDUAL_LENGTH,
        axis_key: int = AXIS_PER_TOKEN,
        axis_value: int = AXIS_PER_TOKEN,
        quantize_prefill: bool = False,
    ) -> None:
        """Initialize the layer.

        Args:
            block_size: Elements sharing one E4M3 scale.
            residual_length: Trailing tokens left un-rounded.
            axis_key: ``-1`` for per-token blocking, ``0`` for per-channel.
            axis_value: Same, for values.
            quantize_prefill: See ``QuantizePrefillMixin``.

        Raises:
            ValueError: If either axis is not ``-1`` or ``0``.
        """
        for name, axis in (("axis_key", axis_key), ("axis_value", axis_value)):
            if axis not in VALID_AXES:
                raise ValueError(
                    f"{name!r} must be {AXIS_PER_TOKEN} (per-token) or "
                    f"{AXIS_PER_CHANNEL} (per-channel, KIVI-style), got {axis}"
                )
        super().__init__(
            nbits=4,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=block_size,
            residual_length=residual_length,
        )
        self.block_size: int = block_size
        self.quantize_prefill: bool = quantize_prefill

    def _quantize(self, tensor: torch.Tensor, axis: int) -> torch.Tensor:
        """Round a KV tensor to the NVFP4 grid, keeping its dtype and shape.

        Args:
            tensor: Shape ``(batch, heads, tokens, head_dim)``.
            axis: ``-1`` to block over ``head_dim``, ``0`` over tokens.

        Returns:
            A new tensor of the same shape and dtype, holding only
            NVFP4-representable values.
        """
        if axis == AXIS_PER_CHANNEL:
            transposed: torch.Tensor = tensor.transpose(-1, -2).contiguous()
            return fake_quantize_nvfp4(transposed, self.block_size).transpose(-1, -2)
        return fake_quantize_nvfp4(tensor, self.block_size)

    def _dequantize(self, qtensor: torch.Tensor) -> torch.Tensor:
        """Return the tensor unchanged, since it was never packed."""
        return qtensor


class FakeNVFP4QuantizedCache(Cache):
    """A :class:`Cache` whose per-layer KV storage is fake-quantized NVFP4."""

    def __init__(
        self,
        config: Any,
        block_size: int = DEFAULT_BLOCK_SIZE,
        residual_length: int = DEFAULT_RESIDUAL_LENGTH,
        axis_key: int = AXIS_PER_TOKEN,
        axis_value: int = AXIS_PER_TOKEN,
        quantize_prefill: bool = False,
    ) -> None:
        """Build one fake-quantized layer per full-attention decoder layer.

        Args:
            config: The model's ``PreTrainedConfig``.
            block_size: Elements sharing one E4M3 scale.
            residual_length: Trailing tokens left un-rounded.
            axis_key: Blocking axis for keys.
            axis_value: Blocking axis for values.
            quantize_prefill: Return quantized KV from the prefill call too
                (for single-pass perplexity); see ``QuantizePrefillMixin``.

        Raises:
            ValueError: If any decoder layer is not full attention
                (sliding-window, chunked or linear-attention layers have their
                own cache layouts that this does not implement).
        """
        super().__init__(
            layers=build_cache_layers(
                config,
                lambda _layer_idx: FakeNVFP4QuantizedLayer(
                    block_size=block_size,
                    residual_length=residual_length,
                    axis_key=axis_key,
                    axis_value=axis_value,
                    quantize_prefill=quantize_prefill,
                ),
                cache_name="FakeNVFP4QuantizedCache",
            )
        )
