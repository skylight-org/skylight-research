"""Per-layer cache construction shared by both NVFP4 backends.

Only models whose decoder layers are **all full attention** are supported: every
layer keeps a token-indexed key/value cache, and every layer gets a quantized
cache layer. Other layer types (sliding-window, chunked, linear attention) have
different cache semantics and are rejected rather than silently mis-stored.
"""

from typing import Any, Callable, List, Set

try:
    from transformers.cache_utils import CacheLayerMixin, get_layer_types_and_kwargs
except ImportError as exc:  # pragma: no cover - depends on the installed transformers
    raise ImportError(
        "KV cache quantization needs the layer-based cache API from "
        "transformers>=5.0. Upgrade transformers, or construct the adapter "
        "without quantize_kv_cache."
    ) from exc

SUPPORTED_LAYER_TYPES: Set[str] = {"full_attention"}


def build_cache_layers(
    config: Any,
    make_quantized_layer: Callable[[int], CacheLayerMixin],
    cache_name: str,
) -> List[Any]:
    """Build one quantized cache layer per decoder layer.

    Args:
        config: The model's ``PreTrainedConfig`` (its text/decoder config is used).
        make_quantized_layer: Called with the decoder layer index; returns a
            fresh quantized layer.
        cache_name: Used in the error message.

    Returns:
        The layers, in decoder order, ready for ``Cache.__init__(layers=...)``.

    Raises:
        ValueError: If any layer is not full attention.
    """
    text_config: Any = config.get_text_config(decoder=True)
    # ``get_layer_types_and_kwargs`` reads fields such as ``sliding_window`` for
    # layer types we reject anyway, so check the raw types first.
    raw_layer_types = getattr(text_config, "layer_types", None)
    if raw_layer_types is not None:
        unsupported: Set[str] = set(raw_layer_types) - SUPPORTED_LAYER_TYPES
        if unsupported:
            raise ValueError(
                f"{cache_name} only supports models whose layers are all full "
                f"attention, not {sorted(unsupported)} layers."
            )
    layer_types, _ = get_layer_types_and_kwargs(text_config)
    unsupported = set(layer_types) - SUPPORTED_LAYER_TYPES
    if unsupported:
        raise ValueError(
            f"{cache_name} only supports models whose layers are all full "
            f"attention, not {sorted(unsupported)} layers."
        )
    return [make_quantized_layer(layer_idx) for layer_idx in range(len(layer_types))]


class QuantizePrefillMixin:
    """Optionally return quantized KV from the very first (prefill) update.

    ``transformers.QuantizedLayer`` quantizes the prompt on the first update
    but returns the *original* keys and values, so prefill attention is exact
    and only later tokens read quantized KV. That is the realistic serving
    behaviour and stays the default.

    Perplexity is measured in a single forward pass, which would then never
    touch quantized KV. With ``quantize_prefill=True`` the first update returns
    the dequantized tensors instead, so every token attends to quantized KV;
    this is the usual way to report perplexity. Mix in before ``QuantizedLayer``.
    """

    quantize_prefill: bool = False

    def update(
        self, key_states: Any, value_states: Any, *args: Any, **kwargs: Any
    ) -> Any:
        first_update: bool = not self.is_initialized  # type: ignore[attr-defined]
        keys, values = super().update(key_states, value_states, *args, **kwargs)  # type: ignore[misc]
        if first_update and self.quantize_prefill:
            return (
                self._dequantize(self._quantized_keys),  # type: ignore[attr-defined]
                self._dequantize(self._quantized_values),  # type: ignore[attr-defined]
            )
        return keys, values
