"""Measure what a quantized KV cache costs a model.

:func:`perplexity` uses the standard KV-quantization protocol: non-overlapping
windows of ``seq_len`` tokens (2048 is the usual choice), one forward pass each, with
a fresh cache per window. Pass a cache built with ``quantize_prefill=True`` so
that attention in that single pass reads quantized KV; a cache without it
would leave the prompt exact and measure nothing.

:func:`kv_reconstruction_error` is a cheaper check that needs no labels: it
prefills one prompt and gives the relative error of the KV each full-attention
layer stored, against an exact cache.
"""

import math
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn


def wikitext2_test_ids(tokenizer: Any) -> torch.Tensor:
    """The whole WikiText-2 test split as one ``(1, tokens)`` tensor, for perplexity."""
    from datasets import load_dataset

    text: str = "\n\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    )
    return tokenizer(text, return_tensors="pt").input_ids


def perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    make_cache: Callable[[], Any],
    seq_len: int = 2048,
    max_windows: Optional[int] = None,
) -> float:
    """Perplexity over non-overlapping windows, one fresh cache per window.

    Args:
        model: A causal LM.
        input_ids: ``(1, tokens)``, e.g. :func:`wikitext2_test_ids`.
        make_cache: Returns a new cache (``DynamicCache(config=...)`` for the
            baseline, or a quantized cache with ``quantize_prefill=True``).
        seq_len: Window length.
        max_windows: Stop after this many windows (for quick runs).

    Returns:
        ``exp(mean token negative log-likelihood)``.
    """
    device: torch.device = next(model.parameters()).device
    windows: int = input_ids.shape[1] // seq_len
    if max_windows is not None:
        windows = min(windows, max_windows)
    if windows == 0:
        raise ValueError(f"Need at least {seq_len} tokens, got {input_ids.shape[1]}.")

    total_nll: float = 0.0
    total_tokens: int = 0
    with torch.no_grad():
        for w in range(windows):
            window: torch.Tensor = input_ids[:, w * seq_len : (w + 1) * seq_len].to(
                device
            )
            logits: torch.Tensor = model(
                window, past_key_values=make_cache(), use_cache=True
            ).logits
            nll: torch.Tensor = nn.functional.cross_entropy(
                logits[0, :-1].float(), window[0, 1:], reduction="sum"
            )
            total_nll += float(nll.item())
            total_tokens += seq_len - 1
    return math.exp(total_nll / total_tokens)


def _stored(layer: Any, attr: str) -> torch.Tensor:
    """A quantized layer's stored history, dequantized if it is packed."""
    stored: Any = getattr(layer, attr)
    return stored if isinstance(stored, torch.Tensor) else layer._dequantize(stored)


def _relative_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    return float(
        (
            torch.linalg.vector_norm(approx.float() - exact.float())
            / torch.linalg.vector_norm(exact.float())
        ).item()
    )


def kv_reconstruction_error(
    model: nn.Module,
    input_ids: torch.Tensor,
    reference_cache: Any,
    quantized_cache: Any,
) -> Dict[int, Dict[str, float]]:
    """Relative Frobenius error of each full-attention layer's quantized KV.

    Prefills the same prompt into both caches and compares what each quantized
    layer stored against the exact keys and values.

    Returns:
        ``{layer_idx: {"keys": err, "values": err}}`` for every quantized layer.
    """
    input_ids = input_ids.to(next(model.parameters()).device)
    with torch.no_grad():
        for cache in (reference_cache, quantized_cache):
            model(input_ids, past_key_values=cache, use_cache=True)
    errors: Dict[int, Dict[str, float]] = {}
    for idx, (ref_layer, q_layer) in enumerate(
        zip(reference_cache.layers, quantized_cache.layers)
    ):
        if getattr(q_layer, "_quantized_keys", None) is None:
            continue
        errors[idx] = {
            "keys": _relative_error(
                _stored(q_layer, "_quantized_keys"), ref_layer.keys
            ),
            "values": _relative_error(
                _stored(q_layer, "_quantized_values"), ref_layer.values
            ),
        }
    return errors
