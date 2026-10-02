"""Backend selection for the quantized KV cache.

Backends are imported inside the factory rather than at module scope, so
choosing ``fake_nvfp4`` does not require nvidia-modelopt to be installed.

There is no default: the adapter's ``quantize_kv_cache`` flag names the backend
outright, so a value that isn't a known backend is a mistake worth raising on
rather than resolving to something plausible.

Attributes:
    NVFP4 (str): Packed NVFP4 via nvidia-modelopt. Saves memory, needs a CUDA
        GPU and modelopt installed.
    FAKE_NVFP4 (str): Simulated NVFP4 — quantize and dequantize in place, in
        pure torch. Same accuracy loss, no memory saving, runs anywhere.
"""

from typing import Any, Dict, Tuple

NVFP4: str = "nvfp4"
FAKE_NVFP4: str = "fake_nvfp4"
BACKENDS: Tuple[str, ...] = (NVFP4, FAKE_NVFP4)


def validate_backend(backend: Any) -> str:
    """Check that a value names a known backend.

    Args:
        backend: The value the caller passed as a backend name.

    Returns:
        The backend name, unchanged.

    Raises:
        ValueError: If it is not a known backend. ``True`` lands here too, on
            purpose: there are two backends with different requirements, so
            there is no sensible reading of "quantize, I don't mind how".
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown KV cache quantization backend {backend!r}. Pass "
            f"quantize_kv_cache=False to disable quantization, or one of: "
            f"{', '.join(repr(name) for name in BACKENDS)}"
        )
    return str(backend)


def create_quantized_kv_cache(config: Any, backend: str, **kwargs: Any) -> Any:
    """Create the quantized KV cache for a model config.

    This is the entry point the HuggingFace adapter calls when
    ``quantize_kv_cache`` names a backend.

    Args:
        config: The model's ``PreTrainedConfig``.
        backend: ``"nvfp4"`` for packed NVFP4 via nvidia-modelopt, or
            ``"fake_nvfp4"`` for simulated NVFP4 in pure torch.
        **kwargs: Forwarded to the cache class: ``block_size``,
            ``residual_length``, ``axis_key``, ``axis_value``,
            and ``quantize_prefill``.

    Returns:
        A fresh cache instance; one is needed per request, since it holds state.

    Raises:
        ValueError: If ``backend`` is not a known backend.
    """
    validate_backend(backend)

    if backend == NVFP4:
        from .nvfp4_cache import NVFP4QuantizedCache

        return NVFP4QuantizedCache(config, **kwargs)

    from .fake_nvfp4_cache import FakeNVFP4QuantizedCache

    return FakeNVFP4QuantizedCache(config, **kwargs)


def describe_backends() -> Dict[str, str]:
    """One-line description of each backend, for help text and logs."""
    return {
        NVFP4: "packed NVFP4 via nvidia-modelopt; saves memory, needs CUDA",
        FAKE_NVFP4: "simulated NVFP4 in torch; accuracy only, runs anywhere",
    }
