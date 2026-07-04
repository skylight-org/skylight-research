"""Model management utilities for ModelServer."""

from typing import Any, Dict, Optional

from .key_generation import hash_kwargs


def generate_model_key(
    model_name: str, gpu_id: Optional[int], model_kwargs: Dict[str, Any]
) -> str:
    """Generate a unique key for a model based on its parameters.

    Args:
        model_name: Name of the model
        gpu_id: GPU ID where model is placed (None for CPU)
        model_kwargs: Additional model creation arguments

    Returns:
        Unique string key for the model
    """
    gpu_str = str(gpu_id) if gpu_id is not None else "cpu"
    kwargs_hash = hash_kwargs(model_kwargs)
    return f"{model_name}|{gpu_str}|{kwargs_hash}"


def generate_tokenizer_key(
    tokenizer_name: str, tokenizer_kwargs: Dict[str, Any]
) -> str:
    """Generate a unique key for a tokenizer based on its parameters.

    Args:
        tokenizer_name: Name of the tokenizer
        tokenizer_kwargs: Additional tokenizer creation arguments

    Returns:
        Unique string key for the tokenizer
    """
    kwargs_hash = hash_kwargs(tokenizer_kwargs)
    return f"{tokenizer_name}|{kwargs_hash}"


def infer_layer_type(
    module: Any, layer_idx: Optional[int], model: Optional[Any] = None
) -> str:
    """Best-effort classification of an attention layer's type.

    Returns the module's explicit ``layer_type`` (or ``config.layer_types[layer_idx]``
    / a boolean ``is_sliding``) when the model exposes it, otherwise ``"unknown"``.
    Used only for density-metric attribution.
    """
    explicit_layer_type = getattr(module, "layer_type", None)
    if isinstance(explicit_layer_type, str):
        return explicit_layer_type

    model_config = getattr(module, "config", None)
    if model_config is None and model is not None:
        model_config = getattr(model, "config", None)

    if model_config is not None and layer_idx is not None:
        layer_types = getattr(model_config, "layer_types", None)
        if (
            isinstance(layer_types, (list, tuple))
            and 0 <= layer_idx < len(layer_types)
            and isinstance(layer_types[layer_idx], str)
        ):
            return layer_types[layer_idx]

    is_sliding = getattr(module, "is_sliding", None)
    if isinstance(is_sliding, bool):
        return "sliding_attention" if is_sliding else "full_attention"

    return "unknown"
