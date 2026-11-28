"""Base classes for sparse attention mechanisms (bare metal)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from ..kv_quantization.base import KVQuantization, KVQuantizationConfig


@dataclass
class SparseAttentionConfig:
    """Configuration class for sparse attention mechanisms.

    Attributes:
        kv_quantization_config: Optional KV quantization applied to the keys and
            values before attention. ``None`` (the default) leaves them untouched.
            Keyword-only so that subclass configs can keep required fields.
    """

    kv_quantization_config: Optional[KVQuantizationConfig] = field(
        default=None, kw_only=True
    )


class SparseAttention(ABC):
    """Abstract base class for sparse attention mechanisms."""

    sparse_attention_config: SparseAttentionConfig
    kv_quantization: Optional[KVQuantization]

    def __init__(self, sparse_attention_config: SparseAttentionConfig) -> None:
        """Initialize sparse attention mechanism.

        Args:
            sparse_attention_config: Configuration for the sparse attention mechanism.
        """
        self.sparse_attention_config = sparse_attention_config
        kv_quantization_config: Optional[KVQuantizationConfig] = getattr(
            sparse_attention_config, "kv_quantization_config", None
        )
        self.kv_quantization = (
            KVQuantization.create_from_config(kv_quantization_config)
            if kv_quantization_config is not None
            else None
        )

    def quantize_kv(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        sparse_meta_data: Dict[Any, Any],
        *,
        num_new_tokens: int,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the keys and values attention should use.

        Without KV quantization this returns the inputs unchanged. Otherwise the
        previous stored history is reconstructed and combined with this call's
        original new K/V before the persistent store is updated/compressed.

        Args:
            keys: Full key history from the HF cache, (b, h_kv, s, d).
            values: Full value history from the HF cache, (b, h_kv, s, d).
            sparse_meta_data: Per-request metadata that holds the quantized store.
            num_new_tokens: Query sequence length for this self-attention call.
            **kwargs: Attention kwargs; must include ``layer_idx``.

        Returns:
            Tuple of keys and values for the attention computation.
        """
        if self.kv_quantization is None:
            return keys, values
        return self.kv_quantization.process(
            kwargs["layer_idx"],
            keys,
            values,
            sparse_meta_data,
            num_new_tokens=num_new_tokens,
            phase=sparse_meta_data.get("kv_quantization_phase", "decode"),
            position_ids=kwargs.get("position_ids"),
        )

    def clear_request_state(self, sparse_meta_data: Dict[Any, Any]) -> None:
        """Free per-request KV quantization state held in ``sparse_meta_data``."""
        if self.kv_quantization is not None:
            self.kv_quantization.clear_request(sparse_meta_data)

    @abstractmethod
    def custom_attention(
        self,
        module: nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        **kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute custom attention mechanism.

        Returns:
            Tuple of attention output and optional attention weights.
        """
        pass

    def pre_attention_hook_generator(self) -> None:
        """Generate pre-attention hooks for model integration."""
        pass

    @classmethod
    def create_from_config(cls, config: SparseAttentionConfig) -> "SparseAttention":
        """Create sparse attention instance from configuration.

        Args:
            config: Configuration for the sparse attention mechanism.

        Returns:
            Instance of the sparse attention mechanism.
        """
        # Import here to avoid circular imports
        from .efficient_attention import EfficientAttention, EfficientAttentionConfig
        from .research_attention import ResearchAttention, ResearchAttentionConfig

        # Check config type and route to appropriate create_from_config method
        if isinstance(config, ResearchAttentionConfig):
            return ResearchAttention.create_from_config(config)
        elif isinstance(config, EfficientAttentionConfig):
            return EfficientAttention.create_from_config(config)
        else:
            # Fallback to default behavior for base config
            return cls(config)
