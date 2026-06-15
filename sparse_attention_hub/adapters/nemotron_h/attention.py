"""Sparse attention integration for NemotronH models."""

from typing import Any, Dict, Optional, Tuple

import torch

from sparse_attention_hub.sparse_attention.research_attention.base import ResearchAttention


class NemotronHSparseAttention:
    """Mixin that replaces scaled_dot_product_attention with hub's sparse attention.
    
    This is not a nn.Module subclass — it is injected at runtime into
    NemotronHAttention's forward pass via enable_sparse_mode().
    """

    @staticmethod
    def sparse_forward(
        original_module: torch.nn.Module,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        research_attention: ResearchAttention,
        sparse_meta_data: Dict[str, Any],
    ) -> torch.Tensor:
        """Replace scaled_dot_product_attention with sparse attention.

        Args:
            original_module: The NemotronHAttention module instance
            query_states: shape (b, num_heads, sq, head_dim)
            key_states: shape (b, num_heads, sk, head_dim)
            value_states: shape (b, num_heads, sq, head_dim)
            attention_mask: optional causal mask
            research_attention: the hub's ResearchAttention instance
            sparse_meta_data: metadata dict passed through forward

        Returns:
            attn_output: shape (b, num_heads, sq, head_dim)
        """
        scaling = original_module.head_dim ** -0.5
        dropout = (
            original_module.attention_dropout if original_module.training else 0.0
        )

        attn_output, _ = research_attention.custom_attention(
            module=original_module,
            queries=query_states,
            keys=key_states,
            values=value_states,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_meta_data=sparse_meta_data,
            layer_idx=original_module.layer_idx,
        )

        return attn_output