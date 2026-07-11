"""Utility functions for masked attention computation."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from .kv_utils import _get_num_key_value_groups, repeat_kv
from .mask import Mask


def get_true_attention_output(
    module: nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    softcap: Optional[float] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Get the true (dense) attention output from the module.

    Args:
        module: The attention module (used for dropout training flag).
        queries: Query tensor of shape (..., seq_len_q, d_k).
        keys: Key tensor of shape (..., seq_len_k, d_k).
        values: Value tensor of shape (..., seq_len_k, d_v).
        attention_mask: Optional mask tensor to apply to attention weights.
        scaling: Scaling factor for attention logits.
        dropout: Dropout probability for attention weights.
        softcap: Optional Gemma-style logit soft cap; forces an eager fp32
            reference path (SDPA cannot apply softcapping).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        Tuple containing:
            - attention_output: Output tensor after applying attention.
            - attention_weights: Softmax-normalized attention weights.
    """
    num_key_value_groups = _get_num_key_value_groups(queries, keys)
    key_states = repeat_kv(keys, num_key_value_groups)
    value_states = repeat_kv(values, num_key_value_groups)

    if softcap is not None:
        # SDPA cannot apply logit softcapping; use an eager fp32 reference
        # mirroring _compute_masked_exp_attention_weights' order of ops.
        q = queries.to(torch.float32)
        k = key_states.to(torch.float32)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
        attn_weights = apply_softcap(attn_weights, softcap)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[
                :, :, :, : key_states.shape[-2]
            ].to(torch.float32)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        if dropout > 0.0 and module.training:
            attn_weights = torch.nn.functional.dropout(
                attn_weights, p=dropout, training=True
            )
        attn_output = torch.matmul(attn_weights, value_states.to(torch.float32))
        attn_output = attn_output.to(queries.dtype).transpose(1, 2).contiguous()
        return attn_output, None

    sdp_kwargs = dict(
        query=queries,
        key=key_states,
        value=value_states,
        attn_mask=attention_mask[:, :, :, : key_states.shape[-2]]
        if attention_mask is not None
        else None,
        dropout_p=dropout if module.training else 0.0,
        is_causal=False,
        scale=scaling,
    )

    if torch.cuda.is_available() and attention_mask is None:
        # Only try flash when no explicit mask; flash kernels reject non-null masks.
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    **sdp_kwargs
                )
        except Exception:
            attn_output = torch.nn.functional.scaled_dot_product_attention(**sdp_kwargs)
    else:
        # Let SDPA pick the best supported backend (math/mem-efficient) when a mask is present.
        attn_output = torch.nn.functional.scaled_dot_product_attention(**sdp_kwargs)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None  # SDPA doesn’t return weights


def apply_inv_mask_sum(input_tensor: torch.Tensor, mask: Mask) -> torch.Tensor:
    """Apply inverse mask to input tensor and sum along the last dimension.

    This function efficiently computes the sum of applying the inverse mask to an input tensor
    using sparse representation and scatter operations, avoiding the need to create dense tensors.

    Args:
        input_tensor: Input tensor of shape (..., n) where n is the last dimension
        mask: Mask object to apply inverse mask with

    Returns:
        Sum tensor of shape (..., 1) with the last dimension reduced

    Note:
        - For full masks: returns sum of all input values
        - For empty masks: returns sum of all input values since empty mask means default case when no mask is built.
        - For sparse masks: efficiently computes sum using sparse operations
    """
    if input_tensor.shape != mask.shape:
        raise ValueError(
            f"input_tensor.shape must be {mask.shape}, got {input_tensor.shape}"
        )

    # Handle special cases
    if mask.is_full or mask.is_empty:
        # Full mask: sum all input values
        return input_tensor.sum(dim=-1, keepdim=True)
    result = mask.apply_inv_mask(input_tensor)
    return result.sum(dim=-1, keepdim=True)


# TODO(is there a better dense version of this function?)
def create_sampling_mask_with_per_head_budget(
    budgets: torch.Tensor,
    sampling_probability: torch.Tensor,
    seq_len_keys: int,
    start_idx: int,
    end_idx: int,
    dtype: torch.dtype = torch.float32,
) -> Mask:
    """Create a sampling mask with per-head budget using direct sparse construction.

    This function efficiently creates a sparse sampling mask by directly constructing
    the sparse representation without creating intermediate dense tensors.

    Args:
        budgets: Budget tensor of shape (b, h, q, 1) indicating how many elements to sample per row
        sampling_probability: Sampling probability tensor of shape (b, h, q, 1)
        seq_len_keys: Length of the key sequence dimension
        start_idx: Starting index for sampling range (inclusive)
        end_idx: Ending index for sampling range (exclusive)
        dtype: Data type for the mask

    Returns:
        Mask object with sparse sampling representation

    Note:
        - Uses direct sparse construction for memory efficiency
        - Generates random indices within [start_idx, end_idx) for each element
        - Creates proper ptr array for sparse representation
        - Assigns sampling probabilities as mask data values

    Important Note:
        - we use random sampling with replacement so the sampling probabilities might lead to be incorrect
    """
    batch_size, num_heads, seq_len_queries, _ = budgets.shape

    # Reshape budget to (num_rows,) for easier processing
    num_rows = batch_size * num_heads * seq_len_queries
    budgets_flat = budgets.view(num_rows)  # (num_rows,)

    # Calculate total number of elements to sample
    total_elements = int(budgets_flat.sum().item())

    # Create ptr array using cumulative sum of budgets
    ptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=budgets.device),
            torch.cumsum(budgets_flat, dim=0),
        ]
    )  # (num_rows + 1,)

    # Generate random indices within sampling range for each element
    idx_in_row = torch.randint(
        low=start_idx,
        high=end_idx,
        size=(total_elements,),
        device=budgets.device,
        dtype=torch.long,
    )  # (total_elements,)

    # Create row indices by repeating each row index according to its budget
    positions = torch.arange(total_elements, device=budgets.device) + 1
    row_id = torch.searchsorted(ptr, positions, right=False) - 1  # (total_elements,)

    # Calculate global indices
    idx_global = idx_in_row + row_id * seq_len_keys  # (total_elements,)

    # Get sampling probabilities for each element
    sampling_prob_flat = sampling_probability.view(num_rows)  # (num_rows,)
    data_global = sampling_prob_flat[row_id]  # (total_elements,)

    # Create the sampling mask directly using sparse index construction
    sampling_mask = Mask.create_mask_from_indices(
        shape=(batch_size, num_heads, seq_len_queries, seq_len_keys),
        indices=idx_global,
        ptr=ptr,
        data=data_global,
        dtype=dtype,
    )

    return sampling_mask


def apply_softcap(scores: torch.Tensor, softcap: float) -> torch.Tensor:
    return softcap * torch.tanh(scores / softcap)


def _compute_masked_exp_attention_weights(
    queries: torch.Tensor,
    keys: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    sparse_attention_mask: Mask,
    dropout: float = 0.0,
    training: bool = False,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    """Compute masked attention weights (common logic for numerator and denominator).

    Args:
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        sparse_attention_mask: Mask object for sparse attention
        dropout: Dropout probability
        training: Whether the model is in training mode

    Returns:
        Masked exponential attention weights tensor of shape (b, h, sq, sk)
    """
    # Calculate num_key_value_groups from tensor shapes
    num_key_value_groups: int = _get_num_key_value_groups(queries, keys)

    # Apply key-value grouping if needed
    key_states: torch.Tensor = repeat_kv(keys, num_key_value_groups)

    q = queries.to(torch.float32)
    k = key_states.to(torch.float32)
    raw_attention_weights: torch.Tensor = torch.matmul(q, k.transpose(2, 3)) * scaling

    # Gemma softcap
    if softcap is not None:
        raw_attention_weights = apply_softcap(raw_attention_weights, softcap)

    if attention_mask is not None:
        raw_attention_weights = raw_attention_weights + attention_mask[
            :, :, :, : key_states.shape[-2]
        ].to(torch.float32)

    row_wise_max: torch.Tensor = torch.max(raw_attention_weights, dim=-1, keepdim=True)[
        0
    ]
    raw_attention_weights = raw_attention_weights - row_wise_max

    exp_attention_weights: torch.Tensor = torch.exp(raw_attention_weights)

    exp_attention_weights = sparse_attention_mask.apply_inv_mask(exp_attention_weights)

    # Apply dropout to attention weights if specified
    if dropout > 0.0 and training:
        exp_attention_weights = torch.nn.functional.dropout(
            exp_attention_weights, p=dropout, training=training
        )

    return exp_attention_weights


def _get_attention_denominator(exp_attention_weights: torch.Tensor) -> torch.Tensor:
    """Get attention denominator from pre-computed exponential attention weights.

    Args:
        exp_attention_weights: Pre-computed exponential attention weights of shape (b, h, sq, sk)

    Returns:
        Denominator tensor of shape (b, h, sq, 1)
    """
    return torch.sum(exp_attention_weights, dim=-1, keepdim=True)


def _get_attention_numerator(
    exp_attention_weights: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """Get attention numerator from pre-computed exponential attention weights and prepared values.

    Args:
        exp_attention_weights: Pre-computed exponential attention weights of shape (b, h, sq, sk)
        value_states: Prepared value tensor of shape (b, h, sq, d) - already grouped if needed

    Returns:
        Numerator tensor of shape (b, h, sq, d)
    """
    return torch.matmul(exp_attention_weights, value_states)


def get_attention_denominator(
    module: Optional[nn.Module],
    queries: torch.Tensor,
    keys: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_attention_mask: Mask,
    softcap: Optional[float] = None,
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    """Get masked attention denominator.

    Args:
        module: The attention module (used to check training mode)
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        dropout: Dropout probability
        sparse_attention_mask: Mask object for sparse attention
        **kwargs: Additional keyword arguments

    Returns:
        Denominator tensor of shape (b, h, sq, 1)
    """
    training: bool = module.training if module is not None else False
    exp_attention_weights: torch.Tensor = _compute_masked_exp_attention_weights(
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        scaling=scaling,
        sparse_attention_mask=sparse_attention_mask,
        dropout=dropout,
        training=training,
        softcap=softcap,
    )

    return _get_attention_denominator(exp_attention_weights)


def get_attention_numerator(
    module: nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_attention_mask: Mask,
    softcap: Optional[float] = None,
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    """Get masked attention numerator.

    Args:
        module: The attention module (used to check training mode)
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        values: Value tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        dropout: Dropout probability
        sparse_attention_mask: Mask object for sparse attention
        **kwargs: Additional keyword arguments

    Returns:
        Numerator tensor of shape (b, h, sq, d)
    """
    training: bool = module.training if module is not None else False
    exp_attention_weights: torch.Tensor = _compute_masked_exp_attention_weights(
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        scaling=scaling,
        sparse_attention_mask=sparse_attention_mask,
        dropout=dropout,
        training=training,
        softcap=softcap,
    )

    # Prepare values by applying key-value grouping
    num_key_value_groups: int = _get_num_key_value_groups(queries, values)
    value_states: torch.Tensor = repeat_kv(values, num_key_value_groups)

    return _get_attention_numerator(
        exp_attention_weights, value_states.to(torch.float32)
    )


def apply_sink_bias(
    num: torch.Tensor,
    den: torch.Tensor,
    exp_attention_weights: Optional[torch.Tensor],
    queries: torch.Tensor,
    keys: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    sinks: Optional[torch.Tensor],
    scaling: float,
    softcap: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if sinks is None:
        return num, den, exp_attention_weights

    assert sinks.ndim == 1, f"sinks must be 1D tensor of shape [H], got {sinks.shape}"
    B, H, Q = queries.shape[0], queries.shape[1], queries.shape[2]
    sinks_row = sinks.view(1, H, 1, 1).expand(B, H, Q, 1)

    # Recompute regular logits to get old row-wise max (WITHOUT sink)
    num_key_value_groups_k = _get_num_key_value_groups(queries, keys)
    key_states = repeat_kv(keys, num_key_value_groups_k)

    q = queries.to(torch.float32)
    k = key_states.to(torch.float32)
    logits = torch.matmul(q, k.transpose(2, 3)) * float(scaling)  # [B,H,Q,K]
    # Match _compute_masked_exp_attention_weights: softcap before mask add, so
    # old_max is measured in the same (capped) logit scale as the exp weights.
    if softcap is not None:
        logits = apply_softcap(logits, softcap)
    if attention_mask is not None:
        logits = logits + attention_mask[:, :, :, : key_states.shape[-2]].to(
            torch.float32
        )

    old_max = logits.max(dim=-1, keepdim=True).values  # [B,H,Q,1]
    new_max = torch.maximum(old_max, sinks_row)

    # Rescale everything that was computed in the (logits - old_max) exp scale
    scale = torch.exp((old_max - new_max).to(torch.float32)).to(den.dtype)  # [B,H,Q,1]
    num = num * scale
    den = den * scale
    if exp_attention_weights is not None:
        exp_attention_weights = exp_attention_weights * scale

    # Add sink contribution in the new_max scale
    sink_exp = torch.exp((sinks_row - new_max).to(torch.float32)).to(
        den.dtype
    )  # [B,H,Q,1]
    den = den + sink_exp

    return num, den, exp_attention_weights


# GET MASKED ATTENTION OUTPUT
def get_masked_attention_output(
    module: nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    sinks: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sparse_attention_mask: Mask,
    softcap: Optional[float] = None,
    return_attention_weights: bool = False,
    **kwargs: Dict[str, Any],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Get masked attention output by dividing numerator by denominator.

    Args:
        module: The attention module (used to check training mode)
        queries: Query tensor of shape (b, h, sk, d)
        keys: Key tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        values: Value tensor of shape (b, h_kv, sq, d) where h_kv may be different from h
        attention_mask: Optional attention mask of shape (b, h, sq, sk)
        scaling: Scaling factor for attention weights
        dropout: Dropout probability
        sparse_attention_mask: Mask object for sparse attention
        return_attention_weights: Whether to return attention weights along with output
        **kwargs: Additional keyword arguments

    Returns:
        If return_attention_weights is False:
            Attention output tensor of shape (b, h, sq, d)
        If return_attention_weights is True:
            Tuple of (attention_output, attention_weights) where:
            - attention_output: tensor of shape (b, h, sq, d)
            - attention_weights: tensor of shape (b, h, sq, sk)
    """
    # Compute exponential attention weights once and reuse
    training: bool = module.training if module is not None else False
    exp_attention_weights: torch.Tensor = _compute_masked_exp_attention_weights(
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        scaling=scaling,
        sparse_attention_mask=sparse_attention_mask,
        dropout=dropout,
        training=training,
        softcap=softcap,
    )

    # Prepare values by applying key-value grouping
    num_key_value_groups: int = _get_num_key_value_groups(queries, values)
    value_states: torch.Tensor = repeat_kv(values, num_key_value_groups).to(
        exp_attention_weights.dtype
    )

    # Use internal helpers with pre-computed weights
    num: torch.Tensor = _get_attention_numerator(exp_attention_weights, value_states)
    den: torch.Tensor = _get_attention_denominator(exp_attention_weights)

    num, den, exp_attention_weights = apply_sink_bias(
        num=num,
        den=den,
        exp_attention_weights=exp_attention_weights
        if return_attention_weights
        else None,
        queries=queries,
        keys=keys,
        attention_mask=attention_mask,
        sinks=sinks,
        scaling=scaling,
        softcap=softcap,
    )

    num = num.to(torch.float32)
    den = den.to(torch.float32)
    # Compute final attention output
    attention_output: torch.Tensor = (
        (num / den).to(queries.dtype).transpose(1, 2).contiguous()
    )

    if return_attention_weights:
        # Normalize exponential weights to get attention probabilities
        attention_weights: torch.Tensor = (
            exp_attention_weights.to(torch.float32) / den
        ).to(queries.dtype)

        return attention_output, attention_weights

    return attention_output
