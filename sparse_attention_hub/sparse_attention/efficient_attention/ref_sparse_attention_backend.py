import torch
import math


def ref_sparse_attention_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_list: torch.Tensor,
    sparse_len: torch.Tensor,
):
    """Reference implementation of sparse attention flash-decoding.

    Args:
        query:        Tensor of shape [B, H, D]
        key:          Tensor of shape [B, H // gqa, S, D]
        value:        Tensor of shape [B, H // gqa, S, D]
        sparse_list:  Tensor of shape [B, H, S] that stores the token indices to
                      attend to. Only the first ``sparse_len[b, h]`` entries of
                      the last dimension are valid.
        sparse_len:   Tensor of shape [B, H] giving the valid length in
                      ``sparse_list`` for every (b, h).

    Returns:
        Tensor of shape [B, H, D] – the attention output for each query head.

    This is a *slow* but very clear reference used for correctness checks. It
    supports grouped-query attention (GQA) where several query heads share the
    same key / value head.  Setting ``gqa = 1`` reduces to standard multi-head
    attention (MHA).
    """

    assert query.ndim == 3, "query must be [B, H, D]"
    assert key.ndim == value.ndim == 4, "key/value must be [B, Kv, S, D]"

    B, H, D = query.shape
    _, Kv, S, _ = key.shape
    device = query.device
    dtype = query.dtype

    # Infer group size from the shapes.  gqa == number of Q heads per KV head.
    gqa_group_size = H // Kv
    assert gqa_group_size * Kv == H, "H must be divisible by Kv (H//gqa)"

    sm_scale = 1.0 / math.sqrt(D)

    # Output tensor
    out = torch.empty_like(query)

    # Iterate over batch and heads – this is a slow reference so clarity beats speed.
    for b in range(B):
        for h in range(H):
            kv_h = h // gqa_group_size  # which KV head this Q head should use

            # Number of tokens that this (b, h) attends to
            L = int(sparse_len[b, h].item())
            if L == 0:
                # Edge-case: no tokens attended -> return zeros (like softmax over empty set)
                out[b, h].zero_()
                continue

            # The token indices we actually attend to (shape [L])
            idx = sparse_list[b, h, :L].to(dtype=torch.long, device=device)

            # Gather the key/value vectors we need (shape [L, D])
            k_vec = key[b, kv_h].index_select(0, idx)  # [L, D]
            v_vec = value[b, kv_h].index_select(0, idx)  # [L, D]

            # Attention logits – [L]
            q_vec = query[b, h]  # [D]
            attn_logits = (k_vec * q_vec).sum(dim=-1).to(torch.float32) * sm_scale

            attn_weights = torch.softmax(attn_logits, dim=-1).to(query.dtype)  # [L]
            out[b, h] = torch.sum(attn_weights.unsqueeze(-1) * v_vec, dim=0)

    return out


def ref_dense_attention_fwd(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    """Vectorised dense attention (reference).

    We replicate key / value along the head dimension so each query head has its
    own slice, then compute attention in batch using two Einsums – this is
    clearer and avoids Python-side loops.
    """

    assert query.ndim == 3 and key.ndim == 4 and value.ndim == 4

    B, H, D = query.shape
    _, Kv, S, _ = key.shape

    gqa_group_size = H // Kv  # heads per KV group
    sm_scale = 1.0 / math.sqrt(D)

    # Repeat key/value so we have one slice per query head: [B, H, S, D]
    key_rep = key.repeat_interleave(gqa_group_size, dim=1)
    value_rep = value.repeat_interleave(gqa_group_size, dim=1)

    # Compute attention logits: [B, H, S]
    attn_logits = torch.einsum("bhd,bhsd->bhs", query, key_rep).to(torch.float32) * sm_scale
    attn_weights = torch.softmax(attn_logits, dim=-1).to(query.dtype)

    # Output: [B, H, D]
    out = torch.einsum("bhs,bhsd->bhd", attn_weights, value_rep)
    return out


if __name__ == "__main__":
    # Simple self-test: when every token is attended, sparse == dense.
    torch.manual_seed(0)
    torch_dtype = torch.float16

    B, H, D, S = 32, 32, 128, 4096
    gqa_group_size = 4  # change as you like – 1 corresponds to MHA
    Kv = H // gqa_group_size

    query = torch.randn(B, H, D, device="cuda", dtype=torch_dtype)
    key = torch.randn(B, Kv, S, D, device="cuda", dtype=torch_dtype)
    value = torch.randn(B, Kv, S, D, device="cuda", dtype=torch_dtype)

    # Build full sparse_list / sparse_len that cover ALL tokens
    sparse_list = torch.arange(S, device="cuda").view(1, 1, S).repeat(B, H, 1)
    sparse_len = torch.full((B, H), S, dtype=torch.long, device="cuda")

    out_sparse = ref_sparse_attention_fwd(query, key, value, sparse_list, sparse_len)
    out_dense = ref_dense_attention_fwd(query, key, value)

    max_abs_err = (out_sparse - out_dense).abs().max().item()
    mean_abs_err = (out_sparse - out_dense).abs().mean().item()
    print(f"[TEST] mean|sparse - dense| = {(out_sparse - out_dense).abs().mean().item():.6e}")
    print(f"[TEST] max|sparse - dense| = {max_abs_err:.6e}")
    # Assert the two results are (almost) identical – tolerance 1e-4 in fp32.
    assert mean_abs_err < 1e-4, "Sparse and dense results differ!"

    print("[TEST] Passed – sparse attention matches dense attention when all tokens are attended.") 