import math
from typing import Tuple

import torch

from ref_sparse_attention_backend import ref_sparse_attention_fwd


def _get_gqa_group_size(H: int, Kv: int) -> int:
    assert H % Kv == 0, "H must be divisible by Kv (H // gqa)"
    return H // Kv


# @torch.no_grad()
# def ref_sparse_attention_fwd(
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     sparse_list: torch.Tensor,
#     sparse_len: torch.Tensor,
# ) -> torch.Tensor:
#     """Reference sparse attention (no bias) – same as earlier helper.

#     Args are identical to the previous spec.
#     """
#     assert query.ndim == 3 and key.ndim == 4 and value.ndim == 4

#     B, H, D = query.shape
#     _, Kv, S, _ = key.shape

#     gqa_group_size = _get_gqa_group_size(H, Kv)
#     sm_scale = 1.0 / math.sqrt(D)

#     out = torch.empty_like(query)

#     for b in range(B):
#         for h in range(H):
#             kv_h = h // gqa_group_size
#             L = int(sparse_len[b, h].item())
#             if L == 0:
#                 out[b, h].zero_()
#                 continue

#             idx = sparse_list[b, h, :L].to(dtype=torch.long, device=query.device)
#             k_vec = key[b, kv_h].index_select(0, idx).to(torch.float32)
#             v_vec = value[b, kv_h].index_select(0, idx).to(torch.float32)
#             q_vec = query[b, h].to(torch.float32)

#             att_logits = (k_vec * q_vec).sum(dim=-1) * sm_scale  # [L]
#             att_weights = torch.softmax(att_logits, dim=-1)
#             out[b, h] = (att_weights.unsqueeze(-1) * v_vec).sum(dim=0).to(query.dtype)

#     return out


@torch.no_grad()
def ref_bias_sparse_attention_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_list: torch.Tensor,
    sparse_len: torch.Tensor,
    weight_list: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of *biased* sparse attention.

    The weight_list supplies a per-(b,h,token) positive weight w. The attention
    weights become w * exp(q·k) / Σ w * exp(q·k).
    """

    assert query.ndim == 3 and key.ndim == 4 and value.ndim == 4
    assert weight_list.shape == sparse_list.shape, "weight_list must be [B,H,S]"

    B, H, D = query.shape
    _, Kv, S, _ = key.shape

    gqa_group_size = _get_gqa_group_size(H, Kv)
    sm_scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(query)

    for b in range(B):
        for h in range(H):
            kv_h = h // gqa_group_size
            L = int(sparse_len[b, h].item())
            if L == 0:
                out[b, h].zero_()
                continue

            idx = sparse_list[b, h, :L].to(dtype=torch.long, device=query.device)
            k_vec = key[b, kv_h].index_select(0, idx)
            v_vec = value[b, kv_h].index_select(0, idx)
            w_vec = weight_list[b, h].index_select(0, idx).to(torch.float32)  # [L]

            # Ensure positivity to avoid log(-)
            w_vec = torch.clamp_min(w_vec, 1e-30)

            q_vec = query[b, h].to(torch.float32)
            att_logits = (k_vec * q_vec).sum(dim=-1).to(torch.float32) * sm_scale  # [L]

            # Incorporate weight as additive bias in log-space
            logits_with_bias = att_logits + torch.log(w_vec)
            att_weights = torch.softmax(logits_with_bias, dim=-1).to(query.dtype)
            out_vec = (att_weights.unsqueeze(-1) * v_vec).sum(dim=0)
            out[b, h] = out_vec.to(query.dtype)

    return out


if __name__ == "__main__":
    # Simple correctness check: when weight == 1, biased == un-biased
    torch.manual_seed(0)

    B, H, D, S = 32, 32, 128, 4096
    gqa_group_size = 2
    Kv = H // gqa_group_size

    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, Kv, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, Kv, S, D, device="cuda", dtype=torch.float16)

    # Sparse pattern
    sparse_list = torch.empty((B, H, S), dtype=torch.long, device="cuda")
    sparse_len = torch.empty((B, H), dtype=torch.long, device="cuda")
    for b in range(B):
        for h in range(H):
            perm = torch.randperm(S, device="cuda")
            sparse_list[b, h] = perm
            sparse_len[b, h] = torch.randint(1, S + 1, (1,), device="cuda")

    # All-ones weight
    weight_list = torch.ones((B, H, S), device="cuda", dtype=torch.float16)

    out_ref = ref_sparse_attention_fwd(q, k, v, sparse_list, sparse_len)
    out_bias = ref_bias_sparse_attention_fwd(q, k, v, sparse_list, sparse_len, weight_list)

    max_err = (out_ref - out_bias).abs().max().item()
    print(f"[BIAS REF TEST] max|no-bias - bias(1)| = {max_err:.6e}")
    assert max_err < 2e-3, "Biased sparse attention (w=1) should equal un-biased!"
    print("[BIAS REF TEST] Passed.") 