import math

import torch
import triton
import triton.language as tl

from sparse_attention_backend import (
    sparse_decode_stage2 as _sparse_decode_stage2,  # stage-2 kernel is weight-agnostic
)

# -----------------------------------------------------------------------------
# Stage-1 kernel – incorporate per-token weight (bias)
# -----------------------------------------------------------------------------


@triton.jit
def _fwd_kernel_bias_sparse_decode_stage1(
    Q,
    K,
    V,
    sm_scale,
    Sparse_List,  # [B, H, S]
    Sparse_Len,  # [B, H]
    Weight_List,  # [B, H, S]
    Mid_O,
    Mid_O_LogExpSum,
    # strides (all element-wise)
    stride_sparse_b,
    stride_sparse_h,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbb,
    stride_kh,
    stride_ks,
    stride_vbb,
    stride_vh,
    stride_vs,
    stride_weight_b,
    stride_weight_h,
    stride_weight_s,
    stride_splen_b,
    stride_splen_h,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    gqa_group_size: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_block_id = tl.program_id(2)

    cur_kv_head = cur_head // gqa_group_size

    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Sequence length of this (b,h)
    cur_seq_len_ptr = Sparse_Len + cur_batch * stride_splen_b + cur_head * stride_splen_h
    cur_seq_len = tl.load(cur_seq_len_ptr)

    block_start = seq_block_id * BLOCK_SEQ
    block_end = tl.minimum(cur_seq_len, block_start + BLOCK_SEQ)

    # Base pointers
    sparse_ptr_base = Sparse_List + cur_batch * stride_sparse_b + cur_head * stride_sparse_h
    weight_ptr_base = Weight_List + cur_batch * stride_weight_b + cur_head * stride_weight_h

    # Load query
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q)

    sum_exp = 0.0
    max_l = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    block_n_size = (
        tl.where(block_end - block_start <= 0, 0, block_end - block_start + BLOCK_N - 1)
        // BLOCK_N
    )

    offs_n = block_start + tl.arange(0, BLOCK_N)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        token_idx = tl.load(
            sparse_ptr_base + offs_n_new,
            mask=offs_n_new < cur_seq_len,
            other=0,
        )
        weight_val = tl.load(
            weight_ptr_base + token_idx * stride_weight_s,
            mask=offs_n_new < cur_seq_len,
            other=0.0,
        ).to(tl.float32)
        weight_val = tl.where(weight_val > 0.0, weight_val, 1e-30)

        base_k_ptr = cur_batch * stride_kbb + cur_kv_head * stride_kh
        off_k = base_k_ptr + token_idx[:, None] * stride_ks + offs_d[None, :]
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_seq_len, other=0.0)
        v = tl.load(V + off_k, mask=offs_n_new[:, None] < cur_seq_len, other=0.0)

        att_val = tl.sum(q[None, :] * k, axis=1) * sm_scale  # [BLOCK_N]
        att_val = att_val + tl.log(weight_val)
        att_val = tl.where(offs_n_new < cur_seq_len, att_val, float("-inf"))

        cur_max = tl.max(att_val, axis=0)
        new_max = tl.maximum(cur_max, max_l)

        exp_l = tl.exp(att_val - new_max)
        scale = tl.exp(max_l - new_max)

        acc *= scale
        acc += tl.sum(exp_l[:, None] * v, axis=0)

        sum_exp = sum_exp * scale + tl.sum(exp_l, axis=0)
        max_l = new_max

    need_store = tl.where(block_n_size == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + seq_block_id * stride_mid_os
            + offs_d
        )
        off_log = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_block_id
        tl.store(Mid_O + off_mid, acc / sum_exp)
        tl.store(Mid_O_LogExpSum + off_log, max_l + tl.log(sum_exp))


# -----------------------------------------------------------------------------
# Python wrappers
# -----------------------------------------------------------------------------


@torch.no_grad()
def bias_sparse_decode_stage1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_list: torch.Tensor,
    sparse_len: torch.Tensor,
    weight_list: torch.Tensor,
    max_len_in_batch: int,
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    block_seq: int,
):
    BLOCK_N = 16
    BLOCK_SEQ = block_seq

    D = q.shape[-1]
    assert D in {16, 32, 64, 128}
    assert k.shape[-1] == D

    sm_scale = 1.0 / math.sqrt(D)

    B, H = q.shape[0], q.shape[1]
    grid = (B, H, triton.cdiv(max_len_in_batch, BLOCK_SEQ))

    gqa_group_size = H // k.shape[1]

    _fwd_kernel_bias_sparse_decode_stage1[grid](
        q,
        k,
        v,
        sm_scale,
        sparse_list,
        sparse_len,
        weight_list,
        mid_out,
        mid_out_logsumexp,
        sparse_list.stride(0),
        sparse_list.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        weight_list.stride(0),
        weight_list.stride(1),
        weight_list.stride(2),
        sparse_len.stride(0),
        sparse_len.stride(1),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out.stride(3),
        mid_out_logsumexp.stride(0),
        mid_out_logsumexp.stride(1),
        mid_out_logsumexp.stride(2),
        gqa_group_size,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=D,
        BLOCK_N=BLOCK_N,
        num_warps=1,
        num_stages=2,
    )


@torch.no_grad()
def bias_sparse_attention_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_list: torch.Tensor,
    sparse_len: torch.Tensor,
    weight_list: torch.Tensor,
    block_seq: int = 256,
):
    """Triton-accelerated biased sparse attention forward."""

    assert all(t.is_cuda for t in [query, key, value, sparse_list, weight_list])

    B, H, D = query.shape
    max_len = int(sparse_len.max().item())

    blk_num = (max_len + block_seq - 1) // block_seq
    mid_out = torch.empty((B, H, blk_num, D), dtype=torch.float32, device=query.device)
    mid_log = torch.empty((B, H, blk_num), dtype=torch.float32, device=query.device)
    out = torch.empty((B, H, D), dtype=query.dtype, device=query.device)

    bias_sparse_decode_stage1(
        query,
        key,
        value,
        sparse_list,
        sparse_len,
        weight_list,
        max_len,
        mid_out,
        mid_log,
        block_seq,
    )

    # stage-2 (weight-independent)
    _sparse_decode_stage2(mid_out, mid_log, sparse_len, out, block_seq)
    return out


# -----------------------------------------------------------------------------
# Quick verification
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from ref_bias_sparse_attention_backend import ref_bias_sparse_attention_fwd

    torch.manual_seed(0)

    B, H, D, S = 32, 32, 128, 4096
    gqa = 4
    Kv = H // gqa

    dtype = torch.float16

    q = torch.randn(B, H, D, device="cuda", dtype=dtype)
    k = torch.randn(B, Kv, S, D, device="cuda", dtype=dtype)
    v = torch.randn(B, Kv, S, D, device="cuda", dtype=dtype)

    sparse_list = torch.empty((B, H, S), dtype=torch.int32, device="cuda")
    sparse_len = torch.empty((B, H), dtype=torch.int32, device="cuda")
    for b in range(B):
        for h in range(H):
            perm = torch.randperm(S, device="cuda", dtype=torch.int32)
            sparse_list[b, h] = perm
            sparse_len[b, h] = torch.randint(1, S + 1, (1,), device="cuda", dtype=torch.int32)

    weight_list = torch.rand((B, H, S), device="cuda", dtype=dtype) * 2.0 + 0.5  # positive weights
    # weight_list = torch.ones((B, H, S), device="cuda", dtype=dtype)

    print(f"{weight_list[0, :5, :5]=}")

    out_ref = ref_bias_sparse_attention_fwd(q, k, v, sparse_list, sparse_len, weight_list)
    out_triton = bias_sparse_attention_fwd(q, k, v, sparse_list, sparse_len, weight_list)

    print(f"{out_ref[0, :5, :5]=}")
    print(f"{out_triton[0, :5, :5]=}")

    max_err = (out_ref - out_triton).abs().max().item()
    mean_err = (out_ref - out_triton).abs().mean().item()
    print(f"[SPARSE ATTENTION TEST] max|ref - triton| = {max_err:.6e}")
    print(f"[SPARSE ATTENTION TEST] mean|ref - triton| = {mean_err:.6e}")
    assert mean_err < 1e-4, "Triton sparse attention does not match reference!"
    print("[SPARSE ATTENTION TEST] Passed!")