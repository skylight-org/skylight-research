import math
from typing import Tuple

import torch
import triton
import triton.language as tl

# -------------------------------
# Kernel: Stage-1 – compute per-block partial results & log-sum-exp
# -------------------------------

@triton.jit
def _fwd_kernel_sparse_decode_stage1(
    Q,  # [B, H, D]
    K,  # [B, Kv, S, D]
    V,  # [B, Kv, S, D]
    sm_scale,  # scalar
    Sparse_List,  # [B, H, S]
    Sparse_Len,  # [B, H] – seq length per (b, h)
    Mid_O,  # [B, H, seq_block_num, D]
    Mid_O_LogExpSum,  # [B, H, seq_block_num]
    # strides – note that all strides are in *elements*
    stride_sparse_b,
    stride_sparse_h,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbb,  # K.stride(0) – batch
    stride_kh,   # K.stride(1) – kv head
    stride_ks,   # K.stride(2) – seq
    stride_vbb,
    stride_vh,
    stride_vs,
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
    """Each program instance processes (b, h, seq_block).

    Within the sequence block (<= BLOCK_SEQ tokens) we iterate in tiles of
    BLOCK_N tokens to compute numerically-stable softmax partials akin to
    Flash-Attention / Flash-Decode stage-1.
    """

    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)

    cur_kv_head = cur_head // gqa_group_size  # shared key/value head

    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Sequence length for (b, h)
    cur_seq_len_ptr = Sparse_Len + cur_batch * stride_splen_b + cur_head * stride_splen_h
    cur_seq_len = tl.load(cur_seq_len_ptr)

    # Start / end position (in sparse_list) of this sequence block
    cur_block_start = seq_start_block * BLOCK_SEQ
    cur_block_end = tl.minimum(cur_seq_len, cur_block_start + BLOCK_SEQ)

    # Pointers base for sparse_list of this head
    sparse_ptr_base = Sparse_List + cur_batch * stride_sparse_b + cur_head * stride_sparse_h

    # Load query vector (shape [D]) – no sequence dim for decode (one query)
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q)  # [D]

    # Prepare accumulators
    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    # Number of micro-blocks within this sequence block
    block_n_size = (
        tl.where(cur_block_end - cur_block_start <= 0, 0,
                  cur_block_end - cur_block_start + BLOCK_N - 1) // BLOCK_N
    )

    offs_n = cur_block_start + tl.arange(0, BLOCK_N)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n  # absolute positions inside sparse_list

        # Load token indices for these positions – mask out-of-range
        token_idx = tl.load(
            sparse_ptr_base + offs_n_new,
            mask=offs_n_new < cur_seq_len,
            other=0,
        )  # [BLOCK_N]

        # Build pointer to K/V: token_idx is [n] so broadcast with d
        base_ptr = cur_batch * stride_kbb + cur_kv_head * stride_kh
        off_k = base_ptr + token_idx[:, None] * stride_ks + offs_d[None, :]
        # Note: stride_kbs == K.stride(2) because K is [B, Kv, S, D] and we want S dimension
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_seq_len, other=0.0)
        v = tl.load(V + off_k, mask=offs_n_new[:, None] < cur_seq_len, other=0.0)

        # Attention scores
        att_value = tl.sum(q[None, :] * k, 1)  # [BLOCK_N]
        att_value *= sm_scale
        att_value = tl.where(offs_n_new < cur_seq_len, att_value, float("-inf"))

        # Numerically-stable softmax merge
        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)

        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v, axis=0)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic

    # Decide whether to store (skip if sequence length 0)
    need_store = tl.where(block_n_size == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + seq_start_block * stride_mid_os
            + offs_d
        )
        off_mid_o_logexpsum = (
            cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block
        )
        tl.store(Mid_O + off_mid_o, acc / sum_exp)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))


# -------------------------------
# Kernel: Stage-2 – reduce across sequence blocks
# identical logic to flash-decode stage-2
# -------------------------------

@triton.jit
def _fwd_kernel_sparse_decode_stage2(
    Sparse_Len,  # [B, H]
    Mid_O,  # [B, H, seq_block_num, D]
    Mid_O_LogExpSum,  # [B, H, seq_block_num]
    O,  # [B, H, D]
    stride_splen_b,
    stride_splen_h,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    stride_obs,
    stride_oh,
    stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Second stage reduction over sequence blocks (identical to Flash-Decode)."""

    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Sequence length for (b,h)
    cur_seq_len_ptr = Sparse_Len + cur_batch * stride_splen_b + cur_head * stride_splen_h
    cur_seq_len = tl.load(cur_seq_len_ptr)

    # Number of blocks covering this sequence
    block_n_size = (
        tl.where(cur_seq_len <= 0, 0, cur_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    )

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    # Precompute starting offsets into Mid tensors
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh

    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)

        new_max_logic = tl.maximum(tlogic, max_logic)

        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    # Write output
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d
    tl.store(O + off_o, acc / sum_exp)


# -------------------------------
# Python helper functions
# -------------------------------


@torch.no_grad()
def sparse_decode_stage1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_list: torch.Tensor,
    sparse_len: torch.Tensor,
    max_len_in_batch: int,
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    block_seq: int,
):
    BLOCK_SEQ = block_seq
    BLOCK_N = 16

    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}

    sm_scale = 1.0 / math.sqrt(Lk)

    batch, head_num = q.shape[0], q.shape[1]
    grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK_SEQ))

    gqa_group_size = head_num // k.shape[1]

    _fwd_kernel_sparse_decode_stage1[grid](
        q,
        k,
        v,
        sm_scale,
        sparse_list,
        sparse_len,
        mid_out,
        mid_out_logsumexp,
        sparse_list.stride(0),
        sparse_list.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),  # stride over B (kbb)
        k.stride(1),
        k.stride(2),  # stride over S (ks)
        v.stride(0),  # stride over B (vbb)
        v.stride(1),
        v.stride(2),  # stride over S (vs)
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
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK_N,
        num_warps=1,
        num_stages=2,
    )


@torch.no_grad()
def sparse_decode_stage2(
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    sparse_len: torch.Tensor,
    O: torch.Tensor,
    block_seq: int,
):
    Lk = mid_out.shape[-1]
    assert Lk in {16, 32, 64, 128}

    batch, head_num = mid_out.shape[0], mid_out.shape[1]
    grid = (batch, head_num)

    _fwd_kernel_sparse_decode_stage2[grid](
        sparse_len,
        mid_out,
        mid_out_logsumexp,
        O,
        sparse_len.stride(0),
        sparse_len.stride(1),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out.stride(3),
        mid_out_logsumexp.stride(0),
        mid_out_logsumexp.stride(1),
        mid_out_logsumexp.stride(2),
        O.stride(0),
        O.stride(1),
        O.stride(2),
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=Lk,
        num_warps=4,
        num_stages=2,
    )


def sparse_attention_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_list: torch.Tensor,
    sparse_len: torch.Tensor,
    block_seq: int = 256,
) -> torch.Tensor:
    """Triton-accelerated sparse attention (flash-decode style).

    Args follow the same convention as the reference implementation.
    Returns: Tensor [B, H, D].
    """

    assert query.is_cuda and key.is_cuda and value.is_cuda and sparse_list.is_cuda

    B, H, D = query.shape
    max_len_in_batch = int(sparse_len.max().item())

    # Allocate intermediate + output
    block_seq_num = (max_len_in_batch + block_seq - 1) // block_seq
    mid_o = torch.empty((B, H, block_seq_num, D), dtype=torch.float32, device=query.device)
    mid_o_log = torch.empty((B, H, block_seq_num), dtype=torch.float32, device=query.device)
    out = torch.empty((B, H, D), dtype=query.dtype, device=query.device)

    sparse_decode_stage1(query, key, value, sparse_list, sparse_len, max_len_in_batch, mid_o, mid_o_log, block_seq)
    sparse_decode_stage2(mid_o, mid_o_log, sparse_len, out, block_seq)

    return out


# -------------------------------
# Quick correctness test vs reference implementation
# -------------------------------


if __name__ == "__main__":
    from ref_sparse_attention_backend import ref_sparse_attention_fwd

    torch.manual_seed(0)

    B, H, D, S = 32, 32, 128, 4096
    gqa_group_size = 4
    Kv = H // gqa_group_size

    dtype = torch.float16

    q = torch.randn(B, H, D, device="cuda", dtype=dtype)
    k = torch.randn(B, Kv, S, D, device="cuda", dtype=dtype)
    v = torch.randn(B, Kv, S, D, device="cuda", dtype=dtype)

    # Build random sparse pattern
    sparse_list = torch.randint(0, S, (B, H, S), device="cuda", dtype=torch.int32)
    sparse_len = torch.randint(1, S + 1, (B, H), device="cuda", dtype=torch.int32)

    print(sparse_list[:5, :5, :10])
    print(sparse_len[:5, :5])

    # Ensure first part of list are unique indices < S (for fairness) – we'll do simple
    for b in range(B):
        for h in range(H):
            perm = torch.randperm(S, device="cuda")
            sparse_list[b, h] = perm
            sparse_len[b, h] = torch.randint(1, S + 1, (1,), device="cuda", dtype=torch.int32)

    out_ref = ref_sparse_attention_fwd(q, k, v, sparse_list, sparse_len)
    out_triton = sparse_attention_fwd(q, k, v, sparse_list, sparse_len)

    print(f"{out_ref[0, :5, :5]=}")
    print(f"{out_triton[0, :5, :5]=}")

    max_err = (out_ref - out_triton).abs().max().item()
    mean_err = (out_ref - out_triton).abs().mean().item()
    print(f"[SPARSE ATTENTION TEST] max|ref - triton| = {max_err:.6e}")
    print(f"[SPARSE ATTENTION TEST] mean|ref - triton| = {mean_err:.6e}")
    assert mean_err < 1e-4, "Triton sparse attention does not match reference!"
    print("[SPARSE ATTENTION TEST] Passed!")
