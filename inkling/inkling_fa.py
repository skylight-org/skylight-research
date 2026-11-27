"""Fused attention kernels for Inkling 2 Mini (Triton), replacing HF's eager attention.

Why: HF Inkling (transformers 5.24.1) only supports eager attention, and every prefill chunk materializes
[heads x queries x keys] tensors several times (relative distance, a hidden-state-conditioned position-bias
gather, the causal mask, the logits). That makes long-context prefill ~10x slower than it needs to be.

Inkling attention, per query token i and head h:
    logits_ij = scale * q_i.k_j + bias_h(i, d_ij)        d_ij = pos_i - pos_j, bias = table_h[i, d] for 0<=d<R else 0
    out_i     = sum_j softmax_j(logits_i, extra logit sink_hi) v_j      (the sink takes mass, has no value)
`table` is the per-token bias-vs-distance profile [B, H, Q, R] that `Inkling2LoopRelativePositionBias`
builds *before* its gather; we keep it ("lazy bias") and apply it inside the kernel, only on tiles that
touch the R=1024 diagonal band.

  * `fa_prefill`   multi-query flash attention: GQA, causal with query/key position offsets, optional
                   sliding window, in-kernel bias band, per-token sinks.  Used for context prefill (all 48
                   visits) and for dense rows of multi-token question chunks.
  * `fa_decode`    flash-decoding for one query per row, split over keys, valid length read from device
                   memory (CUDA-graph safe), bias band, sinks.  Used for dense rows in graphed decode.
  * `install(model)`  registers the "inkling_fa" attention + mask functions and the lazy-bias patches.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import triton
import triton.language as tl
from transformers.models.inkling2_loop import modeling_inkling2_loop as M

LOG2E = 1.4426950408889634
STATE: Dict[str, object] = {"lazy": False, "row_hook": None, "decode_hook": None}


# ----------------------------------------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------------------------------------
@triton.jit
def _fa_prefill_kernel(Q, K, V, TB, SNK, O,
                       sqb, sqh, sqm, skb, skh, skn, svb, svh, svn,
                       stb, sth, stm, ssb, ssh, ssm, sob, som, soh,
                       HQ, M_Q, N_K, q_off, kv_off, qk_scale,
                       G: tl.constexpr, R: tl.constexpr, WINDOW: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // HQ
    h = pid_bh % HQ
    hk = h // G
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_ok = offs_m < M_Q
    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :], mask=m_ok[:, None], other=0.0)
    qpos = q_off + offs_m
    q_first = q_off + pid_m * BLOCK_M
    q_last = tl.minimum(q_first + BLOCK_M, q_off + M_Q) - 1
    hi = tl.minimum(q_last - kv_off + 1, N_K)
    lo = 0
    if WINDOW > 0:
        lo = tl.maximum(q_first - (WINDOW - 1) - kv_off, 0)
        lo = (lo // BLOCK_N) * BLOCK_N
    band = q_first - (R - 1) - kv_off
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    kbase = K + b * skb + hk * skh
    vbase = V + b * svb + hk * svh
    tbase = TB + b * stb + h * sth + offs_m[:, None] * stm
    # fast path: whole tiles at distance >= R from every query of the block -> no mask, no bias
    fast_hi = lo
    if WINDOW == 0:
        fast_hi = tl.minimum(tl.maximum((band // BLOCK_N) * BLOCK_N, lo), hi)
    for start in range(lo, fast_hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        k = tl.load(kbase + offs_n[None, :] * skn + offs_d[:, None])
        s = tl.dot(q, k) * qk_scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vbase + offs_n[:, None] * svn + offs_d[None, :])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    for start in range(fast_hi, hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_ok = offs_n < N_K
        k = tl.load(kbase + offs_n[None, :] * skn + offs_d[:, None], mask=n_ok[None, :], other=0.0)
        s = tl.dot(q, k) * qk_scale
        dist = qpos[:, None] - (kv_off + offs_n)[None, :]
        ok = (dist >= 0) & n_ok[None, :] & m_ok[:, None]
        if WINDOW > 0:
            ok = ok & (dist < WINDOW)
        if start + BLOCK_N > band:
            inb = ok & (dist < R)
            tb = tl.load(tbase + dist, mask=inb, other=0.0).to(tl.float32)
            s = s + tb * 1.4426950408889634
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vbase + offs_n[:, None] * svn + offs_d[None, :], mask=n_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    sk = tl.load(SNK + b * ssb + h * ssh + offs_m * ssm, mask=m_ok, other=0.0).to(tl.float32) * 1.4426950408889634
    m_f = tl.maximum(m_i, sk)
    a = tl.exp2(m_i - m_f)
    l_f = l_i * a + tl.exp2(sk - m_f)
    o = acc * (a / l_f)[:, None]
    tl.store(O + b * sob + offs_m[:, None] * som + h * soh + offs_d[None, :], o.to(O.dtype.element_ty),
             mask=m_ok[:, None])


@triton.jit
def _fa_decode_split_kernel(Q, K, V, PB, LEN, ROWS, OP, ML,
                            sqb, sqh, skb, skh, skn, svb, svh, svn, spb, sph,
                            n_split, qk_scale,
                            HQ: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, R: tl.constexpr,
                            BLOCK_N: tl.constexpr, D: tl.constexpr):
    hk = tl.program_id(0)
    sp = tl.program_id(1)
    ri = tl.program_id(2)
    b = tl.load(ROWS + ri)
    L = tl.load(LEN).to(tl.int32)               # valid keys, current token included
    per = tl.cdiv(tl.cdiv(L, n_split), BLOCK_N) * BLOCK_N
    start = sp * per
    end = tl.minimum(start + per, L)
    offs_g = tl.arange(0, GP)
    g_ok = offs_g < G
    offs_d = tl.arange(0, D)
    heads = hk * G + offs_g
    q = tl.load(Q + b * sqb + heads[:, None] * sqh + offs_d[None, :], mask=g_ok[:, None], other=0.0)
    band = L - R
    m_i = tl.full([GP], float("-inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    kbase = K + b * skb + hk * skh
    vbase = V + b * svb + hk * svh
    for st in range(start, end, BLOCK_N):
        offs_n = st + tl.arange(0, BLOCK_N)
        n_ok = offs_n < end
        k = tl.load(kbase + offs_n[None, :] * skn + offs_d[:, None], mask=n_ok[None, :], other=0.0)
        s = tl.dot(q, k) * qk_scale
        if st + BLOCK_N > band:
            pm = g_ok[:, None] & n_ok[None, :] & (offs_n[None, :] >= band)
            pb = tl.load(PB + b * spb + heads[:, None] * sph + offs_n[None, :], mask=pm, other=0.0)
            s = s + pb.to(tl.float32) * 1.4426950408889634
        s = tl.where(n_ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vbase + offs_n[:, None] * svn + offs_d[None, :], mask=n_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    base = (ri * HQ + heads) * n_split + sp
    tl.store(OP + base[:, None] * D + offs_d[None, :], acc, mask=g_ok[:, None])
    tl.store(ML + base * 2, m_i, mask=g_ok)
    tl.store(ML + base * 2 + 1, l_i, mask=g_ok)


@triton.jit
def _fa_decode_merge_kernel(OP, ML, SNK, ROWS, O, ssb, ssh, sob, soh, n_split,
                            HQ: tl.constexpr, SPLIT_P: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)                      # ri * HQ + h
    ri = pid // HQ
    h = pid % HQ
    b = tl.load(ROWS + ri)
    offs_s = tl.arange(0, SPLIT_P)
    s_ok = offs_s < n_split
    offs_d = tl.arange(0, D)
    m = tl.load(ML + (pid * n_split + offs_s) * 2, mask=s_ok, other=float("-inf"))
    l = tl.load(ML + (pid * n_split + offs_s) * 2 + 1, mask=s_ok, other=0.0)
    acc = tl.load(OP + (pid * n_split + offs_s)[:, None] * D + offs_d[None, :], mask=s_ok[:, None], other=0.0)
    sk = tl.load(SNK + b * ssb + h * ssh).to(tl.float32) * 1.4426950408889634
    mx = tl.maximum(tl.max(m, 0), sk)
    w = tl.exp2(m - mx)
    num = tl.sum(acc * w[:, None], 0)
    den = tl.sum(l * w, 0) + tl.exp2(sk - mx)
    tl.store(O + b * sob + h * soh + offs_d, (num / den).to(O.dtype.element_ty))


# ----------------------------------------------------------------------------------------------------------
# launchers
# ----------------------------------------------------------------------------------------------------------
def fa_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, table: torch.Tensor, sinks: torch.Tensor,
               q_off: int, kv_off: int, scale: float, window: int = 0) -> torch.Tensor:
    """q [B,Hq,M,D] (any strides, last dim contiguous), k/v [B,Hkv,N,D], table [B,Hq,M,R], sinks [B,Hq,M]
    -> out [B, M, Hq, D] (the layout eager attention returns)."""
    B, HQ, MQ, D = q.shape
    HKV, NK = k.shape[1], k.shape[2]
    R = table.shape[-1]
    fixed = []
    for name in ("q", "k", "v", "table", "sinks"):
        t = locals()[name]
        if t.stride(-1) != 1:
            fixed.append((name, tuple(t.shape), t.stride()))
    if fixed:
        if not STATE.get("_warned"):
            print("fa_prefill: non-contiguous inputs", fixed, flush=True)
            STATE["_warned"] = True
        q, k, v, table, sinks = (t if t.stride(-1) == 1 else t.contiguous() for t in (q, k, v, table, sinks))
    out = torch.empty(B, MQ, HQ, D, dtype=q.dtype, device=q.device)
    bm = 128 if MQ >= 128 else 64
    grid = (triton.cdiv(MQ, bm), B * HQ)
    _fa_prefill_kernel[grid](q, k, v, table, sinks, out,
                             q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                             v.stride(0), v.stride(1), v.stride(2),
                             table.stride(0), table.stride(1), table.stride(2),
                             sinks.stride(0), sinks.stride(1), sinks.stride(2),
                             out.stride(0), out.stride(1), out.stride(2),
                             HQ, MQ, NK, int(q_off), int(kv_off), float(scale) * LOG2E,
                             G=HQ // HKV, R=R, WINDOW=int(window), BLOCK_M=bm, BLOCK_N=64, D=D,
                             num_warps=8 if bm == 128 else 4, num_stages=2)
    return out


class FlashDecode:
    """Graph-safe dense decode for a fixed set of rows of a static cache layer (one query token per row)."""

    def __init__(self, rows, n_heads: int, head_dim: int, device, n_split: int = 32, dtype=torch.bfloat16):
        self.rows = torch.tensor(list(rows), dtype=torch.int32, device=device)
        self.n_rows, self.HQ, self.D, self.n_split = len(rows), n_heads, head_dim, n_split
        self.op = torch.empty(self.n_rows * n_heads * n_split, head_dim, dtype=torch.float32, device=device)
        self.ml = torch.empty(self.n_rows * n_heads * n_split, 2, dtype=torch.float32, device=device)

    def __call__(self, q, k, v, pb, length, sinks, out, scale: float):
        """q [B,Hq,D] view, k/v [B,Hkv,max_len,D], pb [B,Hq,max_len] (bias per key slot), length: device
        int tensor (#valid keys), sinks [B,Hq], out [B,Hq,D] written only at `rows`."""
        HKV = k.shape[1]
        G = self.HQ // HKV
        _fa_decode_split_kernel[(HKV, self.n_split, self.n_rows)](
            q, k, v, pb, length, self.rows, self.op, self.ml,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1),
            v.stride(2), pb.stride(0), pb.stride(1), self.n_split, float(scale) * LOG2E,
            HQ=self.HQ, G=G, GP=16, R=1024, BLOCK_N=64, D=self.D, num_warps=4, num_stages=2)
        _fa_decode_merge_kernel[(self.n_rows * self.HQ,)](
            self.op, self.ml, sinks, self.rows, out, sinks.stride(0), sinks.stride(1), out.stride(0),
            out.stride(1), self.n_split, HQ=self.HQ, SPLIT_P=triton.next_power_of_2(self.n_split), D=self.D)
        return out


# ----------------------------------------------------------------------------------------------------------
# lazy relative distance / position bias
# ----------------------------------------------------------------------------------------------------------
class RelDist:
    __slots__ = ("q_len", "kv_len", "q_off", "kv_off")

    def __init__(self, q_len, kv_len, q_off, kv_off):
        self.q_len, self.kv_len, self.q_off, self.kv_off = q_len, kv_len, q_off, kv_off


class LazyBias:
    """What the position-bias module returns in lazy mode: the per-token profile, not the gathered tensor."""
    __slots__ = ("table", "rd")

    def __init__(self, table, rd):
        self.table, self.rd = table, rd

    def materialize(self) -> torch.Tensor:
        rd = self.rd
        dist = ((torch.arange(rd.q_len, device=self.table.device) + rd.q_off)[:, None]
                - (torch.arange(rd.kv_len, device=self.table.device) + rd.kv_off)[None, :])
        R = self.table.shape[-1]
        idx = dist.clamp(0, R - 1).expand(*self.table.shape[:2], -1, -1)
        return self.table.gather(-1, idx).masked_fill((dist < 0) | (dist >= R), 0.0)


_ORIG: Dict[str, Callable] = {}


def _lazy_relative_distance(inputs_embeds, cache, is_sliding):
    if not STATE["lazy"]:
        return _ORIG["rd"](inputs_embeds, cache, is_sliding)
    q_length = inputs_embeds.shape[1]
    if cache is None:
        return RelDist(q_length, q_length, 0, 0)
    if not is_sliding:
        layer_idx = M.find_first_full_attention_index(cache)
    elif True in cache.is_sliding:
        layer_idx = cache.is_sliding.index(True)
    else:
        layer_idx = 0
    kv_length, kv_offset = cache.get_mask_sizes(q_length, layer_idx)
    q_offset = cache.get_query_offset(layer_idx)
    return RelDist(q_length, int(kv_length), int(q_offset), int(kv_offset))


def _lazy_bias_forward(self, hidden_states, relative_distance, position_ids):
    if not isinstance(relative_distance, RelDist):
        return _ORIG["bias"](self, hidden_states, relative_distance, position_ids)
    relative_states = self.linear(hidden_states).view(*hidden_states.shape[:-1], self.num_heads, -1)
    relative_states = self.norm(relative_states, position_ids)
    table = (relative_states @ self.proj).transpose(1, 2)          # [B, H, Q, R]
    return LazyBias(table.contiguous(), relative_distance)


def _no_mask(*args, **kwargs):
    return None


def attention_fn(module, query, key, value, attention_mask, scaling, dropout=0.0, position_bias=None,
                 sinks=None, sliding_window=None, **kwargs):
    kwargs.pop("sparse_meta_data", None)
    if isinstance(position_bias, LazyBias):
        hook = STATE["row_hook"]
        if hook is not None and isinstance(module, M.Inkling2LoopFullAttention):
            res = hook(module, query, key, value, position_bias, sinks, scaling)
            if res is not None:
                return res, None
        rd = position_bias.rd
        if key.shape[2] != rd.kv_len:
            raise RuntimeError(f"inkling_fa: keys {tuple(key.shape)} but mask sizes say kv_len={rd.kv_len} "
                               f"(q_off={rd.q_off}, kv_off={rd.kv_off}, window={sliding_window})")
        out = fa_prefill(query, key, value, position_bias.table, sinks[..., 0], rd.q_off, rd.kv_off, scaling,
                         window=int(sliding_window or 0))
        return out, None
    hook = STATE["decode_hook"]
    if hook is not None and isinstance(module, M.Inkling2LoopFullAttention) and query.shape[2] == 1:
        res = hook(module, query, key, value, position_bias, sinks, scaling)
        if res is not None:
            return res, None
    if isinstance(module, (M.Inkling2LoopFullAttention, M.Inkling2LoopSlidingAttention)) and attention_mask is None:
        raise RuntimeError("inkling_fa: token attention reached eager without a mask")
    return M.eager_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout,
                                     position_bias=position_bias, sinks=sinks, **kwargs)


def install(model: torch.nn.Module) -> None:
    """Register 'inkling_fa' (prefill / lazy mode) and 'inkling_fa_eager' (decode: masks built as usual)."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from inkling_sah import install_visit_tracking

    if "rd" not in _ORIG:
        _ORIG["rd"] = M.get_relative_distance
        _ORIG["bias"] = M.Inkling2LoopRelativePositionBias.forward
        M.get_relative_distance = _lazy_relative_distance
        M.Inkling2LoopRelativePositionBias.forward = _lazy_bias_forward
    install_visit_tracking(model)
    ALL_ATTENTION_FUNCTIONS.register("inkling_fa", attention_fn)
    ALL_ATTENTION_FUNCTIONS.register("inkling_fa_eager", attention_fn)
    ALL_MASK_ATTENTION_FUNCTIONS.register("inkling_fa", _no_mask)
    ALL_MASK_ATTENTION_FUNCTIONS.register("inkling_fa_eager", eager_mask)
    set_mode(model, "prefill")


def set_mode(model: torch.nn.Module, mode: str) -> None:
    """'prefill': lazy bias + fused kernel, no masks. 'decode': HF masks/bias (q_len=1), hooks for full layers."""
    name = "inkling_fa" if mode == "prefill" else "inkling_fa_eager"
    STATE["lazy"] = mode == "prefill"
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = name
