"""Stratified sparse attention (SSA = PQ importance sampling, skylight-org/SSA-Paper-Code) for Inkling 2 Mini.

Reference algorithm (`PQImportance`, per query row, on Inkling's 8 full-attention visits):
  sink(128) + local window(128) dense;  remaining keys scored by PQ (2 sub-vectors x 64 k-means centroids,
  i.e. pq_group_factor=2, pq_bits=6, 10 iterations, euclidean) as  logit = scaling * q.k_pq (+ Inkling's
  position bias, which is part of the true logit and nonzero only within 1024 tokens);
  stratum 1: top `heavy` keys, weight 1;  stratum 2: `m` draws *with replacement* from softmax(logit/T) over
  the rest, each drawn key weighted 1/pi with pi = 1-(1-p)^m.  Attention = weighted numerator / weighted
  denominator (+ Inkling's per-token sink logit).

What makes it fast here: with 2 x 6-bit codes a key's PQ score depends only on its code *pair*, one of
4096 values.  Keys far from every query (distance >= 1024, so no position bias) are bucketed by code pair
once after prefill (counting sort -> CSR).  Then for each query/head
  * scoring is 4096 pair logits (a 2x64 lookup table) + exact per-key logits for the <= few-thousand keys
    near the end (the 1024-token bias band, the question and generated tokens),
  * top-k and the multinomial draws run over those ~5-8k weighted items (bucket count x exp(logit)), and a
    draw that lands in a bucket picks a key uniformly inside it -- the same distribution as drawing keys,
so selection costs O(4096 + band + budget), independent of context length; only the selected ~2% of K/V
is ever read, by the skylight `sparse_optimized` kernel with per-key weights.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch

from sparse_optimized import BatchDecodeWithPagedKVCacheWrapper as SparseDecode

PAGE = 16
SINK, WINDOW = 128, 128
NSUB, NC = 2, 64
NPAIR = NC * NC
R_BIAS = 1024
LN2 = math.log(2.0)
MIN_PI = 1e-4


def kmeans_batched(X: torch.Tensor, k: int, iters: int, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lloyd k-means on P independent problems. X [P, N, d] fp32 -> centroids [P, k, d], codes [P, N]."""
    P, N, d = X.shape
    idx = torch.stack([torch.randperm(N, generator=gen, device=X.device)[:k] for _ in range(P)])
    C = X.gather(1, idx[..., None].expand(-1, -1, d)).clone()
    xx = (X * X).sum(-1, keepdim=True)
    ones = torch.ones(P, N, device=X.device)
    for _ in range(iters):
        codes = (xx - 2 * X @ C.transpose(1, 2) + (C * C).sum(-1)[:, None, :]).argmin(-1)
        sums = torch.zeros(P, k, d, device=X.device).scatter_add_(1, codes[..., None].expand(-1, -1, d), X)
        cnt = torch.zeros(P, k, device=X.device).scatter_add_(1, codes, ones)
        rnd = X.gather(1, torch.randint(0, N, (P, k), generator=gen, device=X.device)[..., None].expand(-1, -1, d))
        C = torch.where((cnt > 0)[..., None], sums / cnt.clamp(min=1)[..., None], rnd)
    codes = (xx - 2 * X @ C.transpose(1, 2) + (C * C).sum(-1)[:, None, :]).argmin(-1)
    return C, codes


class SSAContext:
    """Per full-attention visit, built once from the (densely prefilled) context keys; shared by SSA rows."""

    def __init__(self, keys: torch.Tensor, L0: int, iters: int = 10, seed: int = 0) -> None:
        Hkv, max_len, D = keys.shape
        dev = keys.device
        self.Hkv, self.D, self.dsub, self.max_len, self.L0 = Hkv, D, D // NSUB, max_len, L0
        n = L0 - SINK
        gen = torch.Generator(device=dev).manual_seed(seed)
        X = keys[:, SINK:L0].float().reshape(Hkv, n, NSUB, self.dsub).permute(0, 2, 1, 3).reshape(Hkv * NSUB, n, -1)
        C, codes = kmeans_batched(X, NC, iters, gen)
        del X
        self.C = C.view(Hkv, NSUB, NC, self.dsub).contiguous()
        self.codes = torch.zeros(Hkv, max_len, NSUB, dtype=torch.uint8, device=dev)
        self.codes[:, SINK:L0] = codes.view(Hkv, NSUB, n).permute(0, 2, 1).to(torch.uint8)
        # region A: keys at distance >= 1024 from every query (queries start at L0): bucketed by code pair
        self.A_end = max(SINK + 1, L0 - R_BIAS)
        cA = self.codes[:, SINK:self.A_end].long()
        pair = cA[..., 0] * NC + cA[..., 1]
        self.nA = pair.shape[1]
        self.perm = (torch.argsort(pair, dim=1, stable=True) + SINK).to(torch.int32)    # [Hkv, nA]
        cnt = torch.zeros(Hkv, NPAIR, device=dev).scatter_add_(1, pair, torch.ones_like(pair, dtype=torch.float32))
        self.counts = cnt
        self.bstart = (cnt.cumsum(1) - cnt).long()
        self.bstart32 = self.bstart.to(torch.int32).contiguous()


class SSARow:
    """SSA state of one batch row (its own codes for keys written after the context)."""

    def __init__(self, ctx: SSAContext, b: int) -> None:
        self.ctx, self.b = ctx, b
        self.codes = ctx.codes.clone()

    def code_keys(self, keys_row: torch.Tensor, pos: torch.Tensor) -> None:
        """Assign PQ codes to the keys at `pos` (device LongTensor) of this row ([Hkv, max_len, D])."""
        c = self.ctx
        kn = keys_row.index_select(1, pos).float().view(c.Hkv, -1, NSUB, 1, c.dsub)
        dist = ((kn - c.C[:, None]) ** 2).sum(-1)                                    # [Hkv, n, NSUB, NC]
        self.codes.index_copy_(1, pos, dist.argmin(-1).to(torch.uint8))


class SSASelector:
    """Selection for one SSA row. Static shapes for a fixed number of queries -> CUDA-graph capturable."""

    def __init__(self, row: SSARow, L_end_max: int, n_keys: int, heavy_frac: float, sample_frac: float,
                 scaling: float, temperature: float = 1.0, HQ: int = 24) -> None:
        c = row.ctx
        dev = c.codes.device
        self.row, self.ctx, self.HQ, self.G = row, c, HQ, HQ // c.Hkv
        self.k_h = int(heavy_frac * n_keys)
        self.m = int(sample_frac * n_keys)
        self.S = SINK + WINDOW + self.k_h + self.m
        self.WB = max(1, L_end_max - WINDOW - c.A_end + 1)
        assert c.A_end + self.WB <= c.max_len, (c.A_end, self.WB, c.max_len)
        self.kb = c.A_end + torch.arange(self.WB, device=dev)
        self.scaling, self.T = scaling, temperature
        self.hk_of_head = torch.arange(HQ, device=dev) // self.G
        self.ar_sink = torch.arange(SINK, device=dev)
        self.ar_win = torch.arange(WINDOW, device=dev) - (WINDOW - 1)
        self.ar_heavy = torch.arange(self.k_h, device=dev, dtype=torch.float32) + 0.5

    def _item_to_key(self, item: torch.Tensor, within: torch.Tensor, hk: torch.Tensor) -> torch.Tensor:
        c = self.ctx
        in_b = item >= NPAIR
        start = c.bstart[hk[:, None], item.clamp(max=NPAIR - 1)]
        key_a = c.perm[hk[:, None], (start + within).clamp(0, c.nA - 1)].long()
        key_b = self.kb[(item - NPAIR).clamp(0, self.WB - 1)]
        return torch.where(in_b, key_b, key_a)

    def select(self, q: torch.Tensor, pos: torch.Tensor, bias_of: Callable[[torch.Tensor], torch.Tensor]):
        """q [Qn, HQ, D]; pos [Qn] long (absolute positions); bias_of(idx [Qn, HQ, X]) -> bias fp32 (0 beyond
        the band).  Returns idx [Qn, HQ, S] (long) and weights [Qn, HQ, S] (fp32, 0 = unused slot)."""
        c, HQ, G = self.ctx, self.HQ, self.G
        Qn = q.shape[0]
        rows = Qn * HQ
        codes = self.row.codes
        qs = q.float().view(Qn, c.Hkv, G, NSUB, c.dsub)
        lut = torch.einsum("qkgsd,ksnd->qkgsn", qs, c.C) * self.scaling           # [Qn, Hkv, G, NSUB, NC]
        pair_l = (lut[..., 0, :, None] + lut[..., 1, None, :]).reshape(Qn, c.Hkv, G, NPAIR)
        cnt_a = c.counts[None, :, None, :].expand(Qn, -1, G, -1)
        pair_l = pair_l.masked_fill(cnt_a == 0, float("-inf"))
        # region B: exact per-key PQ logits (+ bias) for keys in [A_end, pos - WINDOW]
        cb = codes[:, c.A_end:c.A_end + self.WB].long()                               # [Hkv, WB, NSUB]
        cb = cb[None, :, None].expand(Qn, -1, G, -1, -1)
        lb = lut[..., 0, :].gather(-1, cb[..., 0]) + lut[..., 1, :].gather(-1, cb[..., 1])  # [Qn,Hkv,G,WB]
        lb = lb.reshape(Qn, HQ, self.WB)
        valid_b = self.kb[None, :] <= (pos[:, None] - WINDOW)                         # [Qn, WB]
        lb = lb + bias_of(self.kb[None, None, :].expand(Qn, HQ, -1))
        lb = lb.masked_fill(~valid_b[:, None, :], float("-inf"))
        logits = torch.cat([pair_l.reshape(Qn, HQ, NPAIR), lb], -1).reshape(rows, -1)
        cnts = torch.cat([cnt_a.reshape(Qn, HQ, NPAIR),
                          valid_b[:, None, :].expand(-1, HQ, -1).float()], -1).reshape(rows, -1)
        NI = logits.shape[1]
        hk = self.hk_of_head.repeat(Qn)                                               # [rows]

        sl, order = logits.sort(-1, descending=True)
        sc = cnts.gather(-1, order)
        cum = sc.cumsum(-1)
        before = cum - sc
        # stratum 1: top-k keys (bucket items expand to their member keys, in bucket order)
        if self.k_h > 0:
            ip = torch.searchsorted(cum, self.ar_heavy[None].expand(rows, -1).contiguous()).clamp(max=NI - 1)
            within = (self.ar_heavy[None] - 0.5 - before.gather(-1, ip)).long()
            heavy_keys = self._item_to_key(order.gather(-1, ip), within, hk)
            heavy_ok = (self.ar_heavy[None] < cum[:, -1:]).float()
            taken = torch.minimum(sc, (self.k_h - before).clamp(min=0))
        else:
            heavy_keys = torch.zeros(rows, 0, dtype=torch.long, device=q.device)
            heavy_ok = torch.zeros(rows, 0, device=q.device)
            taken = torch.zeros_like(sc)
        # stratum 2: m draws with replacement from softmax(logit / T) over the remaining keys
        rem = sc - taken
        mx = sl[:, :1].clamp(min=-1e30)
        e = torch.exp((sl - mx) / self.T)
        cw = (rem * e).cumsum(-1)
        Z = cw[:, -1:]
        u = torch.rand(rows, self.m, device=q.device) * Z
        ip = torch.searchsorted(cw, u.contiguous(), right=True).clamp(max=NI - 1)
        r_it, t_it = rem.gather(-1, ip), taken.gather(-1, ip)
        within = (t_it + (torch.rand(rows, self.m, device=q.device) * r_it).floor().clamp(max=r_it - 1)).long()
        keys_s = self._item_to_key(order.gather(-1, ip), within, hk)
        p = e.gather(-1, ip) / Z
        pi = (-torch.expm1(self.m * torch.log1p(-p))).clamp(MIN_PI, 1.0)
        ok_s = (Z > 0) & (r_it > 0)
        keys_s, srt = keys_s.sort(-1)
        pi, ok_s = pi.gather(-1, srt), ok_s.gather(-1, srt)
        dup = torch.zeros_like(ok_s)
        dup[:, 1:] = keys_s[:, 1:] == keys_s[:, :-1]
        w_s = torch.where(ok_s & ~dup, 1.0 / pi, torch.zeros_like(pi))

        qpos = pos.repeat_interleave(HQ)
        idx = torch.cat([self.ar_sink[None].expand(rows, -1), qpos[:, None] + self.ar_win[None],
                         heavy_keys, keys_s], -1).clamp(0, c.max_len - 1)
        w = torch.cat([torch.ones(rows, SINK + WINDOW, device=q.device), heavy_ok, w_s], -1)
        idx = idx.view(Qn, HQ, -1)
        w = w.view(Qn, HQ, -1) * torch.exp(bias_of(idx))
        return idx, w


class SSAAttention:
    """skylight sparse_optimized over virtual requests (query, kv_head) whose page table = that row's head."""

    def __init__(self, n_q: int, b: int, B: int, Hkv: int, max_len: int, D: int, device,
                 dtype=torch.bfloat16) -> None:
        P = max_len // PAGE
        self.n_q, self.Hkv, self.D, self.rows_v = n_q, Hkv, D, n_q * Hkv
        i32 = dict(dtype=torch.int32, device=device)
        heads = torch.arange(Hkv, device=device).repeat(n_q)                          # [rows_v]
        pages = ((b * Hkv + heads)[:, None] * P + torch.arange(P, device=device)[None]).reshape(-1).to(torch.int32)
        self.ws = torch.zeros(128 << 20, dtype=torch.uint8, device=device)
        self.k = SparseDecode(self.ws, "HND")
        self.k.plan(torch.arange(0, self.rows_v * P + 1, P, **i32), pages, torch.full((self.rows_v,), PAGE, **i32),
                    24 // Hkv, 1, D, PAGE, q_data_type=dtype, kv_data_type=dtype)
        self.len = None

    def run(self, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, idx: torch.Tensor, w: torch.Tensor,
            sinks: torch.Tensor) -> torch.Tensor:
        """q [n_q, HQ, D]; keys/values [B, Hkv, max_len, D]; idx/w [n_q, HQ, S]; sinks [n_q, HQ] -> [n_q, HQ, D]."""
        G = q.shape[1] // self.Hkv
        S = idx.shape[-1]
        if self.len is None or self.len.shape[-1] != 1 or int(self.len.numel()) != self.rows_v * G:
            self.len = torch.full((self.rows_v, G, 1), S, dtype=torch.int32, device=q.device)
        kv = (keys.view(-1, 1, PAGE, self.D), values.view(-1, 1, PAGE, self.D))
        out, lse = self.k.run(q.reshape(self.rows_v, G, self.D), kv, sparse_len=self.len,
                              sparse_idx=idx.reshape(self.rows_v, G, S),
                              sparse_weights=w.reshape(self.rows_v, G, S).contiguous(), return_lse=True)
        gate = torch.sigmoid(lse.float() * LN2 - sinks.reshape(self.rows_v, G).float())
        return (out.float() * gate[..., None]).to(q.dtype).view(self.n_q, -1, self.D)


def reference_select(q: torch.Tensor, pos: int, codes: torch.Tensor, C: torch.Tensor, bias: torch.Tensor,
                     scaling: float, k_h: int, m: int, L: int, T: float = 1.0, gen=None):
    """Per-key PQImportance (SSA-Paper-Code semantics) for ONE query, for tests. q [HQ, D]; codes [Hkv, L, 2];
    C [Hkv, 2, NC, dsub]; bias [HQ, L] (0 beyond band). Returns per-head dicts: logits, heavy set, p over
    the pool (keys not sink/local/heavy), and the drawn keys."""
    HQ, D = q.shape
    Hkv = C.shape[0]
    G = HQ // Hkv
    dsub = D // NSUB
    out = []
    for h in range(HQ):
        hk = h // G
        lut = (q[h].float().view(NSUB, 1, dsub) * C[hk]).sum(-1) * scaling           # [NSUB, NC]
        cd = codes[hk, :L].long()
        lg = lut[0][cd[:, 0]] + lut[1][cd[:, 1]] + bias[h, :L]
        cand = torch.zeros(L, dtype=torch.bool, device=q.device)
        cand[SINK:pos - WINDOW + 1] = True
        lg = lg.masked_fill(~cand, float("-inf"))
        heavy = torch.topk(lg, k_h).indices if k_h > 0 else lg.new_zeros(0, dtype=torch.long)
        pool = lg.clone()
        pool[heavy] = float("-inf")
        p = torch.softmax(pool / T, -1)
        draws = torch.multinomial(p, m, replacement=True, generator=gen) if m > 0 else None
        out.append({"logits": lg, "heavy": heavy, "p": p, "draws": draws})
    return out


# ============================================================================================================
# Fused fast path (Triton): LUT -> item logits, top-k / sorted-uniform multinomial in a few batched torch ops,
# then a split-K gather-attention kernel over each head's own selected keys.
# ============================================================================================================
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

LOG2E = 1.4426950408889634


@triton.jit
def _ssa_lut_kernel(Q, C, LUT, sqn, sqh, HQ: tl.constexpr, G: tl.constexpr, NCc: tl.constexpr,
                    DS: tl.constexpr, scale):
    r = tl.program_id(0)                       # qn * HQ + h
    qn = r // HQ
    h = r % HQ
    hk = h // G
    offs_c = tl.arange(0, NCc)
    offs_d = tl.arange(0, DS)
    for s in tl.static_range(2):
        q = tl.load(Q + qn * sqn + h * sqh + s * DS + offs_d).to(tl.float32)
        cen = tl.load(C + ((hk * 2 + s) * NCc + offs_c[:, None]) * DS + offs_d[None, :])
        tl.store(LUT + (r * 2 + s) * NCc + offs_c, tl.sum(cen * q[None, :], 1) * scale)


@triton.jit
def _ssa_items_kernel(LUT, CNT, CODES, POS, PB, TB, LOGITS, CNTS,
                      A_end, WB, NI, max_len, spb_h, stb_q, stb_h,
                      HQ: tl.constexpr, G: tl.constexpr, NCc: tl.constexpr, NP: tl.constexpr,
                      WINDOW: tl.constexpr, R: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr):
    """Item logits for one (query, head) row and one block of items. Items [0, NP) are code-pair buckets of
    region A (count = #keys with that pair); items NP + j are the single keys A_end + j of region B."""
    r = tl.program_id(0)
    blk = tl.program_id(1)
    qn = r // HQ
    h = r % HQ
    hk = h // G
    it = blk * BLOCK + tl.arange(0, BLOCK)
    lut = LUT + r * 2 * NCc
    pos = tl.load(POS + qn)
    is_pair = it < NP
    c0p = it // NCc
    c1p = it % NCc
    j = it - NP
    in_b = (j >= 0) & (j < WB)
    kb = A_end + j
    c0b = tl.load(CODES + (hk * max_len + kb) * 2, mask=in_b, other=0).to(tl.int32)
    c1b = tl.load(CODES + (hk * max_len + kb) * 2 + 1, mask=in_b, other=0).to(tl.int32)
    c0 = tl.where(is_pair, c0p, c0b)
    c1 = tl.where(is_pair, c1p, c1b)
    live = is_pair | in_b
    v = tl.load(lut + c0, mask=live, other=0.0) + tl.load(lut + NCc + c1, mask=live, other=0.0)
    cnt_p = tl.load(CNT + hk * NP + it, mask=is_pair, other=0.0)
    d = pos - kb
    ok_b = in_b & (d >= WINDOW)
    inb = ok_b & (d < R)
    if MODE == 0:
        bias = tl.load(PB + h * spb_h + kb, mask=inb, other=0.0).to(tl.float32)
    else:
        bias = tl.load(TB + qn * stb_q + h * stb_h + d, mask=inb, other=0.0).to(tl.float32)
    v = tl.where(is_pair, v, v + bias)
    cnt = tl.where(is_pair, cnt_p, tl.where(ok_b, 1.0, 0.0))
    v = tl.where(cnt > 0, v, float("-inf"))
    m = it < NI
    tl.store(LOGITS + r * NI + it, v, mask=m)
    tl.store(CNTS + r * NI + it, cnt, mask=m)


@triton.jit
def _ssa_attn_split_kernel(Q, K, V, IDX, W, OP, ML, b, S, per, sqn, sqh, skb, skh, skn, svb, svh, svn, n_split,
                           qk_scale, HQ: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr, D: tl.constexpr):
    r = tl.program_id(0)
    sp = tl.program_id(1)
    qn = r // HQ
    h = r % HQ
    hk = h // G
    offs_d = tl.arange(0, D)
    q = tl.load(Q + qn * sqn + h * sqh + offs_d).to(tl.float32)
    s0 = sp * per
    s1 = tl.minimum(s0 + per, S)
    m_i = tl.max(tl.full([BLOCK], float("-inf"), tl.float32), 0)
    l_i = tl.sum(tl.zeros([BLOCK], tl.float32), 0)
    acc = tl.zeros([D], tl.float32)
    kbase = K + b * skb + hk * skh
    vbase = V + b * svb + hk * svh
    for st in range(s0, s1, BLOCK):
        o = st + tl.arange(0, BLOCK)
        ok = o < s1
        ix = tl.load(IDX + r * S + o, mask=ok, other=0)
        w = tl.load(W + r * S + o, mask=ok, other=0.0)
        k = tl.load(kbase + ix[:, None] * skn + offs_d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], 1) * qk_scale + tl.log2(w)
        s = tl.where(ok & (w > 0), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 0))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe)
        l_i = l_i * alpha + tl.sum(p, 0)
        vv = tl.load(vbase + ix[:, None] * svn + offs_d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * vv, 0)
        m_i = m_new
    base = r * n_split + sp
    tl.store(OP + base * D + offs_d, acc)
    tl.store(ML + base * 2, m_i)
    tl.store(ML + base * 2 + 1, l_i)


@triton.jit
def _ssa_merge_kernel(OP, ML, SNK, O, n_split, sk_q, sk_h, so_q, so_h,
                      HQ: tl.constexpr, SPLIT_P: tl.constexpr, D: tl.constexpr):
    r = tl.program_id(0)
    qn = r // HQ
    h = r % HQ
    offs_s = tl.arange(0, SPLIT_P)
    s_ok = offs_s < n_split
    offs_d = tl.arange(0, D)
    m = tl.load(ML + (r * n_split + offs_s) * 2, mask=s_ok, other=float("-inf"))
    l = tl.load(ML + (r * n_split + offs_s) * 2 + 1, mask=s_ok, other=0.0)
    acc = tl.load(OP + (r * n_split + offs_s)[:, None] * D + offs_d[None, :], mask=s_ok[:, None], other=0.0)
    sk = tl.load(SNK + qn * sk_q + h * sk_h).to(tl.float32) * 1.4426950408889634
    mx = tl.maximum(tl.max(m, 0), sk)
    w = tl.exp2(m - mx)
    num = tl.sum(acc * w[:, None], 0)
    den = tl.sum(l * w, 0) + tl.exp2(sk - mx)
    tl.store(O + qn * so_q + h * so_h + offs_d, (num / den).to(O.dtype.element_ty))


class SSAFast(SSASelector):
    """SSASelector + fused kernels.  `__call__` = selection + attention for Qn queries of row `b`."""

    BLOCK_ITEMS = 1024

    def __init__(self, row: SSARow, L_end_max: int, n_keys: int, heavy_frac: float, sample_frac: float,
                 scaling: float, temperature: float = 1.0, HQ: int = 24, max_q: int = 1, n_split: int = 8) -> None:
        super().__init__(row, L_end_max, n_keys, heavy_frac, sample_frac, scaling, temperature, HQ)
        dev = row.codes.device
        self.NI = NPAIR + self.WB
        self.KI = min(self.NI, max(self.k_h, 1))
        self.n_split = n_split
        rows = max_q * HQ
        self.lut = torch.empty(rows, 2, NC, dtype=torch.float32, device=dev)
        self.logits = torch.empty(rows, self.NI, dtype=torch.float32, device=dev)
        self.cnts = torch.empty(rows, self.NI, dtype=torch.float32, device=dev)
        self.op = torch.empty(rows * n_split, 128, dtype=torch.float32, device=dev)
        self.ml = torch.empty(rows * n_split, 2, dtype=torch.float32, device=dev)
        self.cnt_flat = row.ctx.counts.contiguous()

    def items(self, q: torch.Tensor, pos: torch.Tensor, pb: Optional[torch.Tensor], tb: Optional[torch.Tensor]):
        c, HQ = self.ctx, self.HQ
        Qn = q.shape[0]
        rows = Qn * HQ
        lut, logits, cnts = self.lut[:rows], self.logits[:rows], self.cnts[:rows]
        _ssa_lut_kernel[(rows,)](q, c.C, lut, q.stride(0), q.stride(1), HQ=HQ, G=self.G, NCc=NC, DS=c.dsub,
                                 scale=self.scaling)
        mode = 0 if pb is not None else 1
        pb_ = pb if pb is not None else logits
        tb_ = tb if tb is not None else logits
        _ssa_items_kernel[(rows, triton.cdiv(self.NI, self.BLOCK_ITEMS))](
            lut, self.cnt_flat, self.row.codes, pos, pb_, tb_, logits, cnts, c.A_end, self.WB, self.NI, c.max_len,
            pb.stride(0) if pb is not None else 0, tb.stride(0) if tb is not None else 0,
            tb.stride(1) if tb is not None else 0, HQ=HQ, G=self.G, NCc=NC, NP=NPAIR, WINDOW=WINDOW, R=R_BIAS,
            MODE=mode, BLOCK=self.BLOCK_ITEMS, num_warps=4)
        return logits, cnts

    def select_fast(self, q, pos, pb=None, tb=None):
        """q [Qn,HQ,D] bf16; pos [Qn] long; decode: pb [HQ, max_len] (bias by key slot); question:
        tb [Qn, HQ, R] (bias by distance). Returns idx [rows, S] long, w [rows, S] fp32 (bias included)."""
        c, HQ = self.ctx, self.HQ
        Qn = q.shape[0]
        rows = Qn * HQ
        logits, cnts = self.items(q, pos, pb, tb)
        hk = self.hk_of_head.repeat(Qn)
        dev = q.device
        # stratum 1: top-k keys = members of the top-KI items by value (each item holds >= 1 key)
        if self.k_h > 0:
            tv, ti = torch.topk(logits, self.KI, dim=-1, sorted=True)
            tc = cnts.gather(-1, ti)
            cum = tc.cumsum(-1)
            before = cum - tc
            ip = torch.searchsorted(cum, self.ar_heavy[None].expand(rows, -1).contiguous()).clamp(max=self.KI - 1)
            within = (self.ar_heavy[None] - 0.5 - before.gather(-1, ip)).long()
            heavy_keys = self._item_to_key(ti.gather(-1, ip), within, hk)
            heavy_ok = (self.ar_heavy[None] < cum[:, -1:]).float()
            taken = torch.zeros_like(cnts).scatter_(-1, ti, torch.minimum(tc, (self.k_h - before).clamp(min=0)))
            mx = tv[:, :1]
        else:
            heavy_keys = torch.zeros(rows, 0, dtype=torch.long, device=dev)
            heavy_ok = torch.zeros(rows, 0, device=dev)
            taken = torch.zeros_like(cnts)
            mx = logits.max(-1, keepdim=True).values
        # stratum 2: m draws with replacement over keys, via one sorted uniform per draw
        if self.m > 0:
            mx = mx.clamp(min=-1e30)
            e = torch.exp((logits - mx) / self.T)
            rem = cnts - taken
            wt = (rem * e).double()
            Wc = wt.cumsum(-1)
            Z = Wc[:, -1:]
            E = -torch.log(torch.rand(rows, self.m + 1, device=dev, dtype=torch.float64).clamp_(min=1e-300))
            Sx = E.cumsum(-1)
            u = (Sx[:, :self.m] / Sx[:, self.m:]) * Z
            ip = torch.searchsorted(Wc, u.contiguous(), right=True).clamp(max=self.NI - 1)
            e_it = e.gather(-1, ip)
            r_it = rem.gather(-1, ip)
            start = Wc.gather(-1, ip) - wt.gather(-1, ip)
            frac = ((u - start) / e_it.double().clamp(min=1e-300)).floor()
            within_s = torch.minimum(frac.clamp(min=0), (r_it - 1).double().clamp(min=0)).long()
            keys_s = self._item_to_key(ip, taken.gather(-1, ip).long() + within_s, hk)
            p = e_it / Z.float()
            pi = (-torch.expm1(self.m * torch.log1p(-p))).clamp(MIN_PI, 1.0)
            same = torch.zeros_like(ip, dtype=torch.bool)
            same[:, 1:] = (ip[:, 1:] == ip[:, :-1]) & (within_s[:, 1:] == within_s[:, :-1])
            w_s = torch.where((~same) & (r_it > 0) & (Z > 0), 1.0 / pi, torch.zeros_like(pi))
        else:
            keys_s = torch.zeros(rows, 0, dtype=torch.long, device=dev)
            w_s = torch.zeros(rows, 0, device=dev)
        qpos = pos.repeat_interleave(HQ)
        idx = torch.cat([self.ar_sink[None].expand(rows, -1), qpos[:, None] + self.ar_win[None], heavy_keys,
                         keys_s], -1).clamp(0, c.max_len - 1)
        w = torch.cat([torch.ones(rows, SINK + WINDOW, device=dev), heavy_ok, w_s], -1)
        if pb is not None:
            bias = pb.gather(-1, idx.view(Qn, HQ, -1)[0]).float().view(rows, -1) if Qn == 1 else \
                pb[None].expand(Qn, -1, -1).gather(-1, idx.view(Qn, HQ, -1)).float().view(rows, -1)
        else:
            d = qpos[:, None] - idx
            okb = (d >= 0) & (d < R_BIAS)
            bias = tb.reshape(rows, -1).gather(-1, d.clamp(0, R_BIAS - 1)).float().masked_fill(~okb, 0.0)
        return idx, w * torch.exp(bias)

    def attend(self, q, keys, values, b: int, idx, w, sinks, out=None):
        """Gather attention over each head's selected keys; q [Qn,HQ,D]; keys/values [B,Hkv,max_len,D];
        sinks [Qn,HQ]; returns [Qn,HQ,D]."""
        Qn = q.shape[0]
        rows = Qn * self.HQ
        S = idx.shape[-1]
        per = triton.cdiv(triton.cdiv(S, self.n_split), 64) * 64
        op, ml = self.op[:rows * self.n_split], self.ml[:rows * self.n_split]
        _ssa_attn_split_kernel[(rows, self.n_split)](
            q, keys, values, idx, w, op, ml, b, S, per, q.stride(0), q.stride(1), keys.stride(0), keys.stride(1),
            keys.stride(2), values.stride(0), values.stride(1), values.stride(2), self.n_split,
            self.scaling * LOG2E, HQ=self.HQ, G=self.G, BLOCK=64, D=q.shape[-1], num_warps=4)
        if out is None:
            out = torch.empty(Qn, self.HQ, q.shape[-1], dtype=q.dtype, device=q.device)
        _ssa_merge_kernel[(rows,)](op, ml, sinks, out, self.n_split, sinks.stride(0), sinks.stride(1),
                                   out.stride(0), out.stride(1), HQ=self.HQ,
                                   SPLIT_P=triton.next_power_of_2(self.n_split), D=q.shape[-1])
        return out

    def __call__(self, q, pos, keys, values, b, sinks, pb=None, tb=None, out=None):
        idx, w = self.select_fast(q, pos, pb, tb)
        return self.attend(q, keys, values, b, idx.contiguous(), w.contiguous(), sinks, out)


# ============================================================================================================
# Fully fused selection: topk (threshold only) + 2 Triton kernels  ->  ~9 launches per (row, visit) in total
# ============================================================================================================
@triton.jit
def _ssa_prefix_kernel(LOGITS, CNTS, TAU, VMAX, HOFF, WCDF, ZT, SU, SEED, NI, KH, M, inv_T_log2e,
                       BLOCK: tl.constexpr, BLOCK_U: tl.constexpr):
    """One program per (query, head) row.  Heavy stratum = keys with logit > tau, plus `KH - count(> tau)`
    keys from items with logit == tau in item order (an exact top-KH).  Writes the inclusive prefix of keys
    taken per item (HOFF), the inclusive fp64 CDF of the remaining mass (WCDF), Z, and m+1 cumulative Exp(1)
    draws (SU) that become m sorted iid uniforms (exponential spacings)."""
    r = tl.program_id(0)
    tau = tl.load(TAU + r)
    vmax = tl.load(VMAX + r)
    base = r * NI
    cnt_gt = tl.sum(tl.zeros([BLOCK], tl.float32), 0)
    for b0 in range(0, NI, BLOCK):
        it = b0 + tl.arange(0, BLOCK)
        ok = it < NI
        v = tl.load(LOGITS + base + it, mask=ok, other=float("-inf"))
        c = tl.load(CNTS + base + it, mask=ok, other=0.0)
        cnt_gt += tl.sum(tl.where(v > tau, c, 0.0), 0)
    need_tie = KH - cnt_gt
    carry_tie = tl.sum(tl.zeros([BLOCK], tl.float32), 0)
    carry_h = tl.sum(tl.zeros([BLOCK], tl.float32), 0)
    carry_w = tl.sum(tl.zeros([BLOCK], tl.float64), 0)
    for b0 in range(0, NI, BLOCK):
        it = b0 + tl.arange(0, BLOCK)
        ok = it < NI
        v = tl.load(LOGITS + base + it, mask=ok, other=float("-inf"))
        c = tl.load(CNTS + base + it, mask=ok, other=0.0)
        ctie = tl.where(v == tau, c, 0.0)
        tie_before = tl.cumsum(ctie, 0) - ctie + carry_tie
        take_eq = tl.minimum(tl.maximum(need_tie - tie_before, 0.0), c)
        taken = tl.where(v > tau, c, tl.where(v == tau, take_eq, 0.0))
        h_incl = tl.cumsum(taken, 0) + carry_h
        tl.store(HOFF + base + it, h_incl, mask=ok)
        e = tl.exp2((v - vmax) * inv_T_log2e)
        e = tl.where(c > 0, e, 0.0)
        wt = ((c - taken) * e).to(tl.float64)
        w_incl = tl.cumsum(wt, 0) + carry_w
        tl.store(WCDF + base + it, w_incl, mask=ok)
        carry_tie += tl.sum(ctie, 0)
        carry_h += tl.sum(taken, 0)
        carry_w += tl.sum(wt, 0)
    tl.store(ZT + r * 2, carry_w.to(tl.float32))
    tl.store(ZT + r * 2 + 1, carry_h)
    seed = tl.load(SEED)
    carry_s = tl.sum(tl.zeros([BLOCK_U], tl.float64), 0)
    for j0 in range(0, M + 1, BLOCK_U):
        j = j0 + tl.arange(0, BLOCK_U)
        okj = j < M + 1
        u = tl.rand(seed, r * (M + 1) + j)
        ex = -tl.log(tl.maximum(u, 1e-12)).to(tl.float64)
        ex = tl.where(okj, ex, 0.0)
        s_incl = tl.cumsum(ex, 0) + carry_s
        tl.store(SU + r * (M + 1) + j, s_incl, mask=okj)
        carry_s += tl.sum(ex, 0)


@triton.jit
def _bsearch(A, base, x, N, LOGN: tl.constexpr):
    """First index i in [0, N) with A[base + i] > x (vectorized over x); N if none."""
    lo = tl.zeros(x.shape, tl.int32)
    hi = tl.zeros(x.shape, tl.int32) + N
    for _ in tl.static_range(LOGN):
        mid = (lo + hi) // 2
        a = tl.load(A + base + mid, mask=mid < N, other=float("inf"))
        right = a <= x
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    return lo


@triton.jit
def _ssa_slots_kernel(LOGITS, CNTS, HOFF, WCDF, ZT, SU, VMAX, BSTART, PERM, POS, PB, TB, IDX, W, SEED,
                      NI, KH, M, S, A_end, nA, spb_h, stb_q, stb_h, inv_T_log2e, n_hblk,
                      HQ: tl.constexpr, G: tl.constexpr, NP: tl.constexpr, SINKc: tl.constexpr,
                      WINc: tl.constexpr, R: tl.constexpr, MODE: tl.constexpr, LOGN: tl.constexpr,
                      BS: tl.constexpr):
    """Fill the S selected (key, weight) slots of one row: sink + local, heavy slot -> key, draw -> key/pi."""
    r = tl.program_id(0)
    pid = tl.program_id(1)
    qn = r // HQ
    h = r % HQ
    hk = h // G
    pos = tl.load(POS + qn)
    base = r * NI
    o = tl.arange(0, BS)
    if pid == 0:
        s = o
        ok = s < SINKc + WINc
        key = tl.where(s < SINKc, s, (pos - (WINc - 1) + (s - SINKc)).to(tl.int32))
        wt = tl.where(ok, 1.0, 0.0)
        slot = s
        if r == 0:
            tl.store(SEED, tl.load(SEED) + 1)
    elif pid <= n_hblk:
        j = (pid - 1) * BS + o
        htot = tl.load(ZT + r * 2 + 1)
        ok = (j < KH) & (j.to(tl.float32) < htot)
        i = _bsearch(HOFF, base, j.to(tl.float32), NI, LOGN)
        i = tl.minimum(i, NI - 1)
        prev = tl.load(HOFF + base + i - 1, mask=(i > 0) & ok, other=0.0)
        within = (j.to(tl.float32) - prev).to(tl.int32)
        is_b = i >= NP
        st = tl.load(BSTART + hk * NP + i, mask=ok & (~is_b), other=0)
        ka = tl.load(PERM + hk * nA + tl.minimum(st + within, nA - 1), mask=ok & (~is_b), other=0)
        key = tl.where(is_b, A_end + (i - NP), ka)
        wt = tl.where(ok, 1.0, 0.0)
        slot = SINKc + WINc + j
        ok = j < KH
    else:
        j = (pid - 1 - n_hblk) * BS + o
        ok = j < M
        Z = tl.load(ZT + r * 2).to(tl.float64)
        stot = tl.load(SU + r * (M + 1) + M)
        vmax = tl.load(VMAX + r)
        u = tl.load(SU + r * (M + 1) + j, mask=ok, other=0.0) / stot * Z
        up = tl.load(SU + r * (M + 1) + j - 1, mask=ok & (j > 0), other=0.0) / stot * Z
        i = tl.minimum(_bsearch(WCDF, base, u, NI, LOGN), NI - 1)
        ip = tl.minimum(_bsearch(WCDF, base, up, NI, LOGN), NI - 1)
        v = tl.load(LOGITS + base + i, mask=ok, other=float("-inf"))
        c = tl.load(CNTS + base + i, mask=ok, other=0.0)
        hi_ = tl.load(HOFF + base + i, mask=ok, other=0.0)
        hl_ = tl.load(HOFF + base + i - 1, mask=ok & (i > 0), other=0.0)
        taken = hi_ - hl_
        e = tl.exp2((v - vmax) * inv_T_log2e)
        wc = tl.load(WCDF + base + i, mask=ok, other=0.0)
        rem = c - taken
        start = wc - (rem * e).to(tl.float64)
        within = tl.minimum(tl.maximum(((u - start) / tl.maximum(e.to(tl.float64), 1e-300)), 0.0),
                            tl.maximum(rem - 1.0, 0.0).to(tl.float64)).to(tl.int32)
        # previous draw (sorted uniforms => equal keys are adjacent)
        vp = tl.load(LOGITS + base + ip, mask=ok & (j > 0), other=float("-inf"))
        cp = tl.load(CNTS + base + ip, mask=ok & (j > 0), other=0.0)
        hip = tl.load(HOFF + base + ip, mask=ok & (j > 0), other=0.0)
        hlp = tl.load(HOFF + base + ip - 1, mask=ok & (j > 0) & (ip > 0), other=0.0)
        ep = tl.exp2((vp - vmax) * inv_T_log2e)
        wcp = tl.load(WCDF + base + ip, mask=ok & (j > 0), other=0.0)
        remp = cp - (hip - hlp)
        startp = wcp - (remp * ep).to(tl.float64)
        withinp = tl.minimum(tl.maximum(((up - startp) / tl.maximum(ep.to(tl.float64), 1e-300)), 0.0),
                             tl.maximum(remp - 1.0, 0.0).to(tl.float64)).to(tl.int32)
        dup = (j > 0) & (ip == i) & (withinp == within)
        is_b = i >= NP
        st = tl.load(BSTART + hk * NP + i, mask=ok & (~is_b), other=0)
        ka = tl.load(PERM + hk * nA + tl.minimum(st + taken.to(tl.int32) + within, nA - 1), mask=ok & (~is_b), other=0)
        key = tl.where(is_b, A_end + (i - NP), ka)
        p = e.to(tl.float64) / tl.maximum(Z, 1e-300)
        pi = 1.0 - tl.exp(M * tl.log(tl.maximum(1.0 - p, 1e-300)))
        pi = tl.minimum(tl.maximum(pi, 1e-4), 1.0).to(tl.float32)
        valid = ok & (~dup) & (rem > 0) & (Z > 0)
        wt = tl.where(valid, 1.0 / pi, 0.0)
        slot = SINKc + WINc + KH + j
    d = pos - key
    if MODE == 0:
        bias = tl.load(PB + h * spb_h + key, mask=ok, other=0.0).to(tl.float32)
    else:
        bias = tl.load(TB + qn * stb_q + h * stb_h + d, mask=ok & (d >= 0) & (d < R), other=0.0).to(tl.float32)
    wt = wt * tl.exp(bias)
    tl.store(IDX + r * S + slot, key.to(tl.int32), mask=ok)
    tl.store(W + r * S + slot, wt, mask=ok)


class SSAFused(SSAFast):
    """SSAFast with the post-items work in two Triton kernels (exact top-k via one torch.topk for the threshold)."""

    BS = 256

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dev = self.row.codes.device
        rows = self.lut.shape[0]
        self.hoff = torch.empty(rows, self.NI, dtype=torch.float32, device=dev)
        self.wcdf = torch.empty(rows, self.NI, dtype=torch.float64, device=dev)
        self.zt = torch.empty(rows, 2, dtype=torch.float32, device=dev)
        self.su = torch.empty(rows, self.m + 1, dtype=torch.float64, device=dev)
        self.idx_buf = torch.empty(rows, self.S, dtype=torch.int32, device=dev)
        self.w_buf = torch.empty(rows, self.S, dtype=torch.float32, device=dev)
        self.seed = torch.randint(0, 2 ** 30, (1,), dtype=torch.int32, device=dev)
        self.ar_kh = torch.tensor([self.k_h - 0.5], device=dev)
        self.logn = max(1, int(self.NI).bit_length())
        self.n_hblk = triton.cdiv(self.k_h, self.BS)
        self.n_dblk = triton.cdiv(self.m, self.BS)
        self.n_split = max(4, min(64, triton.cdiv(self.S, 384)))
        self.op = torch.empty(rows * self.n_split, 128, dtype=torch.float32, device=dev)
        self.ml = torch.empty(rows * self.n_split, 2, dtype=torch.float32, device=dev)

    def select_fused(self, q, pos, pb=None, tb=None):
        c, HQ = self.ctx, self.HQ
        Qn = q.shape[0]
        rows = Qn * HQ
        logits, cnts = self.items(q, pos, pb, tb)
        if self.k_h > 0:
            tv, ti = torch.topk(logits, self.KI, dim=-1, sorted=True)
            cum = cnts.gather(-1, ti).cumsum(-1)
            bpos = torch.searchsorted(cum, self.ar_kh.expand(rows, 1).contiguous()).clamp(max=self.KI - 1)
            tau = tv.gather(-1, bpos).view(-1).contiguous()
            vmax = tv[:, 0].contiguous()
        else:
            vmax = logits.max(-1).values.contiguous()
            tau = torch.full_like(vmax, float("inf"))
        hoff, wcdf, zt, su = self.hoff[:rows], self.wcdf[:rows], self.zt[:rows], self.su[:rows]
        idx, w = self.idx_buf[:rows], self.w_buf[:rows]
        inv = LOG2E / self.T
        _ssa_prefix_kernel[(rows,)](logits, cnts, tau, vmax, hoff, wcdf, zt, su, self.seed, self.NI, self.k_h, self.m,
                                    inv, BLOCK=1024, BLOCK_U=1024, num_warps=8)
        mode = 0 if pb is not None else 1
        pb_ = pb if pb is not None else logits
        tb_ = tb if tb is not None else logits
        _ssa_slots_kernel[(rows, 1 + self.n_hblk + self.n_dblk)](
            logits, cnts, hoff, wcdf, zt, su, vmax, c.bstart32, c.perm, pos, pb_, tb_, idx, w, self.seed, self.NI, self.k_h, self.m, self.S, c.A_end, c.nA,
            pb.stride(0) if pb is not None else 0, tb.stride(0) if tb is not None else 0,
            tb.stride(1) if tb is not None else 0, inv, self.n_hblk, HQ=HQ, G=self.G, NP=NPAIR, SINKc=SINK,
            WINc=WINDOW, R=R_BIAS, MODE=mode, LOGN=self.logn, BS=self.BS, num_warps=4)
        return idx, w

    def __call__(self, q, pos, keys, values, b, sinks, pb=None, tb=None, out=None):
        idx, w = self.select_fused(q, pos, pb, tb)
        return self.attend(q, keys, values, b, idx, w, sinks, out)


# ============================================================================================================
# v3: exact top-k threshold by in-register bitwise search (no torch.topk), parallel sorted uniforms
# ============================================================================================================
@triton.jit
def _ord_key(v):
    """float32 -> uint32 with the same order."""
    b = v.to(tl.uint32, bitcast=True)
    sign = tl.full(b.shape, 0x80000000, tl.uint32)
    return tl.where((b & sign) != 0, ~b, b | sign)


@triton.jit
def _ssa_select_kernel(LOGITS, CNTS, HOFF, WCDF, ZT, VMAX, NI, KH, inv_T_log2e, NI_PAD: tl.constexpr):
    """One program per row, all items in registers.  tau = KH-th largest key logit (counting bucket
    multiplicity), found by a 32-step bitwise search on order-preserving keys; heavy = keys > tau plus
    KH - count(> tau) keys from items == tau in item order (exact top-KH).  Writes HOFF (inclusive prefix of
    keys taken per item), WCDF (fp64 inclusive CDF of the remaining sampling mass), Z, #heavy, vmax."""
    r = tl.program_id(0)
    it = tl.arange(0, NI_PAD)
    ok = it < NI
    v = tl.load(LOGITS + r * NI + it, mask=ok, other=float("-inf"))
    c = tl.load(CNTS + r * NI + it, mask=ok, other=0.0)
    c = tl.where(v == float("-inf"), 0.0, c)
    key = _ord_key(v)
    vmax = tl.max(v, 0)
    total = tl.sum(c, 0)
    kh = tl.minimum(KH * 1.0, total)
    prefix = tl.full([], 0, tl.uint32)
    for k in tl.static_range(32):
        cand = prefix | tl.full([], 1 << (31 - k), tl.uint32)
        cnt = tl.sum(tl.where(key >= cand, c, 0.0), 0)
        prefix = tl.where(cnt >= kh, cand, prefix)
    gt = key > prefix
    eq = key == prefix
    cnt_gt = tl.sum(tl.where(gt, c, 0.0), 0)
    need_tie = kh - cnt_gt
    ctie = tl.where(eq, c, 0.0)
    tie_before = tl.cumsum(ctie, 0) - ctie
    taken = tl.where(gt, c, tl.where(eq, tl.minimum(tl.maximum(need_tie - tie_before, 0.0), c), 0.0))
    taken = tl.where(kh > 0, taken, 0.0)
    tl.store(HOFF + r * NI + it, tl.cumsum(taken, 0), mask=ok)
    e = tl.where(c > 0, tl.exp2((v - vmax) * inv_T_log2e), 0.0)
    wt = ((c - taken) * e).to(tl.float64)
    tl.store(WCDF + r * NI + it, tl.cumsum(wt, 0), mask=ok)
    tl.store(ZT + r * 2, tl.sum(wt, 0).to(tl.float32))
    tl.store(ZT + r * 2 + 1, tl.sum(taken, 0))
    tl.store(VMAX + r, vmax)


@triton.jit
def _ssa_expblk_kernel(SEED, EB, M, nblk_pad, BS: tl.constexpr):
    """Per (row, block of BS draws): sum of the block's Exp(1) variates (the (M+1)-th lives in the last block)."""
    r = tl.program_id(0)
    bk = tl.program_id(1)
    j = bk * BS + tl.arange(0, BS)
    seed = tl.load(SEED)
    u = tl.rand(seed, r * (M + 1) + j)
    ex = tl.where(j < M + 1, -tl.log(tl.maximum(u, 1e-12)).to(tl.float64), 0.0)
    tl.store(EB + r * nblk_pad + bk, tl.sum(ex, 0))


@triton.jit
def _ssa_slots3_kernel(LOGITS, CNTS, HOFF, WCDF, ZT, EB, VMAX, BSTART, PERM, POS, PB, TB, IDX, W, SEED,
                       NI, KH, M, S, A_end, nA, spb_h, stb_q, stb_h, inv_T_log2e, n_hblk, nblk_pad,
                       HQ: tl.constexpr, G: tl.constexpr, NP: tl.constexpr, SINKc: tl.constexpr,
                       WINc: tl.constexpr, R: tl.constexpr, MODE: tl.constexpr, LOGN: tl.constexpr,
                       BS: tl.constexpr, NB_P: tl.constexpr):
    r = tl.program_id(0)
    pid = tl.program_id(1)
    qn = r // HQ
    h = r % HQ
    hk = h // G
    pos = tl.load(POS + qn)
    base = r * NI
    o = tl.arange(0, BS)
    if pid == 0:
        s = o
        ok = s < SINKc + WINc
        key = tl.where(s < SINKc, s, (pos - (WINc - 1) + (s - SINKc)).to(tl.int32))
        wt = tl.where(ok, 1.0, 0.0)
        slot = s
    elif pid <= n_hblk:
        j = (pid - 1) * BS + o
        htot = tl.load(ZT + r * 2 + 1)
        okh = (j < KH) & (j.to(tl.float32) < htot)
        i = tl.minimum(_bsearch(HOFF, base, j.to(tl.float32), NI, LOGN), NI - 1)
        prev = tl.load(HOFF + base + i - 1, mask=(i > 0) & okh, other=0.0)
        within = (j.to(tl.float32) - prev).to(tl.int32)
        is_b = i >= NP
        st = tl.load(BSTART + hk * NP + i, mask=okh & (~is_b), other=0)
        ka = tl.load(PERM + hk * nA + tl.minimum(st + within, nA - 1), mask=okh & (~is_b), other=0)
        key = tl.where(is_b, A_end + (i - NP), ka)
        wt = tl.where(okh, 1.0, 0.0)
        slot = SINKc + WINc + j
        ok = j < KH
    else:
        bk = pid - 1 - n_hblk
        j = bk * BS + o
        ok = j < M
        seed = tl.load(SEED)
        ub = tl.rand(seed, r * (M + 1) + j)
        ex = tl.where(j < M + 1, -tl.log(tl.maximum(ub, 1e-12)).to(tl.float64), 0.0)
        ob = tl.arange(0, NB_P)
        ebs = tl.load(EB + r * nblk_pad + ob, mask=ob < nblk_pad, other=0.0)
        stot = tl.sum(ebs, 0)
        before = tl.sum(tl.where(ob < bk, ebs, 0.0), 0)
        sj = tl.cumsum(ex, 0) + before
        Z = tl.load(ZT + r * 2).to(tl.float64)
        vmax = tl.load(VMAX + r)
        u = sj / stot * Z
        up = (sj - ex) / stot * Z
        i = tl.minimum(_bsearch(WCDF, base, u, NI, LOGN), NI - 1)
        ip = tl.minimum(_bsearch(WCDF, base, up, NI, LOGN), NI - 1)
        v = tl.load(LOGITS + base + i, mask=ok, other=float("-inf"))
        c = tl.load(CNTS + base + i, mask=ok, other=0.0)
        taken = tl.load(HOFF + base + i, mask=ok, other=0.0) - tl.load(HOFF + base + i - 1, mask=ok & (i > 0), other=0.0)
        e = tl.exp2((v - vmax) * inv_T_log2e)
        rem = c - taken
        start = tl.load(WCDF + base + i, mask=ok, other=0.0) - (rem * e).to(tl.float64)
        within = tl.minimum(tl.maximum((u - start) / tl.maximum(e.to(tl.float64), 1e-300), 0.0),
                            tl.maximum(rem - 1.0, 0.0).to(tl.float64)).to(tl.int32)
        okp = ok & (j > 0)
        vp = tl.load(LOGITS + base + ip, mask=okp, other=float("-inf"))
        cp = tl.load(CNTS + base + ip, mask=okp, other=0.0)
        takenp = tl.load(HOFF + base + ip, mask=okp, other=0.0) - tl.load(HOFF + base + ip - 1, mask=okp & (ip > 0), other=0.0)
        ep = tl.exp2((vp - vmax) * inv_T_log2e)
        remp = cp - takenp
        startp = tl.load(WCDF + base + ip, mask=okp, other=0.0) - (remp * ep).to(tl.float64)
        withinp = tl.minimum(tl.maximum((up - startp) / tl.maximum(ep.to(tl.float64), 1e-300), 0.0),
                             tl.maximum(remp - 1.0, 0.0).to(tl.float64)).to(tl.int32)
        dup = (j > 0) & (ip == i) & (withinp == within)
        is_b = i >= NP
        st = tl.load(BSTART + hk * NP + i, mask=ok & (~is_b), other=0)
        ka = tl.load(PERM + hk * nA + tl.minimum(st + taken.to(tl.int32) + within, nA - 1), mask=ok & (~is_b), other=0)
        key = tl.where(is_b, A_end + (i - NP), ka)
        p = e.to(tl.float64) / tl.maximum(Z, 1e-300)
        pi = 1.0 - tl.exp(M * tl.log(tl.maximum(1.0 - p, 1e-300)))
        pi = tl.minimum(tl.maximum(pi, 1e-4), 1.0).to(tl.float32)
        wt = tl.where(ok & (~dup) & (rem > 0) & (Z > 0), 1.0 / pi, 0.0)
        slot = SINKc + WINc + KH + j
    d = pos - key
    if MODE == 0:
        bias = tl.load(PB + h * spb_h + key, mask=ok, other=0.0).to(tl.float32)
    else:
        bias = tl.load(TB + qn * stb_q + h * stb_h + d, mask=ok & (d >= 0) & (d < R), other=0.0).to(tl.float32)
    tl.store(IDX + r * S + slot, key.to(tl.int32), mask=ok)
    tl.store(W + r * S + slot, wt * tl.exp(bias), mask=ok)


@triton.jit
def _bump_seed(SEED):
    tl.store(SEED, tl.load(SEED) + 1)


class SSAFusedV3(SSAFused):
    BS = 256

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dev = self.row.codes.device
        rows = self.lut.shape[0]
        self.vmax = torch.empty(rows, dtype=torch.float32, device=dev)
        self.nblk = triton.cdiv(self.m + 1, self.BS)
        self.eb = torch.zeros(rows, self.nblk, dtype=torch.float64, device=dev)
        self.ni_pad = triton.next_power_of_2(self.NI)
        self.n_split = max(16, min(64, triton.cdiv(self.S, 64)))
        self.op = torch.empty(rows * self.n_split, 128, dtype=torch.float32, device=dev)
        self.ml = torch.empty(rows * self.n_split, 2, dtype=torch.float32, device=dev)

    def select_fused(self, q, pos, pb=None, tb=None):
        c, HQ = self.ctx, self.HQ
        rows = q.shape[0] * HQ
        _bump_seed[(1,)](self.seed)
        logits, cnts = self.items(q, pos, pb, tb)
        hoff, wcdf, zt, vmax = self.hoff[:rows], self.wcdf[:rows], self.zt[:rows], self.vmax[:rows]
        idx, w, eb = self.idx_buf[:rows], self.w_buf[:rows], self.eb[:rows]
        inv = LOG2E / self.T
        _ssa_select_kernel[(rows,)](logits, cnts, hoff, wcdf, zt, vmax, self.NI, self.k_h, inv,
                                    NI_PAD=self.ni_pad, num_warps=16)
        if self.m > 0:
            _ssa_expblk_kernel[(rows, self.nblk)](self.seed, eb, self.m, self.nblk, BS=self.BS)
        mode = 0 if pb is not None else 1
        pb_ = pb if pb is not None else logits
        tb_ = tb if tb is not None else logits
        _ssa_slots3_kernel[(rows, 1 + self.n_hblk + self.n_dblk)](
            logits, cnts, hoff, wcdf, zt, eb, vmax, c.bstart32, c.perm, pos, pb_, tb_, idx, w, self.seed,
            self.NI, self.k_h, self.m, self.S, c.A_end, c.nA, pb.stride(0) if pb is not None else 0,
            tb.stride(0) if tb is not None else 0, tb.stride(1) if tb is not None else 0, inv, self.n_hblk,
            self.nblk, HQ=HQ, G=self.G, NP=NPAIR, SINKc=SINK, WINc=WINDOW, R=R_BIAS, MODE=mode, LOGN=self.logn,
            BS=self.BS, NB_P=triton.next_power_of_2(max(self.nblk, 1)), num_warps=4)
        return idx, w


# ============================================================================================================
# v4: threshold search 2 bits per pass (3-way tuple reduction), exponential block sums folded into the same
# per-row kernel, seed bump folded into the LUT kernel  ->  lut, items, select, slots, attn, merge (6 launches)
# ============================================================================================================
@triton.jit
def _add3(a0, a1, a2, b0, b1, b2):
    return a0 + b0, a1 + b1, a2 + b2


@triton.jit
def _ssa_lut_seed_kernel(Q, C, LUT, SEED, sqn, sqh, HQ: tl.constexpr, G: tl.constexpr, NCc: tl.constexpr,
                         DS: tl.constexpr, scale):
    r = tl.program_id(0)
    if r == 0:
        tl.store(SEED, tl.load(SEED) + 1)
    qn = r // HQ
    h = r % HQ
    hk = h // G
    offs_c = tl.arange(0, NCc)
    offs_d = tl.arange(0, DS)
    for s in tl.static_range(2):
        q = tl.load(Q + qn * sqn + h * sqh + s * DS + offs_d).to(tl.float32)
        cen = tl.load(C + ((hk * 2 + s) * NCc + offs_c[:, None]) * DS + offs_d[None, :])
        tl.store(LUT + (r * 2 + s) * NCc + offs_c, tl.sum(cen * q[None, :], 1) * scale)


@triton.jit
def _ssa_select4_kernel(LOGITS, CNTS, HOFF, WCDF, ZT, VMAX, SEED, EB, NI, KH, M, nblk_pad, inv_T_log2e,
                        NI_PAD: tl.constexpr, BS: tl.constexpr, CH: tl.constexpr):
    r = tl.program_id(0)
    it = tl.arange(0, NI_PAD)
    ok = it < NI
    v = tl.load(LOGITS + r * NI + it, mask=ok, other=float("-inf"))
    c = tl.load(CNTS + r * NI + it, mask=ok, other=0.0)
    c = tl.where(v == float("-inf"), 0.0, c)
    key = _ord_key(v)
    vmax = tl.max(v, 0)
    total = tl.sum(c, 0)
    kh = tl.minimum(KH * 1.0, total)
    prefix = tl.full([], 0, tl.uint32)
    for k in tl.static_range(16):
        c1 = prefix | tl.full([], 1 << (30 - 2 * k), tl.uint32)
        c2 = prefix | tl.full([], 2 << (30 - 2 * k), tl.uint32)
        c3 = prefix | tl.full([], 3 << (30 - 2 * k), tl.uint32)
        s1, s2, s3 = tl.reduce((tl.where(key >= c1, c, 0.0), tl.where(key >= c2, c, 0.0),
                                tl.where(key >= c3, c, 0.0)), 0, _add3)
        prefix = tl.where(s3 >= kh, c3, tl.where(s2 >= kh, c2, tl.where(s1 >= kh, c1, prefix)))
    gt = key > prefix
    eq = key == prefix
    cnt_gt = tl.sum(tl.where(gt, c, 0.0), 0)
    need_tie = kh - cnt_gt
    ctie = tl.where(eq, c, 0.0)
    tie_before = tl.cumsum(ctie, 0) - ctie
    taken = tl.where(gt, c, tl.where(eq, tl.minimum(tl.maximum(need_tie - tie_before, 0.0), c), 0.0))
    taken = tl.where(kh > 0, taken, 0.0)
    tl.store(HOFF + r * NI + it, tl.cumsum(taken, 0), mask=ok)
    e = tl.where(c > 0, tl.exp2((v - vmax) * inv_T_log2e), 0.0)
    wt = ((c - taken) * e).to(tl.float64)
    tl.store(WCDF + r * NI + it, tl.cumsum(wt, 0), mask=ok)
    tl.store(ZT + r * 2, tl.sum(wt, 0).to(tl.float32))
    tl.store(ZT + r * 2 + 1, tl.sum(taken, 0))
    tl.store(VMAX + r, vmax)
    # block sums of the m+1 Exp(1) variates (same Philox stream the slot kernel regenerates)
    seed = tl.load(SEED)
    for j0 in range(0, M + 1, CH * BS):
        j = j0 + tl.arange(0, CH * BS)
        u = tl.rand(seed, r * (M + 1) + j)
        ex = tl.where(j < M + 1, -tl.log(tl.maximum(u, 1e-12)).to(tl.float64), 0.0)
        bs = tl.sum(tl.reshape(ex, (CH, BS)), 1)
        bk = j0 // BS + tl.arange(0, CH)
        tl.store(EB + r * nblk_pad + bk, bs, mask=bk < nblk_pad)


class SSAFusedV4(SSAFusedV3):
    def items(self, q, pos, pb, tb):
        c, HQ = self.ctx, self.HQ
        rows = q.shape[0] * HQ
        lut, logits, cnts = self.lut[:rows], self.logits[:rows], self.cnts[:rows]
        _ssa_lut_seed_kernel[(rows,)](q, c.C, lut, self.seed, q.stride(0), q.stride(1), HQ=HQ, G=self.G, NCc=NC,
                                      DS=c.dsub, scale=self.scaling)
        mode = 0 if pb is not None else 1
        pb_ = pb if pb is not None else logits
        tb_ = tb if tb is not None else logits
        _ssa_items_kernel[(rows, triton.cdiv(self.NI, self.BLOCK_ITEMS))](
            lut, self.cnt_flat, self.row.codes, pos, pb_, tb_, logits, cnts, c.A_end, self.WB, self.NI, c.max_len,
            pb.stride(0) if pb is not None else 0, tb.stride(0) if tb is not None else 0,
            tb.stride(1) if tb is not None else 0, HQ=HQ, G=self.G, NCc=NC, NP=NPAIR, WINDOW=WINDOW, R=R_BIAS,
            MODE=mode, BLOCK=self.BLOCK_ITEMS, num_warps=4)
        return logits, cnts

    def select_fused(self, q, pos, pb=None, tb=None):
        c, HQ = self.ctx, self.HQ
        rows = q.shape[0] * HQ
        logits, cnts = self.items(q, pos, pb, tb)
        hoff, wcdf, zt, vmax = self.hoff[:rows], self.wcdf[:rows], self.zt[:rows], self.vmax[:rows]
        idx, w, eb = self.idx_buf[:rows], self.w_buf[:rows], self.eb[:rows]
        inv = LOG2E / self.T
        _ssa_select4_kernel[(rows,)](logits, cnts, hoff, wcdf, zt, vmax, self.seed, eb, self.NI, self.k_h, self.m,
                                     self.nblk, inv, NI_PAD=self.ni_pad, BS=self.BS, CH=4, num_warps=16)
        mode = 0 if pb is not None else 1
        pb_ = pb if pb is not None else logits
        tb_ = tb if tb is not None else logits
        _ssa_slots3_kernel[(rows, 1 + self.n_hblk + self.n_dblk)](
            logits, cnts, hoff, wcdf, zt, eb, vmax, c.bstart32, c.perm, pos, pb_, tb_, idx, w, self.seed,
            self.NI, self.k_h, self.m, self.S, c.A_end, c.nA, pb.stride(0) if pb is not None else 0,
            tb.stride(0) if tb is not None else 0, tb.stride(1) if tb is not None else 0, inv, self.n_hblk,
            self.nblk, HQ=HQ, G=self.G, NP=NPAIR, SINKc=SINK, WINc=WINDOW, R=R_BIAS, MODE=mode, LOGN=self.logn,
            BS=self.BS, NB_P=triton.next_power_of_2(max(self.nblk, 1)), num_warps=4)
        return idx, w
