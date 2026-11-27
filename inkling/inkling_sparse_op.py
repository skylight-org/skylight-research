"""Inkling 2 Mini sparse decode operator: Quest page selection on top of skylight_kernels.

One full-attention decode step (q_len = 1) for H_q=24 / H_kv=8 / D=128, paged KV in HND layout:

  A. explicit tokens   sink [0, S) U local [L-W, L)   -> sparse_optimized kernel, with Inkling's
                       relative-position bias as per-token weights (weighted = qk + log w)
  B. selected pages    top-k "middle" pages per (request, kv head) by the Quest upper bound
                       score(page) = relu(q).kmax + min(q,0).kmin   (two batched matmuls, shared
                       across the 3 q-heads of a kv group) -> the *dense* paged kernel, run over
                       virtual requests (request, kv_head) whose page tables list only the chosen
                       pages: an HND cache [P, H_kv, page, D] views as [P*H_kv, 1, page, D] for free
  merge                A and B by log-sum-exp, then Inkling's per-token sink logit
                       out *= sigmoid(lse - sink)

No new CUDA: only existing kernels + PyTorch. Prototype assumptions (benchmark layout): every
request has the same length L (a multiple of the page size) and its pages are physically
contiguous; page min/max summaries are kept per request as [B, H_kv, n_pages, D].
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from original_optimized import BatchDecodeWithPagedKVCacheWrapper as DenseDecode
from sparse_optimized import BatchDecodeWithPagedKVCacheWrapper as SparseDecode

LN2 = math.log(2.0)


def page_summaries(k_cache: torch.Tensor, B: int, n_pages: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """k_cache HND [B*n_pages, H_kv, page, D] -> (kmin, kmax) each [B, H_kv, n_pages, D]."""
    k = k_cache.view(B, n_pages, *k_cache.shape[1:])                  # [B, P, H, page, D]
    return (k.amin(dim=3).permute(0, 2, 1, 3).contiguous(),
            k.amax(dim=3).permute(0, 2, 1, 3).contiguous())


class InklingQuestDecode:
    def __init__(self, B: int, seq_len: int, top_pages: int, *, H_q: int = 24, H_kv: int = 8,
                 D: int = 128, page: int = 16, sink_tokens: int = 64, local_tokens: int = 1024,
                 dtype: torch.dtype = torch.bfloat16, device: str = "cuda") -> None:
        assert seq_len % page == 0 and sink_tokens % page == 0 and local_tokens % page == 0
        self.B, self.L, self.H_q, self.H_kv, self.D, self.page = B, seq_len, H_q, H_kv, D, page
        self.G = H_q // H_kv
        self.n_pages = seq_len // page
        self.mid_start = sink_tokens // page
        self.mid_end = self.n_pages - local_tokens // page
        self.k = min(top_pages, self.mid_end - self.mid_start)
        self.sink_tokens, self.local_tokens = sink_tokens, local_tokens
        i32 = dict(dtype=torch.int32, device=device)

        def ws() -> torch.Tensor:
            return torch.zeros(256 << 20, dtype=torch.uint8, device=device)

        # A: explicit sink+local tokens over the real page table (H_q=24, H_kv=8)
        self.sparse = SparseDecode(ws(), "HND")
        self.sparse.plan(torch.arange(0, B * self.n_pages + 1, self.n_pages, **i32),
                         torch.arange(B * self.n_pages, **i32), torch.full((B,), page, **i32),
                         H_q, H_kv, D, page, q_data_type=dtype, kv_data_type=dtype)
        idx = torch.cat([torch.arange(sink_tokens, device=device),
                         torch.arange(seq_len - local_tokens, seq_len, device=device)])
        self.idx_a = idx.expand(B, H_q, -1).contiguous()                  # int64 [B, H_q, S_A]
        self.len_a = torch.full((B, H_q, 1), idx.numel(), **i32)
        self.ones_sink = torch.ones(B, H_q, sink_tokens, device=device)

        # B: dense kernel over virtual (request, kv_head) requests, k pages each
        nv = B * H_kv
        self.vidx = torch.zeros(nv * self.k, **i32)                       # rewritten every step
        self.dense = DenseDecode(ws(), "HND")
        self.dense.plan(torch.arange(0, nv * self.k + 1, self.k, **i32), self.vidx,
                        torch.full((nv,), page, **i32), self.G, 1, D, page,
                        q_data_type=dtype, kv_data_type=dtype)
        self.req_page0 = (torch.arange(B, device=device) * self.n_pages).view(B, 1, 1)
        self.kv_head = torch.arange(H_kv, device=device).view(1, H_kv, 1)
        self.last_pages: Optional[torch.Tensor] = None

    def select(self, q: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor) -> torch.Tensor:
        """Top-k middle pages per (request, kv head): logical page ids [B, H_kv, k]."""
        qg = q.view(self.B, self.H_kv, self.G, self.D)
        lo, hi = self.mid_start, self.mid_end
        s = (qg.clamp(min=0) @ kmax[:, :, lo:hi].transpose(-1, -2)
             + qg.clamp(max=0) @ kmin[:, :, lo:hi].transpose(-1, -2))     # [B, H_kv, G, P_mid]
        s = s.amax(dim=2)
        return torch.topk(s, self.k, dim=-1, sorted=False).indices + lo

    def __call__(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                 kmin: torch.Tensor, kmax: torch.Tensor, rel_bias: torch.Tensor,
                 sinks: torch.Tensor, timings: Optional[Dict[str, list]] = None) -> torch.Tensor:
        """q [B,H_q,D]; caches HND [B*n_pages, H_kv, page, D]; rel_bias [B,H_q,W] (distance 0..W-1);
        sinks [B,H_q] (natural-log logits). Returns [B, H_q, D]."""
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if timings is not None else None
        if ev:
            ev[0].record()
        pages = self.select(q, kmin, kmax)
        self.vidx.copy_(((self.req_page0 + pages) * self.H_kv + self.kv_head).reshape(-1))
        self.last_pages = pages
        if ev:
            ev[1].record()
        w_a = torch.cat([self.ones_sink, rel_bias.flip(-1).float().exp()], dim=-1)
        out_a, lse_a = self.sparse.run(q, (k_cache, v_cache), sparse_len=self.len_a,
                                       sparse_idx=self.idx_a, sparse_weights=w_a, return_lse=True)
        if ev:
            ev[2].record()
        nv = self.B * self.H_kv
        kv_virtual = (k_cache.view(-1, 1, self.page, self.D), v_cache.view(-1, 1, self.page, self.D))
        out_b, lse_b = self.dense.run(q.view(nv, self.G, self.D), kv_virtual, return_lse=True)
        out_b, lse_b = out_b.view_as(q), lse_b.view(self.B, self.H_q)
        if ev:
            ev[3].record()
        m = torch.maximum(lse_a, lse_b)
        wa, wb = torch.exp2(lse_a - m), torch.exp2(lse_b - m)
        lse = (m + torch.log2(wa + wb)) * LN2
        out = (out_a.float() * wa[..., None] + out_b.float() * wb[..., None]) / (wa + wb)[..., None]
        out = (out * torch.sigmoid(lse - sinks.float())[..., None]).to(q.dtype)
        if ev:
            ev[4].record()
            torch.cuda.synchronize()
            for name, (a, b) in {"select": (0, 1), "sink_local": (1, 2), "pages": (2, 3),
                                 "merge": (3, 4)}.items():
                timings.setdefault(name, []).append(ev[a].elapsed_time(ev[b]))
        return out

    def reference(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                  rel_bias: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
        """PyTorch fp32 attention over exactly the tokens the operator used (after a __call__)."""
        B, P, page = self.B, self.n_pages, self.page
        K = k_cache.view(B, P, self.H_kv, page, self.D).permute(0, 2, 1, 3, 4).reshape(B, self.H_kv, -1, self.D)
        V = v_cache.view(B, P, self.H_kv, page, self.D).permute(0, 2, 1, 3, 4).reshape(B, self.H_kv, -1, self.D)
        out = torch.empty(B, self.H_q, self.D, device=q.device)
        base = self.idx_a[0, 0]
        for b in range(B):
            for h in range(self.H_q):
                kvh = h // self.G
                pg = self.last_pages[b, kvh]
                toks = torch.cat([base, (pg[:, None] * page + torch.arange(page, device=q.device)).reshape(-1)])
                logits = (q[b, h].float() @ K[b, kvh, toks].float().T) * self.D ** -0.5
                dist = (self.L - 1) - toks
                near = dist < rel_bias.shape[-1]
                logits = logits + torch.where(near, rel_bias[b, h].float()[dist.clamp(max=rel_bias.shape[-1] - 1)], 0.0)
                p = torch.softmax(torch.cat([logits, sinks[b, h:h + 1].float()]), dim=-1)[:-1]
                out[b, h] = p @ V[b, kvh, toks].float()
        return out
