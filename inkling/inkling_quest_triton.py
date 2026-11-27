"""Fused Triton selection + merge for the Inkling Quest decode operator, plus CUDA-graph capture.

Replaces the PyTorch glue of `InklingQuestDecode` (inkling_sparse_op.py):
  * quest_score:   one pass over kmin/kmax -> max over the GQA group of
                   sum_d max(q*kmax, q*kmin)  (same value as relu(q).kmax + min(q,0).kmin)
                   written as fp32 [B*H_kv, P_mid]. Replaces 2 matmuls + add + amax.
  * top-k:         torch.topk on the (16x smaller than token-level) page scores.
  * page_ids:      logical page -> virtual page id (phys_page * H_kv + kv_head), int32, in place.
  * bias_weights:  exp(rel_bias reversed) straight into the persistent sparse_weights buffer.
  * merge:         LSE merge of the two partial outputs + Inkling's per-token sink, one kernel.
`GraphedDecode` captures the whole step (all of the above + the two skylight kernels) in one
CUDA graph, removing ~10 kernel-launch latencies per step.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import triton
import triton.language as tl

from inkling_sparse_op import InklingQuestDecode


@triton.jit
def _quest_score_kernel(q_ptr, kmin_ptr, kmax_ptr, out_ptr, n_pages, mid_lo, n_mid,
                        G: tl.constexpr, D: tl.constexpr, BLOCK_P: tl.constexpr):
    row = tl.program_id(0)                      # b * H_kv + kv_head
    offs_p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < n_mid
    offs_d = tl.arange(0, D)
    ptrs = row.to(tl.int64) * n_pages * D + (mid_lo + offs_p)[:, None].to(tl.int64) * D + offs_d[None, :]
    kmx = tl.load(kmax_ptr + ptrs, mask=mask_p[:, None], other=0.0).to(tl.float32)
    kmn = tl.load(kmin_ptr + ptrs, mask=mask_p[:, None], other=0.0).to(tl.float32)
    best = tl.full([BLOCK_P], float("-inf"), tl.float32)
    for g in tl.static_range(G):
        qg = tl.load(q_ptr + (row * G + g) * D + offs_d).to(tl.float32)
        s = tl.sum(tl.maximum(qg[None, :] * kmx, qg[None, :] * kmn), axis=1)
        best = tl.maximum(best, s)
    tl.store(out_ptr + row.to(tl.int64) * n_mid + offs_p, best, mask=mask_p)


@triton.jit
def _page_ids_kernel(top_ptr, out_ptr, n_pages, mid_lo, H_KV: tl.constexpr, K: tl.constexpr,
                     BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    m = offs < K
    logical = tl.load(top_ptr + row * K + offs, mask=m, other=0) + mid_lo
    b = row // H_KV
    h = row % H_KV
    vid = (b * n_pages + logical) * H_KV + h
    tl.store(out_ptr + row * K + offs, vid.to(tl.int32), mask=m)


@triton.jit
def _bias_weights_kernel(bias_ptr, w_ptr, S_A, sink: tl.constexpr, W: tl.constexpr):
    row = tl.program_id(0)                      # b * H_q + q_head
    offs = tl.arange(0, W)
    bias = tl.load(bias_ptr + row * W + (W - 1 - offs))   # local token i has distance W-1-i
    tl.store(w_ptr + row * S_A + sink + offs, tl.exp(bias.to(tl.float32)))


@triton.jit
def _merge_kernel(oa_ptr, la_ptr, ob_ptr, lb_ptr, sink_ptr, out_ptr, D: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, D)
    la = tl.load(la_ptr + row)
    lb = tl.load(lb_ptr + row)                  # log2 units (FlashInfer convention)
    m = tl.maximum(la, lb)
    wa = tl.exp2(la - m)
    wb = tl.exp2(lb - m)
    lse = (m + tl.log2(wa + wb)) * 0.6931471805599453
    gate = tl.sigmoid(lse - tl.load(sink_ptr + row).to(tl.float32))
    oa = tl.load(oa_ptr + row * D + offs).to(tl.float32)
    ob = tl.load(ob_ptr + row * D + offs).to(tl.float32)
    o = (oa * wa + ob * wb) / (wa + wb) * gate
    tl.store(out_ptr + row * D + offs, o.to(out_ptr.dtype.element_ty))


class InklingQuestDecodeFused(InklingQuestDecode):
    BLOCK_P = 32

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dev = self.vidx.device
        self.n_mid = self.mid_end - self.mid_start
        self.scores = torch.empty(self.B * self.H_kv, self.n_mid, dtype=torch.float32, device=dev)
        self.w_a = torch.ones(self.B, self.H_q, self.idx_a.shape[-1], dtype=torch.float32, device=dev)
        self.out = torch.empty(self.B, self.H_q, self.D, dtype=torch.bfloat16, device=dev)

    def select(self, q: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor) -> torch.Tensor:
        rows = self.B * self.H_kv
        grid = (rows, triton.cdiv(self.n_mid, self.BLOCK_P))
        _quest_score_kernel[grid](q, kmin, kmax, self.scores, self.n_pages, self.mid_start, self.n_mid,
                                  G=self.G, D=self.D, BLOCK_P=self.BLOCK_P, num_warps=4)
        top = torch.topk(self.scores, self.k, dim=-1, sorted=False).indices      # [rows, k] int64
        _page_ids_kernel[(rows,)](top, self.vidx, self.n_pages, self.mid_start, H_KV=self.H_kv,
                                  K=self.k, BLOCK_K=triton.next_power_of_2(self.k))
        self.last_pages = (top + self.mid_start).view(self.B, self.H_kv, self.k)
        return self.last_pages

    def __call__(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                 kmin: torch.Tensor, kmax: torch.Tensor, rel_bias: torch.Tensor,
                 sinks: torch.Tensor, timings: Optional[Dict[str, list]] = None) -> torch.Tensor:
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if timings is not None else None
        if ev:
            ev[0].record()
        self.select(q, kmin, kmax)
        if ev:
            ev[1].record()
        rows_q = self.B * self.H_q
        _bias_weights_kernel[(rows_q,)](rel_bias, self.w_a, self.w_a.shape[-1],
                                        sink=self.sink_tokens, W=self.local_tokens)
        out_a, lse_a = self.sparse.run(q, (k_cache, v_cache), sparse_len=self.len_a,
                                       sparse_idx=self.idx_a, sparse_weights=self.w_a, return_lse=True)
        if ev:
            ev[2].record()
        nv = self.B * self.H_kv
        kv_virtual = (k_cache.view(-1, 1, self.page, self.D), v_cache.view(-1, 1, self.page, self.D))
        out_b, lse_b = self.dense.run(q.view(nv, self.G, self.D), kv_virtual, return_lse=True)
        if ev:
            ev[3].record()
        _merge_kernel[(rows_q,)](out_a, lse_a.contiguous(), out_b, lse_b.contiguous(), sinks,
                                 self.out, D=self.D)
        if ev:
            ev[4].record()
            torch.cuda.synchronize()
            for name, (a, b) in {"select": (0, 1), "sink_local": (1, 2), "pages": (2, 3),
                                 "merge": (3, 4)}.items():
                timings.setdefault(name, []).append(ev[a].elapsed_time(ev[b]))
        return self.out


class GraphedDecode:
    """Capture one decode step of `op` into a CUDA graph over static input buffers."""

    def __init__(self, op: InklingQuestDecodeFused, q, k_cache, v_cache, kmin, kmax, rel_bias, sinks):
        self.op = op
        self.static = [t.clone() if t is not None else None for t in (q, rel_bias, sinks)]
        self.cache = (k_cache, v_cache, kmin, kmax)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):                  # warm up (Triton JIT, allocator) off the main stream
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.result = self._step()

    def _step(self) -> torch.Tensor:
        q, bias, sinks = self.static
        k, v, kmin, kmax = self.cache
        return self.op(q, k, v, kmin, kmax, bias, sinks)

    def __call__(self, q, rel_bias, sinks) -> torch.Tensor:
        for dst, src in zip(self.static, (q, rel_bias, sinks)):
            dst.copy_(src)
        self.graph.replay()
        return self.result
