"""Wire the fused Quest operator into Inkling 2 Mini's real HF decode loop.

After the (dense, chunked) context prefill, the K/V storage of the 8 full-attention cache layers
(visits 5, 11, 17, 23, 29, 35, 41, 47) moves into paged HND buffers with per-page key min/max
summaries; the layers keep their short-convolution state. During decode (q_len = 1):

  * `layer.update` appends the new K/V into its page (and updates that page's summary) instead of
    `torch.cat`-ing the whole cache; `layer.get_seq_length` reports the paged length, so masks and
    Inkling's relative distances stay correct.
  * the attention callable sends each full-attention visit to `PagedQuestState.attend`:
      A  sink [0,64) U local [16*p0, L), p0 = (L-1024)//16   -> sparse_optimized, with weights
         exp(position_bias) (Inkling's bias is already 0 beyond 1024 tokens)
      B  top-k pages of [4, p0) by the Quest bound (fused Triton score + torch.topk) -> dense kernel
         on virtual (request, kv_head) page tables
      merge by LSE + Inkling's per-token sink (fused Triton)
    Sliding visits and the attention-residual modules run Inkling's own eager kernel.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
import triton
from transformers.models.inkling2_loop import modeling_inkling2_loop as M

from inkling_quest_triton import _merge_kernel, _page_ids_kernel, _quest_score_kernel
from inkling_sah import InklingModelAdapterHF
from original_optimized import BatchDecodeWithPagedKVCacheWrapper as DenseDecode
from sparse_optimized import BatchDecodeWithPagedKVCacheWrapper as SparseDecode

PAGE, SINK, LOCAL = 16, 64, 1024
FULL_VISITS = (5, 11, 17, 23, 29, 35, 41, 47)
REGISTRY: Dict[int, "PagedQuestState"] = {}
TIMING: Dict[str, float] = defaultdict(float)


class PagedQuestState:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor, top_frac: float, max_len: int,
                 workspace: Optional[torch.Tensor] = None) -> None:
        B, H, L0, D = keys.shape
        dev, dt = keys.device, keys.dtype
        self.B, self.H_kv, self.D, self.G = B, H, D, 24 // H
        self.H_q = self.G * H
        self.max_pages = math.ceil(max_len / PAGE) + 1
        self.len = L0
        P = B * self.max_pages
        self.k_pages = torch.zeros(P, H, PAGE, D, dtype=dt, device=dev)
        self.v_pages = torch.zeros(P, H, PAGE, D, dtype=dt, device=dev)
        self.kmin = torch.zeros(B, H, self.max_pages, D, dtype=dt, device=dev)
        self.kmax = torch.zeros(B, H, self.max_pages, D, dtype=dt, device=dev)
        n0 = math.ceil(L0 / PAGE)
        pad = n0 * PAGE - L0
        for src, dst in ((keys, self.k_pages), (values, self.v_pages)):
            x = F.pad(src, (0, 0, 0, pad)).view(B, H, n0, PAGE, D).permute(0, 2, 1, 3, 4)
            dst.view(B, self.max_pages, H, PAGE, D)[:, :n0] = x
        kp = F.pad(keys, (0, 0, 0, pad)).view(B, H, n0, PAGE, D)
        valid = (torch.arange(n0 * PAGE, device=dev) < L0).view(1, 1, n0, PAGE, 1)
        self.kmin[:, :, :n0] = kp.masked_fill(~valid, float("inf")).amin(dim=3)
        self.kmax[:, :, :n0] = kp.masked_fill(~valid, float("-inf")).amax(dim=3)

        n_mid0 = (L0 - LOCAL) // PAGE - SINK // PAGE
        assert n_mid0 >= 1, "context too short for sink+local+pages"
        self.k = max(1, min(n_mid0, round(top_frac * n_mid0)))
        i32 = dict(dtype=torch.int32, device=dev)
        ws = lambda: torch.zeros(128 << 20, dtype=torch.uint8, device=dev)  # noqa: E731
        self.sparse = SparseDecode(ws(), "HND")
        self.sparse.plan(torch.arange(0, P + 1, self.max_pages, **i32), torch.arange(P, **i32),
                         torch.full((B,), PAGE, **i32), self.H_q, H, D, PAGE, q_data_type=dt, kv_data_type=dt)
        s_max = SINK + LOCAL + PAGE
        self.idx_a = torch.zeros(B, self.H_q, s_max, dtype=torch.int64, device=dev)
        self.idx_a[..., :SINK] = torch.arange(SINK, device=dev)
        self.ar = torch.arange(LOCAL + PAGE, device=dev)
        self.len_a = torch.zeros(B, self.H_q, 1, **i32)
        self.w_a = torch.ones(B, self.H_q, s_max, dtype=torch.float32, device=dev)
        nv = B * H
        self.vidx = torch.zeros(nv * self.k, **i32)
        self.dense = DenseDecode(ws(), "HND")
        self.dense.plan(torch.arange(0, nv * self.k + 1, self.k, **i32), self.vidx,
                        torch.full((nv,), PAGE, **i32), self.G, 1, D, PAGE, q_data_type=dt, kv_data_type=dt)
        self.scores = torch.empty(nv, self.max_pages, dtype=torch.float32, device=dev)
        self.out = torch.empty(B, self.H_q, D, dtype=dt, device=dev)
        self.kv_virtual = (self.k_pages.view(-1, 1, PAGE, D), self.v_pages.view(-1, 1, PAGE, D))

    # ---- cache-layer hooks ----
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any):
        q = key_states.shape[2]
        assert q == 1, "paged full-attention cache only supports single-token decode"
        pos = self.len
        p, slot = divmod(pos, PAGE)
        k, v = key_states[:, :, 0], value_states[:, :, 0]
        pages = self.k_pages.view(self.B, self.max_pages, self.H_kv, PAGE, self.D)
        pages[:, p, :, slot] = k
        self.v_pages.view(self.B, self.max_pages, self.H_kv, PAGE, self.D)[:, p, :, slot] = v
        if slot == 0:
            self.kmin[:, :, p] = k
            self.kmax[:, :, p] = k
        else:
            self.kmin[:, :, p] = torch.minimum(self.kmin[:, :, p], k)
            self.kmax[:, :, p] = torch.maximum(self.kmax[:, :, p], k)
        self.len += 1
        return key_states, value_states

    def get_seq_length(self) -> int:
        return self.len

    # ---- attention ----
    def attend(self, q: torch.Tensor, position_bias: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
        """q [B,H_q,D]; position_bias [B,H_q,L] (per key position); sinks [B,H_q]."""
        L = self.len
        p0 = (L - LOCAL) // PAGE
        local_start = p0 * PAGE
        n_local = L - local_start
        n_mid = p0 - SINK // PAGE
        mid_lo = SINK // PAGE
        # A: sink + local, bias as weights
        self.idx_a[..., SINK:SINK + n_local] = self.ar[:n_local] + local_start
        self.len_a.fill_(SINK + n_local)
        self.w_a[..., SINK:SINK + n_local] = position_bias[..., local_start:L].float().exp()
        out_a, lse_a = self.sparse.run(q, (self.k_pages, self.v_pages), sparse_len=self.len_a,
                                       sparse_idx=self.idx_a, sparse_weights=self.w_a, return_lse=True)
        # B: Quest page selection + dense kernel over chosen pages
        rows = self.B * self.H_kv
        scores = self.scores.view(-1)[: rows * n_mid].view(rows, n_mid)
        _quest_score_kernel[(rows, triton.cdiv(n_mid, 32))](
            q, self.kmin, self.kmax, scores, self.max_pages, mid_lo, n_mid,
            G=self.G, D=self.D, BLOCK_P=32, num_warps=4)
        top = torch.topk(scores, self.k, dim=-1, sorted=False).indices.contiguous()
        _page_ids_kernel[(rows,)](top, self.vidx, self.max_pages, mid_lo, H_KV=self.H_kv, K=self.k,
                                  BLOCK_K=triton.next_power_of_2(self.k))
        out_b, lse_b = self.dense.run(q.view(rows, self.G, self.D), self.kv_virtual, return_lse=True)
        _merge_kernel[(self.B * self.H_q,)](out_a, lse_a.contiguous(), out_b, lse_b.contiguous(),
                                            sinks.float().contiguous(), self.out, D=self.D)
        return self.out


# Sliding visits attend over only the keys their window can reach (exact). With transformers 5.24.1
# this is a no-op guard: LinearAttentionAndSlidingWindowAttentionLayer already keeps only 1023 keys.
SLICE_SLIDING = True
TIMING_ON = False      # time full-attention calls (adds a sync per call; only during decode)


def install_paged(cache: Any, top_frac: float, max_len: int) -> None:
    """Move the full-attention visits' K/V into paged Quest state (after prefill)."""
    REGISTRY.clear()
    for visit in FULL_VISITS:
        layer = cache.layers[visit]
        st = PagedQuestState(layer.keys, layer.values, top_frac, max_len)
        layer.keys = layer.values = torch.empty(0, device=layer.keys.device)
        layer.update = st.update
        layer.get_seq_length = st.get_seq_length
        REGISTRY[visit] = st
    st = REGISTRY[FULL_VISITS[0]]       # compile the Triton kernels now, not inside the timed decode
    st.attend(torch.zeros(st.B, st.H_q, st.D, dtype=st.out.dtype, device=st.out.device),
              torch.zeros(st.B, st.H_q, st.len, device=st.out.device),
              torch.zeros(st.B, st.H_q, device=st.out.device))
    torch.cuda.synchronize()


def make_paged_attention_fn() -> Callable:
    def fn(module, query, key, value, attention_mask, scaling, dropout=0.0, position_bias=None,
           sinks=None, **kwargs):
        kwargs.pop("sparse_meta_data", None)
        is_full = isinstance(module, M.Inkling2LoopFullAttention)
        timed = TIMING_ON and is_full
        if timed:
            torch.cuda.synchronize()
            t = time.perf_counter()
        st = REGISTRY.get(getattr(module, "_sah_visit_idx", -1)) if is_full else None
        if st is not None:
            assert query.shape[2] == 1
            out = st.attend(query[:, :, 0].contiguous(), position_bias[:, :, 0], sinks[:, :, 0, 0])
            res = (out.unsqueeze(1), None)
        else:
            if (SLICE_SLIDING and isinstance(module, M.Inkling2LoopSlidingAttention)
                    and module.sliding_window):
                # queries at positions [L-q, L) can only reach keys >= L-q-window: drop the rest.
                keep = query.shape[2] + module.sliding_window + 16
                if key.shape[2] > keep:
                    key, value = key[:, :, -keep:], value[:, :, -keep:]
                    if attention_mask is not None:
                        attention_mask = attention_mask[..., -keep:]
                    if position_bias is not None:
                        position_bias = position_bias[..., -keep:]
            res = M.eager_attention_forward(module, query, key, value, attention_mask, scaling=scaling,
                                            dropout=dropout, position_bias=position_bias, sinks=sinks, **kwargs)
        if timed:
            torch.cuda.synchronize()
            TIMING["full_attn_s"] += time.perf_counter() - t
            TIMING["full_attn_calls"] += 1
        return res

    return fn


class InklingPagedAdapterHF(InklingModelAdapterHF):
    """ModelAdapterHF whose sparse mode = paged fused-Quest decode on the 8 full-attention visits.

    The custom attention callable is installed for *all* forwards (prefill included), so the
    sliding-window slicing also speeds up the chunked prefill. `paged=False` keeps the
    full-attention visits dense (Inkling eager), the matching baseline. Question tokens are fed
    one at a time (hybrid=True)."""

    def __init__(self, *args: Any, top_frac: float = 0.02, paged: bool = True,
                 max_new_tokens: int = 256, decode_experts: Optional[str] = "batched_mm",
                 **kwargs: Any) -> None:
        kwargs.setdefault("hybrid", True)
        super().__init__(*args, **kwargs)
        self.top_frac, self.paged, self.max_new_tokens_cap = top_frac, paged, max_new_tokens
        # MoE implementation used only while decoding (q_len = 1). `batched_mm` gathers the 8 chosen
        # experts' weights per token: cheap for one token, ~10 GB for a 256-token prefill chunk, so
        # prefill keeps the model's default (grouped_mm).
        self.decode_experts = decode_experts
        self._experts = [m for m in self.model.modules() if isinstance(m, M.Inkling2LoopExperts)]
        self.last_timing: Dict[str, float] = {}
        name = self._ensure_attention_registered()
        for module in self.model.modules():
            if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
                module.config._attn_implementation = name

    def get_custom_attention_function(self, sparse_attention: Any) -> Callable:
        return make_paged_attention_fn()

    def _generate_response(self, question_tokens, context_outputs, sparse_meta_data, generation_kwargs, **kwargs):
        global TIMING_ON
        cache = context_outputs.past_key_values
        TIMING.clear()
        if self.paged:
            L0 = cache.layers[FULL_VISITS[0]].get_seq_length()
            install_paged(cache, self.top_frac,
                          L0 + question_tokens.shape[1] + generation_kwargs.get("max_new_tokens", 256) + 8)
        prefill_impl = self._experts[0].config._experts_implementation
        if self.decode_experts:
            for m in self._experts:
                m.config._experts_implementation = self.decode_experts
        torch.cuda.synchronize()
        TIMING_ON = True
        t = time.perf_counter()
        try:
            text = super()._generate_response(question_tokens, context_outputs, sparse_meta_data,
                                              generation_kwargs, **kwargs)
        finally:
            torch.cuda.synchronize()
            TIMING_ON = False
            for m in self._experts:
                m.config._experts_implementation = prefill_impl
            n_steps = TIMING.get("full_attn_calls", 0) / len(FULL_VISITS)
            self.last_timing = {
                "decode_wall_s": round(time.perf_counter() - t, 3),
                "steps": int(n_steps),
                "ms_per_step": round(1000 * (time.perf_counter() - t) / max(n_steps, 1), 2),
                "full_attn_ms_per_step": round(1000 * TIMING.get("full_attn_s", 0.0) / max(n_steps, 1), 3),
            }
            REGISTRY.clear()
        return text
