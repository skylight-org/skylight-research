"""CUDA-graph-safe Inkling decode: HF StaticCache + a graph-safe paged Quest operator.

HF's StaticCache keeps every visit's K/V in fixed buffers [B, H_kv, max_len, D], tracks the length in a
device tensor (`cumulative_length`) and writes with `index_copy_`, so masks / relative distances have a
fixed size (max_len) and a decode step has no host-dependent shapes -- it can be captured in a CUDA graph.

`StaticPagedQuest` runs Inkling's 8 full-attention visits sparsely *directly on those buffers*:
  [B, H_kv, max_len, D] viewed as [B*H_kv*P, 1, 16, D] = per-(request, kv_head) pages, so both
  skylight kernels run on B*H_kv virtual requests (3 q-heads each, GQA group 3). Everything that depends
  on the current length L (local-window start, #middle pages, the page summary to refresh, the bias
  gather) is computed on device from the layer's `cumulative_length`: no host syncs, static shapes.
"""

from __future__ import annotations

import copy
import warnings
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

import torch
import triton
import triton.language as tl
from transformers.models.inkling2_loop import modeling_inkling2_loop as M

from inkling_quest_triton import _merge_kernel
from original_optimized import BatchDecodeWithPagedKVCacheWrapper as DenseDecode
from sparse_optimized import BatchDecodeWithPagedKVCacheWrapper as SparseDecode

PAGE, SINK, LOCAL = 16, 64, 1024
FULL_VISITS = (5, 11, 17, 23, 29, 35, 41, 47)
STATIC_REGISTRY: Dict[int, "StaticPagedQuest"] = {}


@triton.jit
def _quest_score_dev_kernel(q_ptr, kmin_ptr, kmax_ptr, out_ptr, n_mid_ptr, n_pages, mid_lo, max_mid,
                            G: tl.constexpr, D: tl.constexpr, BLOCK_P: tl.constexpr):
    """Like `_quest_score_kernel`, but #valid middle pages is read from device memory; the rest of the
    fixed-width score row is -inf (so top-k never picks a page beyond the current length)."""
    row = tl.program_id(0)
    offs_p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
    n_mid = tl.load(n_mid_ptr)
    in_row = offs_p < max_mid
    valid = offs_p < n_mid
    offs_d = tl.arange(0, D)
    ptrs = row.to(tl.int64) * n_pages * D + (mid_lo + offs_p)[:, None].to(tl.int64) * D + offs_d[None, :]
    kmx = tl.load(kmax_ptr + ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
    kmn = tl.load(kmin_ptr + ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
    best = tl.full([BLOCK_P], float("-inf"), tl.float32)
    for g in tl.static_range(G):
        qg = tl.load(q_ptr + (row * G + g) * D + offs_d).to(tl.float32)
        best = tl.maximum(best, tl.sum(tl.maximum(qg[None, :] * kmx, qg[None, :] * kmn), axis=1))
    best = tl.where(valid, best, float("-inf"))
    tl.store(out_ptr + row.to(tl.int64) * max_mid + offs_p, best, mask=in_row)


class StaticPagedQuest:
    def __init__(self, layer: Any, top_frac: float) -> None:
        keys = layer.keys                                   # [B, H, max_len, D] static buffer
        B, H, max_len, D = keys.shape
        assert max_len % PAGE == 0
        dev, dt = keys.device, keys.dtype
        self.layer, self.B, self.H, self.D, self.G = layer, B, H, D, 24 // H
        self.P = max_len // PAGE
        self.rows = B * H
        self.max_len = max_len
        L0 = int(layer.cumulative_length)                   # host read once, at install time only
        self.mid_lo = SINK // PAGE
        self.max_mid = self.P - self.mid_lo - LOCAL // PAGE - 1
        n_mid0 = (L0 - LOCAL) // PAGE - self.mid_lo
        assert n_mid0 >= 1
        self.k = max(1, min(n_mid0, round(top_frac * n_mid0)))
        i32 = dict(dtype=torch.int32, device=dev)
        # page summaries for all pages, from the current contents (rows beyond L0 are refreshed later)
        kp = keys.view(B, H, self.P, PAGE, D)
        valid = (torch.arange(max_len, device=dev) < L0).view(1, 1, self.P, PAGE, 1)
        self.kmin = kp.masked_fill(~valid, float("inf")).amin(3).contiguous()
        self.kmax = kp.masked_fill(~valid, float("-inf")).amax(3).contiguous()
        self.slot_ar = torch.arange(PAGE, device=dev).view(1, 1, PAGE, 1)
        ws = lambda: torch.zeros(128 << 20, dtype=torch.uint8, device=dev)  # noqa: E731
        rows, G = self.rows, self.G
        # A: sink + local over virtual requests (one per (b, kv_head)); page table = that head's pages
        self.sparse = SparseDecode(ws(), "HND")
        self.sparse.plan(torch.arange(0, rows * self.P + 1, self.P, **i32), torch.arange(rows * self.P, **i32),
                         torch.full((rows,), PAGE, **i32), G, 1, D, PAGE, q_data_type=dt, kv_data_type=dt)
        s_max = SINK + LOCAL + PAGE
        self.idx_a = torch.zeros(rows, G, s_max, dtype=torch.int64, device=dev)
        self.idx_a[..., :SINK] = torch.arange(SINK, device=dev)
        self.ar_local = torch.arange(LOCAL + PAGE, device=dev)
        self.len_a = torch.zeros(rows, G, 1, **i32)
        self.w_a = torch.ones(rows, G, s_max, dtype=torch.float32, device=dev)
        # B: dense kernel over the k chosen middle pages of each virtual request
        self.vidx = torch.zeros(rows * self.k, **i32)
        self.dense = DenseDecode(ws(), "HND")
        self.dense.plan(torch.arange(0, rows * self.k + 1, self.k, **i32), self.vidx,
                        torch.full((rows,), PAGE, **i32), G, 1, D, PAGE, q_data_type=dt, kv_data_type=dt)
        self.row_base = (torch.arange(rows, device=dev) * self.P + self.mid_lo).view(rows, 1)
        self.scores = torch.empty(rows, self.max_mid, dtype=torch.float32, device=dev)
        self.n_mid = torch.zeros(1, dtype=torch.int32, device=dev)
        self.out = torch.empty(B, 24, D, dtype=dt, device=dev)

    def attend(self, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
               position_bias: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
        """q [B,24,D]; keys/values = the layer's static buffers; position_bias [B,24,max_len]; sinks [B,24]."""
        B, H, P, D, G, rows = self.B, self.H, self.P, self.D, self.G, self.rows
        L = self.layer.cumulative_length                    # device scalar, already includes this token
        # refresh the summary of the page holding the newest token (masked to its filled slots)
        pos = L - 1
        p = torch.div(pos, PAGE, rounding_mode="floor").view(1)
        tile = keys.view(B, H, P, PAGE, D).index_select(2, p)[:, :, 0]           # [B, H, PAGE, D]
        filled = self.slot_ar <= (pos % PAGE)
        self.kmin.index_copy_(2, p, tile.masked_fill(~filled, float("inf")).amin(2, keepdim=True))
        self.kmax.index_copy_(2, p, tile.masked_fill(~filled, float("-inf")).amax(2, keepdim=True))
        # A: sink + local window [16*p0, L), bias as weights
        p0 = torch.div(L - LOCAL, PAGE, rounding_mode="floor")
        local_start = p0 * PAGE
        self.idx_a[..., SINK:] = (local_start + self.ar_local).clamp(max=self.max_len - 1)
        self.len_a.copy_((SINK + L - local_start).to(torch.int32).expand_as(self.len_a))
        pb = position_bias.reshape(rows, G, -1)
        self.w_a[..., SINK:] = pb.gather(-1, self.idx_a[..., SINK:]).float().exp()
        qv = q.reshape(rows, G, D)
        kv = (keys.view(-1, 1, PAGE, D), values.view(-1, 1, PAGE, D))
        out_a, lse_a = self.sparse.run(qv, kv, sparse_len=self.len_a, sparse_idx=self.idx_a,
                                       sparse_weights=self.w_a, return_lse=True)
        # B: Quest top-k over the fixed-width middle-page range
        self.n_mid.copy_((p0 - self.mid_lo).to(torch.int32).view(1))
        _quest_score_dev_kernel[(rows, triton.cdiv(self.max_mid, 32))](
            q, self.kmin, self.kmax, self.scores, self.n_mid, P, self.mid_lo, self.max_mid,
            G=G, D=D, BLOCK_P=32, num_warps=4)
        top = torch.topk(self.scores, self.k, dim=-1, sorted=False).indices
        self.vidx.copy_((self.row_base + top).view(-1))
        out_b, lse_b = self.dense.run(qv, kv, return_lse=True)
        _merge_kernel[(B * 24,)](out_a, lse_a.contiguous(), out_b, lse_b.contiguous(),
                                 sinks.float().contiguous(), self.out, D=D)
        return self.out


def patch_static_sliding_update() -> None:
    """StaticSlidingWindowLayer.update builds `torch.tensor([-1], device=...)` on every full-window
    single-token update: a host->device copy that is illegal during CUDA-graph capture. Same logic,
    with the index tensor created once (outside capture, on the first eager decode step)."""
    from transformers import cache_utils as CU

    if getattr(CU.StaticSlidingWindowLayer, "_inkling_patched", False):
        return
    original = CU.StaticSlidingWindowLayer.update

    def update(self, key_states, value_states, *args, **kwargs):
        if (self.is_initialized and key_states.shape[-2] == 1
                and self.cumulative_length_int >= self.max_cache_len):
            self.cumulative_length_int += 1
            idx = self.__dict__.get("_last_idx")
            if idx is None:
                idx = self._last_idx = torch.tensor([self.max_cache_len - 1], device=self.device)
            new_keys = self.keys.roll(-1, dims=-2)
            new_values = self.values.roll(-1, dims=-2)
            new_keys.index_copy_(2, idx, key_states)
            new_values.index_copy_(2, idx, value_states)
            self.keys.copy_(new_keys)
            self.values.copy_(new_values)
            return self.keys, self.values
        return original(self, key_states, value_states, *args, **kwargs)

    CU.StaticSlidingWindowLayer.update = update
    CU.StaticSlidingWindowLayer._inkling_patched = True


def install_static_paged(cache: Any, top_frac: float) -> None:
    STATIC_REGISTRY.clear()
    for v in FULL_VISITS:
        STATIC_REGISTRY[v] = StaticPagedQuest(cache.layers[v], top_frac)


def make_static_attention_fn() -> Callable:
    def fn(module, query, key, value, attention_mask, scaling, dropout=0.0, position_bias=None,
           sinks=None, **kwargs):
        kwargs.pop("sparse_meta_data", None)
        st = (STATIC_REGISTRY.get(getattr(module, "_sah_visit_idx", -1))
              if isinstance(module, M.Inkling2LoopFullAttention) else None)
        if st is not None and query.shape[2] == 1:
            out = st.attend(query[:, :, 0].contiguous(), key, value, position_bias[:, :, 0], sinks[:, :, 0, 0])
            return out.unsqueeze(1), None
        return M.eager_attention_forward(module, query, key, value, attention_mask, scaling=scaling,
                                         dropout=dropout, position_bias=position_bias, sinks=sinks, **kwargs)

    return fn


def install_attention(model: torch.nn.Module, name: str = "inkling_static") -> None:
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from inkling_sah import install_visit_tracking

    install_visit_tracking(model)
    ALL_ATTENTION_FUNCTIONS.register(name, make_static_attention_fn())
    ALL_MASK_ATTENTION_FUNCTIONS.register(name, eager_mask)
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = name


def set_experts(model: torch.nn.Module, impl: str) -> None:
    for m in model.modules():
        if isinstance(m, M.Inkling2LoopExperts):
            m.config._experts_implementation = impl


def find_sync_sites(fn: Callable[[], Any], depth: int = 6) -> List[tuple]:
    """Run `fn` with CUDA sync debug mode on; return the distinct (stack tail) sites that synced."""
    import traceback

    sites: Counter = Counter()
    old = warnings.showwarning

    def hook(message, category, filename, lineno, file=None, line=None):
        stack = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno}:{f.name}"
                 for f in traceback.extract_stack()[:-2] if "warnings" not in f.filename]
        sites[(str(message)[:60],) + tuple(stack[-depth:])] += 1

    warnings.showwarning = hook
    warnings.simplefilter("always")
    torch.cuda.set_sync_debug_mode("warn")
    try:
        fn()
        torch.cuda.synchronize()
    finally:
        torch.cuda.set_sync_debug_mode("default")
        warnings.showwarning = old
    return [(n, s) for s, n in sites.most_common()]


class GraphedDecodeStep:
    """Capture one q_len=1 decode step of `model` over a StaticCache in a CUDA graph.

    `position_ids` live in a device buffer that the step itself increments (HF would otherwise derive
    them from the sliding layers' python-int length, which a replay never advances). Warm-up runs real
    decode steps on `warmup_tokens` (advancing the cache); the capture records without executing."""

    def __init__(self, model: torch.nn.Module, cache: Any, warmup_tokens: torch.Tensor, start_pos: int) -> None:
        self.model, self.cache = model, cache
        self.ids = warmup_tokens[:, :1].clone()
        self.pos = torch.full((1, 1), start_pos, dtype=torch.long, device=self.ids.device)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(warmup_tokens.shape[1]):
                self.ids.copy_(warmup_tokens[:, i:i + 1])
                self.warm_logits = self._fwd().clone()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self._fwd()

    def _fwd(self) -> torch.Tensor:
        out = self.model(input_ids=self.ids, position_ids=self.pos, past_key_values=self.cache, use_cache=True,
                         logits_to_keep=1, sparse_meta_data={}).logits[:, -1]
        self.pos.add_(1)
        return out

    def step(self, token: torch.Tensor) -> torch.Tensor:
        self.ids.copy_(token.view(1, 1))
        self.graph.replay()
        return self.logits


class GraphedGreedyDecode(GraphedDecodeStep):
    """Greedy decoding entirely on device: the captured step writes argmax back into its own input
    buffer and records the token, so `run(n)` is n bare graph replays with no host round-trips."""

    def __init__(self, model, cache, first_token: torch.Tensor, start_pos: int, max_steps: int,
                 warmup: int = 2) -> None:
        dev = first_token.device
        self.start = start_pos
        self.tokens = torch.zeros(max_steps, dtype=torch.long, device=dev)
        self.model, self.cache = model, cache
        self.ids = first_token.view(1, 1).clone()
        self.pos = torch.full((1, 1), start_pos, dtype=torch.long, device=dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._fwd()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.n_done = warmup
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self._fwd()

    def _fwd(self) -> torch.Tensor:
        out = self.model(input_ids=self.ids, position_ids=self.pos, past_key_values=self.cache, use_cache=True,
                         logits_to_keep=1, sparse_meta_data={}).logits[:, -1]
        nxt = out.argmax(-1).view(1, 1)
        self.tokens.index_copy_(0, (self.pos - self.start).view(1), nxt.view(1))
        self.ids.copy_(nxt)
        self.pos.add_(1)
        return out

    def run(self, n: int) -> None:
        for _ in range(n):
            self.graph.replay()
        self.n_done += n
