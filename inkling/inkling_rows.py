"""Several attention configs decoded side by side from ONE dense context prefill.

The context is prefilled once (B=1, fused kernel), the StaticCache is repeated to R rows, and the rows then
differ only on Inkling's 8 full-attention visits: dense rows use the fused kernels, sparse rows use SSA
(stratified PQ importance sampling) or its top-k-only special case (PQCache).  Question tokens and every
generated token go through each row's own attention (the SSA paper's protocol); everything else (MoE,
sliding layers, ...) runs batched.  Decode for all rows is one CUDA graph, so R configs cost about as much
as one (Inkling decode at batch 1 is launch-bound).  Sampling uses the same Gumbel noise in every row
(common random numbers), so rows only diverge where their logits do.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import inkling_fa as FA
from inkling_ssa import SSAAttention, SSAContext, SSAFast, SSAFused, SSAFusedV3, SSAFusedV4, SSARow, SSASelector

FULL_VISITS = (5, 11, 17, 23, 29, 35, 41, 47)
Q_CHUNK = 128


def _rep(t, R: int):
    if torch.is_tensor(t):
        return t.repeat(R, *([1] * (t.dim() - 1))) if t.dim() >= 2 and t.shape[0] == 1 else t.clone()
    return t


def repeat_cache(cache, R: int):
    """Copy a B=1 cache into R identical rows (batch dim of every per-row tensor repeated). Inkling's layers
    are LinearAttentionAndStatic*Layer: keys/values plus conv_states held in a dict {state_idx: tensor}."""
    new = copy.copy(cache)
    new.layers = []
    for layer in cache.layers:
        nl = copy.copy(layer)
        for k, v in list(vars(layer).items()):
            if torch.is_tensor(v):
                setattr(nl, k, _rep(v, R))
            elif isinstance(v, dict) and any(torch.is_tensor(t) for t in v.values()):
                setattr(nl, k, {kk: _rep(t, R) for kk, t in v.items()})
            elif isinstance(v, (list, tuple)) and v and any(torch.is_tensor(t) for t in v):
                setattr(nl, k, type(v)(_rep(t, R) for t in v))
            elif isinstance(v, dict):
                setattr(nl, k, dict(v))
        if isinstance(getattr(nl, "batch_size", None), int):
            nl.batch_size = R
        new.layers.append(nl)
    for k, v in list(vars(cache).items()):
        if k != "layers" and torch.is_tensor(v) and v.dim() >= 2 and v.shape[0] == 1:
            setattr(new, k, v.repeat(R, *([1] * (v.dim() - 1))))
    return new


def describe_cache(cache) -> dict:
    out = {"type": type(cache).__name__, "n_layers": len(cache.layers)}
    for i in (0, 5):
        if i < len(cache.layers):
            L = cache.layers[i]
            out[f"layer{i}"] = {k: (type(v).__name__, list(v.shape) if torch.is_tensor(v) else
                                    ([list(t.shape) for t in v] if isinstance(v, (list, tuple)) and v and
                                     torch.is_tensor(v[0]) else str(v)[:40]))
                                for k, v in vars(L).items()}
            out[f"layer{i}"]["__class__"] = type(L).__name__
    out["cache_attrs"] = {k: type(v).__name__ for k, v in vars(cache).items() if k != "layers"}
    return out


class Rows:
    """modes: list like ["dense", "ssa", "pqc"]; dense rows must come first."""

    def __init__(self, cache, modes: Sequence[str], L0: int, n_prompt: int, max_new: int, scaling: float,
                 budget: float = 0.02, heavy_share: float = 1.0 / 3.0, temperature: float = 1.0,
                 kmeans_iter: int = 10, seed: int = 0, overrides: Optional[Dict[str, Tuple[float, float]]] = None,
                 fast: bool = True):
        self.cache, self.modes, self.R, self.fast = cache, list(modes), len(modes), fast
        self.n_dense = sum(m == "dense" for m in modes)
        assert all(m == "dense" for m in modes[:self.n_dense]) and all(m != "dense" for m in modes[self.n_dense:])
        k0 = cache.layers[FULL_VISITS[0]].keys
        _, self.Hkv, self.max_len, self.D = k0.shape
        self.HQ, self.scaling, dev = 24, scaling, k0.device
        self.fd = FA.FlashDecode(range(self.n_dense), self.HQ, self.D, dev) if self.n_dense else None
        fracs = {"ssa": (budget * heavy_share, budget * (1 - heavy_share)), "pqc": (budget, 0.0),
                 "full": (1.0, 0.0)}
        fracs.update(overrides or {})
        self.sparse = [(b, m) for b, m in enumerate(modes) if m != "dense"]
        self.ctx: Dict[int, SSAContext] = {}
        self.state: Dict[Tuple[int, int], SSARow] = {}
        self.sel: Dict[Tuple[int, int], SSASelector] = {}
        if self.sparse:
            for v in FULL_VISITS:
                self.ctx[v] = SSAContext(cache.layers[v].keys[self.sparse[0][0]], L0, iters=kmeans_iter,
                                         seed=seed + v)
                for b, m in self.sparse:
                    hf, sf = fracs[m]
                    self.state[(v, b)] = SSARow(self.ctx[v], b)
                    cls = {"v4": SSAFusedV4, "v3": SSAFusedV3, "v2": SSAFused}.get(fast, SSAFast if fast else SSASelector)
                    kw = {"max_q": Q_CHUNK} if fast else {}
                    self.sel[(v, b)] = cls(self.state[(v, b)], n_prompt + max_new, n_prompt, hf, sf, scaling,
                                           temperature, **kw)
        self.attn_dec = ({} if fast else
                         {b: SSAAttention(1, b, self.R, self.Hkv, self.max_len, self.D, dev) for b, _ in self.sparse})
        self.attn_q: Dict[Tuple[int, int], SSAAttention] = {}
        self.out = torch.empty(self.R, self.HQ, self.D, dtype=k0.dtype, device=dev)
        self.budget_tokens = {m: (self.sel[(FULL_VISITS[0], b)].S if (FULL_VISITS[0], b) in self.sel else None)
                              for b, m in self.sparse}

    # -- question chunk (multi-token, lazy bias) -------------------------------------------------------------
    def row_hook(self, module, query, key, value, lb, sinks, scaling):
        v = module._sah_visit_idx
        rd = lb.rd
        Qn = query.shape[2]
        out = torch.empty(self.R, Qn, self.HQ, self.D, dtype=query.dtype, device=query.device)
        nd = self.n_dense
        if nd:
            out[:nd] = FA.fa_prefill(query[:nd], key[:nd], value[:nd], lb.table[:nd], sinks[:nd, ..., 0],
                                     rd.q_off, rd.kv_off, scaling)
        pos_all = rd.q_off + torch.arange(Qn, device=query.device)
        Rb = lb.table.shape[-1]
        for b, _ in self.sparse:
            st, sel = self.state[(v, b)], self.sel[(v, b)]
            st.code_keys(key[b], pos_all)
            for c0 in range(0, Qn, Q_CHUNK):
                c1 = min(Qn, c0 + Q_CHUNK)
                n = c1 - c0
                qq = query[b, :, c0:c1].transpose(0, 1).contiguous()
                pos = pos_all[c0:c1]
                tab = lb.table[b, :, c0:c1].transpose(0, 1)                         # [n, HQ, R]

                def bias_of(idx, pos=pos, tab=tab):
                    d = pos[:, None, None] - idx
                    ok = (d >= 0) & (d < Rb)
                    return tab.gather(-1, d.clamp(0, Rb - 1)).float().masked_fill(~ok, 0.0)

                if self.fast:
                    out[b, c0:c1] = sel(query[b, :, c0:c1].transpose(0, 1), pos, key, value, b,
                                        sinks[b, :, c0:c1, 0].transpose(0, 1), tb=tab)
                    continue
                idx, w = sel.select(qq, pos, bias_of)
                if (b, n) not in self.attn_q:
                    self.attn_q[(b, n)] = SSAAttention(n, b, self.R, self.Hkv, self.max_len, self.D, query.device)
                out[b, c0:c1] = self.attn_q[(b, n)].run(qq, key, value, idx, w,
                                                        sinks[b, :, c0:c1, 0].transpose(0, 1))
        return out

    # -- single-token decode (graph-safe) --------------------------------------------------------------------
    def decode_hook(self, module, query, key, value, position_bias, sinks, scaling):
        v = module._sah_visit_idx
        L = self.cache.layers[v].cumulative_length
        q = query[:, :, 0]
        pb = position_bias[:, :, 0]
        sk = sinks[:, :, 0, 0]
        out = self.out
        if self.n_dense:
            self.fd(q, key, value, pb, L, sk, out, scaling)
        if self.sparse:
            pos = (L - 1).view(1).long()
            for b, _ in self.sparse:
                self.state[(v, b)].code_keys(key[b], pos)
                pbb = pb[b]

                def bias_of(idx, pbb=pbb):
                    return pbb.gather(-1, idx[0]).float()[None]

                if self.fast:
                    self.sel[(v, b)](q[b][None], pos, key, value, b, sk[b][None], pb=pbb, out=out[b][None])
                    continue
                qb = q[b][None].contiguous()
                idx, w = self.sel[(v, b)].select(qb, pos, bias_of)
                out[b] = self.attn_dec[b].run(qb, key, value, idx, w, sk[b][None])[0]
        return out.unsqueeze(1)

    def install(self) -> None:
        FA.STATE["row_hook"] = self.row_hook
        FA.STATE["decode_hook"] = self.decode_hook

    @staticmethod
    def uninstall() -> None:
        FA.STATE["row_hook"] = None
        FA.STATE["decode_hook"] = None


class GraphedRowsDecode:
    """R-row decode in one CUDA graph; sampling (Gumbel-max, shared noise) or greedy, all on device."""

    def __init__(self, model, cache, first: torch.Tensor, start_pos: int, max_steps: int, temperature: float,
                 warmup: int = 2) -> None:
        dev = first.device
        R = first.shape[0]
        self.model, self.cache, self.T, self.start = model, cache, float(temperature), start_pos
        self.tokens = torch.zeros(R, max_steps, dtype=torch.long, device=dev)
        self.tokens[:, 0] = first.view(-1)
        self.ids = first.view(R, 1).clone()
        self.pos = torch.full((R, 1), start_pos, dtype=torch.long, device=dev)
        self.max_steps = max_steps
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._fwd()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.n_done = 1 + warmup
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._fwd()

    def _fwd(self) -> None:
        logits = self.model(input_ids=self.ids, position_ids=self.pos, past_key_values=self.cache, use_cache=True,
                            logits_to_keep=1).logits[:, -1].float()
        if self.T > 0:
            u = torch.rand(logits.shape[-1], device=logits.device).clamp_(1e-10, 1 - 1e-7)
            logits = logits / self.T - torch.log(-torch.log(u))[None]
        nxt = logits.argmax(-1)
        step = (self.pos[0] - self.start + 1).clamp(max=self.max_steps - 1)
        self.tokens.index_copy_(1, step, nxt.view(-1, 1))
        self.ids.copy_(nxt.view(-1, 1))
        self.pos.add_(1)

    def run_until(self, stop_ids: Sequence[int], burst: int = 16) -> int:
        stop = torch.tensor(list(stop_ids), device=self.tokens.device)
        while self.n_done < self.max_steps - 1:
            n = min(burst, self.max_steps - 1 - self.n_done)
            for _ in range(n):
                self.graph.replay()
            self.n_done += n
            hit = torch.isin(self.tokens[:, :self.n_done], stop).any(-1)
            if bool(hit.all()):
                break
        return self.n_done


def sample_first(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """logits [R, V] -> [R, 1] with shared Gumbel noise."""
    logits = logits.float()
    if temperature > 0:
        u = torch.rand(logits.shape[-1], device=logits.device).clamp_(1e-10, 1 - 1e-7)
        logits = logits / temperature - torch.log(-torch.log(u))[None]
    return logits.argmax(-1, keepdim=True)
