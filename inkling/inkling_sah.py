"""Sparse-attention-hub integration for Inkling 2 Mini (transformers `inkling2_loop`).

Why the stock `ModelAdapterHF` path is not enough for Inkling
--------------------------------------------------------------
1. Non-token attention calls. Inkling also routes its depth-wise "attention residuals"
   (`Inkling2LoopBlockAttentionResidual`, `Inkling2LoopMessageAttentionResidual`) through
   `ALL_ATTENTION_FUNCTIONS`. Those attend over *residual blocks*, not tokens, and pass no
   `sparse_meta_data`, so the stock callable raises. We send them to Inkling's own eager kernel.
2. Learned relative-position logits + data-dependent sinks. Inkling's eager kernel adds a
   `position_bias` of shape (b, h, q, k) to the logits and a per-token sink logit `sinks`
   of shape (b, h, q, 1). sparse-attention-hub ignores `position_bias` and only supports a
   1-D per-head `s_aux` sink, and its dense fallback (`get_true_attention_output`, SDPA) drops
   both. We fold `position_bias` into the additive mask (so maskers *see* it too) and handle
   per-row sinks in our own masked softmax.
3. Looped layers alias `layer_idx`. Physical layers 6-17 are visited 3 times with the same
   module (same `layer_idx`) but separate KV caches (`cache_offset`). Stateful maskers key
   `sparse_meta_data` by `layer_idx`, so passes would clobber each other. We record a unique
   per-visit id (= the KV-cache layer index, layer_idx + cache_offset) and hand that to maskers.
4. Only 8 of the 48 layer visits are global attention ("hybrid" layers 5, 11x3, 17x3, 23);
   the other 40 are already 1024-token sliding window. By default only "hybrid" visits go sparse;
   sliding visits run Inkling's exact eager kernel.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from transformers.models.inkling2_loop import modeling_inkling2_loop as M

from sparse_attention_hub.adapters.huggingface import ModelAdapterHF
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttention
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
    repeat_kv,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

TOKEN_ATTENTION_CLASSES = (M.Inkling2LoopSlidingAttention, M.Inkling2LoopFullAttention)


class SparseStats:
    """Per-visit diagnostics: density and relative output error vs dense attention."""

    def __init__(self) -> None:
        self.enabled = True
        self.reset()

    def reset(self) -> None:
        self.calls: Dict[int, int] = defaultdict(int)
        self.density: Dict[int, List[float]] = defaultdict(list)
        self.error: Dict[int, List[float]] = defaultdict(list)
        self.dense_fallback_calls = 0

    def summary(self) -> Dict[str, Any]:
        def mean(xs: Iterable[float]) -> float:
            xs = list(xs)
            return sum(xs) / max(len(xs), 1)

        return {
            "sparse_calls_per_visit": dict(sorted(self.calls.items())),
            "mean_density_per_visit": {k: round(mean(v), 4) for k, v in sorted(self.density.items())},
            "mean_rel_err_per_visit": {k: round(mean(v), 4) for k, v in sorted(self.error.items())},
            "dense_fallback_calls": self.dense_fallback_calls,
        }


STATS = SparseStats()


def install_visit_tracking(model: torch.nn.Module) -> int:
    """Wrap each token-attention module so it records `layer_idx + cache_offset` before running."""
    n = 0
    for module in model.modules():
        if isinstance(module, TOKEN_ATTENTION_CLASSES) and not getattr(module, "_sah_wrapped", False):
            original_forward = module.forward

            def forward(*args: Any, _orig: Callable = original_forward, _mod: Any = module,
                        cache_offset: int = 0, **kwargs: Any) -> Any:
                _mod._sah_visit_idx = _mod.layer_idx + cache_offset
                return _orig(*args, cache_offset=cache_offset, **kwargs)

            module.forward = forward
            module._sah_wrapped = True
            n += 1
    return n


def _masked_attention_with_row_sinks(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    additive_mask: Optional[torch.Tensor],
    sinks: Optional[torch.Tensor],
    scaling: float,
    sparse_mask: Optional[Mask],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """softmax([qk*s + mask, sink]) restricted to `sparse_mask` (with 1/p inverse weighting)."""
    groups = _get_num_key_value_groups(queries, keys)
    k = repeat_kv(keys, groups).float()
    v = repeat_kv(values, groups).float()
    logits = torch.matmul(queries.float(), k.transpose(2, 3)) * scaling
    if additive_mask is not None:
        logits = logits + additive_mask[..., : k.shape[-2]].float()
    row_max = logits.amax(dim=-1, keepdim=True)
    if sinks is not None:
        row_max = torch.maximum(row_max, sinks.float())
    exp_w = torch.exp(logits - row_max)
    if sparse_mask is not None:
        exp_w = sparse_mask.apply_inv_mask(exp_w)
    den = exp_w.sum(dim=-1, keepdim=True)
    if sinks is not None:
        den = den + torch.exp(sinks.float() - row_max)
    out = torch.matmul(exp_w, v) / den
    return out.to(queries.dtype).transpose(1, 2).contiguous(), (exp_w / den).to(queries.dtype)


def make_inkling_attention_fn(
    sparse_attention: ResearchAttention,
    sparse_layer_types: Tuple[str, ...] = ("hybrid",),
    stats: SparseStats = STATS,
    compute_error: bool = True,
    bias_in_selection: bool = False,
) -> Callable:
    """Build an `ALL_ATTENTION_FUNCTIONS`-compatible callable for Inkling.

    Maskers see the plain causal mask by default: some (e.g. Quest) read an additive float mask
    as "negative => forbidden", which Inkling's signed position bias would corrupt. The bias is
    always applied in the final softmax. `bias_in_selection=True` lets score-ranking maskers
    (oracle top-k) rank with the bias included.
    """

    def inkling_sparse_attention(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        position_bias: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        sparse_meta_data = kwargs.pop("sparse_meta_data", None)
        is_token_attn = isinstance(module, TOKEN_ATTENTION_CLASSES)
        goes_sparse = (
            is_token_attn
            and sparse_meta_data is not None
            and module.config.layer_types[module.layer_idx] in sparse_layer_types
        )
        if not goes_sparse:
            if is_token_attn:
                stats.dense_fallback_calls += 1
            return M.eager_attention_forward(
                module, query, key, value, attention_mask, scaling=scaling, dropout=dropout,
                position_bias=position_bias, sinks=sinks, **kwargs,
            )

        visit_idx: int = getattr(module, "_sah_visit_idx", module.layer_idx)
        additive = attention_mask
        if position_bias is not None:
            additive = position_bias if additive is None else additive + position_bias

        b, h, q_len, _ = query.shape
        k_len = key.shape[2]
        mask = Mask.create_empty_mask((b, h, q_len, k_len), dtype=query.dtype, device=query.device)
        for masker in sparse_attention.maskers:
            mask = masker.add_mask(
                keys=key, queries=query, values=value,
                attention_mask=additive if bias_in_selection else attention_mask, scaling=scaling,
                dropout=dropout, sparse_meta_data=sparse_meta_data, previous_mask=mask,
                layer_idx=visit_idx, **kwargs,
            )

        out, weights = _masked_attention_with_row_sinks(query, key, value, additive, sinks, scaling, mask)

        if stats.enabled:
            stats.calls[visit_idx] += 1
            stats.density[visit_idx].append(float(mask.get_density()))
            if compute_error:
                dense, _ = _masked_attention_with_row_sinks(query, key, value, additive, sinks, scaling, None)
                err = torch.linalg.vector_norm((out - dense).float()) / torch.linalg.vector_norm(dense.float())
                stats.error[visit_idx].append(float(err))
        return out, weights

    return inkling_sparse_attention


class InklingModelAdapterHF(ModelAdapterHF):
    """`ModelAdapterHF` with the Inkling-specific attention callable + chat-template handling."""

    def __init__(self, *args: Any, sparse_layer_types: Tuple[str, ...] = ("hybrid",),
                 reasoning_effort: str = "none", prefill_chunk_size: int = 1024,
                 memoize_prefill: bool = True, **kwargs: Any) -> None:
        self.sparse_layer_types = sparse_layer_types
        self.reasoning_effort = reasoning_effort
        model_name = kwargs.get("model_name", args[0] if args else None)
        if not kwargs.get("model_registry_path") and model_name is not None:
            kwargs["model_registry_path"] = _write_inkling_registry(model_name)
        super().__init__(*args, **kwargs)
        install_visit_tracking(self.model)
        self.prefill_memo: Optional[Dict[str, Any]] = {} if memoize_prefill else None
        _wrap_model_forward(self.model, prefill_chunk_size, self.prefill_memo)

    def set_sparse_config(self, sparse_attention_config: Any) -> None:
        self._cleanup_attention_registration()
        self.sparse_attention = ResearchAttention.create_from_config(sparse_attention_config)
        self._sparse_attention_available = True

    def get_custom_attention_function(self, sparse_attention: Any) -> Callable:
        return make_inkling_attention_fn(sparse_attention, self.sparse_layer_types)

    def _preprocess_context_and_questions(self, context: str, questions: List[str], answer_prefix: str):
        context = context + self.random_separator
        context = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": context}], tokenize=False, add_generation_prompt=True,
            reasoning_effort=self.reasoning_effort,
        )
        head, tail = context.split(self.random_separator)
        return head, [q + tail + answer_prefix for q in questions]


def _write_inkling_registry(model_name: str) -> str:
    """The stock ModelServerHF uses AutoModelForCausalLM, which has no mapping for the multimodal
    `inkling2_loop` config. Point the framework's model registry at the right class instead
    (text-only `inkling2_loop_text` checkpoints already work with AutoModelForCausalLM)."""
    import json
    import tempfile

    from transformers import AutoConfig

    model_type = AutoConfig.from_pretrained(model_name).model_type
    if model_type != "inkling2_loop":
        return ""
    cls = "transformers.models.inkling2_loop.modeling_inkling2_loop.Inkling2LoopForConditionalGeneration"
    path = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False).name
    with open(path, "w") as f:  # JSON is valid YAML
        json.dump({"models": {model_name: {"model_class": cls}}}, f)
    return path


def _alias_cache(out: Any) -> Any:
    """transformers 5.x Inkling returns `.cache`; the adapter reads `.past_key_values`."""
    if getattr(out, "past_key_values", None) is None and getattr(out, "cache", None) is not None:
        try:
            out["past_key_values"] = out.cache
        except Exception:  # noqa: BLE001
            object.__setattr__(out, "past_key_values", out.cache)
    return out


def _wrap_model_forward(model: torch.nn.Module, chunk_size: int,
                        memo: Optional[Dict[str, Any]]) -> None:
    """Chunked (and optionally memoized) context prefill.

    Inkling only supports eager attention, which materialises (heads x q x k) fp32 logits, so a
    one-shot prefill OOMs an 80 GB card at ~8k tokens. A cache-less, multi-token, `use_cache=True`
    call longer than `chunk_size` (i.e. ModelAdapterHF's context prefill) is instead fed in chunks
    through the KV cache. Peak memory becomes ~heads x chunk x k. With `memo`, the last context's
    prefill is reused (deep-copied) so sweeping sparse configs over one context prefills it once.
    """
    import copy
    import hashlib

    original_forward = model.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        has_cache = kwargs.get("past_key_values") is not None or kwargs.get("cache") is not None
        chunkable = (
            chunk_size and isinstance(input_ids, torch.Tensor) and input_ids.dim() == 2
            and input_ids.shape[1] > chunk_size and not has_cache and kwargs.get("use_cache")
            and kwargs.get("inputs_embeds") is None and len(args) <= 1
        )
        if not chunkable:
            return _alias_cache(original_forward(*args, **kwargs))

        key = hashlib.sha1(input_ids.detach().cpu().numpy().tobytes()).hexdigest()
        if memo is not None and key in memo:
            return copy.deepcopy(memo[key])

        extra = {k: v for k, v in kwargs.items()
                 if k not in ("input_ids", "past_key_values", "cache", "use_cache", "logits_to_keep")}
        cache, out = None, None
        for start in range(0, input_ids.shape[1], chunk_size):
            out = _alias_cache(original_forward(
                input_ids=input_ids[:, start:start + chunk_size], past_key_values=cache,
                use_cache=True, logits_to_keep=1, **extra))
            cache = out.past_key_values
        if memo is not None:
            memo.clear()
            memo[key] = out
            return copy.deepcopy(out)
        return out

    model.forward = forward
