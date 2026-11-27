"""CPU-only validation of the Inkling <-> sparse-attention-hub integration (no GPU, ~cents).

Builds a tiny randomly-initialised Inkling2Loop text model with the *real* architecture
(24 layers, loop 6-18 x3 passes, hybrid/sliding pattern, relative bias, sinks, attn residuals)
and the real tokenizer, then checks:
  1. the stock ModelAdapterHF callable breaks on Inkling (why we need a custom one),
  2. full-mask sparse path == Inkling eager (numerical equivalence),
  3. sparse visits are exactly the 8 global-attention visits with unique per-pass ids,
  4. real sparse configs run end-to-end through ModelAdapterHF.process_request.

    modal run validate_cpu.py
"""

import modal

app = modal.App("inkling-sparse-cpu-validate")
vol = modal.Volume.from_name("inkling-workspace")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.9.1", "torchvision==0.24.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "transformers==5.24.1", "accelerate", "safetensors", "tiktoken", "pillow", "sentencepiece",
        "numpy", "ray[tune]>=2.48.0", "hyperopt", "pydantic", "tqdm", "pandas", "scikit-learn",
        "matplotlib", "seaborn", "plotly", "psutil", "pynvml", "colorama", "fire", "datasets",
    )
    .env({"PYTHONPATH": "/vol/repos/skylight-research", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("inkling_sah")
)

MODEL_PATH = "/vol/models/Inkling-2-Mini"
TOK_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]


@app.function(image=image, cpu=8, memory=16384, volumes={"/vol": vol}, timeout=20 * 60)
def validate() -> dict:
    import copy
    import json
    import shutil

    import torch
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM

    import inkling_sah as ink
    from sparse_attention_hub.adapters import ModelAdapterHF, Request
    from sparse_attention_hub.sparse_attention.research_attention import (
        ResearchAttention, ResearchAttentionConfig,
    )
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
        LocalMaskerConfig, OracleTopKConfig, SinkMaskerConfig,
    )

    torch.manual_seed(0)
    res: dict = {}

    cfg = Inkling2LoopTextConfig(
        hidden_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=32,
        sliding_window_size=32, rel_extent=32, d_rel=4, moe_intermediate_size=32,
        n_routed_experts=8, num_experts_per_tok=2, loop_gate_rank=8,
    )
    cfg._attn_implementation = "eager"
    model = Inkling2LoopForCausalLM(cfg).eval()
    tiny = "/tmp/tiny-inkling"
    model.save_pretrained(tiny)
    for f in TOK_FILES:
        shutil.copy(f"{MODEL_PATH}/{f}", f"{tiny}/{f}")
    res["tiny_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    res["layer_types"] = cfg.layer_types

    full_cfg = ResearchAttentionConfig(masker_configs=[LocalMaskerConfig(window_size=10**9)])
    adapter = ink.InklingModelAdapterHF(model_name=tiny, sparse_attention_config=full_cfg,
                                        model_kwargs={"dtype": torch.float32}, device="cpu",
                                        prefill_chunk_size=64)
    model, tok = adapter.model.eval(), adapter.tokenizer
    res["loaded_class"] = type(model).__name__

    ids = tok("The grass is green. The sky is blue. " * 40, return_tensors="pt").input_ids[:, :300]
    with torch.no_grad():
        pre = model(ids[:, :-1], use_cache=True)
        cache = pre.past_key_values
        res["cache_layers"] = len(cache.layers)
        dense = model(ids[:, -1:], past_key_values=copy.deepcopy(cache), use_cache=True).logits[0, -1]

        # 1. stock adapter callable
        stock = ModelAdapterHF.get_custom_attention_function(adapter, adapter.sparse_attention)
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register("stock_sah", stock)
        prev = model.config._attn_implementation
        for m in model.modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = "stock_sah"
        try:
            out = model(ids[:, -1:], past_key_values=copy.deepcopy(cache), use_cache=True,
                        sparse_meta_data={}).logits[0, -1]
            res["stock_adapter"] = {"raised": False, "max_abs_diff_vs_eager": float((out - dense).abs().max())}
        except Exception as e:  # noqa: BLE001
            res["stock_adapter"] = {"raised": True, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        for m in model.modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = prev

        # 2+3. our callable, full mask
        ink.STATS.reset()
        with adapter.enable_sparse_mode():
            sp = model(ids[:, -1:], past_key_values=copy.deepcopy(cache), use_cache=True,
                       sparse_meta_data={}).logits[0, -1]
        res["equivalence_full_mask"] = {
            "max_abs_logit_diff": float((dense - sp).abs().max()),
            "top1_match": bool(dense.argmax() == sp.argmax()),
            **ink.STATS.summary(),
        }

        # same check but with ALL layer visits (incl. sliding) routed through the sparse path
        adapter_all = ink.make_inkling_attention_fn(adapter.sparse_attention, ("hybrid", "hybrid_sliding"))
        adapter._cleanup_attention_registration()
        adapter.get_custom_attention_function = lambda sa: adapter_all  # type: ignore[method-assign]
        ink.STATS.reset()
        with adapter.enable_sparse_mode():
            sp_all = model(ids[:, -1:], past_key_values=copy.deepcopy(cache), use_cache=True,
                           sparse_meta_data={}).logits[0, -1]
        res["equivalence_all_visits_full_mask"] = {
            "max_abs_logit_diff": float((dense - sp_all).abs().max()),
            "num_sparse_visits": len(ink.STATS.calls),
        }
        del adapter.get_custom_attention_function
        adapter._cleanup_attention_registration()

    # 5. chunked prefill (chunk=64, sliding window=32, rel_extent=32) == one-shot prefill
    with torch.no_grad():
        ids2 = tok("The grass is green. The sky is blue. The pass key is 4417. " * 30,
                   return_tensors="pt").input_ids[:, :301]
        ref = ink._alias_cache(type(model).forward(model, input_ids=ids2[:, :-1], use_cache=True))
        adapter.prefill_memo.clear()
        chk = model(ids2[:, :-1], past_key_values=None, use_cache=True)  # positional, like the adapter
        chk2 = model(ids2[:, :-1], past_key_values=None, use_cache=True)  # memo hit
        nxt_ref = model(ids2[:, -1:], past_key_values=ref.past_key_values, use_cache=True).logits[0, -1]
        nxt_chk = model(ids2[:, -1:], past_key_values=chk.past_key_values, use_cache=True).logits[0, -1]
        nxt_chk2 = model(ids2[:, -1:], past_key_values=chk2.past_key_values, use_cache=True).logits[0, -1]
    res["chunked_prefill"] = {
        "prefill_last_logit_max_diff": float((ref.logits[0, -1] - chk.logits[0, -1]).abs().max()),
        "next_step_logit_max_diff": float((nxt_ref - nxt_chk).abs().max()),
        "memo_hit_next_step_max_diff": float((nxt_ref - nxt_chk2).abs().max()),
        "logit_scale": float(nxt_ref.abs().max()),
    }

    # 4. end-to-end through process_request with real maskers
    req = Request(context="The grass is green. The sky is blue. " * 60 + "The pass key is 123.",
                  questions="What is the pass key?", answer_prefix="")
    res["process_request"] = {}
    for name, c in {
        "streaming": ResearchAttentionConfig(masker_configs=[SinkMaskerConfig(sink_size=8),
                                                             LocalMaskerConfig(window_size=16)]),
        "oracle_top10%": ResearchAttentionConfig(masker_configs=[SinkMaskerConfig(sink_size=8),
                                                                 LocalMaskerConfig(window_size=16),
                                                                 OracleTopKConfig(heavy_size=0.1)]),
    }.items():
        adapter.set_sparse_config(c)
        ink.STATS.reset()
        r = adapter.process_request(req, generation_kwargs={"max_new_tokens": 4}, request_kwargs={})
        res["process_request"][name] = {"output": r.responses, **ink.STATS.summary()}
    print(json.dumps(res, indent=2, default=str))
    return res


@app.local_entrypoint()
def main():
    validate.remote()
