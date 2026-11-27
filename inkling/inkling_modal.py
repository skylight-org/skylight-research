"""Run Inkling 2 Mini + sparse-attention-hub on a Modal A100-80GB.

    modal run inkling_modal.py                 # smoke + equivalence + needle test
    modal run inkling_modal.py --context-tokens 8000

Budget guard: every GPU function has a hard timeout; at A100-80GB ($2.50/h) + 4 CPU + 48 GiB RAM
(~$0.57/h) the worst case per call is TIMEOUT_MIN/60 * ~$3.07.
"""

import modal

TIMEOUT_MIN = 40  # hard ceiling per run => <= ~$2.05 worst case

app = modal.App("inkling-sparse")
vol = modal.Volume.from_name("inkling-workspace")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.9.1", "torchvision==0.24.1")
    .pip_install(
        "transformers==5.24.1", "accelerate", "safetensors", "tiktoken", "pillow", "sentencepiece",
        "huggingface_hub[hf_xet]", "numpy", "ray[tune]>=2.48.0", "hyperopt", "pydantic", "tqdm",
        "pandas", "scikit-learn", "matplotlib", "seaborn", "plotly", "psutil", "pynvml", "colorama",
        "fire", "datasets",
    )
    .env({"PYTHONPATH": "/vol/repos/skylight-research", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("inkling_sah")
)

MODEL_PATH = "/vol/models/Inkling-2-Mini"

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, volumes={"/vol": vol},
              timeout=TIMEOUT_MIN * 60)
def run(context_tokens: int = 4000, needle_depth: float = 0.3, max_new_tokens: int = 160,
        suite: str = "default") -> dict:
    import json
    import os
    import time
    import traceback

    results: dict = {}
    try:
        _run_impl(context_tokens, needle_depth, max_new_tokens, results, suite)
    except Exception:  # noqa: BLE001 - return partial results + a plain-text traceback
        results["error"] = traceback.format_exc()[-4000:]
        print(results["error"], flush=True)
    os.makedirs("/vol/results", exist_ok=True)
    with open(f"/vol/results/run_{int(time.time())}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    vol.commit()
    return results


def _run_impl(context_tokens: int, needle_depth: float, max_new_tokens: int, results: dict,
              suite: str = "default") -> None:
    import copy
    import json
    import os
    import time

    import torch

    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)  # noqa: E731

    from sparse_attention_hub.adapters import Request
    from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
        LocalMaskerConfig, OracleTopKConfig, QuestTopKMaskerConfig, SinkMaskerConfig,
    )

    import inkling_sah as ink

    log("GPU:", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))

    # ---------------- A. load + dense smoke test ----------------
    full_cfg = ResearchAttentionConfig(masker_configs=[LocalMaskerConfig(window_size=10**9)])
    adapter = ink.InklingModelAdapterHF(
        model_name=MODEL_PATH,
        sparse_attention_config=full_cfg,
        model_kwargs={"dtype": torch.bfloat16, "device_map": "cuda"},
        device="cuda",
    )
    model, tok = adapter.model, adapter.tokenizer
    model.eval()
    results["load_s"] = round(time.time() - t0, 1)
    results["weights_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    results["attn_impl"] = model.config.get_text_config()._attn_implementation
    log("loaded", type(model).__name__, results)

    msgs = [{"role": "user", "content": "In two sentences, why is the sky blue?"}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, reasoning_effort="none",
                                     return_tensors="pt", return_dict=True).to("cuda")
    ts = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    results["dense_smoke"] = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    results["dense_decode_tok_s"] = round(n_new / (time.time() - ts), 2)
    log("smoke:", results["dense_smoke"], f"({results['dense_decode_tok_s']} tok/s)")

    # ---------------- B. equivalence: full-mask sparse path == eager ----------------
    ids = tok(FILLER * 80, return_tensors="pt").input_ids[:, :1500].cuda()
    with torch.no_grad():
        pre = model(ids[:, :-1], use_cache=True)
        cache = pre.past_key_values
        dense_logits = model(ids[:, -1:], past_key_values=copy.deepcopy(cache), use_cache=True).logits[0, -1].float()
        ink.STATS.reset()
        with adapter.enable_sparse_mode():
            sp_logits = model(ids[:, -1:], past_key_values=copy.deepcopy(cache), use_cache=True,
                              sparse_meta_data={}).logits[0, -1].float()
    results["equivalence"] = {
        "max_abs_logit_diff": float((dense_logits - sp_logits).abs().max()),
        "cosine": float(torch.nn.functional.cosine_similarity(dense_logits, sp_logits, dim=0)),
        "top1_match": bool(dense_logits.argmax() == sp_logits.argmax()),
        "stats": ink.STATS.summary(),
    }
    log("equivalence:", json.dumps(results["equivalence"]))
    del cache, pre

    # ---------------- C. passkey retrieval: dense vs sparse decode ----------------
    passkey = "68297"
    n_fill = max(1, context_tokens // 22)
    depth = int(n_fill * needle_depth)
    context = (FILLER * depth + f" The pass key is {passkey}. Remember it. {passkey} is the pass key. "
               + FILLER * (n_fill - depth))
    request = Request(context=context, questions="What is the pass key? Reply with only the number.",
                      answer_prefix="")
    configs = {
        "dense": None,
        "sink128+local512 (streaming)": ResearchAttentionConfig(
            masker_configs=[SinkMaskerConfig(sink_size=128), LocalMaskerConfig(window_size=512)]),
        "sink128+local512+oracle_top5%": ResearchAttentionConfig(
            masker_configs=[SinkMaskerConfig(sink_size=128), LocalMaskerConfig(window_size=512),
                            OracleTopKConfig(heavy_size=0.05)]),
        "sink32+local64+oracle_top2%": ResearchAttentionConfig(
            masker_configs=[SinkMaskerConfig(sink_size=32), LocalMaskerConfig(window_size=64),
                            OracleTopKConfig(heavy_size=0.02)]),
    }
    if suite == "quest":
        sl = [SinkMaskerConfig(sink_size=64), LocalMaskerConfig(window_size=1024)]
        configs = {
            "dense": None,
            "sink64+local1024": ResearchAttentionConfig(masker_configs=list(sl)),
            "sink64+local1024+oracle2%": ResearchAttentionConfig(
                masker_configs=sl + [OracleTopKConfig(heavy_size=0.02)]),
            "sink64+local1024+quest_p16_2%": ResearchAttentionConfig(
                masker_configs=sl + [QuestTopKMaskerConfig(heavy_size=0.02, page_size=16)]),
            "sink64+local1024+quest_p16_1%": ResearchAttentionConfig(
                masker_configs=sl + [QuestTopKMaskerConfig(heavy_size=0.01, page_size=16)]),
        }
    results["needle"] = {}
    for name, cfg in configs.items():
        torch.cuda.reset_peak_memory_stats()
        ink.STATS.reset()
        if cfg is None:
            adapter._sparse_attention_available = False
        else:
            adapter.set_sparse_config(cfg)
        ts = time.time()
        resp = adapter.process_request(request, generation_kwargs={"max_new_tokens": max_new_tokens},
                                       request_kwargs={"max_context_length": 10**9})
        results["needle"][name] = {
            "answer": resp.responses.strip(),
            "correct": passkey in resp.responses,
            "seconds": round(time.time() - ts, 1),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "stats": ink.STATS.summary(),
        }
        log(name, json.dumps(results["needle"][name]))

    results["context_tokens_requested"] = context_tokens
    results["total_s"] = round(time.time() - t0, 1)


@app.function(image=image, cpu=2, memory=8192, volumes={"/vol": vol}, timeout=600)
def check_imports() -> str:
    """CPU-only (pennies): verify image, repo on volume, and integration imports."""
    import transformers

    import inkling_sah as ink
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
        OracleTopKConfig,
    )

    cfg = transformers.AutoConfig.from_pretrained(MODEL_PATH)
    tc = cfg.get_text_config()
    return (f"transformers {transformers.__version__}; {ink.TOKEN_ATTENTION_CLASSES}; "
            f"layer_types={tc.layer_types}; loop=({tc.loop_layer_index_start},{tc.loop_layer_index_end},"
            f"{tc.loop_num_passes}); OracleTopK ok={OracleTopKConfig(heavy_size=0.05).heavy_size}")


@app.local_entrypoint()
def main(context_tokens: int = 4000, needle_depth: float = 0.3, max_new_tokens: int = 160,
         suite: str = "default"):
    import json

    print(json.dumps(run.remote(context_tokens, needle_depth, max_new_tokens, suite), indent=2, default=str))
