# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "accelerate==1.14.0",
#     "datasets==5.0.1",
#     "fuzzywuzzy==0.18.0",
#     "jieba==0.42.1",
#     "nvidia-modelopt==0.46.0",
#     "pandas",
#     "python-Levenshtein",
#     "rouge==1.0.1",
#     "torch==2.11.0",
#     "transformers==5.15.1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # nvfp4 kv cache on qwen3-4b

    ## 1. setup

    usual boilerplate. i check torch sees the gpu before anything else because molab has handed me a cpu-only runtime more than once and i'd rather find out here than forty minutes in.

    the time budget is the part that matters. the session gets killed somewhere around 12h, so both long loops further down stop picking up new samples at 11h and let the notebook fall through to the save cells. whatever finished is on disk, and the next run resumes from the checkpoint.
    """)
    return


@app.cell
def _():
    import time as _time

    # molab has a hard ~12h ceiling. Both benchmark loops stop starting new
    # samples once this budget is spent, so the notebook can still reach the
    # aggregation/save cells and leave valid output on disk before the
    # session gets killed, rather than dying mid-generate(). The next run
    # (fresh 12h budget) resumes from checkpoint for whatever's left -- see
    # the checkpointed loop cells below.
    SESSION_START = _time.time()
    TIME_BUDGET_SECONDS = 11 * 3600
    return SESSION_START, TIME_BUDGET_SECONDS


@app.cell
def _():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no gpu in this runtime: restart molab with a gpu before running anything below")
    print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
    return (torch,)


@app.cell
def _(mo):
    # the nvfp4 cache comes from skylight-research's kv_quantization package. only that
    # package is loaded (not all of sparse_attention_hub), so the rest of the repo's
    # dependencies aren't needed here.
    import importlib.util as _importlib_util
    import sys as _sys

    from transformers import AutoModelForCausalLM, AutoTokenizer

    _candidates = [
        mo.notebook_dir().parent / "sparse_attention_hub" / "kv_quantization",  # notebooks/ inside the repo
        mo.notebook_dir() / "skylight-research" / "sparse_attention_hub" / "kv_quantization",  # molab copy
    ]
    _package_dir = next((p for p in _candidates if (p / "nvfp4_cache.py").exists()), None)
    if _package_dir is None:
        raise FileNotFoundError(f"kv_quantization package not found in {[str(p) for p in _candidates]}")
    _spec = _importlib_util.spec_from_file_location(
        "kv_quantization", _package_dir / "__init__.py", submodule_search_locations=[str(_package_dir)]
    )
    _module = _importlib_util.module_from_spec(_spec)
    _sys.modules["kv_quantization"] = _module
    _spec.loader.exec_module(_module)

    from kv_quantization.nvfp4_cache import NVFP4QuantizedCache, NVFP4QuantizedLayer

    print(f"kv_quantization loaded from {_package_dir}")
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        NVFP4QuantizedCache,
        NVFP4QuantizedLayer,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. the cache itself

    this is the thing actually being tested: `NVFP4QuantizedCache` from the `kv_quantization` package in skylight-research, real packed nvfp4 through modelopt. it plugs into hf's `generate()` like any other cache.

    axis `-1` blocks along head_dim (per token). axis `0` transposes first so the blocking runs along the token axis instead, which is the kivi style per channel thing (token counts that aren't a multiple of 16 get padded and trimmed by the package). keys usually want `0` and values `-1`, but both are configurable since that's half of what i want to sweep later.

    `residual_length` is how many of the most recent tokens stay in bf16 instead of being packed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. model

    plain Qwen3-4B, bf16, nothing else done to it. weights stay full precision on purpose so that anything measured below is attributable to the cache and nothing else.
    """)
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, torch):
    MODEL_NAME = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    model
    return model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. shared setup for both benchmarks

    the 21 standard longbench tasks (i skip the `_e` variants, they're largely the same contexts resampled by length), the results/checkpoint directory, and the imports that more than one cell needs. marimo wants every global owned by exactly one cell, so they're centralised here rather than re-imported locally.
    """)
    return


@app.cell
def _():
    # All 21 standard LongBench tasks (excludes the 13 length-stratified "_e"
    # variants, which resample largely the same underlying contexts). Shared
    # by both the Frobenius-norm sweep and the real-metric benchmark below.
    STANDARD_LONGBENCH_TASKS = [
        "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
        "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report",
        "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum",
        "lsht", "passage_count", "passage_retrieval_en",
        "passage_retrieval_zh", "lcc", "repobench-p",
    ]
    return (STANDARD_LONGBENCH_TASKS,)


@app.cell
def _(mo):
    # Shared output/checkpoint directory. molab sessions can be cut off
    # mid-run, so both benchmark loops below append completed samples to a
    # JSONL checkpoint file here as they go, and skip re-running whatever a
    # prior (possibly interrupted) run already finished. benchmark_results/ is
    # git-ignored, so running this from the repo never adds results to a commit.
    RESULTS_DIR = mo.notebook_dir() / "benchmark_results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return (RESULTS_DIR,)


@app.cell
def _():
    # Third-party imports shared by multiple cells below. marimo requires
    # each global name to be defined by exactly one cell, so these are
    # centralized here rather than re-imported locally in every consumer.
    import json
    import time

    import pandas as pd
    from datasets import load_dataset

    return json, load_dataset, pd, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. benchmark 1: how much does the cache actually change

    no generation in this half. prefill the same context twice, once with a normal `DynamicCache` and once with the nvfp4 one, then compare k and v layer by layer with a frobenius norm.

    only the context matters for this (the question never affects cache error), so contexts get deduped per task. saves a lot of redundant prefills on the multi-doc qa tasks.

    two numbers per layer. `manual_` runs the quantizer straight on the bf16 tensor, `real_` reads back what the cache actually stored. right after a prefill the cache has quantized the whole context (the residual buffer only fills up while decoding), so the two should match exactly. they're kept as a check that the cache stores what the quantizer produces, not as two different measurements.
    """)
    return


@app.cell
def _(STANDARD_LONGBENCH_TASKS, load_dataset):
    MAX_SAMPLES_PER_TASK = None  # set an int to cap each task for a quick smoke test; None = full task

    # LongBench has multiple questions per context (esp. multi-doc QA tasks);
    # since this benchmark only measures KV-cache quantization error, which
    # depends purely on the context (not the question), dedupe per task so
    # each unique context is only prefilled once.
    longbench_samples = []
    for _task in STANDARD_LONGBENCH_TASKS:
        _ds = load_dataset("Xnhyacinth/LongBench", _task, split="test")
        _seen_contexts = set()
        _n_added = 0
        for _row in _ds:
            if _row["context"] in _seen_contexts:
                continue
            _seen_contexts.add(_row["context"])
            longbench_samples.append({"task": _task, "context": _row["context"]})
            _n_added += 1
            if MAX_SAMPLES_PER_TASK is not None and _n_added >= MAX_SAMPLES_PER_TASK:
                break
        print(f"  {_task}: {_n_added} unique context(s)")

    print(f"Total unique contexts across {len(STANDARD_LONGBENCH_TASKS)} tasks: {len(longbench_samples)}")
    return (longbench_samples,)


@app.cell
def _(NVFP4QuantizedCache, NVFP4QuantizedLayer, model, tokenizer, torch):
    # Two measurements per layer from the same two prefill passes (a plain
    # DynamicCache and an NVFP4QuantizedCache):
    #
    #   manual_*: the bf16 tensor run through NVFP4QuantizedLayer._quantize /
    #     _dequantize directly.
    #   real_*: read back from what the NVFP4QuantizedCache stored (dequantized
    #     packed part + bf16 residual, concatenated along the token dim).
    #
    # After a prefill, HF's QuantizedLayer has quantized the whole context and
    # the residual buffer is still empty (it only fills while decoding), so the
    # two are expected to be identical. real_* is a check that the cache stores
    # what the quantizer produces.
    MAX_CONTEXT_TOKENS = 8192  # tune down further if you hit OOM or time limits

    def _truncate_to_length(text: str, max_tokens: int) -> str:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= max_tokens:
            return text
        half = max_tokens // 2
        truncated_ids = token_ids[:half] + token_ids[-half:]
        return tokenizer.decode(truncated_ids)

    @torch.no_grad()
    def per_layer_kv_frobenius(
        context: str,
        block_size: int = 16,
        residual_length: int = 128,
        axis_key: int = -1,
        axis_value: int = -1,
    ):
        from transformers import DynamicCache

        text = _truncate_to_length(context, MAX_CONTEXT_TOKENS)
        model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

        bf16_cache = DynamicCache()
        model(**model_inputs, past_key_values=bf16_cache, use_cache=True)

        nvfp4_cache = NVFP4QuantizedCache(
            model.config, block_size=block_size, residual_length=residual_length,
            axis_key=axis_key, axis_value=axis_value,
        )
        model(**model_inputs, past_key_values=nvfp4_cache, use_cache=True)

        probe_layer = NVFP4QuantizedLayer(
            block_size=block_size, residual_length=residual_length, axis_key=axis_key, axis_value=axis_value
        )

        per_layer = []
        for layer_idx, (bf16_layer, q_layer) in enumerate(zip(bf16_cache.layers, nvfp4_cache.layers)):
            layer_result = {"layer": layer_idx}
            for name, orig_tensor, axis, quantized_attr, residual_attr in (
                ("k", bf16_layer.keys, axis_key, "_quantized_keys", "keys"),
                ("v", bf16_layer.values, axis_value, "_quantized_values", "values"),
            ):
                orig = orig_tensor.float()
                orig_norm = torch.linalg.vector_norm(orig).item()
                layer_result[f"{name}_orig_norm"] = orig_norm

                # --- manual: quantize/dequantize the whole tensor directly ---
                manual_quantized = probe_layer._quantize(orig_tensor, axis=axis)
                manual_dequant = probe_layer._dequantize(manual_quantized).float()
                manual_diff = manual_dequant - orig
                layer_result[f"{name}_manual_dequant_norm"] = torch.linalg.vector_norm(manual_dequant).item()
                layer_result[f"{name}_manual_diff_norm"] = torch.linalg.vector_norm(manual_diff).item()
                layer_result[f"{name}_manual_rel_error"] = (
                    layer_result[f"{name}_manual_diff_norm"] / max(orig_norm, 1e-12)
                )

                # --- real: reconstruct from the actual NVFP4QuantizedCache ---
                # dequantize the physically-packed (older) portion and
                # concatenate the still-bf16 residual (most recent) portion.
                quantized_part = getattr(q_layer, quantized_attr, None)
                residual_part = getattr(q_layer, residual_attr, None)
                parts = []
                if quantized_part is not None:
                    parts.append(q_layer._dequantize(quantized_part).float())
                if residual_part is not None and residual_part.numel() > 0:
                    parts.append(residual_part.float())
                real_full = torch.cat(parts, dim=-2) if len(parts) > 1 else parts[0]
                real_diff = real_full - orig
                layer_result[f"{name}_real_dequant_norm"] = torch.linalg.vector_norm(real_full).item()
                layer_result[f"{name}_real_diff_norm"] = torch.linalg.vector_norm(real_diff).item()
                layer_result[f"{name}_real_rel_error"] = (
                    layer_result[f"{name}_real_diff_norm"] / max(orig_norm, 1e-12)
                )
            per_layer.append(layer_result)

        del bf16_cache, nvfp4_cache, model_inputs
        torch.cuda.empty_cache()
        return per_layer

    return (per_layer_kv_frobenius,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### running it

    this is the long one. every finished sample gets appended to a jsonl immediately and anything already in there is skipped on re-run, so a killed session isn't a wasted session.
    """)
    return


@app.cell
def _(
    RESULTS_DIR,
    SESSION_START,
    TIME_BUDGET_SECONDS,
    json,
    longbench_samples,
    per_layer_kv_frobenius,
    time,
):
    # Every completed sample is appended to this JSONL file immediately. On
    # re-run (e.g. after a molab session got cut off), already-checkpointed
    # samples are loaded back in and skipped rather than recomputed.
    _checkpoint_path = RESULTS_DIR / "frobenius_checkpoint.jsonl"

    results = {}
    if _checkpoint_path.exists():
        with open(_checkpoint_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    _entry = json.loads(_line)
                    results[_entry["sample_idx"]] = _entry
        print(f"Resuming: {len(results)} sample(s) already checkpointed.")

    _todo = [(i, s) for i, s in enumerate(longbench_samples) if i not in results]
    print(f"{len(_todo)} sample(s) remaining out of {len(longbench_samples)}.")

    _start = time.time()
    with open(_checkpoint_path, "a") as _ckpt_f:
        for _done_count, (_idx, _sample) in enumerate(_todo, start=1):
            if time.time() - SESSION_START >= TIME_BUDGET_SECONDS:
                print(
                    f"Time budget ({TIME_BUDGET_SECONDS / 3600:.1f}h) reached -- stopping early, "
                    f"{len(_todo) - _done_count + 1} sample(s) left for the next run."
                )
                break
            _t0 = time.time()
            try:
                _per_layer = per_layer_kv_frobenius(_sample["context"])
                _entry = {"task": _sample["task"], "sample_idx": _idx, "per_layer": _per_layer}
            except Exception as e:
                print(f"Skipping sample {_idx} ({_sample['task']}): {e}")
                _entry = {"task": _sample["task"], "sample_idx": _idx, "error": str(e)}

            results[_idx] = _entry
            _ckpt_f.write(json.dumps(_entry) + "\n")
            _ckpt_f.flush()

            _elapsed = time.time() - _start
            _avg = _elapsed / _done_count
            _eta_min = _avg * (len(_todo) - _done_count) / 60
            print(
                f"[{_done_count}/{len(_todo)} remaining] {_sample['task']} "
                f"({time.time() - _t0:.2f}s, avg {_avg:.2f}s/sample, ETA {_eta_min:.1f} min)"
            )
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### aggregating

    manual and real stay as separate tables and separate files the whole way through. i tried one combined csv early on and it got confusing quickly.
    """)
    return


@app.cell
def _(pd, results):
    # Two distinct error measurements (see per_layer_kv_frobenius) are split
    # into two separate DataFrames/output files rather than one combined
    # table, so "manual quantize/dequantize error" and "real NVFP4QuantizedCache
    # error" stay clearly separate artifacts, same as benchmark 1 (Frobenius)
    # vs benchmark 2 (real-metric) already being separate files.
    _rows = []
    for _idx, _entry in results.items():
        if "per_layer" not in _entry:
            continue
        for _layer_stats in _entry["per_layer"]:
            _rows.append({"sample_idx": _idx, "task": _entry["task"], **_layer_stats})

    _all_df = pd.DataFrame(_rows)
    _shared_cols = ["sample_idx", "task", "layer", "k_orig_norm", "v_orig_norm"]

    if len(_all_df):
        _manual_cols = [c for c in _all_df.columns if "_manual_" in c]
        _real_cols = [c for c in _all_df.columns if "_real_" in c]
        frobenius_manual_df = _all_df[_shared_cols + _manual_cols]
        frobenius_real_df = _all_df[_shared_cols + _real_cols]

        manual_summary = (
            frobenius_manual_df.groupby("layer")[["k_manual_rel_error", "v_manual_rel_error", "k_manual_diff_norm", "v_manual_diff_norm"]]
            .mean()
            .reset_index()
        )
        manual_task_summary = (
            frobenius_manual_df.groupby("task")[["k_manual_rel_error", "v_manual_rel_error"]]
            .mean()
            .reset_index()
            .sort_values("k_manual_rel_error", ascending=False)
        )
        print("--- manual (quantize/dequantize the whole tensor) ---")
        print(manual_summary.to_string(index=False))
        print(manual_task_summary.to_string(index=False))

        real_summary = (
            frobenius_real_df.groupby("layer")[["k_real_rel_error", "v_real_rel_error", "k_real_diff_norm", "v_real_diff_norm"]]
            .mean()
            .reset_index()
        )
        real_task_summary = (
            frobenius_real_df.groupby("task")[["k_real_rel_error", "v_real_rel_error"]]
            .mean()
            .reset_index()
            .sort_values("k_real_rel_error", ascending=False)
        )
        print("--- real (actual NVFP4QuantizedCache) ---")
        print(real_summary.to_string(index=False))
        print(real_task_summary.to_string(index=False))
    else:
        frobenius_manual_df = _all_df
        frobenius_real_df = _all_df
        manual_summary = manual_task_summary = _all_df
        real_summary = real_task_summary = _all_df
        print("No completed samples yet.")
    return (
        frobenius_manual_df,
        frobenius_real_df,
        manual_summary,
        manual_task_summary,
        real_summary,
        real_task_summary,
    )


@app.cell
def _(
    RESULTS_DIR,
    frobenius_manual_df,
    frobenius_real_df,
    manual_summary,
    manual_task_summary,
    real_summary,
    real_task_summary,
):
    # Two separate files per error type -- these are full re-derived
    # snapshots of the checkpoint (frobenius_checkpoint.jsonl) each time this
    # runs; the checkpoint is the resumable source of truth, these CSVs are
    # just for downstream analysis.
    if len(frobenius_manual_df):
        frobenius_manual_df.to_csv(RESULTS_DIR / "frobenius_manual_raw.csv", index=False)
        manual_summary.to_csv(RESULTS_DIR / "frobenius_manual_layer_summary.csv", index=False)
        manual_task_summary.to_csv(RESULTS_DIR / "frobenius_manual_task_summary.csv", index=False)

        frobenius_real_df.to_csv(RESULTS_DIR / "frobenius_real_raw.csv", index=False)
        real_summary.to_csv(RESULTS_DIR / "frobenius_real_layer_summary.csv", index=False)
        real_task_summary.to_csv(RESULTS_DIR / "frobenius_real_task_summary.csv", index=False)
        print(f"Saved Frobenius-norm results (manual + real, 6 files) to {RESULTS_DIR}/")
    else:
        print("Nothing to save yet.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. benchmark 2: does it actually hurt the answers

    norms are reassuring but they don't tell me whether the output got worse. so: same 21 tasks, but now generate an answer per question twice, bf16 cache and nvfp4 cache, and score both.

    no deduping here, f1 depends on the question. this is much slower, a full `generate()` per sample per variant, which is why the time budget exists.
    """)
    return


@app.cell
def _(STANDARD_LONGBENCH_TASKS, load_dataset):
    # Separate from the Frobenius corpus above: this half needs a full
    # generate() call per (context, question) pair, per cache variant, so
    # it's far more expensive per-sample. Runs the same full 21-task corpus
    # as the Frobenius sweep -- every question (not deduped by context, since
    # F1 depends on the question) gets scored, so this is the expensive half.
    METRIC_TASKS = STANDARD_LONGBENCH_TASKS
    MAX_SAMPLES_PER_METRIC_TASK = None  # None = full task; set an int to cap for a quick smoke test

    metric_samples = []
    for _task in METRIC_TASKS:
        _ds = load_dataset("Xnhyacinth/LongBench", _task, split="test")
        if MAX_SAMPLES_PER_METRIC_TASK is not None:
            _ds = _ds.select(range(min(MAX_SAMPLES_PER_METRIC_TASK, len(_ds))))
        for _row in _ds:
            metric_samples.append(
                {
                    "task": _task,
                    "context": _row["context"],
                    "question": _row["question"],
                    "answers": list(_row["answers"]),
                    "answer_prefix": _row.get("answer_prefix", ""),
                    "all_classes": list(_row["all_classes"]) if _row.get("all_classes") is not None else None,
                }
            )
        print(f"  {_task}: {len(_ds)} question(s)")

    print(f"Loaded {len(metric_samples)} (context, question) pairs across {len(METRIC_TASKS)} tasks")
    return (metric_samples,)


@app.cell
def _(model, tokenizer, torch):
    # Approximate per-task generation budgets, following the shape of
    # LongBench's own dataset2maxlen.json (short answers for QA/retrieval/
    # counting, long ones for summarization) -- not copied verbatim from that
    # file, so tune individual values if you have it and want exact parity.
    MAX_NEW_TOKENS_BY_TASK = {
        "narrativeqa": 128,
        "qasper": 128,
        "multifieldqa_en": 64,
        "multifieldqa_zh": 64,
        "hotpotqa": 32,
        "2wikimqa": 32,
        "musique": 32,
        "dureader": 128,
        "gov_report": 512,
        "qmsum": 512,
        "multi_news": 512,
        "vcsum": 512,
        "trec": 64,
        "triviaqa": 32,
        "samsum": 128,
        "lsht": 64,
        "passage_count": 32,
        "passage_retrieval_en": 32,
        "passage_retrieval_zh": 32,
        "lcc": 64,
        "repobench-p": 64,
    }
    DEFAULT_MAX_NEW_TOKENS = 64
    MAX_METRIC_CONTEXT_TOKENS = 8192  # same truncation budget as the Frobenius benchmark

    def _truncate_to_length(text: str, max_tokens: int) -> str:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= max_tokens:
            return text
        half = max_tokens // 2
        truncated_ids = token_ids[:half] + token_ids[-half:]
        return tokenizer.decode(truncated_ids)

    @torch.no_grad()
    def generate_answer(context: str, question: str, answer_prefix: str, task: str, cache_factory):
        context = _truncate_to_length(context, MAX_METRIC_CONTEXT_TOKENS)
        # Not the official per-task LongBench prompt template (that lives in
        # LongBench's own dataset2prompt.json, not reproduced here) -- fine
        # for a relative baseline-vs-NVFP4 comparison since both variants use
        # the exact same prompt, but don't treat these scores as directly
        # comparable to published LongBench leaderboard numbers.
        prompt = f"{context}\n\nQuestion: {question}\n{answer_prefix}"
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        max_new_tokens = MAX_NEW_TOKENS_BY_TASK.get(task, DEFAULT_MAX_NEW_TOKENS)

        out = model.generate(
            **model_inputs,
            past_key_values=cache_factory(),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated = out[0, model_inputs["input_ids"].shape[-1] :]
        answer = tokenizer.decode(generated, skip_special_tokens=True)

        del model_inputs, out
        torch.cuda.empty_cache()
        return answer

    return (generate_answer,)


@app.cell
def _(
    NVFP4QuantizedCache,
    RESULTS_DIR,
    SESSION_START,
    TIME_BUDGET_SECONDS,
    generate_answer,
    json,
    metric_samples,
    model,
    time,
):
    # Same checkpointing pattern as the Frobenius loop: each completed sample
    # (both variants) is appended to a JSONL file immediately, and a re-run
    # skips whatever a prior (possibly interrupted) run already finished.
    from transformers import DynamicCache

    _checkpoint_path = RESULTS_DIR / "metric_checkpoint.jsonl"

    metric_results = {}
    if _checkpoint_path.exists():
        with open(_checkpoint_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    _entry = json.loads(_line)
                    metric_results[_entry["sample_idx"]] = _entry
        print(f"Resuming: {len(metric_results)} sample(s) already checkpointed.")

    _todo = [(i, s) for i, s in enumerate(metric_samples) if i not in metric_results]
    print(f"{len(_todo)} sample(s) remaining out of {len(metric_samples)}.")

    _start = time.time()
    with open(_checkpoint_path, "a") as _ckpt_f:
        for _done_count, (_idx, _sample) in enumerate(_todo, start=1):
            if time.time() - SESSION_START >= TIME_BUDGET_SECONDS:
                print(
                    f"Time budget ({TIME_BUDGET_SECONDS / 3600:.1f}h) reached -- stopping early, "
                    f"{len(_todo) - _done_count + 1} sample(s) left for the next run."
                )
                break
            _t0 = time.time()
            _row_result = {
                "sample_idx": _idx,
                "task": _sample["task"],
                "answers": _sample["answers"],
                "all_classes": _sample["all_classes"],
            }
            try:
                _row_result["baseline_answer"] = generate_answer(
                    _sample["context"], _sample["question"], _sample["answer_prefix"], _sample["task"],
                    cache_factory=lambda: DynamicCache(),
                )
                _row_result["nvfp4_answer"] = generate_answer(
                    _sample["context"], _sample["question"], _sample["answer_prefix"], _sample["task"],
                    cache_factory=lambda: NVFP4QuantizedCache(model.config),
                )
            except Exception as e:
                print(f"Skipping sample {_idx} ({_sample['task']}): {e}")
                _row_result["error"] = str(e)

            metric_results[_idx] = _row_result
            _ckpt_f.write(json.dumps(_row_result) + "\n")
            _ckpt_f.flush()

            _elapsed = time.time() - _start
            _avg = _elapsed / _done_count
            _eta_min = _avg * (len(_todo) - _done_count) / 60
            print(
                f"[{_done_count}/{len(_todo)} remaining] {_sample['task']} "
                f"({time.time() - _t0:.2f}s, avg {_avg:.2f}s/sample, ETA {_eta_min:.1f} min)"
            )
    return (metric_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### scoring

    per task metrics following `calculate_metrics.py` from sparse-attention-hub, which is longbench's own mapping (f1 for qa, rouge-l for summarisation, exact match for classification, paragraph id for retrieval, fuzzy line match for code). worth being honest about one thing: the prompt template here is mine, not longbench's official `dataset2prompt.json`. fine for baseline vs nvfp4 since both sides see the identical prompt, not comparable to published leaderboard numbers.
    """)
    return


@app.cell
def _(metric_results, pd):
    # sparse-attention-hub/benchmark/longbench/calculate_metrics.py 
    # matches LongBench's official per-task metric choices: F1 for QA,
    # ROUGE-L for summarization, exact-match-in-prediction for classification,
    # paragraph-id extraction for retrieval, digit extraction for counting,
    # fuzzy line match for code) so all 21 tasks get their correct metric
    # rather than one score type applied uniformly.
    import re
    import string
    from collections import Counter

    import jieba
    from fuzzywuzzy import fuzz
    from rouge import Rouge

    def _normalize_answer(s: str) -> str:
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        return white_space_fix(remove_articles(remove_punc(s.lower())))

    def _normalize_zh_answer(s: str) -> str:
        cn_punctuation = (
            "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、"
            "〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        )
        all_punctuation = set(string.punctuation + cn_punctuation)

        def white_space_fix(text):
            return "".join(text.split())

        def remove_punc(text):
            return "".join(ch for ch in text if ch not in all_punctuation)

        return white_space_fix(remove_punc(s.lower()))

    def _f1(prediction_tokens, ground_truth_tokens) -> float:
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    def qa_f1_score(prediction, ground_truth, **kwargs) -> float:
        return _f1(_normalize_answer(prediction).split(), _normalize_answer(ground_truth).split())

    def qa_f1_zh_score(prediction, ground_truth, **kwargs) -> float:
        prediction_tokens = [_normalize_zh_answer(t) for t in jieba.cut(prediction, cut_all=False)]
        ground_truth_tokens = [_normalize_zh_answer(t) for t in jieba.cut(ground_truth, cut_all=False)]
        prediction_tokens = [t for t in prediction_tokens if len(t) > 0]
        ground_truth_tokens = [t for t in ground_truth_tokens if len(t) > 0]
        return _f1(prediction_tokens, ground_truth_tokens)

    def rouge_score(prediction, ground_truth, **kwargs) -> float:
        try:
            scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
        except Exception:
            return 0.0
        return scores["rouge-l"]["f"]

    def rouge_zh_score(prediction, ground_truth, **kwargs) -> float:
        prediction = " ".join(jieba.cut(prediction, cut_all=False))
        ground_truth = " ".join(jieba.cut(ground_truth, cut_all=False))
        return rouge_score(prediction, ground_truth)

    def classification_score(prediction, ground_truth, **kwargs) -> float:
        all_classes = kwargs["all_classes"]
        em_match_list = [c for c in all_classes if c in prediction]
        for match_term in list(em_match_list):
            if match_term in ground_truth and match_term != ground_truth:
                em_match_list.remove(match_term)
        if ground_truth in em_match_list:
            return 1.0 / len(em_match_list)
        return 0.0

    def retrieval_score(prediction, ground_truth, **kwargs) -> float:
        matches = re.findall(r"Paragraph (\d+)", ground_truth)
        ground_truth_id = matches[0]
        numbers = re.findall(r"\d+", prediction)
        right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
        return 0.0 if len(numbers) == 0 else right_num / len(numbers)

    def retrieval_zh_score(prediction, ground_truth, **kwargs) -> float:
        matches = re.findall(r"段落(\d+)", ground_truth)
        ground_truth_id = matches[0]
        numbers = re.findall(r"\d+", prediction)
        right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
        return 0.0 if len(numbers) == 0 else right_num / len(numbers)

    def count_score(prediction, ground_truth, **kwargs) -> float:
        numbers = re.findall(r"\d+", prediction)
        right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
        return 0.0 if len(numbers) == 0 else right_num / len(numbers)

    def code_sim_score(prediction, ground_truth, **kwargs) -> float:
        all_lines = prediction.lstrip("\n").split("\n")
        prediction_line = ""
        for line in all_lines:
            if ("`" not in line) and ("#" not in line) and ("//" not in line):
                prediction_line = line
                break
        return fuzz.ratio(prediction_line, ground_truth) / 100

    dataset2metric = {
        "narrativeqa": qa_f1_score,
        "qasper": qa_f1_score,
        "multifieldqa_en": qa_f1_score,
        "multifieldqa_zh": qa_f1_zh_score,
        "hotpotqa": qa_f1_score,
        "2wikimqa": qa_f1_score,
        "musique": qa_f1_score,
        "dureader": rouge_zh_score,
        "gov_report": rouge_score,
        "qmsum": rouge_score,
        "multi_news": rouge_score,
        "vcsum": rouge_zh_score,
        "trec": classification_score,
        "triviaqa": qa_f1_score,
        "samsum": rouge_score,
        "lsht": classification_score,
        "passage_retrieval_en": retrieval_score,
        "passage_count": count_score,
        "passage_retrieval_zh": retrieval_zh_score,
        "lcc": code_sim_score,
        "repobench-p": code_sim_score,
    }

    # Matches calculate_metrics.py::scorer: these four tasks' answers are
    # single-line, so only the first line of the prediction is scored.
    _first_line_only_tasks = {"trec", "triviaqa", "samsum", "lsht"}

    def _best_score(task: str, prediction: str, ground_truths, all_classes) -> float:
        if task in _first_line_only_tasks:
            prediction = prediction.lstrip("\n").split("\n")[0]
        metric_fn = dataset2metric[task]
        return max(
            (metric_fn(prediction, gt, all_classes=all_classes) for gt in ground_truths),
            default=0.0,
        )

    # baseline (never quantized) and nvfp4 (quantized) scores are kept as two
    # separate rows/tables/files rather than one combined table -- same split
    # as benchmark 1's manual/real Frobenius files.
    _baseline_rows = []
    _nvfp4_rows = []
    for _entry in metric_results.values():
        if "error" in _entry:
            continue
        _baseline_rows.append(
            {
                "sample_idx": _entry["sample_idx"],
                "task": _entry["task"],
                "answers": _entry["answers"],
                "answer": _entry["baseline_answer"],
                "score": _best_score(
                    _entry["task"], _entry["baseline_answer"], _entry["answers"], _entry["all_classes"]
                ),
            }
        )
        _nvfp4_rows.append(
            {
                "sample_idx": _entry["sample_idx"],
                "task": _entry["task"],
                "answers": _entry["answers"],
                "answer": _entry["nvfp4_answer"],
                "score": _best_score(
                    _entry["task"], _entry["nvfp4_answer"], _entry["answers"], _entry["all_classes"]
                ),
            }
        )

    metric_baseline_df = pd.DataFrame(_baseline_rows)
    metric_nvfp4_df = pd.DataFrame(_nvfp4_rows)

    if len(metric_baseline_df):
        metric_baseline_task_summary = (
            metric_baseline_df.groupby("task")["score"].mean().reset_index().sort_values("score")
        )
        metric_nvfp4_task_summary = (
            metric_nvfp4_df.groupby("task")["score"].mean().reset_index().sort_values("score")
        )
        print("--- baseline (never quantized) ---")
        print(metric_baseline_task_summary.to_string(index=False))
        print("--- nvfp4 (quantized) ---")
        print(metric_nvfp4_task_summary.to_string(index=False))
        print(
            f"\nOverall: baseline score={metric_baseline_df['score'].mean() * 100:.2f}, "
            f"NVFP4 score={metric_nvfp4_df['score'].mean() * 100:.2f}"
        )
    else:
        metric_baseline_task_summary = metric_baseline_df
        metric_nvfp4_task_summary = metric_nvfp4_df
        print("No completed samples yet.")
    return (
        metric_baseline_df,
        metric_baseline_task_summary,
        metric_nvfp4_df,
        metric_nvfp4_task_summary,
    )


@app.cell
def _(
    RESULTS_DIR,
    metric_baseline_df,
    metric_baseline_task_summary,
    metric_nvfp4_df,
    metric_nvfp4_task_summary,
):
    if len(metric_baseline_df):
        metric_baseline_df.to_csv(RESULTS_DIR / "metric_baseline_raw.csv", index=False)
        metric_baseline_task_summary.to_csv(RESULTS_DIR / "metric_baseline_task_summary.csv", index=False)

        metric_nvfp4_df.to_csv(RESULTS_DIR / "metric_nvfp4_raw.csv", index=False)
        metric_nvfp4_task_summary.to_csv(RESULTS_DIR / "metric_nvfp4_task_summary.csv", index=False)
        print(f"Saved real-metric results (baseline + nvfp4, 4 files) to {RESULTS_DIR}/")
    else:
        print("Nothing to save yet.")
    return


@app.cell(hide_code=True)
def benchmark_title(mo):
    mo.md(r"""
    ---
    ## 7. benchmark

    results from a previous full run on the molab gpu (qwen3-4b, bf16 weights, contexts truncated to 8192 tokens, nvfp4 cache with block size 16, residual 128, per token blocking for both k and v). copied here as text so they're readable without re-running anything; the live cells above write the same numbers to `benchmark_results/` when they run.
    """)
    return


@app.cell(hide_code=True)
def benchmark_frobenius(mo):
    mo.md(r"""
    ### benchmark 1: kv cache error (frobenius)

    relative error = `||nvfp4 - bf16|| / ||bf16||` per layer, averaged over all contexts. the manual and real measurements came out **identical to every printed digit**, which is expected: right after a prefill the whole context is quantized and the residual buffer is still empty, so both measure the same thing. one table covers both.

    #### per layer

    | layer | k rel error | v rel error | k diff norm | v diff norm |
    |---|---|---|---|---|
    | 0 | 0.041884 | 0.092537 | 1451.020110 | 7.516764 |
    | 1 | 0.087686 | 0.093377 | 703.765417 | 12.504006 |
    | 2 | 0.091669 | 0.094032 | 415.745877 | 17.466104 |
    | 3 | 0.092945 | 0.093708 | 449.057129 | 23.426127 |
    | 4 | 0.090280 | 0.094563 | 789.969282 | 30.093416 |
    | 5 | 0.088019 | 0.093913 | 1166.733636 | 34.366171 |
    | 6 | 0.093076 | 0.094332 | 634.766895 | 44.355147 |
    | 7 | 0.091876 | 0.093989 | 428.254853 | 59.480669 |
    | 8 | 0.091302 | 0.094240 | 768.375154 | 75.338805 |
    | 9 | 0.092800 | 0.094278 | 418.802659 | 77.295119 |
    | 10 | 0.091691 | 0.094074 | 633.963258 | 100.585865 |
    | 11 | 0.092229 | 0.093785 | 635.323525 | 70.557270 |
    | 12 | 0.091599 | 0.093717 | 528.656796 | 80.088275 |
    | 13 | 0.093171 | 0.092875 | 430.511657 | 74.050031 |
    | 14 | 0.091863 | 0.092835 | 560.138991 | 96.995498 |
    | 15 | 0.092739 | 0.092427 | 446.080411 | 95.076929 |
    | 16 | 0.090967 | 0.093480 | 441.518535 | 116.056853 |
    | 17 | 0.093264 | 0.092150 | 451.991766 | 107.600040 |
    | 18 | 0.092173 | 0.092780 | 455.256617 | 118.627525 |
    | 19 | 0.093037 | 0.092959 | 462.687567 | 146.742036 |
    | 20 | 0.093524 | 0.092781 | 461.103347 | 143.451216 |
    | 21 | 0.092391 | 0.092819 | 472.179348 | 156.703511 |
    | 22 | 0.091525 | 0.093762 | 494.488753 | 188.052560 |
    | 23 | 0.092489 | 0.093526 | 552.374090 | 187.340436 |
    | 24 | 0.093345 | 0.094160 | 454.760584 | 249.907859 |
    | 25 | 0.092060 | 0.094106 | 486.501771 | 223.118850 |
    | 26 | 0.092622 | 0.094488 | 428.085400 | 258.701302 |
    | 27 | 0.092068 | 0.094203 | 417.535973 | 293.923758 |
    | 28 | 0.092205 | 0.094295 | 442.907375 | 318.920193 |
    | 29 | 0.092803 | 0.094629 | 436.396624 | 485.754848 |
    | 30 | 0.092183 | 0.094244 | 480.550149 | 505.207241 |
    | 31 | 0.092464 | 0.094510 | 435.442383 | 608.907807 |
    | 32 | 0.093161 | 0.094451 | 433.261157 | 738.426759 |
    | 33 | 0.092968 | 0.094511 | 405.574420 | 1135.233754 |
    | 34 | 0.092085 | 0.093975 | 451.803989 | 1013.690823 |
    | 35 | 0.092786 | 0.094026 | 718.033585 | 712.616466 |

    keys sit at 8.8-9.4% relative error in every layer except layer 0 (4.2%), values at 9.2-9.5% in every layer. the diff norms grow with depth for values because the value norms themselves grow, the relative error stays flat.

    #### per task

    | task | k rel error | v rel error |
    |---|---|---|
    | trec | 0.090762 | 0.093758 |
    | dureader | 0.090749 | 0.093788 |
    | multifieldqa_zh | 0.090745 | 0.093722 |
    | musique | 0.090727 | 0.093750 |
    | passage_retrieval_zh | 0.090725 | 0.093773 |
    | lsht | 0.090723 | 0.093787 |
    | hotpotqa | 0.090721 | 0.093731 |
    | 2wikimqa | 0.090713 | 0.093719 |
    | multi_news | 0.090700 | 0.093741 |
    | narrativeqa | 0.090690 | 0.093878 |
    | repobench-p | 0.090682 | 0.093684 |
    | qasper | 0.090682 | 0.093656 |
    | triviaqa | 0.090681 | 0.093759 |
    | passage_count | 0.090680 | 0.093696 |
    | multifieldqa_en | 0.090672 | 0.093702 |
    | lcc | 0.090670 | 0.093711 |
    | vcsum | 0.090666 | 0.093797 |
    | gov_report | 0.090663 | 0.093671 |
    | passage_retrieval_en | 0.090659 | 0.093747 |
    | qmsum | 0.090625 | 0.093789 |
    | samsum | 0.090614 | 0.093866 |

    basically the same error on every task (k 9.061-9.076%, v 9.366-9.388%): the error depends on the number format, not on what the text is about.
    """)
    return


@app.cell(hide_code=True)
def benchmark_longbench(mo):
    mo.md(r"""
    ### benchmark 2: longbench answers (baseline vs nvfp4)

    scores are the per task longbench metric x 100 (averaged per task, sorted by baseline). prompt template is mine, not the official one, so compare the two columns with each other, not with published leaderboard numbers.

    | task | metric | baseline (bf16) | nvfp4 | change (points) |
    |---|---|---|---|---|
    | narrativeqa | f1 | 3.07 | 2.99 | -0.08 |
    | passage_count | count | 5.04 | 4.08 | -0.96 |
    | lsht | classification | 7.35 | 7.42 | +0.07 |
    | musique | f1 | 7.48 | 7.71 | +0.23 |
    | qasper | f1 | 11.43 | 11.60 | +0.17 |
    | hotpotqa | f1 | 11.47 | 11.16 | -0.30 |
    | vcsum | rouge-l (zh) | 12.11 | 10.43 | -1.67 |
    | 2wikimqa | f1 | 13.09 | 12.16 | -0.92 |
    | qmsum | rouge-l | 20.22 | 20.03 | -0.19 |
    | multifieldqa_zh | f1 (zh) | 22.13 | 23.15 | +1.02 |
    | multi_news | rouge-l | 23.38 | 22.76 | -0.61 |
    | dureader | rouge-l (zh) | 24.48 | 24.69 | +0.20 |
    | multifieldqa_en | f1 | 24.55 | 24.86 | +0.31 |
    | gov_report | rouge-l | 29.41 | 29.58 | +0.16 |
    | samsum | rouge-l | 41.89 | 40.68 | -1.21 |
    | lcc | code sim | 50.77 | 51.54 | +0.78 |
    | trec | classification | 58.00 | 58.00 | +0.00 |
    | passage_retrieval_en | retrieval | 61.17 | 58.83 | -2.33 |
    | repobench-p | code sim | 63.72 | 63.14 | -0.58 |
    | triviaqa | f1 | 83.42 | 84.16 | +0.74 |
    | passage_retrieval_zh | retrieval (zh) | 97.22 | 97.14 | -0.08 |
    | **overall** | | **35.24** | **35.03** | **-0.21** |

    nvfp4 is higher on 9 tasks, lower on 11 and equal on 1. overall it costs **0.21 points** (35.24 -> 35.03). the biggest drops are passage_retrieval_en (-2.33), vcsum (-1.67), samsum (-1.21) and passage_count (-0.96); the biggest gains multifieldqa_zh (+1.02), lcc (+0.78) and triviaqa (+0.74).
    """)
    return


if __name__ == "__main__":
    app.run()
