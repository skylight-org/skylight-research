"""Inkling 2 Mini real decode loop with the paged fused-Quest operator, on an A100-80GB.

    modal run paged_modal.py --context-tokens 32000 --check     # wiring check + 32k passkey
    modal run paged_modal.py --context-tokens 128000 --prefill-chunk 256
"""

import modal

app = modal.App("inkling-paged-decode")
vol = modal.Volume.from_name("inkling-workspace")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "g++")
    .pip_install("torch==2.9.1", "torchvision==0.24.1", "ninja", "numpy", "packaging", "setuptools", "wheel")
    .pip_install(
        "transformers==5.24.1", "accelerate", "safetensors", "tiktoken", "pillow", "sentencepiece",
        "ray[tune]>=2.48.0", "hyperopt", "pydantic", "tqdm", "pandas", "scikit-learn", "matplotlib",
        "seaborn", "plotly", "psutil", "pynvml", "colorama", "fire", "datasets", "pyyaml",
    )
    .run_commands(
        "git clone --quiet https://x-access-token:$GITHUB_TOKEN@github.com/skylight-org/skylight_kernels.git /opt/sk"
        " && git -C /opt/sk remote set-url origin https://github.com/skylight-org/skylight_kernels.git",
        secrets=[modal.Secret.from_name("github-readonly")],
    )
    .add_local_file("patch_gs3.py", "/opt/patch_gs3.py", copy=True)
    .run_commands(
        "python /opt/patch_gs3.py /opt/sk",
        "cd /opt/sk && CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8 python setup.py build_ext --inplace > /opt/build.log 2>&1"
        " || (tail -40 /opt/build.log; exit 1)",
    )
    .env({"PYTHONPATH": "/opt/sk:/vol/repos/skylight-research", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("inkling_sah", "inkling_sparse_op", "inkling_quest_triton", "inkling_paged")
)

MODEL_PATH = "/vol/models/Inkling-2-Mini"
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, volumes={"/vol": vol}, timeout=40 * 60)
def run(context_tokens: int = 32000, prefill_chunk: int = 1024, check: bool = False,
        max_new_tokens: int = 160) -> dict:
    import copy
    import json
    import os
    import time
    import traceback

    import torch

    t0 = time.time()
    res: dict = {"context_tokens_requested": context_tokens}
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    try:
        from sparse_attention_hub.adapters import Request
        from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMaskerConfig,
        )

        import inkling_paged as ip

        adapter = ip.InklingPagedAdapterHF(
            model_name=MODEL_PATH,
            sparse_attention_config=ResearchAttentionConfig(masker_configs=[LocalMaskerConfig(window_size=8)]),
            model_kwargs={"dtype": torch.bfloat16, "device_map": "cuda"}, device="cuda",
            prefill_chunk_size=prefill_chunk,
        )
        model, tok = adapter.model.eval(), adapter.tokenizer
        log("loaded")

        if check:
            ids = tok(FILLER * 500, return_tensors="pt").input_ids[:, :8193].cuda()
            res["check"] = {}
            with torch.no_grad():
                ip.SLICE_SLIDING = False
                pre = model(ids[:, :-1], past_key_values=None, use_cache=True)
                res["cache_layer_types"] = sorted({type(l).__name__ for l in pre.past_key_values.layers})
                res["sliding_layer0_keys_len"] = int(pre.past_key_values.layers[0].keys.shape[-2])
                for frac in (1.0, 0.02):
                    c_dense = copy.deepcopy(pre.past_key_values)
                    c_paged = copy.deepcopy(pre.past_key_values)
                    ip.install_paged(c_paged, frac, 8192 + 32)
                    nxt, rows = ids[:, -1:], []
                    for _ in range(4):
                        ip.SLICE_SLIDING = False      # reference: plain Inkling eager everywhere
                        ld = model(nxt, past_key_values=c_dense, use_cache=True).logits[0, -1].float()
                        ip.SLICE_SLIDING = True       # paged full attention + sliding-window slicing
                        with adapter.enable_sparse_mode():
                            lp = model(nxt, past_key_values=c_paged, use_cache=True,
                                       sparse_meta_data={}).logits[0, -1].float()
                        rows.append({"cos": round(float(torch.nn.functional.cosine_similarity(ld, lp, dim=0)), 6),
                                     "max_abs": round(float((ld - lp).abs().max()), 4),
                                     "top1_match": bool(ld.argmax() == lp.argmax())})
                        nxt = ld.argmax().view(1, 1)
                    res["check"][f"top{frac:g}"] = {"k_pages": ip.REGISTRY[5].k, "steps": rows}
                    ip.REGISTRY.clear()
                    del c_dense, c_paged
            adapter.prefill_memo.clear()
            torch.cuda.empty_cache()
            log("check", json.dumps(res["check"]))

        passkey = "68297"
        n_fill = max(1, context_tokens // 22)
        depth = int(n_fill * 0.3)
        context = (FILLER * depth + f" The pass key is {passkey}. Remember it. {passkey} is the pass key. "
                   + FILLER * (n_fill - depth))
        req = Request(context=context, questions="What is the pass key? Reply with only the number.",
                      answer_prefix="")
        res["needle"] = {}
        for name, paged, slice_, frac in (("dense + sliding fix", False, True, None),
                                          ("dense eager (no fix)", False, False, None),
                                          ("paged quest 1% + sliding fix", True, True, 0.01),
                                          ("paged quest 2% + sliding fix", True, True, 0.02)):
            adapter.paged, adapter.top_frac = paged, frac or 0.0
            ip.SLICE_SLIDING = slice_
            torch.cuda.reset_peak_memory_stats()
            ts = time.time()
            r = adapter.process_request(req, generation_kwargs={"max_new_tokens": max_new_tokens},
                                        request_kwargs={"max_context_length": 10**9})
            res["needle"][name] = {"correct": passkey in r.responses, "answer_tail": r.responses.strip()[-120:],
                                   "seconds_total": round(time.time() - ts, 1),
                                   "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                   **adapter.last_timing}
            log(name, json.dumps(res["needle"][name]))
    except Exception:  # noqa: BLE001
        res["error"] = traceback.format_exc()[-4000:]
        print(res["error"], flush=True)
    res["total_s"] = round(time.time() - t0, 1)
    os.makedirs("/vol/results", exist_ok=True)
    with open(f"/vol/results/paged_{context_tokens}_{int(t0)}.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    vol.commit()
    return res


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, volumes={"/vol": vol}, timeout=30 * 60)
def profile(contexts: str = "32000,128000", steps: int = 5) -> dict:
    """torch.profiler breakdown of single decode steps (dense eager vs paged Quest 2%)."""
    import json
    import time
    import traceback
    from collections import defaultdict

    import torch
    from torch.profiler import ProfilerActivity, record_function
    from transformers.models.inkling2_loop import modeling_inkling2_loop as M

    t0 = time.time()
    res: dict = {}
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    try:
        from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMaskerConfig,
        )

        import inkling_paged as ip

        adapter = ip.InklingPagedAdapterHF(
            model_name=MODEL_PATH,
            sparse_attention_config=ResearchAttentionConfig(masker_configs=[LocalMaskerConfig(window_size=8)]),
            model_kwargs={"dtype": torch.bfloat16, "device_map": "cuda"}, device="cuda", prefill_chunk_size=256)
        model, tok = adapter.model.eval(), adapter.tokenizer
        res["gpu"] = torch.cuda.get_device_name(0)

        # named ranges: every Inkling module class + lm_head + mask/distance helpers
        want = {n for n in dir(M) if n.startswith("Inkling2Loop") and isinstance(getattr(M, n), type)
                and issubclass(getattr(M, n), torch.nn.Module)}
        for name, mod in model.named_modules():
            cls = "lm_head" if name.endswith("lm_head") else type(mod).__name__
            if cls not in want and cls != "lm_head":
                continue

            def pre(m, args, _n=cls):
                rf = record_function("mod::" + _n)
                rf.__enter__()
                m.__dict__.setdefault("_rf", []).append(rf)

            def post(m, args, out):
                m._rf.pop().__exit__(None, None, None)

            mod.register_forward_pre_hook(pre)
            mod.register_forward_hook(post)
        for fname in ("create_causal_mask", "create_sliding_window_causal_mask",
                      "create_recurrent_attention_mask", "get_relative_distance"):
            orig = getattr(M, fname)

            def wrapped(*a, _o=orig, _n=fname, **k):
                with record_function("fn::" + _n):
                    return _o(*a, **k)

            setattr(M, fname, wrapped)

        for L in [int(x) for x in contexts.split(",")]:
            ids = tok(FILLER * (L // 20 + 10), return_tensors="pt").input_ids[:, :L + 1].cuda()
            adapter.prefill_memo.clear()
            torch.cuda.empty_cache()
            with torch.no_grad():
                ts = time.time()
                pre_out = model(ids[:, :-1], past_key_values=None, use_cache=True)
                log(f"prefill {L}: {time.time() - ts:.1f}s")
                for mode in ("dense", "paged2%"):
                    cache = pre_out.past_key_values if mode == "dense" else \
                        model(ids[:, :-1], past_key_values=None, use_cache=True).past_key_values  # memo copy
                    if mode != "dense":
                        ip.install_paged(cache, 0.02, L + 64)
                    nxt = ids[:, -1:]

                    def step():
                        nonlocal nxt
                        with adapter.enable_sparse_mode():
                            out = model(input_ids=nxt, past_key_values=cache, use_cache=True,
                                        logits_to_keep=1, sparse_meta_data={})
                        nxt = out.logits[:, -1:].argmax(-1)

                    for _ in range(3):
                        step()
                    torch.cuda.synchronize()
                    ts = time.perf_counter()
                    for _ in range(steps):
                        step()
                    torch.cuda.synchronize()
                    wall = 1000 * (time.perf_counter() - ts) / steps
                    with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        for _ in range(steps):
                            step()
                        torch.cuda.synchronize()
                    ka = prof.key_averages()
                    per = lambda us: round(us / 1000 / steps, 3)  # noqa: E731
                    is_kernel = lambda e: getattr(e.device_type, "name", "") == "CUDA"  # noqa: E731
                    kernels = sorted([e for e in ka if is_kernel(e)], key=lambda e: -e.self_device_time_total)
                    gpu_busy = per(sum(e.self_device_time_total for e in kernels))
                    ranges = sorted([e for e in ka if e.key.startswith(("mod::", "fn::"))],
                                    key=lambda e: -e.cpu_time_total)
                    host_ops = [e for e in ka if not is_kernel(e) and not e.key.startswith(("mod::", "fn::"))]
                    ops = sorted(host_ops, key=lambda e: -e.device_time_total)[:15]
                    cpu_ops = sorted(host_ops, key=lambda e: -e.self_cpu_time_total)[:12]
                    launches = sum(e.count for e in ka if e.key.startswith(("cudaLaunchKernel", "cuLaunchKernel")))
                    res[f"{L}_{mode}"] = {
                        "wall_ms_per_step_unprofiled": round(wall, 2),
                        "gpu_kernel_ms_per_step": gpu_busy,
                        "kernel_launches_per_step": round(launches / steps),
                        "ranges_ms_per_step(cpu_incl, gpu_incl, calls)": [
                            (e.key, per(e.cpu_time_total), per(e.device_time_total), round(e.count / steps))
                            for e in ranges],
                        "top_ops_by_gpu_time(ms)": [(e.key[:60], per(e.device_time_total), round(e.count / steps))
                                                    for e in ops],
                        "top_kernels(ms)": [(e.key[:70], per(e.self_device_time_total), round(e.count / steps))
                                            for e in kernels[:10]],
                        "top_cpu_ops(self_ms)": [(e.key[:60], per(e.self_cpu_time_total), round(e.count / steps))
                                                 for e in cpu_ops],
                    }
                    log(L, mode, "wall", round(wall, 1), "gpu", gpu_busy, "launches", round(launches / steps))
                    ip.REGISTRY.clear()
                    del cache
                del pre_out
    except Exception:  # noqa: BLE001
        res["error"] = traceback.format_exc()[-4000:]
        print(res["error"], flush=True)
    return res


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, volumes={"/vol": vol}, timeout=35 * 60)
def experts_bench(contexts: str = "32000,128000", steps: int = 20, passkey_only: bool = False) -> dict:
    """Decode ms/step for MoE implementations {grouped_mm (default), eager, batched_mm} x {dense, paged 2%}."""
    import copy
    import json
    import time
    import traceback

    import torch
    from torch.profiler import ProfilerActivity
    from transformers.models.inkling2_loop import modeling_inkling2_loop as M

    t0 = time.time()
    res: dict = {}
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    try:
        from sparse_attention_hub.adapters import Request
        from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
            LocalMaskerConfig,
        )

        import inkling_paged as ip

        adapter = ip.InklingPagedAdapterHF(
            model_name=MODEL_PATH,
            sparse_attention_config=ResearchAttentionConfig(masker_configs=[LocalMaskerConfig(window_size=8)]),
            model_kwargs={"dtype": torch.bfloat16, "device_map": "cuda"}, device="cuda", prefill_chunk_size=256)
        model, tok = adapter.model.eval(), adapter.tokenizer
        experts = [m for m in model.modules() if isinstance(m, M.Inkling2LoopExperts)]
        res["default_experts_implementation"] = experts[0].config._experts_implementation

        def set_impl(name: str) -> None:
            for m in experts:
                m.config._experts_implementation = name

        for L in ([] if passkey_only else [int(x) for x in contexts.split(",")]):
            ids = tok(FILLER * (L // 20 + 10), return_tensors="pt").input_ids[:, :L + 1].cuda()
            adapter.prefill_memo.clear()
            torch.cuda.empty_cache()
            set_impl("grouped_mm")
            with torch.no_grad():
                ts = time.time()
                base = model(ids[:, :-1], past_key_values=None, use_cache=True)
                log(f"prefill {L}: {time.time() - ts:.1f}s")
                # correctness: one decode step per implementation vs grouped_mm
                ref_logits = None
                for impl in ("grouped_mm", "eager", "batched_mm"):
                    set_impl(impl)
                    c = copy.deepcopy(base.past_key_values)
                    lg = model(ids[:, -1:], past_key_values=c, use_cache=True).logits[0, -1].float()
                    if ref_logits is None:
                        ref_logits = lg
                    res[f"{L}_check_{impl}"] = {
                        "cos_vs_grouped_mm": round(float(torch.nn.functional.cosine_similarity(lg, ref_logits, dim=0)), 6),
                        "top1_match": bool(lg.argmax() == ref_logits.argmax())}
                    del c
                for mode in ("dense", "paged2%"):
                    for impl in ("grouped_mm", "eager", "batched_mm"):
                        set_impl(impl)
                        cache = copy.deepcopy(base.past_key_values)
                        if mode != "dense":
                            ip.install_paged(cache, 0.02, L + 4 * steps + 64)
                        nxt = ids[:, -1:]

                        def step():
                            nonlocal nxt
                            with adapter.enable_sparse_mode():
                                out = model(input_ids=nxt, past_key_values=cache, use_cache=True,
                                            logits_to_keep=1, sparse_meta_data={})
                            nxt = out.logits[:, -1:].argmax(-1)

                        for _ in range(3):
                            step()
                        torch.cuda.synchronize()
                        ts = time.perf_counter()
                        for _ in range(steps):
                            step()
                        torch.cuda.synchronize()
                        wall = 1000 * (time.perf_counter() - ts) / steps
                        with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                            for _ in range(2):
                                step()
                            torch.cuda.synchronize()
                        ka = prof.key_averages()
                        launches = sum(e.count for e in ka if e.key in ("cudaLaunchKernel", "cuLaunchKernel")) / 2
                        gpu = sum(e.self_device_time_total for e in ka
                                  if getattr(e.device_type, "name", "") == "CUDA") / 1000 / 2
                        syncs = sum(e.count for e in ka if e.key in ("cudaStreamSynchronize", "cudaDeviceSynchronize",
                                                                      "cudaMemcpyAsync")) / 2
                        res[f"{L}_{mode}_{impl}"] = {"ms_per_step": round(wall, 2), "launches": round(launches),
                                                     "gpu_kernel_ms": round(gpu, 2), "memcpy_or_sync": round(syncs)}
                        log(L, mode, impl, json.dumps(res[f"{L}_{mode}_{impl}"]))
                        ip.REGISTRY.clear()
                        del cache
                        torch.cuda.empty_cache()
            del base

        # end-to-end check at 32k with the fastest paged implementation
        L = int(contexts.split(",")[0])
        best = "batched_mm" if passkey_only else min(("grouped_mm", "eager", "batched_mm"),
                                                     key=lambda i: res[f"{L}_paged2%_{i}"]["ms_per_step"])
        set_impl("grouped_mm")                 # prefill impl; the adapter switches for decode only
        adapter.decode_experts = best
        passkey = "68297"
        n_fill = max(1, L // 22)
        depth = int(n_fill * 0.3)
        context = (FILLER * depth + f" The pass key is {passkey}. Remember it. {passkey} is the pass key. "
                   + FILLER * (n_fill - depth))
        req = Request(context=context, questions="What is the pass key? Reply with only the number.",
                      answer_prefix="")
        adapter.prefill_memo.clear()
        adapter.paged, adapter.top_frac = True, 0.02
        for dec in ("grouped_mm", best):      # same container: default vs decode-only switch
            adapter.decode_experts = dec
            r = adapter.process_request(req, generation_kwargs={"max_new_tokens": 160},
                                        request_kwargs={"max_context_length": 10**9})
            res[f"passkey_{L}_paged2%_decode_{dec}"] = {"correct": passkey in r.responses,
                                                        "answer_tail": r.responses.strip()[-100:],
                                                        **adapter.last_timing}
            log("passkey", dec, json.dumps(res[f"passkey_{L}_paged2%_decode_{dec}"]))
    except Exception:  # noqa: BLE001
        res["error"] = traceback.format_exc()[-4000:]
        print(res["error"], flush=True)
    return res


@app.local_entrypoint()
def experts(contexts: str = "32000,128000", steps: int = 20, passkey_only: bool = False):
    import json

    print("RESULT_JSON " + json.dumps(experts_bench.remote(contexts, steps, passkey_only), default=str))


@app.local_entrypoint()
def prof(contexts: str = "32000,128000", steps: int = 5):
    import json

    print("RESULT_JSON " + json.dumps(profile.remote(contexts, steps), default=str))


@app.local_entrypoint()
def main(context_tokens: int = 32000, prefill_chunk: int = 1024, check: bool = False,
         max_new_tokens: int = 160):
    import json

    print("RESULT_JSON " + json.dumps(run.remote(context_tokens, prefill_chunk, check, max_new_tokens),
                                      default=str))
