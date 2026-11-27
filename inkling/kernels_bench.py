"""Benchmark skylight_kernels at Inkling 2 Mini's full-attention shapes on an A100-80GB.

Inkling full-attention layers: H_q=24, H_kv=8 (GQA group 3), head_dim=128, bf16.
  1. correctness: sink+local+top-k kernel vs dense kernel (topk=1.0) and vs the PyTorch reference
  2. Inkling extras without kernel changes: relative-position bias via `sparse_weights`
     (weighted = qk + log w) and per-token sink logits via the returned LSE
  3. latency sweep vs dense paged decode, 32K..1M context, batch 1..64

    modal run kernels_bench.py
"""

import modal

app = modal.App("inkling-kernels-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "g++")
    .pip_install("torch==2.9.1", "ninja", "numpy", "pytest", "packaging", "setuptools", "wheel")
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
        "ls /opt/sk/*/_cuda_ops*.so",
    )
    .env({"PYTHONPATH": "/opt/sk"})
    .add_local_python_source("inkling_sparse_op", "inkling_quest_triton")
)

HQ, HKV, D, PAGE = 24, 8, 128, 16


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=32768, timeout=25 * 60)
def bench() -> dict:
    import math
    import subprocess
    import time

    import torch
    from original_optimized import BatchDecodeWithPagedKVCacheWrapper as Dense
    from sparse_optimized import BatchDecodeWithPagedKVCacheWrapper as Sparse
    from sparse_oracle_topk_sink_local import BatchDecodeWithPagedKVCacheWrapper as SinkLocalRef
    from sparse_oracle_topk_sink_local_optimized import (
        BatchDecodeWithPagedKVCacheWrapper as SinkLocalOpt,
    )

    t0 = time.time()
    res: dict = {"gpu": torch.cuda.get_device_name(0)}
    dev, dt = "cuda", torch.bfloat16
    log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)  # noqa: E731

    def make(B: int, ctx: int, seed: int = 0):
        g = torch.Generator(device=dev).manual_seed(seed)
        ppr = math.ceil(ctx / PAGE)
        P = B * ppr
        indptr = torch.arange(0, P + 1, ppr, dtype=torch.int32, device=dev)
        indices = torch.arange(P, dtype=torch.int32, device=dev)
        last = torch.full((B,), ctx % PAGE or PAGE, dtype=torch.int32, device=dev)
        kv = torch.randn(P, 2, PAGE, HKV, D, dtype=dt, device=dev, generator=g)
        q = torch.randn(B, HQ, D, dtype=dt, device=dev, generator=g)
        return indptr, indices, last, kv, q

    def planned(cls, indptr, indices, last, **ctor):
        w = cls(torch.zeros(256 << 20, dtype=torch.uint8, device=dev), "NHD", **ctor)
        w.plan(indptr, indices, last, HQ, HKV, D, PAGE, q_data_type=dt, kv_data_type=dt)
        return w

    def timeit(fn, warmup: int = 5, iters: int = 20) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return round(s.elapsed_time(e) / iters, 4)

    # ---------- 1. correctness at Inkling shapes ----------
    try:
        indptr, indices, last, kv, q = make(2, 8192)
        dense = planned(Dense, indptr, indices, last)
        opt = planned(SinkLocalOpt, indptr, indices, last)
        ref = planned(SinkLocalRef, indptr, indices, last)
        d_out = dense.run(q, kv).float()
        full_out = opt.run(q, kv, 1.0, sink_size=0, local_size=0).float()
        o = opt.run(q, kv, 0.05, sink_size=64, local_size=1024).float()
        r = ref.run(q, kv, 0.05, sink_size=64, local_size=1024).float()
        res["correctness"] = {
            "gs3_full_vs_dense_max_abs": float((full_out - d_out).abs().max()),
            "gs3_sparse5%_vs_reference_max_abs": float((o - r).abs().max()),
            "output_abs_scale": float(d_out.abs().max()),
        }
        log("correctness", res["correctness"])

    except Exception as exc:  # noqa: BLE001 - keep going, report the stage error
        res['correctness_error'] = f'{type(exc).__name__}: {exc}'[:600]
        log('correctness failed', res['correctness_error'])
    # ---------- 2. Inkling relative-position bias + per-token sinks, no kernel change ----------
    try:
        ctx = 8192
        indptr, indices, last, kv, q = make(1, ctx, seed=1)
        K = kv[:, 0].reshape(-1, HKV, D)[:ctx].float()
        V = kv[:, 1].reshape(-1, HKV, D)[:ctx].float()
        g = torch.Generator(device=dev).manual_seed(2)
        mid = torch.arange(64, ctx - 1024, device=dev)
        n_mid = int(0.05 * mid.numel())
        sel = torch.stack([torch.cat([torch.arange(64, device=dev),
                                      mid[torch.randperm(mid.numel(), generator=g, device=dev)[:n_mid]],
                                      torch.arange(ctx - 1024, ctx, device=dev)]) for _ in range(HQ)])
        S = sel.shape[-1]
        dist = (ctx - 1) - sel                                         # query sits at position ctx-1
        bias_tab = torch.randn(HQ, 1024, device=dev, generator=g) * 2  # stands in for rel_logits
        bias = torch.where(dist < 1024, bias_tab.gather(1, dist.clamp(max=1023)), torch.zeros_like(dist, dtype=torch.float))
        sinks = torch.randn(HQ, device=dev, generator=g) * 2           # per-token sink logit (b=1, q=1)
        sp = planned(Sparse, indptr, indices, last)
        out, lse = sp.run(q, kv, sparse_len=torch.full((1, HQ, 1), S, dtype=torch.int32, device=dev),
                          sparse_idx=sel[None].contiguous(), sparse_weights=bias.exp()[None].float().contiguous(),
                          return_lse=True)
        lse_nat = lse.float() * math.log(2.0)                          # kernel LSE is log2
        out_ink = out.float() * torch.sigmoid(lse_nat - sinks[None])[..., None]
        ref_rows = []
        for h in range(HQ):
            kvh = h // (HQ // HKV)
            logits = (q[0, h].float() @ K[sel[h], kvh].T) * D ** -0.5 + bias[h]
            p = torch.softmax(torch.cat([logits, sinks[h:h + 1]]), dim=-1)[:-1]
            ref_rows.append(p @ V[sel[h], kvh])
        ref_ink = torch.stack(ref_rows)[None]
        res["inkling_bias_and_sinks"] = {
            "selected_per_head": S,
            "max_abs_diff": float((out_ink - ref_ink).abs().max()),
            "rel_err": float((out_ink - ref_ink).norm() / ref_ink.norm()),
        }
        log("inkling bias+sinks", res["inkling_bias_and_sinks"])
        del kv, K, V
        torch.cuda.empty_cache()

    except Exception as exc:  # noqa: BLE001 - keep going, report the stage error
        res['inkling_bias_and_sinks_error'] = f'{type(exc).__name__}: {exc}'[:600]
        log('inkling_bias_and_sinks failed', res['inkling_bias_and_sinks_error'])
    # ---------- repo's own test-suite on A100 (sanity of the sm80 build) ----------
    try:
        t = subprocess.run(["python", "-m", "pytest", "-q", "-x", "test_sparse_oracle_topk_sink_local_optimized.py"],
                           cwd="/opt/sk", capture_output=True, text=True, timeout=420)
        res["repo_pytest_sink_local_opt"] = (t.stdout.strip().splitlines() or ["?"])[-1]
    except subprocess.TimeoutExpired:
        res["repo_pytest_sink_local_opt"] = "timeout"
    log("pytest", res["repo_pytest_sink_local_opt"])

    # ---------- 3. latency sweep ----------
    combos = [(1, 32768), (1, 131072), (1, 524288), (1, 1048576),
              (8, 32768), (8, 131072), (8, 524288), (32, 32768), (32, 131072), (64, 32768)]
    res["latency_ms"] = []
    for B, ctx in combos:
        indptr, indices, last, kv, q = make(B, ctx)
        dense = planned(Dense, indptr, indices, last)
        opt = planned(SinkLocalOpt, indptr, indices, last)
        row = {"B": B, "ctx": ctx, "kv_gib": round(kv.numel() * 2 / 2**30, 2),
               "dense": timeit(lambda: dense.run(q, kv))}
        for ch in (8, 128):
            for frac in (0.05, 0.02, 0.01):
                k = round(frac * (ctx - 1088))
                row[f"ch{ch}_top{int(frac * 100)}%"] = timeit(
                    lambda: opt.run(q, kv, frac, sink_size=64, local_size=1024, channel_num=ch))
                row[f"density_top{int(frac * 100)}%"] = round((1088 + k) / ctx, 4)
        res["latency_ms"].append(row)
        log(row)
        del kv, dense, opt
        torch.cuda.empty_cache()
    res["total_s"] = round(time.time() - t0, 1)
    return res


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=32768, timeout=25 * 60)
def bench_quest() -> dict:
    """Page-level (Quest) Inkling operator vs dense paged decode and the token-level oracle pipeline."""
    import math
    import time

    import torch
    from original_optimized import BatchDecodeWithPagedKVCacheWrapper as Dense
    from sparse_oracle_topk_sink_local_optimized import (
        BatchDecodeWithPagedKVCacheWrapper as SinkLocalOpt,
    )

    from inkling_sparse_op import InklingQuestDecode, page_summaries

    t0 = time.time()
    dev, dt = "cuda", torch.bfloat16
    res: dict = {"gpu": torch.cuda.get_device_name(0)}
    log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)  # noqa: E731
    i32 = dict(dtype=torch.int32, device=dev)

    def make(B: int, L: int, seed: int = 0):
        g = torch.Generator(device=dev).manual_seed(seed)
        P = B * (L // PAGE)
        k = torch.randn(P, HKV, PAGE, D, dtype=dt, device=dev, generator=g)
        v = torch.randn(P, HKV, PAGE, D, dtype=dt, device=dev, generator=g)
        q = torch.randn(B, HQ, D, dtype=dt, device=dev, generator=g)
        bias = torch.randn(B, HQ, 1024, device=dev, generator=g) * 2
        sinks = torch.randn(B, HQ, device=dev, generator=g) * 2
        return k, v, q, bias, sinks

    def timeit(fn, warmup: int = 5, iters: int = 20) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return round(s.elapsed_time(e) / iters, 4)

    # ---- correctness: operator vs fp32 PyTorch attention over the same token set ----
    try:
        B, L = 2, 16384
        k, v, q, bias, sinks = make(B, L, seed=3)
        kmin, kmax = page_summaries(k, B, L // PAGE)
        op = InklingQuestDecode(B, L, top_pages=48)
        out = op(q, k, v, kmin, kmax, bias, sinks).float()
        ref = op.reference(q, k, v, bias, sinks)
        out2 = op(q * -1, k, v, kmin, kmax, bias, sinks).float()   # new selection, same plan
        ref2 = op.reference(q * -1, k, v, bias, sinks)
        res["correctness"] = {
            "rel_err": float((out - ref).norm() / ref.norm()),
            "max_abs": float((out - ref).abs().max()),
            "rel_err_after_reselect": float((out2 - ref2).norm() / ref2.norm()),
            "tokens_per_head": int(op.idx_a.shape[-1] + op.k * PAGE),
        }
    except Exception as exc:  # noqa: BLE001
        res["correctness_error"] = f"{type(exc).__name__}: {exc}"[:800]
    log("correctness", res.get("correctness", res.get("correctness_error")))

    combos = [(1, 32768), (1, 131072), (1, 524288), (1, 1048576),
              (8, 32768), (8, 131072), (8, 524288), (32, 32768), (32, 131072), (64, 32768)]
    res["latency_ms"] = []
    for B, L in combos:
        npr = L // PAGE
        k, v, q, bias, sinks = make(B, L)
        dense = Dense(torch.zeros(256 << 20, dtype=torch.uint8, device=dev), "HND")
        dense.plan(torch.arange(0, B * npr + 1, npr, **i32), torch.arange(B * npr, **i32),
                   torch.full((B,), PAGE, **i32), HQ, HKV, D, PAGE, q_data_type=dt, kv_data_type=dt)
        row = {"B": B, "ctx": L, "dense": timeit(lambda: dense.run(q, (k, v)))}
        kmin, kmax = page_summaries(k, B, npr)
        mid = npr - 4 - 64
        for frac in (0.05, 0.02, 0.01):
            op = InklingQuestDecode(B, L, top_pages=max(1, round(frac * mid)))
            tag = f"quest{int(frac * 100)}%"
            row[tag] = timeit(lambda: op(q, k, v, kmin, kmax, bias, sinks))
            row[f"density_{tag}"] = round((1088 + op.k * PAGE) / L, 4)
            tm: dict = {}
            for _ in range(10):
                op(q, k, v, kmin, kmax, bias, sinks, timings=tm)
            row[f"{tag}_breakdown"] = {n: round(sum(x) / len(x), 3) for n, x in tm.items()}
            del op
        # token-level oracle pipeline (NHD copy) at 2%, channel_num=8, for comparison
        kv_nhd = torch.stack([k.transpose(1, 2), v.transpose(1, 2)], dim=1).contiguous()
        del dense
        torch.cuda.empty_cache()
        tok = SinkLocalOpt(torch.zeros(256 << 20, dtype=torch.uint8, device=dev), "NHD")
        tok.plan(torch.arange(0, B * npr + 1, npr, **i32), torch.arange(B * npr, **i32),
                 torch.full((B,), PAGE, **i32), HQ, HKV, D, PAGE, q_data_type=dt, kv_data_type=dt)
        row["token_ch8_2%"] = timeit(lambda: tok.run(q, kv_nhd, 0.02, sink_size=64, local_size=1024, channel_num=8))
        res["latency_ms"].append(row)
        log({k_: v_ for k_, v_ in row.items() if "breakdown" not in k_})
        del k, v, kv_nhd, kmin, kmax, tok
        torch.cuda.empty_cache()
    res["total_s"] = round(time.time() - t0, 1)
    return res


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=32768, timeout=25 * 60)
def bench_fused() -> dict:
    """Fused-Triton Quest operator (eager and CUDA-graphed) vs PyTorch-glue Quest vs dense."""
    import time

    import torch
    from original_optimized import BatchDecodeWithPagedKVCacheWrapper as Dense

    from inkling_quest_triton import GraphedDecode, InklingQuestDecodeFused
    from inkling_sparse_op import InklingQuestDecode, page_summaries

    t0 = time.time()
    dev, dt = "cuda", torch.bfloat16
    res: dict = {"gpu": torch.cuda.get_device_name(0), "triton": __import__("triton").__version__}
    log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)  # noqa: E731
    i32 = dict(dtype=torch.int32, device=dev)

    def make(B: int, L: int, seed: int = 0):
        g = torch.Generator(device=dev).manual_seed(seed)
        P = B * (L // PAGE)
        k = torch.randn(P, HKV, PAGE, D, dtype=dt, device=dev, generator=g)
        v = torch.randn(P, HKV, PAGE, D, dtype=dt, device=dev, generator=g)
        q = torch.randn(B, HQ, D, dtype=dt, device=dev, generator=g)
        bias = torch.randn(B, HQ, 1024, device=dev, generator=g) * 2
        sinks = torch.randn(B, HQ, device=dev, generator=g) * 2
        return k, v, q, bias, sinks

    def timeit(fn, warmup: int = 5, iters: int = 30) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return round(s.elapsed_time(e) / iters, 4)

    # ---- correctness ----
    try:
        B, L = 2, 16384
        k, v, q, bias, sinks = make(B, L, seed=3)
        kmin, kmax = page_summaries(k, B, L // PAGE)
        fused = InklingQuestDecodeFused(B, L, top_pages=48)
        base = InklingQuestDecode(B, L, top_pages=48)
        out_f = fused(q, k, v, kmin, kmax, bias, sinks).float()
        ref_f = fused.reference(q, k, v, bias, sinks)
        out_t = base(q, k, v, kmin, kmax, bias, sinks).float()
        pf, pt = fused.last_pages, base.last_pages
        overlap = sum(len(set(pf[b, h].tolist()) & set(pt[b, h].tolist()))
                      for b in range(B) for h in range(HKV)) / (B * HKV * fused.k)
        corr = {"fused_vs_fp32_ref_rel_err": float((out_f - ref_f).norm() / ref_f.norm()),
                "page_selection_overlap_with_torch_version": round(overlap, 4),
                "fused_vs_torch_version_rel_err": float((out_f - out_t).norm() / out_t.norm())}
        try:
            gd = GraphedDecode(fused, q, k, v, kmin, kmax, bias, sinks)
            out_g = gd(q * -1, bias, sinks).float()
            ref_g = fused.reference(q * -1, k, v, bias, sinks)
            corr["graphed_new_query_vs_fp32_ref_rel_err"] = float((out_g - ref_g).norm() / ref_g.norm())
        except Exception as exc:  # noqa: BLE001
            corr["graph_error"] = f"{type(exc).__name__}: {exc}"[:500]
        res["correctness"] = corr
    except Exception as exc:  # noqa: BLE001
        res["correctness_error"] = f"{type(exc).__name__}: {exc}"[:800]
    log("correctness", res.get("correctness", res.get("correctness_error")))

    combos = [(1, 32768), (1, 131072), (1, 524288), (1, 1048576),
              (8, 32768), (8, 131072), (8, 524288), (32, 32768), (32, 131072), (64, 32768)]
    res["latency_ms"] = []
    for B, L in combos:
        npr = L // PAGE
        k, v, q, bias, sinks = make(B, L)
        dense = Dense(torch.zeros(256 << 20, dtype=torch.uint8, device=dev), "HND")
        dense.plan(torch.arange(0, B * npr + 1, npr, **i32), torch.arange(B * npr, **i32),
                   torch.full((B,), PAGE, **i32), HQ, HKV, D, PAGE, q_data_type=dt, kv_data_type=dt)
        row = {"B": B, "ctx": L, "dense": timeit(lambda: dense.run(q, (k, v)))}
        kmin, kmax = page_summaries(k, B, npr)
        mid = npr - 4 - 64
        for frac in (0.05, 0.02, 0.01):
            kp = max(1, round(frac * mid))
            tag = f"{int(frac * 100)}%"
            fused = InklingQuestDecodeFused(B, L, top_pages=kp)
            row[f"fused{tag}"] = timeit(lambda: fused(q, k, v, kmin, kmax, bias, sinks))
            row[f"density{tag}"] = round((1088 + fused.k * PAGE) / L, 4)
            try:
                gd = GraphedDecode(fused, q, k, v, kmin, kmax, bias, sinks)
                row[f"graphed{tag}"] = timeit(lambda: gd(q, bias, sinks))
            except Exception as exc:  # noqa: BLE001
                row[f"graphed{tag}"] = None
                row["graph_error"] = f"{type(exc).__name__}: {exc}"[:300]
            if frac == 0.02:
                tm: dict = {}
                for _ in range(10):
                    fused(q, k, v, kmin, kmax, bias, sinks, timings=tm)
                row["fused2%_breakdown"] = {n: round(sum(x) / len(x), 3) for n, x in tm.items()}
                base = InklingQuestDecode(B, L, top_pages=kp)
                row["torchglue2%"] = timeit(lambda: base(q, k, v, kmin, kmax, bias, sinks))
                del base
            del fused
            torch.cuda.empty_cache()
        res["latency_ms"].append(row)
        log({k_: v_ for k_, v_ in row.items() if "breakdown" not in k_})
        del k, v, kmin, kmax, dense
        torch.cuda.empty_cache()
    res["total_s"] = round(time.time() - t0, 1)
    return res


@app.local_entrypoint()
def main(which: str = "kernels"):
    import json

    fn = {"kernels": bench, "quest": bench_quest, "fused": bench_fused}[which]
    print("RESULT_JSON " + json.dumps(fn.remote()))
