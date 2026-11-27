"""CUDA-graph decode experiments for Inkling 2 Mini.

    modal run graph_modal.py::tiny      # small random Inkling on an L4: correctness + sync sites + capture
"""

import modal

from paged_modal import image as paged_image

app = modal.App("inkling-graphs")
vol = modal.Volume.from_name("inkling-workspace")
image = paged_image.add_local_python_source("inkling_static", "paged_modal")
MODEL_PATH = "/vol/models/Inkling-2-Mini"


def _cmp(a, b) -> dict:
    import torch

    a, b = a.float().flatten(), b.float().flatten()
    return {"cos": round(float(torch.nn.functional.cosine_similarity(a, b, dim=0)), 6),
            "max_abs": round(float((a - b).abs().max()), 4), "top1": bool(a.argmax() == b.argmax())}


@app.function(image=image, gpu="L4", cpu=4, memory=16384, volumes={"/vol": vol}, timeout=20 * 60)
def tiny(L0: int = 4096, steps: int = 6) -> dict:
    import copy
    import time
    import traceback

    import torch
    from transformers import DynamicCache, StaticCache
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM

    import inkling_static as ist

    res: dict = {}
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)  # noqa: E731
    try:
        torch.manual_seed(0)
        cfg = Inkling2LoopTextConfig(hidden_size=256, num_attention_heads=24, num_key_value_heads=8, head_dim=128,
                                     swa_num_attention_heads=24, swa_num_key_value_heads=8, swa_head_dim=128,
                                     sliding_window_size=1024, rel_extent=1024, d_rel=16,
                                     moe_intermediate_size=64, n_routed_experts=16, num_experts_per_tok=8,
                                     loop_gate_rank=16)
        cfg._attn_implementation = "eager"
        model = Inkling2LoopForCausalLM._from_config(cfg, dtype=torch.bfloat16).cuda().eval()
        ist.install_attention(model)
        ist.patch_static_sliding_update()
        ist.set_experts(model, "batched_mm")
        ids = torch.randint(0, 200000, (1, L0 + steps + 8), device="cuda")
        max_len = ((L0 + steps + 64 + 15) // 16) * 16 + 1024
        chunk = 512

        def prefill(cache):
            for s in range(0, L0, chunk):
                model(ids[:, s:min(s + chunk, L0)], past_key_values=cache, use_cache=True)

        def decode(cache, n, start=0):
            out = []
            for t in range(start, start + n):
                out.append(model(ids[:, L0 + t:L0 + t + 1], past_key_values=cache, use_cache=True,
                                 logits_to_keep=1, sparse_meta_data={}).logits[0, -1].float())
            return out

        with torch.no_grad():
            dyn = DynamicCache(config=cfg)
            prefill(dyn)
            ref = decode(dyn, steps)
            st = StaticCache(config=cfg, max_cache_len=max_len)
            prefill(st)
            snap = copy.deepcopy(st)
            stat = decode(copy.deepcopy(snap), steps)
            res["A_static_vs_dynamic"] = [_cmp(a, b) for a, b in zip(stat, ref)]
            log("A", res["A_static_vs_dynamic"])

            c = copy.deepcopy(snap)
            sites = ist.find_sync_sites(lambda: decode(c, 1))
            res["B_sync_sites_dense_static"] = sites[:15]
            log("B", len(sites), sites[:6])

            try:
                c = copy.deepcopy(snap)
                g = ist.GraphedDecodeStep(model, c, ids[:, L0:L0 + 2], start_pos=L0)
                rep = [g.step(ids[:, L0 + t]).clone()[0].float() for t in range(2, steps)]
                res["C_graph_dense_vs_eager_static"] = [_cmp(a, b) for a, b in zip(rep, stat[2:])]
            except Exception as exc:  # noqa: BLE001
                res["C_graph_dense_error"] = traceback.format_exc()[-2500:]
            log("C", res.get("C_graph_dense_vs_eager_static", res.get("C_graph_dense_error", "")[-600:]))

            for frac in (1.0, 0.02):
                tag = f"top{frac:g}"
                c = copy.deepcopy(snap)
                ist.install_static_paged(c, frac)
                paged = decode(c, steps)
                res[f"D_{tag}_paged_eager_vs_dense_static"] = [_cmp(a, b) for a, b in zip(paged, stat)]
                c = copy.deepcopy(snap)
                ist.install_static_paged(c, frac)
                res[f"D_{tag}_sync_sites"] = ist.find_sync_sites(lambda: decode(c, 1))[:10]
                try:
                    c = copy.deepcopy(snap)
                    ist.install_static_paged(c, frac)
                    g = ist.GraphedDecodeStep(model, c, ids[:, L0:L0 + 2], start_pos=L0)
                    rep = [g.step(ids[:, L0 + t]).clone()[0].float() for t in range(2, steps)]
                    res[f"D_{tag}_graph_vs_paged_eager"] = [_cmp(a, b) for a, b in zip(rep, paged[2:])]
                    # timing: replay vs eager step
                    torch.cuda.synchronize(); ts = time.perf_counter()
                    for t in range(20):
                        g.step(ids[:, L0 + 2])
                    torch.cuda.synchronize()
                    res[f"D_{tag}_graph_ms"] = round(1000 * (time.perf_counter() - ts) / 20, 3)
                except Exception:  # noqa: BLE001
                    res[f"D_{tag}_graph_error"] = traceback.format_exc()[-2500:]
                ist.STATIC_REGISTRY.clear()
                log("D", tag, {k: v for k, v in res.items() if k.startswith(f"D_{tag}") and "sync" not in k})
            c = copy.deepcopy(snap)
            ist.install_static_paged(c, 0.02)
            decode(c, 2)
            torch.cuda.synchronize(); ts = time.perf_counter()
            for t in range(10):
                model(ids[:, L0 + 2:L0 + 3], past_key_values=c, use_cache=True, logits_to_keep=1, sparse_meta_data={})
            torch.cuda.synchronize()
            res["E_eager_paged_ms"] = round(1000 * (time.perf_counter() - ts) / 10, 3)
            ist.STATIC_REGISTRY.clear()
            log("E", res["E_eager_paged_ms"])
    except Exception:  # noqa: BLE001
        res["error"] = traceback.format_exc()[-3000:]
        print(res["error"], flush=True)
    return res


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, volumes={"/vol": vol}, timeout=40 * 60)
def real(contexts: str = "32000,128000", new_tokens: int = 160, chunk: int = 256) -> dict:
    """Real Inkling 2 Mini: eager vs CUDA-graphed greedy decode on a StaticCache, dense vs paged Quest."""
    import copy
    import time
    import traceback

    import torch
    from transformers import AutoTokenizer, StaticCache

    import inkling_static as ist

    res: dict = {}
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    try:
        from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForConditionalGeneration
        model = Inkling2LoopForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=torch.bfloat16,
                                                                     device_map="cuda").eval()
        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        ist.install_attention(model)
        ist.patch_static_sliding_update()
        res["gpu"] = torch.cuda.get_device_name(0)
        for L in [int(x) for x in contexts.split(",")]:
            passkey = "68297"
            n_fill = max(1, L // 22)
            depth = int(n_fill * 0.3)
            ctx = (filler * depth + f" The pass key is {passkey}. Remember it. {passkey} is the pass key. "
                   + filler * (n_fill - depth))
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": ctx + "\nWhat is the pass key? Reply with only the number."}],
                add_generation_prompt=True, reasoning_effort="none", return_tensors="pt", return_dict=True)
            ids = prompt["input_ids"].cuda()
            n = ids.shape[1]
            max_len = ((n + new_tokens + 64 + 15) // 16) * 16
            ist.set_experts(model, "grouped_mm")
            with torch.no_grad():
                cache = StaticCache(config=model.config, max_cache_len=max_len)
                ts = time.time()
                for s in range(0, n - 1, chunk):
                    model(ids[:, s:min(s + chunk, n - 1)], past_key_values=cache, use_cache=True)
                torch.cuda.synchronize()
                log(f"prefill {n} tokens: {time.time() - ts:.1f}s")
                snap = cache
                ist.set_experts(model, "batched_mm")
                first, start = ids[:, n - 1:n], n - 1
                row = {"prompt_tokens": n}
                # eager static decode (dense) for reference timing
                c = copy.deepcopy(snap)
                nxt = first
                for i in range(3):
                    nxt = model(nxt, past_key_values=c, use_cache=True, logits_to_keep=1,
                                sparse_meta_data={}).logits[:, -1].argmax(-1, keepdim=True)
                torch.cuda.synchronize(); ts = time.perf_counter()
                for i in range(10):
                    nxt = model(nxt, past_key_values=c, use_cache=True, logits_to_keep=1,
                                sparse_meta_data={}).logits[:, -1].argmax(-1, keepdim=True)
                torch.cuda.synchronize()
                row["eager_dense_ms"] = round(1000 * (time.perf_counter() - ts) / 10, 2)
                del c
                for mode, frac in (("graph_dense", None), ("graph_paged2%", 0.02), ("graph_paged1%", 0.01)):
                    try:
                        c = copy.deepcopy(snap)
                        if frac:
                            ist.install_static_paged(c, frac)
                        g = ist.GraphedGreedyDecode(model, c, first, start, new_tokens + 4)
                        torch.cuda.synchronize(); ts = time.perf_counter()
                        g.run(new_tokens - g.n_done)
                        torch.cuda.synchronize()
                        ms = 1000 * (time.perf_counter() - ts) / (new_tokens - 2)
                        text = tok.decode(g.tokens[:new_tokens].tolist(), skip_special_tokens=True)
                        row[mode] = {"ms_per_step": round(ms, 2), "correct": passkey in text,
                                     "tail": text.strip()[-90:]}
                        del g
                    except Exception:  # noqa: BLE001
                        row[mode] = {"error": traceback.format_exc()[-1500:]}
                    ist.STATIC_REGISTRY.clear()
                    del c
                    torch.cuda.empty_cache()
                    log(L, mode, row[mode] if "error" not in row[mode] else row[mode]["error"][-500:])
                res[str(L)] = row
                del snap, cache
                torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        res["error"] = traceback.format_exc()[-3000:]
        print(res["error"], flush=True)
    res["total_s"] = round(time.time() - t0, 1)
    return res


RULER_TASKS = ["cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multiquery",
               "niah_multivalue", "niah_single_1", "niah_single_2", "niah_single_3", "qa_1", "qa_2", "vt"]
RULER_CONFIGS = (("dense", None), ("quest5%", 0.05), ("quest2%", 0.02), ("quest1%", 0.01))


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, volumes={"/vol": vol}, timeout=30 * 60,
              secrets=[modal.Secret.from_name("hf-token")])
def ruler_shard(items: list, chunk: int = 1024) -> list:
    """Run RULER-32k samples [(task, row_idx), ...]: one prefill each, then graphed greedy decode per config."""
    import copy
    import json
    import os
    import time
    import traceback

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, StaticCache
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForConditionalGeneration

    import inkling_static as ist

    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    model = Inkling2LoopForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=torch.bfloat16,
                                                                 device_map="cuda").eval()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    ist.install_attention(model)
    ist.patch_static_sliding_update()
    stop_ids = {i for i in [tok.convert_tokens_to_ids("<|end_message|>"), tok.eos_token_id,
                            model.generation_config.eos_token_id] if isinstance(i, int)}
    data = {}
    out = []
    for task, idx in items:
        rec = {"task": task, "idx": idx}
        try:
            if task not in data:
                data[task] = load_dataset("xAlg-AI/att-hub-ruler-32k", task, split=task)
            row = data[task][idx]
            rec["answer"] = list(row["answer"])
            msgs = [{"role": "user", "content": row["context"] + row["question"]},
                    {"role": "assistant", "content": row["answer_prefix"]}]
            ids = tok.apply_chat_template(msgs, continue_final_message=True, reasoning_effort="none",
                                          return_tensors="pt", return_dict=True)["input_ids"].cuda()
            n = ids.shape[1]
            new = int(min(row["max_new_tokens"], 128))
            rec["prompt_tokens"], rec["new_tokens"] = n, new
            max_len = ((n + new + 64 + 15) // 16) * 16
            with torch.no_grad():
                ist.set_experts(model, "grouped_mm")
                cache = StaticCache(config=model.config, max_cache_len=max_len)
                ts = time.time()
                for s in range(0, n - 1, chunk):
                    model(ids[:, s:min(s + chunk, n - 1)], past_key_values=cache, use_cache=True)
                torch.cuda.synchronize()
                rec["prefill_s"] = round(time.time() - ts, 1)
                ist.set_experts(model, "batched_mm")
                rec["pred"], rec["ms_per_token"] = {}, {}
                for name, frac in RULER_CONFIGS:
                    c = copy.deepcopy(cache)
                    if frac:
                        ist.install_static_paged(c, frac)
                    g = ist.GraphedGreedyDecode(model, c, ids[:, n - 1:n], n - 1, new + 4)
                    torch.cuda.synchronize()
                    ts = time.perf_counter()
                    g.run(new - g.n_done)
                    torch.cuda.synchronize()
                    rec["ms_per_token"][name] = round(1000 * (time.perf_counter() - ts) / max(new - 2, 1), 2)
                    toks = g.tokens[:new].tolist()
                    cut = next((j for j, t in enumerate(toks) if t in stop_ids), len(toks))
                    rec["pred"][name] = tok.decode(toks[:cut], skip_special_tokens=True).strip()
                    del g, c
                    ist.STATIC_REGISTRY.clear()
                    torch.cuda.empty_cache()
                del cache
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            rec["error"] = traceback.format_exc()[-1500:]
        out.append(rec)
        log(task, idx, json.dumps({k: rec.get(k) for k in ("prompt_tokens", "prefill_s", "ms_per_token")}),
            json.dumps(rec.get("pred", {}))[:300], rec.get("answer"), (rec.get("error") or "")[-300:])
    os.makedirs("/vol/results/ruler32k", exist_ok=True)
    with open(f"/vol/results/ruler32k/shard_{int(t0)}_{items[0][0]}_{items[0][1]}.json", "w") as f:
        json.dump(out, f, indent=1)
    vol.commit()
    return out


def _score(records: list) -> dict:
    """Repo's RULER metric: string_match_part for qa_*, string_match_all otherwise (x100, per task)."""
    import re

    clean = re.compile(r"[\x00-\x1f]")
    by: dict = {}
    for r in records:
        if "pred" not in r:
            continue
        for cfg, pred in r["pred"].items():
            p = clean.sub("", pred).strip().lower()
            refs = [a.lower() for a in r["answer"]]
            if r["task"].startswith("qa"):
                s = max(1.0 if a in p else 0.0 for a in refs)
            else:
                s = sum(1.0 if a in p else 0.0 for a in refs) / len(refs)
            by.setdefault(cfg, {}).setdefault(r["task"], []).append(s)
    table = {cfg: {t: round(100 * sum(v) / len(v), 1) for t, v in sorted(tasks.items())} for cfg, tasks in by.items()}
    for cfg, tasks in table.items():
        tasks["_overall(mean of tasks)"] = round(sum(tasks.values()) / len(tasks), 1)
    return table


@app.local_entrypoint()
def ruler_main(per_task: int = 6, shards: int = 6, tasks: str = ",".join(RULER_TASKS)):
    import json

    items = [(t, i) for i in range(per_task) for t in tasks.split(",")]
    groups = [items[k::shards] for k in range(shards) if items[k::shards]]
    records = [r for shard in ruler_shard.map(groups) for r in shard]
    scores = _score(records)
    errors = [(r["task"], r["idx"], r["error"][-300:]) for r in records if "error" in r]
    timing = {}
    for r in records:
        for cfg, ms in r.get("ms_per_token", {}).items():
            timing.setdefault(cfg, []).append(ms)
    summary = {"n_samples": len(records), "errors": errors, "scores": scores,
               "mean_ms_per_token": {c: round(sum(v) / len(v), 2) for c, v in timing.items()},
               "mean_prefill_s": round(sum(r.get("prefill_s", 0) for r in records) / max(len(records), 1), 1),
               "mean_prompt_tokens": round(sum(r.get("prompt_tokens", 0) for r in records) / max(len(records), 1))}
    print("RESULT_JSON " + json.dumps(summary, default=str))
    print("RECORDS_JSON " + json.dumps(records, default=str))


@app.local_entrypoint()
def real_main(contexts: str = "32000,128000"):
    import json

    print("RESULT_JSON " + json.dumps(real.remote(contexts), default=str))


@app.local_entrypoint()
def tiny_main(prefill_len: int = 4096):
    import json

    print("RESULT_JSON " + json.dumps(tiny.remote(prefill_len), default=str))
