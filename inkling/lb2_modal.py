"""LongBench v2 on Inkling 2 Mini (A100-80GB): dense vs stratified sparse attention (SSA) vs PQCache.

Per question: dense fused-kernel prefill of the context (once), then R rows (e.g. dense / ssa / pqc) that
share it, process the question and generate side by side (one CUDA graph).  Prompts come from
lb2_prep.py (official 0-shot template, middle truncation to 120k tokens, official answer regex).

    modal run lb2_modal.py::checks_main                  # real-model equivalence, kernel timings, SSA vs reference
    modal run lb2_modal.py::pilot_main --efforts high,low,none --n 8
    modal run lb2_modal.py::run_main --effort low --shards 4 ...
"""

import modal

from paged_modal import image as paged_image

app = modal.App("inkling-lb2")
vol = modal.Volume.from_name("inkling-workspace")
image = paged_image.add_local_python_source("inkling_static", "paged_modal", "inkling_fa", "inkling_ssa",
                                            "inkling_rows", "ssa_modal")
MODEL_PATH = "/vol/models/Inkling-2-Mini"
DATA = "/vol/data/lb2/ml120000_high"
EOS = 200006


# ------------------------------------------------------------------------------------------------------------
def load():
    import time

    import torch
    from transformers import AutoTokenizer
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForConditionalGeneration

    import inkling_fa as FA
    import inkling_static as ist

    t = time.time()
    model = Inkling2LoopForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=torch.bfloat16,
                                                                 device_map="cuda").eval()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    FA.install(model)
    ist.patch_static_sliding_update()
    print(f"model loaded in {time.time() - t:.0f}s", flush=True)
    return model, tok


def effort_ids(tok, effort: str):
    s = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, reasoning_effort=effort)
    return tok.encode(s[:s.index("<|message_user|>")], add_special_tokens=False)


def prompt_ids(tok, meta: dict, effort: str):
    import numpy as np

    ids = np.load(f"{DATA}/{meta['_id']}.npy").tolist()
    hi, new = effort_ids(tok, "high"), effort_ids(tok, effort)
    assert ids[:len(hi)] == hi, "unexpected prompt prefix"
    ids = new + ids[len(hi):]
    return ids, meta["n_ctx"] + len(new) - len(hi)


def extract_answer(response: str):
    import re

    response = response.replace("*", "")
    m = re.search(r"The correct answer is \(([A-D])\)", response) or re.search(r"The correct answer is ([A-D])", response)
    return m.group(1) if m else None


def parse(tok, toks):
    cut = toks.index(EOS) if EOS in toks else len(toks)
    text = tok.decode(toks[:cut], skip_special_tokens=False)
    think = ""
    if "<|content_thinking|>" in text:
        think = text.split("<|content_thinking|>", 1)[1].split("<|end_message|>", 1)[0]
    answer = text.rsplit("<|content_text|>", 1)[1].split("<|end_message|>", 1)[0] if "<|content_text|>" in text else ""
    n_think = len(tok.encode(think, add_special_tokens=False)) if think else 0
    return {"pred": extract_answer(answer), "finished": EOS in toks, "n_gen": cut, "n_think": n_think,
            "answer_text": answer.strip()[-300:]}


def run_sample(model, tok, meta, modes, effort, max_new, temperature, chunk=8192, seed=0, budget=0.02,
               log=print):
    import time

    import torch
    from transformers import StaticCache

    import inkling_fa as FA
    import inkling_rows as RW
    import inkling_static as ist

    ids, n_ctx = prompt_ids(tok, meta, effort)
    n = len(ids)
    rec = {"_id": meta["_id"], "domain": meta["domain"], "difficulty": meta["difficulty"], "length": meta["length"],
           "answer": meta["answer"], "n_prompt": n, "n_ctx": n_ctx, "effort": effort, "modes": list(modes)}
    max_len = ((n + max_new + 64 + 15) // 16) * 16
    dev = "cuda"
    x = torch.tensor(ids, device=dev)[None]
    torch.manual_seed(seed)
    RW.Rows.uninstall()
    with torch.no_grad():
        FA.set_mode(model, "prefill")
        ist.set_experts(model, "grouped_mm")
        cache = StaticCache(config=model.config, max_cache_len=max_len)
        torch.cuda.synchronize()
        t = time.time()
        for s in range(0, n_ctx, chunk):
            model(x[:, s:min(s + chunk, n_ctx)], past_key_values=cache, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        rec["prefill_s"] = round(time.time() - t, 2)
        R = len(modes)
        cR = RW.repeat_cache(cache, R)
        del cache
        torch.cuda.empty_cache()
        t = time.time()
        rows = RW.Rows(cR, modes, L0=n_ctx, n_prompt=n, max_new=max_new, scaling=128 ** -0.5, budget=budget,
                       seed=seed)
        rows.install()
        rec["ssa_build_s"] = round(time.time() - t, 2)
        rec["budget_tokens"] = rows.budget_tokens
        t = time.time()
        q = x[:, n_ctx:].expand(R, -1)
        for a in range(0, q.shape[1], 1024):
            lg = model(q[:, a:a + 1024], past_key_values=cR, use_cache=True, logits_to_keep=1).logits[:, -1]
        first = RW.sample_first(lg, temperature)
        torch.cuda.synchronize()
        rec["question_s"] = round(time.time() - t, 2)
        FA.set_mode(model, "decode")
        ist.set_experts(model, "batched_mm")
        t = time.time()
        g = RW.GraphedRowsDecode(model, cR, first, n, max_new, temperature)
        torch.cuda.synchronize()
        t1 = time.time()
        steps = g.run_until([EOS], burst=32)
        torch.cuda.synchronize()
        rec["graph_capture_s"] = round(t1 - t, 2)
        rec["decode_steps"] = steps
        rec["ms_per_step"] = round(1000 * (time.time() - t1) / max(steps - 3, 1), 2)
        toks = g.tokens[:, :steps].tolist()
        rec["rows"] = {}
        for b, m in enumerate(modes):
            r = parse(tok, toks[b])
            r["judge"] = r["pred"] == meta["answer"]
            rec["rows"][m] = r
        del g, rows, cR
    RW.Rows.uninstall()
    torch.cuda.empty_cache()
    log(meta["_id"], meta["length"], n, effort, {k: rec[k] for k in ("prefill_s", "question_s", "decode_steps",
                                                                    "ms_per_step")},
        {m: (r["pred"], r["judge"], r["n_gen"], r["finished"]) for m, r in rec["rows"].items()}, "ans", meta["answer"])
    return rec


def _pick(n: int, seed: int = 0, exclude=()):
    """Stratified pick over (length) with a fixed seed."""
    import json
    import random

    meta = json.load(open(f"{DATA}/meta.json"))
    rng = random.Random(seed)
    out = []
    for L in ("short", "medium", "long"):
        pool = [m for m in meta if m["length"] == L and m["_id"] not in exclude]
        rng.shuffle(pool)
        out += pool[:(n + 2) // 3]
    return out[:n]


# ------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, volumes={"/vol": vol}, timeout=45 * 60)
def checks() -> dict:
    import copy
    import json
    import time
    import traceback

    import torch
    from transformers import StaticCache

    import inkling_fa as FA
    import inkling_rows as RW
    import inkling_ssa as SS
    import inkling_static as ist
    from ssa_modal import _attn_ref, _cmp

    res: dict = {}
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    model, tok = load()
    res["load_s"] = round(time.time() - t0, 1)
    meta_all = json.load(open(f"{DATA}/meta.json"))
    dev = "cuda"

    def use_eager():
        ist.install_attention(model)
        FA.STATE["lazy"] = False

    # ---- C1: real-model prefill FA vs eager on a 12k prefix, then 16 greedy tokens each --------------------
    try:
        m = next(x for x in meta_all if 14000 < x["n_ctx"] < 30000)
        ids, n_ctx = prompt_ids(tok, m, "none")
        Lp = 12000
        x = torch.tensor(ids[:Lp], device=dev)[None]
        follow = torch.tensor(ids[Lp:Lp + 1], device=dev)[None]
        out = {}
        with torch.no_grad():
            ist.set_experts(model, "grouped_mm")
            caches = {}
            for name, chunk in (("eager1024", 1024), ("fa4096", 4096)):
                c = StaticCache(config=model.config, max_cache_len=Lp + 64)
                if name.startswith("eager"):
                    use_eager()
                else:
                    FA.set_mode(model, "prefill")
                torch.cuda.synchronize()
                t = time.time()
                for s in range(0, Lp, chunk):
                    lg = model(x[:, s:s + chunk], past_key_values=c, use_cache=True, logits_to_keep=1).logits[0, -1]
                torch.cuda.synchronize()
                out[f"{name}_s"] = round(time.time() - t, 2)
                caches[name] = (c, lg.float())
            out["last_logits"] = _cmp(caches["fa4096"][1], caches["eager1024"][1])
            use_eager()
            ist.set_experts(model, "batched_mm")
            toks = {}
            for name, (c, _) in caches.items():
                nxt, tt = follow, []
                for _ in range(16):
                    nxt = model(nxt, past_key_values=c, use_cache=True, logits_to_keep=1).logits[:, -1].argmax(-1, keepdim=True)
                    tt.append(int(nxt))
                toks[name] = tt
            out["greedy16_match"] = sum(a == b for a, b in zip(toks["fa4096"], toks["eager1024"]))
            out["text_fa"] = tok.decode(toks["fa4096"])
            del caches
        torch.cuda.empty_cache()
        res["C1_prefill_equivalence_12k"] = out
    except Exception:  # noqa: BLE001
        res["C1_error"] = traceback.format_exc()[-2500:]
    log("C1", res.get("C1_prefill_equivalence_12k", res.get("C1_error")))

    # ---- C2: prefill throughput at ~120k, chunk 8192, with a profile of one late chunk ----------------------
    big = max(meta_all, key=lambda x: x["n_ctx"])
    ids, n_ctx = prompt_ids(tok, big, "none")
    n = len(ids)
    max_new = 64
    max_len = ((n + max_new + 64 + 15) // 16) * 16
    x = torch.tensor(ids, device=dev)[None]
    cache = None
    try:
        with torch.no_grad():
            FA.set_mode(model, "prefill")
            ist.set_experts(model, "grouped_mm")
            cache = StaticCache(config=model.config, max_cache_len=max_len)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.time()
            last = (n_ctx - 1) // 8192 * 8192
            for s in range(0, last, 8192):
                model(x[:, s:s + 8192], past_key_values=cache, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
            t_main = time.time() - t
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                model(x[:, last:n_ctx], past_key_values=cache, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize()
            ka = prof.key_averages()
            tot = sum(k.device_time_total for k in ka)
            top = sorted(ka, key=lambda k: -k.device_time_total)[:14]
            res["C2_prefill_120k"] = {"n_ctx": n_ctx, "seconds_excl_last_chunk": round(t_main, 1),
                                      "tokens_per_s": round(last / t_main), "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1),
                                      "last_chunk_kernel_share": {k.key[:70]: round(k.device_time_total / tot, 3) for k in top}}
    except Exception:  # noqa: BLE001
        res["C2_error"] = traceback.format_exc()[-2500:]
    log("C2", res.get("C2_prefill_120k", res.get("C2_error")))

    # ---- C3: rows on the real 120k context: dense-row tokens vs B=1 eager; SSA vs reference SSA error -------
    try:
        with torch.no_grad():
            ref_c = copy.deepcopy(cache)
            modes = ["dense", "ssa", "pqc"]
            cR = RW.repeat_cache(cache, 3)
            cache = None
            torch.cuda.empty_cache()
            rows = RW.Rows(cR, modes, L0=n_ctx, n_prompt=n, max_new=max_new, scaling=128 ** -0.5)
            rows.install()
            grab = {}
            orig = rows.row_hook

            def spy(module, query, key, value, lb, sinks, scaling):
                if module._sah_visit_idx == 29:
                    grab.update(q=query[0, :, -1].clone(), table=lb.table[0, :, -1].clone(),
                                sinks=sinks[0, :, -1, 0].clone(), pos=lb.rd.q_off + query.shape[2] - 1,
                                q_off=lb.rd.q_off, Qn=query.shape[2])
                return orig(module, query, key, value, lb, sinks, scaling)

            FA.STATE["row_hook"] = spy
            FA.set_mode(model, "prefill")
            lg = model(x[:, n_ctx:].expand(3, -1), past_key_values=cR, use_cache=True, logits_to_keep=1).logits[:, -1]
            FA.STATE["row_hook"] = rows.row_hook
            first = lg.argmax(-1, keepdim=True)
            # B=1 reference: FA question pass (dense) then eager decode
            RW.Rows.uninstall()
            FA.set_mode(model, "prefill")
            lq = model(x[:, n_ctx:], past_key_values=ref_c, use_cache=True, logits_to_keep=1).logits[:, -1]
            use_eager()
            ist.set_experts(model, "batched_mm")
            nxt, ref_toks = lq.argmax(-1, keepdim=True), [int(lq.argmax())]
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(15):
                nxt = model(nxt, past_key_values=ref_c, use_cache=True, logits_to_keep=1).logits[:, -1].argmax(-1, keepdim=True)
                ref_toks.append(int(nxt))
            torch.cuda.synchronize()
            eager_ms = 1000 * (time.time() - t) / 15
            del ref_c
            torch.cuda.empty_cache()
            rows.install()
            FA.set_mode(model, "decode")
            g = RW.GraphedRowsDecode(model, cR, first, n, 40, temperature=0.0)
            torch.cuda.synchronize()
            t = time.time()
            g.run_until([-1], burst=37)
            torch.cuda.synchronize()
            ms = 1000 * (time.time() - t) / 37
            toks = g.tokens[:, :16].tolist()
            res["C3_rows_120k"] = {"dense_row_vs_eager_tokens_match": sum(a == b for a, b in zip(toks[0], ref_toks)),
                                   "ssa_vs_dense_match": sum(a == b for a, b in zip(toks[1], toks[0])),
                                   "pqc_vs_dense_match": sum(a == b for a, b in zip(toks[2], toks[0])),
                                   "graph_ms_per_step_R3": round(ms, 2), "eager_B1_ms_per_step": round(eager_ms, 2),
                                   "budget_tokens": rows.budget_tokens,
                                   "text": {m: tok.decode(toks[b]) for b, m in enumerate(modes)},
                                   "ref_text": tok.decode(ref_toks)}
            del g
            # SSA vs reference SSA on the dense row's last question query at visit 29
            v = 29
            ctx = rows.ctx[v]
            keys, vals = cR.layers[v].keys, cR.layers[v].values
            row0 = SS.SSARow(ctx, 0)
            row0.code_keys(keys[0], torch.arange(grab["q_off"], grab["q_off"] + grab["Qn"], device=dev))
            pos, L = grab["pos"], grab["pos"] + 1
            qv = grab["q"]
            dd = pos - torch.arange(keys.shape[2], device=dev)
            band = grab["table"].float().gather(-1, dd.clamp(0, 1023)[None].expand(24, -1)).masked_fill(
                ((dd < 0) | (dd >= 1024))[None], 0)
            dense = _attn_ref(qv[None, :, None], keys[:1, :, :L], vals[:1, :, :L], grab["table"][None, :, None],
                              grab["sinks"][None, :, None], pos, 0, 128 ** -0.5, 0)[0, 0].float()
            kk = keys[0].repeat_interleave(3, 0)
            vv = vals[0].repeat_interleave(3, 0)

            def weighted(idx_h, w_h, h):
                s = (qv[h].float() @ kk[h, idx_h].float().T) * 128 ** -0.5 + band[h, idx_h]
                mx = s.max()
                e = w_h * torch.exp(s - mx)
                return (e @ vv[h, idx_h].float()) / (e.sum() + torch.exp(grab["sinks"][h].float() - mx))

            errs = {}
            for mode, (hf, sf) in (("ssa", (0.02 / 3, 0.04 / 3)), ("pqc", (0.02, 0.0))):
                sel = SS.SSASelector(row0, n + 8, n, hf, sf, 128 ** -0.5)
                att = SS.SSAAttention(1, 0, 3, 8, keys.shape[2], 128, dev)
                mine, refe = [], []
                for d in range(12 if sf > 0 else 1):
                    idx, w = sel.select(qv[None], torch.tensor([pos], device=dev),
                                        lambda i: band.gather(-1, i[0])[None])
                    o = att.run(qv[None].contiguous(), keys, vals, idx, w, grab["sinks"][None])[0].float()
                    mine.append(float(((o - dense).norm(dim=-1) / dense.norm(dim=-1)).mean()))
                    rr = SS.reference_select(qv.float(), pos, row0.codes, ctx.C, band, 128 ** -0.5, sel.k_h, sel.m, L)
                    oo = []
                    for h in range(24):
                        base = torch.cat([torch.arange(SS.SINK, device=dev), torch.arange(pos - SS.WINDOW + 1, pos + 1, device=dev),
                                          rr[h]["heavy"]])
                        wb = torch.ones(base.numel(), device=dev)
                        if sf > 0:
                            u = torch.unique(rr[h]["draws"])
                            pi = (-torch.expm1(sel.m * torch.log1p(-rr[h]["p"][u]))).clamp(SS.MIN_PI, 1)
                            base, wb = torch.cat([base, u]), torch.cat([wb, 1 / pi])
                        oo.append(weighted(base, wb, h))
                    oo = torch.stack(oo)
                    refe.append(float(((oo - dense).norm(dim=-1) / dense.norm(dim=-1)).mean()))
                errs[mode] = {"mine_mean_rel_err": round(sum(mine) / len(mine), 4),
                              "reference_mean_rel_err": round(sum(refe) / len(refe), 4), "draws": len(mine),
                              "budget": sel.S, "of_keys": L}
            res["C3_ssa_vs_reference_visit29"] = errs

            # ---- C4: per-visit operator timings at this length ------------------------------------------------
            from original_optimized import BatchDecodeWithPagedKVCacheWrapper as DenseDecode

            def timeit(fn, it=30):
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(it):
                    fn()
                e1.record()
                torch.cuda.synchronize()
                return round(e0.elapsed_time(e1) / it * 1000, 1)       # microseconds

            Lt = torch.tensor(L, device=dev)
            pb = band[None].to(torch.bfloat16).expand(3, -1, -1).contiguous()
            sk3 = grab["sinks"][None].expand(3, -1).contiguous()
            q3 = qv[None].expand(3, -1, -1).contiguous()
            outb = torch.empty(3, 24, 128, dtype=torch.bfloat16, device=dev)
            fd1 = FA.FlashDecode([0], 24, 128, dev)
            P = keys.shape[2] // 16
            i32 = dict(dtype=torch.int32, device=dev)
            npg = (L + 15) // 16
            dd8 = DenseDecode(torch.zeros(128 << 20, dtype=torch.uint8, device=dev), "HND")
            dd8.plan(torch.arange(0, 8 * npg + 1, npg, **i32),
                     (torch.arange(8, device=dev)[:, None] * P + torch.arange(npg, device=dev)[None]).reshape(-1).to(torch.int32),
                     torch.full((8,), L - (npg - 1) * 16, **i32), 3, 1, 128, 16, q_data_type=torch.bfloat16,
                     kv_data_type=torch.bfloat16)
            kvp = (keys.view(-1, 1, 16, 128), vals.view(-1, 1, 16, 128))
            sel = SS.SSASelector(row0, n + 8, n, 0.02 / 3, 0.04 / 3, 128 ** -0.5)
            att = SS.SSAAttention(1, 0, 3, 8, keys.shape[2], 128, dev)
            posT = torch.tensor([pos], device=dev)

            def ssa_op():
                idx, w = sel.select(qv[None], posT, lambda i: pb[0].gather(-1, i[0]).float()[None])
                return att.run(qv[None].contiguous(), keys, vals, idx, w, grab["sinks"][None])

            def ssa_graphable():
                return ssa_op()

            gph = torch.cuda.CUDAGraph()
            s_ = torch.cuda.Stream()
            s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                for _ in range(3):
                    ssa_graphable()
            torch.cuda.current_stream().wait_stream(s_)
            with torch.cuda.graph(gph):
                ssa_graphable()
            res["C4_op_us_per_visit"] = {
                "context": L,
                "dense_flashinfer_nobias": timeit(lambda: dd8.run(qv.view(8, 3, 128), kvp)),
                "dense_ours_flashdecode": timeit(lambda: fd1(q3, keys, vals, pb, Lt, sk3, outb, 128 ** -0.5)),
                "dense_hf_eager": timeit(lambda: M_eager(model, qv, keys, vals, band, grab["sinks"], L), it=10),
                "ssa2%_eager_launch": timeit(ssa_op),
                "ssa2%_cuda_graph": timeit(gph.replay),
            }
            del cR, rows
    except Exception:  # noqa: BLE001
        res["C3C4_error"] = traceback.format_exc()[-3000:]
    log("C3", res.get("C3_rows_120k", res.get("C3C4_error")))
    log("C3b", res.get("C3_ssa_vs_reference_visit29"))
    log("C4", res.get("C4_op_us_per_visit"))
    res["total_s"] = round(time.time() - t0, 1)
    import os
    os.makedirs("/vol/results/lb2", exist_ok=True)
    with open(f"/vol/results/lb2/checks_{int(t0)}.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    vol.commit()
    return res


def M_eager(model, q, keys, vals, band, sinks, L):
    """HF eager attention for one query over the static buffer (what the old graphed dense path ran)."""
    import torch
    from transformers.models.inkling2_loop import modeling_inkling2_loop as M

    mod = next(m for m in model.modules() if isinstance(m, M.Inkling2LoopFullAttention))
    max_len = keys.shape[2]
    mask = torch.zeros(1, 1, 1, max_len, device=q.device, dtype=torch.bfloat16)
    mask[..., L:] = float("-inf")
    return M.eager_attention_forward(mod, q[None, :, None], keys[:1], vals[:1], mask, scaling=128 ** -0.5,
                                     position_bias=band[None, :, None].to(torch.bfloat16),
                                     sinks=sinks[None, :, None, None].to(torch.bfloat16))


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, volumes={"/vol": vol}, timeout=35 * 60)
def pilot(effort: str, n: int = 8, max_new: int = 3072, temperature: float = 1.0, seed: int = 0,
          modes: tuple = ("dense",)) -> list:
    import json
    import os
    import time
    import traceback

    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    model, tok = load()
    out = []
    os.makedirs("/vol/results/lb2", exist_ok=True)
    path = f"/vol/results/lb2/pilot_{effort}_{int(t0)}.jsonl"
    for i, meta in enumerate(_pick(n, seed=seed)):
        try:
            rec = run_sample(model, tok, meta, list(modes), effort, max_new, temperature, seed=seed + i, log=log)
        except Exception:  # noqa: BLE001
            rec = {"_id": meta["_id"], "error": traceback.format_exc()[-2000:]}
            log("ERROR", rec["error"][-800:])
        out.append(rec)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        vol.commit()
    return out


@app.local_entrypoint()
def checks_main():
    import json

    print("RESULT_JSON " + json.dumps(checks.remote(), default=str))


@app.local_entrypoint()
def pilot_main(efforts: str = "high,low,none", n: int = 8, max_new: int = 2560, modes: str = "dense"):
    import json

    calls = {e: pilot.spawn(e, n, max_new, 1.0, 0, tuple(modes.split(","))) for e in efforts.split(",")}
    for e, c in calls.items():
        recs = c.get()
        print("PILOT_JSON " + json.dumps({"effort": e, "records": recs}, default=str))


# ------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, volumes={"/vol": vol}, timeout=150 * 60)
def shard(tag: str, ids: list, effort: str, modes: list, max_new: int, temperature: float = 1.0,
          seed: int = 0, deadline_s: float = 140 * 60) -> dict:
    """Run a list of LongBench v2 _ids; append one JSON line per sample (resumable); stop cleanly before
    `deadline_s` so the container never hits its hard timeout mid-sample."""
    import json
    import os
    import time
    import traceback

    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    meta = {m["_id"]: m for m in json.load(open(f"{DATA}/meta.json"))}
    out_dir = f"/vol/results/lb2/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    done = set()
    for fn in os.listdir(out_dir):
        if fn.endswith(".jsonl"):
            for line in open(f"{out_dir}/{fn}"):
                r = json.loads(line)
                if "rows" in r:
                    done.add(r["_id"])
    todo = [i for i in ids if i not in done]
    model, tok = load()
    path = f"{out_dir}/shard_{int(t0)}_{ids[0][:8]}.jsonl"
    n_ok, per_sample = 0, []
    for k, _id in enumerate(todo):
        avg = sum(per_sample) / len(per_sample) if per_sample else 240.0
        if time.time() - t0 + 1.5 * avg > deadline_s:
            log(f"deadline: stopping with {len(todo) - k} left")
            break
        ts = time.time()
        try:
            rec = run_sample(model, tok, meta[_id], modes, effort, max_new, temperature,
                             seed=seed + hash(_id) % 100000, log=log)
            n_ok += 1
        except Exception:  # noqa: BLE001
            rec = {"_id": _id, "error": traceback.format_exc()[-2000:]}
            log("ERROR", _id, rec["error"][-600:])
            import torch
            torch.cuda.empty_cache()
        per_sample.append(time.time() - ts)
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        vol.commit()
    return {"done": n_ok, "todo": len(todo), "seconds": round(time.time() - t0)}


def _score(recs: list, modes: list) -> dict:
    """Official LongBench v2 breakdown (overall / easy / hard / short / medium / long), % correct, plus paired
    agreement of every sparse row with the dense row."""
    out = {}
    for m in modes:
        rows = [r for r in recs if "rows" in r and m in r["rows"]]

        def acc(sel):
            s = [r for r in rows if sel(r)]
            return (round(100 * sum(r["rows"][m]["judge"] for r in s) / len(s), 1), len(s)) if s else (None, 0)

        out[m] = {"overall": acc(lambda r: True), "easy": acc(lambda r: r["difficulty"] == "easy"),
                  "hard": acc(lambda r: r["difficulty"] == "hard"), "short": acc(lambda r: r["length"] == "short"),
                  "medium": acc(lambda r: r["length"] == "medium"), "long": acc(lambda r: r["length"] == "long"),
                  "no_answer": sum(r["rows"][m]["pred"] is None for r in rows),
                  "unfinished": sum(not r["rows"][m]["finished"] for r in rows),
                  "mean_gen_tokens": round(sum(r["rows"][m]["n_gen"] for r in rows) / max(len(rows), 1))}
        if m != "dense" and rows and "dense" in rows[0]["rows"]:
            both = [(r["rows"]["dense"]["judge"], r["rows"][m]["judge"]) for r in rows]
            out[m]["paired_vs_dense"] = {"both_right": sum(a and b for a, b in both),
                                         "dense_only": sum(a and not b for a, b in both),
                                         "sparse_only": sum(b and not a for a, b in both),
                                         "same_letter": sum(r["rows"]["dense"]["pred"] == r["rows"][m]["pred"]
                                                            for r in rows)}
    return out


@app.function(image=image, volumes={"/vol": vol}, cpu=1, memory=2048, timeout=600)
def collect(tag: str) -> list:
    import json
    import os

    vol.reload()
    recs = {}
    d = f"/vol/results/lb2/{tag}"
    for fn in sorted(os.listdir(d)):
        for line in open(f"{d}/{fn}"):
            r = json.loads(line)
            if "rows" in r or "direct" in r or r["_id"] not in recs:
                recs[r["_id"]] = r
    return list(recs.values())


@app.local_entrypoint()
def run_main(tag: str, effort: str = "low", modes: str = "dense,ssa,pqc", max_new: int = 2048,
             shards: int = 4, deadline_min: float = 140, limit: int = 0, temperature: float = 1.0):
    """Spread all 503 questions over `shards` containers, balanced by estimated cost (n^2 attention + n)."""
    import json

    ids = order_ids.remote(shards, limit)
    calls = [shard.spawn(tag, g, effort, modes.split(","), max_new, temperature, 0, deadline_min * 60)
             for g in ids if g]
    for c in calls:
        print("SHARD", c.get())
    recs = collect.remote(tag)
    print("RESULT_JSON " + json.dumps({"tag": tag, "n": len(recs), "errors": sum("error" in r for r in recs),
                                       "scores": _score(recs, modes.split(","))}))


@app.function(image=image, volumes={"/vol": vol}, cpu=1, memory=2048, timeout=300)
def order_ids(shards: int, limit: int = 0) -> list:
    import json

    meta = json.load(open(f"{DATA}/meta.json"))
    if limit:
        meta = meta[:limit]
    cost = lambda m: (m["n_ctx"] + m["n_q"]) / 1e3 + ((m["n_ctx"] + m["n_q"]) / 1e3) ** 2 / 400 + 60  # noqa: E731
    meta.sort(key=lambda m: -cost(m))
    groups, loads = [[] for _ in range(shards)], [0.0] * shards
    for m in meta:
        k = loads.index(min(loads))
        groups[k].append(m["_id"])
        loads[k] += cost(m)
    return groups


@app.local_entrypoint()
def score_main(tag: str, modes: str = "dense,ssa,pqc"):
    import json

    recs = collect.remote(tag)
    print("RESULT_JSON " + json.dumps({"tag": tag, "n": len(recs), "errors": sum("error" in r for r in recs),
                                       "scores": _score(recs, modes.split(","))}))


# ------------------------------------------------------------------------------------------------------------
def _graph_time_us(fn, it: int = 50) -> float:
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(it):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return round(e0.elapsed_time(e1) / it * 1000, 1)


def op_bench(keys, vals, L: int, n_prompt: int, dev, tag: str) -> dict:
    """Per full-attention visit, one query, graph-replayed: dense FlashInfer (no bias), our dense
    flash-decode (bias + sinks), SSA 2% and PQCache 2% (fused), old torch SSA."""
    import torch
    from original_optimized import BatchDecodeWithPagedKVCacheWrapper as DenseDecode

    import inkling_fa as FA
    import inkling_ssa as SS

    out = {"context": L}
    torch.manual_seed(0)
    q = (torch.randn(24, 128, device=dev) * 1.0).to(torch.bfloat16)
    sinks = torch.randn(1, 24, device=dev)
    max_len = keys.shape[2]
    dd = (L - 1) - torch.arange(max_len, device=dev)
    pb = (torch.randn(24, max_len, device=dev) * 0.5).masked_fill(((dd < 0) | (dd >= 1024))[None], 0).to(torch.bfloat16)
    B = keys.shape[0]
    P = max_len // 16
    npg = (L + 15) // 16
    i32 = dict(dtype=torch.int32, device=dev)
    fi = DenseDecode(torch.zeros(128 << 20, dtype=torch.uint8, device=dev), "HND")
    fi.plan(torch.arange(0, 8 * npg + 1, npg, **i32),
            (torch.arange(8, device=dev)[:, None] * P + torch.arange(npg, device=dev)[None]).reshape(-1).to(torch.int32),
            torch.full((8,), L - (npg - 1) * 16, **i32), 3, 1, 128, 16, q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16)
    kvp = (keys.view(-1, 1, 16, 128), vals.view(-1, 1, 16, 128))
    out["dense_flashinfer_nobias_us"] = _graph_time_us(lambda: fi.run(q.view(8, 3, 128), kvp))
    fd = FA.FlashDecode([0], 24, 128, dev)
    Lt = torch.tensor(L, device=dev)
    qB = q[None].expand(B, -1, -1).contiguous()
    pbB = pb[None].expand(B, -1, -1).contiguous()
    skB = sinks.expand(B, -1).contiguous()
    o = torch.empty(B, 24, 128, dtype=torch.bfloat16, device=dev)
    out["dense_ours_us"] = _graph_time_us(lambda: fd(qB, keys, vals, pbB, Lt, skB, o, 128 ** -0.5))
    ctx = SS.SSAContext(keys[0], L0=L - 1, iters=10, seed=0)
    row = SS.SSARow(ctx, 0)
    posT = torch.tensor([L - 1], device=dev)
    row.code_keys(keys[0], posT)
    for name, (hf, sf) in (("ssa2%", (0.02 / 3, 0.04 / 3)), ("pqc2%", (0.02, 0.0))):
        f = SS.SSAFast(row, n_prompt + 64, n_prompt, hf, sf, 128 ** -0.5, max_q=1)
        out[f"{name}_budget_keys_per_head"] = f.S
        out[f"{name}_fused_us"] = _graph_time_us(lambda: f(q[None], posT, keys, vals, 0, sinks, pb=pb))
        out[f"{name}_select_only_us"] = _graph_time_us(lambda: f.select_fast(q[None], posT, pb=pb))
        f2 = SS.SSAFused(row, n_prompt + 64, n_prompt, hf, sf, 128 ** -0.5, max_q=1)
        out[f"{name}_fused_v2_us"] = _graph_time_us(lambda: f2(q[None], posT, keys, vals, 0, sinks, pb=pb))
        out[f"{name}_select_v2_us"] = _graph_time_us(lambda: f2.select_fused(q[None], posT, pb=pb))
        f3 = SS.SSAFusedV3(row, n_prompt + 64, n_prompt, hf, sf, 128 ** -0.5, max_q=1)
        out[f"{name}_fused_v3_us"] = _graph_time_us(lambda: f3(q[None], posT, keys, vals, 0, sinks, pb=pb))
        out[f"{name}_select_v3_us"] = _graph_time_us(lambda: f3.select_fused(q[None], posT, pb=pb))
    sel = SS.SSASelector(row, n_prompt + 64, n_prompt, 0.02 / 3, 0.04 / 3, 128 ** -0.5)
    att = SS.SSAAttention(1, 0, B, 8, max_len, 128, dev)

    def old():
        idx, w = sel.select(q[None], posT, lambda i: pb.gather(-1, i[0]).float()[None])
        return att.run(q[None].contiguous(), keys, vals, idx, w, sinks)
    out["ssa2%_torch_first_version_us"] = _graph_time_us(old)
    out["speedup_ssa_v3_vs_flashinfer_dense"] = round(out["dense_flashinfer_nobias_us"] / out["ssa2%_fused_v3_us"], 2)
    out["speedup_ssa_v3_vs_our_dense"] = round(out["dense_ours_us"] / out["ssa2%_fused_v3_us"], 2)
    print(tag, out, flush=True)
    return out


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, volumes={"/vol": vol}, timeout=25 * 60)
def bench() -> dict:
    import json
    import time
    import traceback

    import torch
    from transformers import StaticCache

    import inkling_fa as FA
    import inkling_rows as RW
    import inkling_static as ist

    res: dict = {}
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    dev = "cuda"
    # ---- synthetic long contexts (before the model takes 49 GB) ------------------------------------------
    for L in (524288, 1048576):
        try:
            max_len = L + 1024
            keys = torch.randn(1, 8, max_len, 128, device=dev, dtype=torch.bfloat16)
            vals = torch.randn(1, 8, max_len, 128, device=dev, dtype=torch.bfloat16)
            res[f"op_synthetic_{L}"] = op_bench(keys, vals, L, L, dev, f"synthetic{L}")
            del keys, vals
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            res[f"op_synthetic_{L}_error"] = traceback.format_exc()[-1500:]
            log(res[f"op_synthetic_{L}_error"][-500:])
    model, tok = load()
    meta_all = json.load(open(f"{DATA}/meta.json"))
    for target in (32000, 120000):
        m = min(meta_all, key=lambda x: abs(x["n_ctx"] - target))
        ids, n_ctx = prompt_ids(tok, m, "none")
        n = len(ids)
        max_new = 64
        max_len = ((n + max_new + 64 + 15) // 16) * 16
        x = torch.tensor(ids, device=dev)[None]
        try:
            with torch.no_grad():
                FA.set_mode(model, "prefill")
                ist.set_experts(model, "grouped_mm")
                cache = StaticCache(config=model.config, max_cache_len=max_len)
                torch.cuda.synchronize()
                t = time.time()
                last = (n_ctx - 1) // 8192 * 8192
                for s in range(0, last, 8192):
                    model(x[:, s:s + 8192], past_key_values=cache, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize()
                t_main = time.time() - t
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    model(x[:, last:n_ctx], past_key_values=cache, use_cache=True, logits_to_keep=1)
                    torch.cuda.synchronize()
                ka = prof.key_averages()
                tot = sum(k.device_time_total for k in ka)
                top = sorted(ka, key=lambda k: -k.device_time_total)[:25]
                res[f"prefill_{target}"] = {"n_ctx": n_ctx, "s_excl_last": round(t_main, 1),
                                            "tok_per_s": round(last / max(t_main, 1e-9)),
                                            "last_chunk_tokens": n_ctx - last, "last_chunk_ms": round(tot / 1000, 1),
                                            "kernels": {k.key[:90]: round(k.device_time_total / tot, 3) for k in top}}
                log("prefill", target, res[f"prefill_{target}"])
                # operator timings on the real keys of visit 29
                res[f"op_real_{target}"] = op_bench(cache.layers[29].keys, cache.layers[29].values, n_ctx, n, dev,
                                                    f"real{target}")
                # whole decode step: R=1 dense, R=3 dense/ssa/pqc, with two MoE implementations
                for modes, impl in ((["dense"], True), (["dense", "ssa", "pqc"], True), (["dense", "ssa", "pqc"], "v3"),
                                    (["dense", "ssa"], "v3"), (["ssa"], "v3")):
                    for experts in ("batched_mm",):
                        key = f"step_ms_{target}_{'+'.join(modes)}_{impl if isinstance(impl, str) else 'v1'}"
                        try:
                            cR = RW.repeat_cache(cache, len(modes))
                            rows = RW.Rows(cR, modes, L0=n_ctx, n_prompt=n, max_new=max_new, scaling=128 ** -0.5,
                                           fast=impl)
                            rows.install()
                            FA.set_mode(model, "prefill")
                            ist.set_experts(model, "grouped_mm")
                            lg = model(x[:, n_ctx:].expand(len(modes), -1), past_key_values=cR, use_cache=True,
                                       logits_to_keep=1).logits[:, -1]
                            FA.set_mode(model, "decode")
                            ist.set_experts(model, experts)
                            g = RW.GraphedRowsDecode(model, cR, lg.argmax(-1, keepdim=True), n, 48, temperature=1.0)
                            torch.cuda.synchronize()
                            t = time.time()
                            g.run_until([-1], burst=40)
                            torch.cuda.synchronize()
                            res[key] = round(1000 * (time.time() - t) / 40, 2)
                            del g, rows, cR
                        except Exception:  # noqa: BLE001
                            res[key] = "error: " + traceback.format_exc()[-400:]
                        RW.Rows.uninstall()
                        torch.cuda.empty_cache()
                        log(key, res[key])
                del cache
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            res[f"real_{target}_error"] = traceback.format_exc()[-2500:]
            log(res[f"real_{target}_error"][-800:])
    res["total_s"] = round(time.time() - t0, 1)
    import os
    os.makedirs("/vol/results/lb2", exist_ok=True)
    with open(f"/vol/results/lb2/bench_{int(t0)}.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    vol.commit()
    return res


@app.local_entrypoint()
def bench_main():
    import json

    print("RESULT_JSON " + json.dumps(bench.remote(), default=str))


# ------------------------------------------------------------------------------------------------------------
def answer_tokens(tok):
    """Common prefix of 'The correct answer is (X)' over X in ABCD, and the first token that differs per X."""
    seqs = {c: tok.encode(f"The correct answer is ({c})", add_special_tokens=False) for c in "ABCD"}
    k = 0
    while all(len(s) > k for s in seqs.values()) and len({s[k] for s in seqs.values()}) == 1:
        k += 1
    prefix = seqs["A"][:k]
    letters = {c: s[k] for c, s in seqs.items()}
    assert len(set(letters.values())) == 4, seqs
    return prefix, letters


def run_sample2(model, tok, meta, modes, effort, reasoning: bool, max_new: int, temperature: float = 1.0,
                chunk: int = 8192, seed: int = 0, budget: float = 0.02, log=print):
    """One dense context prefill, then (a) forced direct answer for every row (letter = argmax over the four
    option tokens right after 'The correct answer is (' -- the LongBench v2 'w/o CoT' reading, deterministic),
    and (b) optionally free reasoning + answer for every row, from the same prefill."""
    import time

    import torch
    from transformers import StaticCache

    import inkling_fa as FA
    import inkling_rows as RW
    import inkling_static as ist

    ids, n_ctx = prompt_ids(tok, meta, effort)
    n = len(ids)
    prefix, letters = answer_tokens(tok)
    forced = ids[n_ctx:] + tok.encode("<|message_model|><|content_text|>", add_special_tokens=False) + prefix
    rec = {"_id": meta["_id"], "domain": meta["domain"], "difficulty": meta["difficulty"], "length": meta["length"],
           "answer": meta["answer"], "n_prompt": n, "n_ctx": n_ctx, "effort": effort, "modes": list(modes)}
    max_len = ((max(n + max_new, n_ctx + len(forced)) + 64 + 15) // 16) * 16
    dev = "cuda"
    x = torch.tensor(ids, device=dev)[None]
    torch.manual_seed(seed)
    RW.Rows.uninstall()
    R = len(modes)
    with torch.no_grad():
        FA.set_mode(model, "prefill")
        ist.set_experts(model, "grouped_mm")
        cache = StaticCache(config=model.config, max_cache_len=max_len)
        torch.cuda.synchronize()
        t = time.time()
        for s in range(0, n_ctx, chunk):
            model(x[:, s:min(s + chunk, n_ctx)], past_key_values=cache, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        rec["prefill_s"] = round(time.time() - t, 2)

        # (a) direct answer
        t = time.time()
        cR = RW.repeat_cache(cache, R)
        rows = RW.Rows(cR, modes, L0=n_ctx, n_prompt=n_ctx + len(forced), max_new=8, scaling=128 ** -0.5,
                       budget=budget, seed=seed)
        rows.install()
        fx = torch.tensor(forced, device=dev)[None].expand(R, -1)
        for a in range(0, fx.shape[1], 1024):
            lg = model(fx[:, a:a + 1024], past_key_values=cR, use_cache=True, logits_to_keep=1).logits[:, -1].float()
        lid = torch.tensor([letters[c] for c in "ABCD"], device=dev)
        lp = torch.log_softmax(lg[:, lid], -1)
        rec["direct"] = {}
        for b, m in enumerate(modes):
            pred = "ABCD"[int(lp[b].argmax())]
            rec["direct"][m] = {"pred": pred, "judge": pred == meta["answer"],
                                "logp": [round(float(v), 4) for v in lp[b]],
                                "p_letters_total": round(float(torch.softmax(lg[b], -1)[lid].sum()), 4)}
        rec["budget_tokens"] = rows.budget_tokens
        RW.Rows.uninstall()
        del rows, cR
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        rec["direct_s"] = round(time.time() - t, 2)

        # (b) reasoning
        if reasoning:
            t = time.time()
            cR = RW.repeat_cache(cache, R)
            rows = RW.Rows(cR, modes, L0=n_ctx, n_prompt=n, max_new=max_new, scaling=128 ** -0.5, budget=budget,
                           seed=seed)
            rows.install()
            FA.set_mode(model, "prefill")
            ist.set_experts(model, "grouped_mm")
            q = x[:, n_ctx:].expand(R, -1)
            for a in range(0, q.shape[1], 1024):
                lg = model(q[:, a:a + 1024], past_key_values=cR, use_cache=True, logits_to_keep=1).logits[:, -1]
            first = RW.sample_first(lg, temperature)
            FA.set_mode(model, "decode")
            ist.set_experts(model, "batched_mm")
            g = RW.GraphedRowsDecode(model, cR, first, n, max_new, temperature)
            torch.cuda.synchronize()
            t1 = time.time()
            steps = g.run_until([EOS], burst=32)
            torch.cuda.synchronize()
            rec["decode_steps"] = steps
            rec["ms_per_step"] = round(1000 * (time.time() - t1) / max(steps - 3, 1), 2)
            toks = g.tokens[:, :steps].tolist()
            rec["reasoning"] = {}
            for b, m in enumerate(modes):
                r = parse(tok, toks[b])
                r["judge"] = r["pred"] == meta["answer"]
                rec["reasoning"][m] = r
            del g, rows, cR
            RW.Rows.uninstall()
            FA.set_mode(model, "prefill")
            torch.cuda.synchronize()
            rec["reasoning_s"] = round(time.time() - t, 2)
        del cache
    torch.cuda.empty_cache()
    log(meta["_id"], meta["length"], n, {k: rec.get(k) for k in ("prefill_s", "direct_s", "reasoning_s",
                                                                   "decode_steps", "ms_per_step")},
        "direct", {m: (r["pred"], r["judge"]) for m, r in rec["direct"].items()},
        "reasoning", {m: (r["pred"], r["judge"], r["n_gen"]) for m, r in rec.get("reasoning", {}).items()},
        "ans", meta["answer"])
    return rec


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, volumes={"/vol": vol}, timeout=27 * 60)
def shard2(tag: str, ids: list, reason_ids: list, effort: str, modes: list, max_new: int,
           temperature: float = 1.0, deadline_s: float = 75 * 60) -> dict:
    import json
    import os
    import time
    import traceback

    import torch

    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)  # noqa: E731
    meta = {m["_id"]: m for m in json.load(open(f"{DATA}/meta.json"))}
    out_dir = f"/vol/results/lb2/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    vol.reload()
    done = set()
    for fn in os.listdir(out_dir):
        if fn.endswith(".jsonl"):
            for line in open(f"{out_dir}/{fn}"):
                r = json.loads(line)
                if "direct" in r and (r["_id"] not in reason_ids or "reasoning" in r):
                    done.add(r["_id"])
    todo = [i for i in ids if i not in done]
    model, tok = load()
    path = f"{out_dir}/shard_{int(t0)}_{ids[0][:8]}.jsonl"
    rs = set(reason_ids)
    n_ok, est = 0, []
    for k, _id in enumerate(todo):
        guess = (sorted(est)[len(est) // 2] if est else 45.0) + (2.2 * max_new * 0.08 if _id in rs else 0.0)
        if time.time() - t0 + 1.3 * guess + 30 > deadline_s:
            log(f"deadline: stopping with {len(todo) - k} left")
            break
        ts = time.time()
        try:
            rec = run_sample2(model, tok, meta[_id], modes, effort, _id in rs, max_new, temperature,
                              seed=sum(map(ord, _id)), log=log)
            n_ok += 1
            if _id not in rs:
                est.append(time.time() - ts)
        except Exception:  # noqa: BLE001
            rec = {"_id": _id, "error": traceback.format_exc()[-2000:]}
            log("ERROR", _id, rec["error"][-600:])
            torch.cuda.empty_cache()
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        vol.commit()
    return {"done": n_ok, "todo": len(todo), "seconds": round(time.time() - t0)}


@app.function(image=image, volumes={"/vol": vol}, cpu=1, memory=2048, timeout=300)
def pick_ids(n: int, seed: int) -> list:
    return [m["_id"] for m in _pick(n, seed=seed)]


def _score2(recs: list, modes: list) -> dict:
    out = {}
    for part in ("direct", "reasoning"):
        rows = [r for r in recs if part in r]
        if not rows:
            continue
        res = {"n": len(rows)}
        for m in modes:
            def acc(sel):
                s = [r for r in rows if sel(r)]
                return (round(100 * sum(r[part][m]["judge"] for r in s) / len(s), 1), len(s)) if s else (None, 0)

            d = {"overall": acc(lambda r: True), "easy": acc(lambda r: r["difficulty"] == "easy"),
                 "hard": acc(lambda r: r["difficulty"] == "hard"), "short": acc(lambda r: r["length"] == "short"),
                 "medium": acc(lambda r: r["length"] == "medium"), "long": acc(lambda r: r["length"] == "long")}
            if part == "reasoning":
                d["unfinished"] = sum(not r[part][m]["finished"] for r in rows)
                d["mean_gen_tokens"] = round(sum(r[part][m]["n_gen"] for r in rows) / len(rows))
            if m != "dense":
                pairs = [(r[part]["dense"]["judge"], r[part][m]["judge"]) for r in rows]
                d["paired_vs_dense"] = {"both": sum(a and b for a, b in pairs), "dense_only": sum(a and not b for a, b in pairs),
                                        "sparse_only": sum(b and not a for a, b in pairs),
                                        "same_letter": sum(r[part]["dense"]["pred"] == r[part][m]["pred"] for r in rows)}
            res[m] = d
        out[part] = res
    return out


@app.local_entrypoint()
def run2_main(tag: str, effort: str = "low", modes: str = "dense,ssa,pqc", n_reason: int = 48, max_new: int = 2048,
              shards: int = 4, deadline_min: float = 75, limit: int = 0):
    import json

    groups = order_ids.remote(shards, limit)
    if n_reason < 0:                      # smoke test: reasoning on the cheapest sample of each shard
        reason = [g[-1] for g in groups if g]
    else:
        reason = pick_ids.remote(n_reason, 1) if n_reason else []
    calls = [shard2.spawn(tag, g, [i for i in reason if i in set(g)], effort, modes.split(","), max_new, 1.0,
                          deadline_min * 60) for g in groups if g]
    for c in calls:
        print("SHARD", c.get())
    recs = collect.remote(tag)
    print("RESULT_JSON " + json.dumps({"tag": tag, "n": len(recs), "errors": sum("error" in r for r in recs),
                                       "scores": _score2(recs, modes.split(","))}))


@app.local_entrypoint()
def score2_main(tag: str, modes: str = "dense,ssa,pqc"):
    import json

    recs = collect.remote(tag)
    print("RESULT_JSON " + json.dumps({"tag": tag, "n": len(recs), "errors": sum("error" in r for r in recs),
                                       "scores": _score2(recs, modes.split(","))}))


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, volumes={"/vol": vol}, timeout=12 * 60)
def opbreak(contexts: str = "32768,131072,1048576") -> dict:
    """Stage-by-stage graph timings of the fused SSA op on synthetic keys (no model)."""
    import torch
    import triton

    import inkling_ssa as SS

    dev = "cuda"
    res = {}
    for L in [int(x) for x in contexts.split(",")]:
        torch.manual_seed(0)
        max_len = L + 1024
        keys = torch.randn(1, 8, max_len, 128, device=dev, dtype=torch.bfloat16)
        vals = torch.randn(1, 8, max_len, 128, device=dev, dtype=torch.bfloat16)
        q = torch.randn(1, 24, 128, device=dev).to(torch.bfloat16)
        sinks = torch.randn(1, 24, device=dev)
        pb = torch.zeros(24, max_len, device=dev, dtype=torch.bfloat16)
        ctx = SS.SSAContext(keys[0], L0=L - 1, iters=10, seed=0)
        row = SS.SSARow(ctx, 0)
        posT = torch.tensor([L - 1], device=dev)
        f = SS.SSAFused(row, L + 64, L, 0.02 / 3, 0.04 / 3, 128 ** -0.5, max_q=1)
        out = {"NI": f.NI, "KI": f.KI, "S": f.S, "n_split": f.n_split}
        out["items"] = _graph_time_us(lambda: f.items(q, posT, pb, None))
        logits, cnts = f.items(q, posT, pb, None)
        out["topk"] = _graph_time_us(lambda: torch.topk(logits, f.KI, dim=-1, sorted=True))
        out["select_total"] = _graph_time_us(lambda: f.select_fused(q, posT, pb=pb))
        idx, w = f.select_fused(q, posT, pb=pb)
        for ns in (4, 8, 16, 32, 64):
            f.n_split = ns
            f.op = torch.empty(24 * ns, 128, device=dev)
            f.ml = torch.empty(24 * ns, 2, device=dev)
            out[f"attend_split{ns}"] = _graph_time_us(lambda: f.attend(q, keys, vals, 0, idx, w, sinks))
        res[str(L)] = out
        print(L, out, flush=True)
        del keys, vals, ctx, row, f
        torch.cuda.empty_cache()
    return res


@app.local_entrypoint()
def opbreak_main(contexts: str = "32768,131072,1048576"):
    import json

    print("RESULT_JSON " + json.dumps(opbreak.remote(contexts)))


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=49152, volumes={"/vol": vol}, timeout=12 * 60)
def opbreak3(contexts: str = "32768,131072,1048576") -> dict:
    import torch
    import triton

    import inkling_ssa as SS

    dev = "cuda"
    res = {}
    for L in [int(x) for x in contexts.split(",")]:
        torch.manual_seed(0)
        max_len = L + 1024
        keys = torch.randn(1, 8, max_len, 128, device=dev, dtype=torch.bfloat16)
        vals = torch.randn(1, 8, max_len, 128, device=dev, dtype=torch.bfloat16)
        q = torch.randn(1, 24, 128, device=dev).to(torch.bfloat16)
        sinks = torch.randn(1, 24, device=dev)
        pb = torch.zeros(24, max_len, device=dev, dtype=torch.bfloat16)
        ctx = SS.SSAContext(keys[0], L0=L - 1, iters=10, seed=0)
        row = SS.SSARow(ctx, 0)
        posT = torch.tensor([L - 1], device=dev)
        f = SS.SSAFusedV3(row, L + 64, L, 0.02 / 3, 0.04 / 3, 128 ** -0.5, max_q=1)
        rows, c, inv = 24, ctx, SS.LOG2E
        out = {"NI": f.NI, "ni_pad": f.ni_pad, "S": f.S, "n_split": f.n_split, "nblk": f.nblk}
        out["bump"] = _graph_time_us(lambda: SS._bump_seed[(1,)](f.seed))
        out["lut"] = _graph_time_us(lambda: SS._ssa_lut_kernel[(rows,)](q, c.C, f.lut, q.stride(0), q.stride(1), HQ=24, G=3,
                                                                       NCc=SS.NC, DS=c.dsub, scale=f.scaling))
        logits, cnts = f.items(q, posT, pb, None)
        out["lut+items"] = _graph_time_us(lambda: f.items(q, posT, pb, None))
        for nw in (4, 8, 16):
            out[f"select_k_w{nw}"] = _graph_time_us(lambda: SS._ssa_select_kernel[(rows,)](
                logits, cnts, f.hoff, f.wcdf, f.zt, f.vmax, f.NI, f.k_h, inv, NI_PAD=f.ni_pad, num_warps=nw))
        out["expblk"] = _graph_time_us(lambda: SS._ssa_expblk_kernel[(rows, f.nblk)](f.seed, f.eb, f.m, f.nblk, BS=f.BS))
        out["select_total"] = _graph_time_us(lambda: f.select_fused(q, posT, pb=pb))
        idx, w = f.select_fused(q, posT, pb=pb)
        out["slots_only~"] = round(out["select_total"] - out["lut+items"] - out["select_k_w16"] - out["expblk"] - out["bump"], 1)
        out["attend"] = _graph_time_us(lambda: f.attend(q, keys, vals, 0, idx, w, sinks))
        out["total"] = _graph_time_us(lambda: f(q, posT, keys, vals, 0, sinks, pb=pb))
        f4 = SS.SSAFusedV4(row, L + 64, L, 0.02 / 3, 0.04 / 3, 128 ** -0.5, max_q=1)
        out["v4_select_k"] = _graph_time_us(lambda: SS._ssa_select4_kernel[(rows,)](
            logits, cnts, f4.hoff, f4.wcdf, f4.zt, f4.vmax, f4.seed, f4.eb, f4.NI, f4.k_h, f4.m, f4.nblk, inv,
            NI_PAD=f4.ni_pad, BS=f4.BS, CH=4, num_warps=16))
        out["v4_select_total"] = _graph_time_us(lambda: f4.select_fused(q, posT, pb=pb))
        out["v4_total"] = _graph_time_us(lambda: f4(q, posT, keys, vals, 0, sinks, pb=pb))
        p4 = SS.SSAFusedV4(row, L + 64, L, 0.02, 0.0, 128 ** -0.5, max_q=1)
        out["v4_pqc_total"] = _graph_time_us(lambda: p4(q, posT, keys, vals, 0, sinks, pb=pb))
        from original_optimized import BatchDecodeWithPagedKVCacheWrapper as DenseDecode
        P = max_len // 16
        npg = (L + 15) // 16
        i32 = dict(dtype=torch.int32, device=dev)
        fi = DenseDecode(torch.zeros(128 << 20, dtype=torch.uint8, device=dev), "HND")
        fi.plan(torch.arange(0, 8 * npg + 1, npg, **i32),
                (torch.arange(8, device=dev)[:, None] * P + torch.arange(npg, device=dev)[None]).reshape(-1).to(torch.int32),
                torch.full((8,), L - (npg - 1) * 16, **i32), 3, 1, 128, 16, q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16)
        kvp = (keys.view(-1, 1, 16, 128), vals.view(-1, 1, 16, 128))
        out["dense_flashinfer"] = _graph_time_us(lambda: fi.run(q[0].view(8, 3, 128), kvp))
        out["speedup_v4_vs_flashinfer"] = round(out["dense_flashinfer"] / out["v4_total"], 2)
        res[str(L)] = out
        print(L, out, flush=True)
        del keys, vals, ctx, row, f
        torch.cuda.empty_cache()
    return res


@app.local_entrypoint()
def opbreak3_main(contexts: str = "32768,131072,1048576"):
    import json

    print("RESULT_JSON " + json.dumps(opbreak3.remote(contexts)))


def _analyze(recs: list, modes: list) -> dict:
    """Accuracy (official breakdown), paired stats vs dense (exact McNemar p, bootstrap 95% CI of the
    accuracy difference), mean log-prob of the right letter, density and timing."""
    import math
    import random

    def mcnemar(b, c):
        n = b + c
        if n == 0:
            return 1.0
        k = min(b, c)
        return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

    out = {"n_records": len(recs), "errors": [r["_id"] for r in recs if "error" in r]}
    for part in ("direct", "reasoning"):
        rows = [r for r in recs if part in r]
        if not rows:
            continue
        P = {"n": len(rows)}
        for m in modes:
            J = [bool(r[part][m]["judge"]) for r in rows]
            d = {"acc": round(100 * sum(J) / len(J), 1)}
            for key, sel in (("easy", lambda r: r["difficulty"] == "easy"), ("hard", lambda r: r["difficulty"] == "hard"),
                             ("short", lambda r: r["length"] == "short"), ("medium", lambda r: r["length"] == "medium"),
                             ("long", lambda r: r["length"] == "long")):
                s = [r for r in rows if sel(r)]
                d[key] = (round(100 * sum(r[part][m]["judge"] for r in s) / len(s), 1), len(s)) if s else None
            if part == "direct":
                lp = [r[part][m]["logp"]["ABCD".index(r["answer"])] for r in rows]
                d["mean_logp_correct"] = round(sum(lp) / len(lp), 4)
            else:
                d["unfinished"] = sum(not r[part][m]["finished"] for r in rows)
                d["no_answer"] = sum(r[part][m]["pred"] is None for r in rows)
                d["mean_gen_tokens"] = round(sum(r[part][m]["n_gen"] for r in rows) / len(rows))
            if m != "dense":
                D = [bool(r[part]["dense"]["judge"]) for r in rows]
                b = sum(a and not s for a, s in zip(D, J))
                c = sum(s and not a for a, s in zip(D, J))
                rng = random.Random(0)
                diffs = []
                for _ in range(2000):
                    ii = [rng.randrange(len(rows)) for _ in rows]
                    diffs.append(100 * sum(J[i] - D[i] for i in ii) / len(ii))
                diffs.sort()
                d["vs_dense"] = {"diff_pts": round(100 * (sum(J) - sum(D)) / len(J), 1),
                                 "ci95": [round(diffs[50], 1), round(diffs[1949], 1)],
                                 "dense_only_right": b, "sparse_only_right": c, "mcnemar_p": round(mcnemar(b, c), 3),
                                 "same_letter_pct": round(100 * sum(r[part]["dense"]["pred"] == r[part][m]["pred"]
                                                                    for r in rows) / len(rows), 1)}
                if part == "direct":
                    dl = [r[part][m]["logp"]["ABCD".index(r["answer"])] - r[part]["dense"]["logp"]["ABCD".index(r["answer"])]
                          for r in rows]
                    d["vs_dense"]["mean_delta_logp_correct"] = round(sum(dl) / len(dl), 4)
            P[m] = d
        out[part] = P
    dens = [r["budget_tokens"]["ssa"] / r["n_prompt"] for r in recs if "budget_tokens" in r and r.get("budget_tokens", {}).get("ssa")]
    if dens:
        dens.sort()
        out["ssa_effective_density_pct"] = {"median": round(100 * dens[len(dens) // 2], 2), "min": round(100 * dens[0], 2),
                                            "max": round(100 * dens[-1], 2)}
    pf = [(r["n_ctx"], r["prefill_s"]) for r in recs if "prefill_s" in r]
    if pf:
        out["prefill"] = {"total_tokens": sum(a for a, _ in pf), "total_s": round(sum(b for _, b in pf)),
                          "tok_per_s": round(sum(a for a, _ in pf) / sum(b for _, b in pf))}
    ms = [r["ms_per_step"] for r in recs if r.get("ms_per_step")]
    if ms:
        out["reasoning_ms_per_step_3rows"] = round(sum(ms) / len(ms), 1)
    return out


@app.local_entrypoint()
def analyze_main(tag: str = "lb2_main", modes: str = "dense,ssa,pqc"):
    import json

    recs = collect.remote(tag)
    print("ANALYSIS_JSON " + json.dumps(_analyze(recs, modes.split(","))))
