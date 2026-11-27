"""Tests and benchmarks for the fused prefill kernel, flash-decode, and SSA (stratified PQ importance sampling).

    modal run ssa_modal.py::tiny_main        # L4, small random Inkling: kernels vs references, SSA vs reference
"""

import modal

from paged_modal import image as paged_image

app = modal.App("inkling-ssa")
vol = modal.Volume.from_name("inkling-workspace")
image = paged_image.add_local_python_source("inkling_static", "paged_modal", "inkling_fa", "inkling_ssa",
                                            "inkling_rows")
MODEL_PATH = "/vol/models/Inkling-2-Mini"


def _cmp(a, b) -> dict:
    import torch

    a, b = a.float().flatten(), b.float().flatten()
    return {"cos": round(float(torch.nn.functional.cosine_similarity(a, b, dim=0)), 6),
            "max_abs": float(f"{float((a - b).abs().max()):.3g}"),
            "rel": float(f"{float((a - b).norm() / b.norm().clamp(min=1e-12)):.3g}")}


def _attn_ref(q, k, v, table, sinks, q_off, kv_off, scale, window):
    """Exact Inkling attention in fp32. q [B,HQ,M,D], k/v [B,HKV,N,D], table [B,HQ,M,R], sinks [B,HQ,M]."""
    import torch

    B, HQ, MQ, D = q.shape
    N = k.shape[2]
    G = HQ // k.shape[1]
    R = table.shape[-1]
    kk, vv = k.repeat_interleave(G, 1).float(), v.repeat_interleave(G, 1).float()
    s = q.float() @ kk.transpose(-1, -2) * scale
    dist = ((torch.arange(MQ, device=q.device) + q_off)[:, None] - (torch.arange(N, device=q.device) + kv_off)[None])
    bias = table.float().gather(-1, dist.clamp(0, R - 1).expand(B, HQ, -1, -1)).masked_fill((dist < 0) | (dist >= R), 0)
    ok = dist >= 0
    if window:
        ok = ok & (dist < window)
    s = (s + bias).masked_fill(~ok, float("-inf"))
    p = torch.softmax(torch.cat([s, sinks.float()[..., None]], -1), -1)[..., :-1]
    return (p @ vv).transpose(1, 2)


def _tiny_model():
    import torch
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM

    torch.manual_seed(0)
    cfg = Inkling2LoopTextConfig(hidden_size=256, num_attention_heads=24, num_key_value_heads=8, head_dim=128,
                                 swa_num_attention_heads=24, swa_num_key_value_heads=8, swa_head_dim=128,
                                 sliding_window_size=1024, rel_extent=1024, d_rel=16,
                                 moe_intermediate_size=64, n_routed_experts=16, num_experts_per_tok=8,
                                 loop_gate_rank=16)
    cfg._attn_implementation = "eager"
    return Inkling2LoopForCausalLM._from_config(cfg, dtype=torch.bfloat16).cuda().eval(), cfg


@app.function(image=image, gpu="L4", cpu=4, memory=24576, volumes={"/vol": vol}, timeout=25 * 60)
def tiny_tests() -> dict:
    import copy
    import time
    import traceback

    import torch
    from transformers import StaticCache

    import inkling_fa as FA
    import inkling_rows as RW
    import inkling_ssa as SS
    import inkling_static as ist

    res: dict = {}
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)  # noqa: E731
    dev = "cuda"
    torch.manual_seed(0)

    # ---- T1: fused prefill kernel vs exact formula -------------------------------------------------------
    try:
        out = {}
        for name, (B, M, N, q_off, kv_off, win) in {
                "full_mid_chunk": (2, 300, 2000, 1700, 0, 0), "full_from0": (1, 777, 777, 0, 0, 0),
                "sliding_offset": (2, 500, 1523, 2200, 1177, 1024), "one_query": (1, 1, 4000, 3999, 0, 0)}.items():
            q = torch.randn(B, 24, M, 128, device=dev, dtype=torch.bfloat16) * 0.6
            k = torch.randn(B, 8, N, 128, device=dev, dtype=torch.bfloat16) * 0.6
            v = torch.randn(B, 8, N, 128, device=dev, dtype=torch.bfloat16)
            tab = torch.randn(B, 24, M, 1024, device=dev, dtype=torch.bfloat16) * 2
            sk = torch.randn(B, 24, M, device=dev, dtype=torch.bfloat16) * 2
            qv = q.transpose(1, 2).contiguous().transpose(1, 2)          # non-contiguous like HF's q
            got = FA.fa_prefill(qv, k, v, tab, sk, q_off, kv_off, 128 ** -0.5, window=win)
            out[name] = _cmp(got, _attn_ref(q, k, v, tab, sk, q_off, kv_off, 128 ** -0.5, win))
        res["T1_fa_prefill_vs_exact"] = out
    except Exception:  # noqa: BLE001
        res["T1_error"] = traceback.format_exc()[-2500:]
    log("T1", res.get("T1_fa_prefill_vs_exact", res.get("T1_error")))

    # ---- T2: flash-decode (device length) vs exact --------------------------------------------------------
    try:
        B, max_len, L = 3, 4096, 3001
        q = torch.randn(B, 24, 128, device=dev, dtype=torch.bfloat16) * 0.6
        k = torch.randn(B, 8, max_len, 128, device=dev, dtype=torch.bfloat16) * 0.6
        v = torch.randn(B, 8, max_len, 128, device=dev, dtype=torch.bfloat16)
        tab = torch.randn(B, 24, 1, 1024, device=dev, dtype=torch.bfloat16) * 2
        dist = (L - 1) - torch.arange(max_len, device=dev)
        pb = tab[:, :, 0].float().gather(-1, dist.clamp(0, 1023).expand(B, 24, -1)).masked_fill(
            (dist < 0) | (dist >= 1024), 0).to(torch.bfloat16)
        sk = torch.randn(B, 24, device=dev, dtype=torch.bfloat16)
        fd = FA.FlashDecode([0, 2], 24, 128, dev)
        o = torch.zeros(B, 24, 128, device=dev, dtype=torch.bfloat16)
        fd(q, k, v, pb, torch.tensor(L, device=dev), sk, o, 128 ** -0.5)
        ref = _attn_ref(q[:, :, None], k[:, :, :L], v[:, :, :L], tab, sk[:, :, None], L - 1, 0, 128 ** -0.5, 0)[:, 0]
        res["T2_flash_decode_vs_exact"] = {"row0": _cmp(o[0], ref[0]), "row2": _cmp(o[2], ref[2]),
                                           "row1_untouched": bool(o[1].abs().max() == 0)}
    except Exception:  # noqa: BLE001
        res["T2_error"] = traceback.format_exc()[-2500:]
    log("T2", res.get("T2_flash_decode_vs_exact", res.get("T2_error")))

    # ---- T3: SSA selection vs per-key reference; SSA output vs exact formula on its own selection --------
    try:
        L, max_len, Hkv, D = 6000, 6144, 8, 128
        centers = torch.randn(Hkv, 40, D, device=dev)
        keys = torch.zeros(1, Hkv, max_len, D, device=dev, dtype=torch.bfloat16)
        assign = torch.randint(0, 40, (Hkv, L), device=dev)
        keys[0, :, :L] = (centers.gather(1, assign[..., None].expand(-1, -1, D)) +
                          0.4 * torch.randn(Hkv, L, D, device=dev)).to(torch.bfloat16)
        vals = torch.randn(1, Hkv, max_len, D, device=dev, dtype=torch.bfloat16)
        ctx = SS.SSAContext(keys[0], L0=L - 200, iters=10, seed=1)
        row = SS.SSARow(ctx, 0)
        row.code_keys(keys[0], torch.arange(L - 200, L, device=dev))
        sel = SS.SSASelector(row, L_end_max=L + 64, n_keys=L, heavy_frac=0.02 / 3, sample_frac=0.04 / 3,
                             scaling=128 ** -0.5)
        pos = L - 1
        q = (torch.randn(24, D, device=dev) * 1.5).to(torch.bfloat16)
        band = torch.zeros(24, max_len, device=dev)
        dd = pos - torch.arange(max_len, device=dev)
        inb = (dd >= 0) & (dd < 1024)
        band[:, inb] = torch.randn(24, int(inb.sum()), device=dev) * 0.5

        def bias_of(idx):
            return band.gather(-1, idx[0])[None]

        posT = torch.tensor([pos], device=dev)
        ref = SS.reference_select(q.float(), pos, row.codes, ctx.C, band, 128 ** -0.5, sel.k_h, sel.m, L)
        fast = SS.SSAFast(row, L_end_max=L + 64, n_keys=L, heavy_frac=0.02 / 3, sample_frac=0.04 / 3,
                          scaling=128 ** -0.5)
        sinks = torch.randn(1, 24, device=dev)
        fused2 = SS.SSAFused(row, L_end_max=L + 64, n_keys=L, heavy_frac=0.02 / 3, sample_frac=0.04 / 3,
                             scaling=128 ** -0.5)
        fused3 = SS.SSAFusedV3(row, L_end_max=L + 64, n_keys=L, heavy_frac=0.02 / 3, sample_frac=0.04 / 3,
                               scaling=128 ** -0.5)
        fused4 = SS.SSAFusedV4(row, L_end_max=L + 64, n_keys=L, heavy_frac=0.02 / 3, sample_frac=0.04 / 3,
                               scaling=128 ** -0.5)
        att = SS.SSAAttention(1, 0, 1, Hkv, max_len, D, dev)
        variants = {
            "torch": (lambda: [t.view(24, -1) for t in sel.select(q[None], posT, bias_of)],
                      lambda ii, ww: att.run(q[None].contiguous(), keys, vals, ii.view(1, 24, -1), ww.view(1, 24, -1), sinks)[0]),
            "fused": (lambda: list(fast.select_fast(q[None], posT, pb=band)),
                      lambda ii, ww: fast.attend(q[None], keys, vals, 0, ii.contiguous(), ww.contiguous(), sinks)[0]),
            "fused2": (lambda: [t.long().clone() if t.dtype == torch.int32 else t.clone()
                                for t in fused2.select_fused(q[None], posT, pb=band)],
                       lambda ii, ww: fused2.attend(q[None], keys, vals, 0, ii.to(torch.int32).contiguous(),
                                                    ww.contiguous(), sinks)[0]),
            "fused3": (lambda: [t.long().clone() if t.dtype == torch.int32 else t.clone()
                                for t in fused3.select_fused(q[None], posT, pb=band)],
                       lambda ii, ww: fused3.attend(q[None], keys, vals, 0, ii.to(torch.int32).contiguous(),
                                                    ww.contiguous(), sinks)[0]),
            "fused4": (lambda: [t.long().clone() if t.dtype == torch.int32 else t.clone()
                                for t in fused4.select_fused(q[None], posT, pb=band)],
                       lambda ii, ww: fused4.attend(q[None], keys, vals, 0, ii.to(torch.int32).contiguous(),
                                                    ww.contiguous(), sinks)[0]),
        }
        kk = keys[0].repeat_interleave(3, 0).float()
        vv = vals[0].repeat_interleave(3, 0).float()
        sfull = (q.float()[:, None, :] * kk[:, :L]).sum(-1) * 128 ** -0.5 + band[:, :L]
        pfull = torch.softmax(torch.cat([sfull, sinks[0][:, None]], -1), -1)[:, :-1]
        dense = (pfull[..., None] * vv[:, :L]).sum(1)
        k_h, m0 = sel.k_h, SS.SINK + SS.WINDOW
        out_t3 = {"budget_S": sel.S, "k_heavy": k_h, "m_draws": sel.m}
        for vname, (select_fn, attend_fn) in variants.items():
            idx, w = select_fn()
            heavy_dev, pi_dev = [], []
            for h in range(24):
                mine = idx[h, m0:m0 + k_h]
                lg = ref[h]["logits"]
                heavy_dev.append(float((lg[mine].sort().values - lg[ref[h]["heavy"]].sort().values).abs().max()))
                ws, ks = w[h, m0 + k_h:], idx[h, m0 + k_h:]
                live = ws > 0
                pi_m = torch.exp(band[h, ks[live]]) / ws[live]
                pi_r = (-torch.expm1(sel.m * torch.log1p(-ref[h]["p"][ks[live]]))).clamp(min=SS.MIN_PI)
                pi_dev.append(float(((pi_m - pi_r).abs() / pi_r).max()))
            reps = 300
            freq = torch.zeros(3, max_len, device=dev)
            for _ in range(reps):
                ii, ww = select_fn()
                for h in range(3):
                    s_idx, s_w = ii[h, m0 + k_h:], ww[h, m0 + k_h:]
                    freq[h].index_add_(0, s_idx[s_w > 0], torch.ones(int((s_w > 0).sum()), device=dev))
            freq /= reps
            fr = []
            for h in range(3):
                pi_r = -torch.expm1(sel.m * torch.log1p(-ref[h]["p"]))
                big = pi_r > 0.05
                fr.append({"n": int(big.sum()), "mae": round(float((freq[h, :L][big] - pi_r[big]).abs().mean()), 4),
                           "sd": round(float((pi_r[big] * (1 - pi_r[big]) / reps).sqrt().mean()), 4),
                           "mass_err": round(float((freq[h, :L].sum() - pi_r.sum()).abs() / pi_r.sum()), 4)})
            o = attend_fn(idx, w)
            ii = idx
            sx = (q.float()[:, None, :] * kk.gather(1, ii[..., None].expand(-1, -1, D))).sum(-1) * 128 ** -0.5
            mx = sx.max(-1, keepdim=True).values
            ex = w * torch.exp(sx - mx)
            num = (ex[..., None] * vv.gather(1, ii[..., None].expand(-1, -1, D))).sum(1)
            den = ex.sum(-1, keepdim=True) + torch.exp(sinks[0][:, None] - mx)
            acc = torch.zeros_like(dense)
            for _ in range(100):
                acc += attend_fn(*select_fn()).float()
            out_t3[vname] = {"heavy_logit_maxdev": max(heavy_dev), "pi_rel_err_max": max(pi_dev),
                             "inclusion_freq_vs_pi": fr, "kernel_vs_formula": _cmp(o, num / den),
                             "single_draw_vs_dense": _cmp(o, dense), "mean100_vs_dense": _cmp(acc / 100, dense)}
        res["T3_ssa"] = out_t3
    except Exception:  # noqa: BLE001
        res["T3_error"] = traceback.format_exc()[-3000:]
    log("T3", res.get("T3_ssa", res.get("T3_error")))

    # ---- T4/T5: tiny Inkling end to end -----------------------------------------------------------------------
    try:
        model, cfg = _tiny_model()
        FA.install(model)
        ist.patch_static_sliding_update()
        L0, Qn, steps = 3000, 40, 8
        ids = torch.randint(0, 200000, (1, L0 + Qn), device=dev)
        max_len = ((L0 + Qn + steps + 64 + 15) // 16) * 16

        def use_eager():
            ist.install_attention(model)
            FA.STATE["lazy"] = False

        def prefill(chunk, mode):
            c = StaticCache(config=cfg, max_cache_len=max_len)
            if mode == "eager":
                use_eager()
            else:
                FA.set_mode(model, "prefill")
            with torch.no_grad():
                for s in range(0, L0, chunk):
                    lg = model(ids[:, s:min(s + chunk, L0)], past_key_values=c, use_cache=True, logits_to_keep=1).logits
            return c, lg[0, -1]

        def eager_decode(c, first, n):
            use_eager()
            outs, toks, nxt = [], [], first
            with torch.no_grad():
                for _ in range(n):
                    lg = model(nxt, past_key_values=c, use_cache=True, logits_to_keep=1).logits[:, -1]
                    outs.append(lg[0].float())
                    nxt = lg.argmax(-1, keepdim=True)
                    toks.append(int(nxt))
            return outs, toks

        ist.set_experts(model, "grouped_mm")
        c_e, lg_e = prefill(512, "eager")
        c_f, lg_f = prefill(700, "fa")
        res["T4_prefill_fa_vs_eager_last_logits"] = _cmp(lg_f, lg_e)
        # question tokens + decode, B=1 reference: FA question pass, then eager decode
        ref_c = copy.deepcopy(c_f)
        FA.set_mode(model, "prefill")
        with torch.no_grad():
            lq = model(ids[:, L0:], past_key_values=ref_c, use_cache=True, logits_to_keep=1).logits[:, -1]
        first = lq.argmax(-1, keepdim=True)
        ref_logits, ref_toks = eager_decode(ref_c, first, steps)
        # eager question pass (to check the FA multi-token question path)
        e_c = copy.deepcopy(c_e)
        use_eager()
        with torch.no_grad():
            lqe = model(ids[:, L0:], past_key_values=e_c, use_cache=True, logits_to_keep=1).logits[:, -1]
        res["T4_question_fa_vs_eager"] = _cmp(lq[0], lqe[0])
        res["T4_cache_layout"] = RW.describe_cache(c_f)

        # rows: dense, exact-sparse (heavy=100%), ssa 2%, pqc 2%
        cR = RW.repeat_cache(c_f, 4)
        rows = RW.Rows(cR, ["dense", "full", "ssa", "pqc"], L0=L0, n_prompt=L0 + Qn, max_new=steps + 8,
                       scaling=128 ** -0.5)
        rows.install()
        FA.set_mode(model, "prefill")
        with torch.no_grad():
            lr = model(ids[:, L0:].expand(4, -1), past_key_values=cR, use_cache=True, logits_to_keep=1).logits[:, -1]
        res["T5_question_rows_vs_ref"] = {"dense": _cmp(lr[0], lq[0]), "full": _cmp(lr[1], lq[0]),
                                          "ssa2%": _cmp(lr[2], lq[0]), "pqc2%": _cmp(lr[3], lq[0])}
        FA.set_mode(model, "decode")
        ist.set_experts(model, "batched_mm")
        firstR = first.expand(4, 1).contiguous()
        g = RW.GraphedRowsDecode(model, cR, firstR, L0 + Qn, steps + 2, temperature=0.0)
        g.run_until([-1], burst=steps + 2)
        toks = g.tokens[:, 1:steps + 1].tolist()
        res["T5_graph_tokens"] = {"ref": ref_toks, "dense": toks[0], "full": toks[1], "ssa": toks[2], "pqc": toks[3]}
        res["T5_budget_tokens"] = rows.budget_tokens
        RW.Rows.uninstall()
        del g, rows, cR, c_e, e_c
        torch.cuda.empty_cache()
        cR3 = RW.repeat_cache(c_f, 3)
        rows3 = RW.Rows(cR3, ["dense", "full", "ssa"], L0=L0, n_prompt=L0 + Qn, max_new=steps + 8,
                        scaling=128 ** -0.5, fast="v4")
        rows3.install()
        FA.set_mode(model, "prefill")
        ist.set_experts(model, "grouped_mm")
        with torch.no_grad():
            lr3 = model(ids[:, L0:].expand(3, -1), past_key_values=cR3, use_cache=True, logits_to_keep=1).logits[:, -1]
        res["T5_v3_question_rows_vs_ref"] = {"dense": _cmp(lr3[0], lq[0]), "full": _cmp(lr3[1], lq[0]),
                                             "ssa2%": _cmp(lr3[2], lq[0])}
        FA.set_mode(model, "decode")
        ist.set_experts(model, "batched_mm")
        g3 = RW.GraphedRowsDecode(model, cR3, first.expand(3, 1).contiguous(), L0 + Qn, steps + 2, temperature=0.0)
        g3.run_until([-1], burst=steps + 2)
        t3 = g3.tokens[:, 1:steps + 1].tolist()
        res["T5_v3_graph_tokens"] = {"ref": ref_toks, "dense": t3[0], "full": t3[1], "ssa": t3[2]}
        RW.Rows.uninstall()
    except Exception:  # noqa: BLE001
        res["T45_error"] = traceback.format_exc()[-3000:]
    log("T4/T5", {k: v for k, v in res.items() if k.startswith(("T4", "T5")) and k != "T4_cache_layout"})
    res["seconds"] = round(time.time() - t0, 1)
    return res


@app.local_entrypoint()
def tiny_main():
    import json

    print("RESULT_JSON " + json.dumps(tiny_tests.remote(), default=str))
