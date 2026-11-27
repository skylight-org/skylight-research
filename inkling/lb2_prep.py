"""LongBench v2 prep on a CPU container: download, render prompts, tokenize, store ids on the volume.

Prompt = official 0-shot template (THUDM/LongBench prompts/0shot.txt), middle-truncated like the official
`pred.py` (first max_len//2 + last max_len//2 tokens of the user prompt). The prompt is split into
  context  = chat prefix + "Please read ... <text> DOC </text>\n\n"   (prefilled densely in every config)
  question = "What is the correct answer ... " + chat suffix           (sparse in the sparse configs)
which is the sparse-attention-hub / SSA-paper protocol. Both halves are tokenized separately and
concatenated, and every config uses exactly the same ids.

    modal run lb2_prep.py --max-len 120000 --effort high
"""

import modal

from paged_modal import image as _img

app = modal.App("inkling-lb2-prep")
vol = modal.Volume.from_name("inkling-workspace")
image = _img.add_local_python_source("paged_modal")
MODEL_PATH = "/vol/models/Inkling-2-Mini"

TEMPLATE_CTX = "Please read the following text and answer the question below.\n\n<text>\n$DOC$\n</text>\n\n"
TEMPLATE_Q = ("What is the correct answer to this question: $Q$\nChoices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n"
              "(D) $C_D$\n\nFormat your response as follows: \"The correct answer is (insert answer here)\".")


@app.function(image=image, volumes={"/vol": vol}, cpu=8, memory=32768, timeout=40 * 60,
              secrets=[modal.Secret.from_name("hf-token")])
def prep(max_len: int = 120000, effort: str = "high") -> dict:
    import json
    import os
    import time

    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    ds = load_dataset("THUDM/LongBench-v2", split="train", cache_dir="/vol/data/hf_cache")
    out_dir = f"/vol/data/lb2/ml{max_len}_{effort}"
    os.makedirs(out_dir, exist_ok=True)
    marker = "\u0000SPLIT\u0000"
    meta = []
    for i, it in enumerate(ds):
        doc = it["context"].strip()
        q = (TEMPLATE_Q.replace("$Q$", it["question"].strip()).replace("$C_A$", it["choice_A"].strip())
             .replace("$C_B$", it["choice_B"].strip()).replace("$C_C$", it["choice_C"].strip())
             .replace("$C_D$", it["choice_D"].strip()))
        user = TEMPLATE_CTX.replace("$DOC$", doc) + q
        full_user_ids = tok.encode(user, add_special_tokens=False)
        n_raw = len(full_user_ids)
        if n_raw > max_len:   # official middle truncation of the user prompt, then decode back to text
            user = tok.decode(full_user_ids[:max_len // 2] + full_user_ids[-max_len // 2:],
                              skip_special_tokens=True)
        cut = user.rfind("What is the correct answer to this question:")
        assert cut > 0, it["_id"]
        rendered = tok.apply_chat_template([{"role": "user", "content": user[:cut] + marker + user[cut:]}],
                                           tokenize=False, add_generation_prompt=True, reasoning_effort=effort)
        a, b = rendered.split(marker)
        ids_a = tok.encode(a, add_special_tokens=False)
        ids_b = tok.encode(b, add_special_tokens=False)
        np.save(f"{out_dir}/{it['_id']}.npy", np.array(ids_a + ids_b, dtype=np.int32))
        meta.append({"_id": it["_id"], "domain": it["domain"], "sub_domain": it["sub_domain"],
                     "difficulty": it["difficulty"], "length": it["length"], "answer": it["answer"],
                     "n_raw": n_raw, "n_ctx": len(ids_a), "n_q": len(ids_b)})
        if i == 0:
            example = {"head": rendered[:300], "tail": rendered[-700:]}
        if i % 50 == 0:
            print(f"[{time.time() - t0:6.1f}s] {i} n_raw={n_raw} n_ctx={len(ids_a)} n_q={len(ids_b)}", flush=True)
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f)
    vol.commit()

    raw = np.array([m["n_raw"] for m in meta])
    tot = np.array([m["n_ctx"] + m["n_q"] for m in meta])
    stats = {"n": len(meta), "tokens_total_truncated": int(tot.sum()), "mean_truncated": float(tot.mean()),
             "q_tokens_mean": float(np.mean([m["n_q"] for m in meta])),
             "q_tokens_max": int(max(m["n_q"] for m in meta)),
             "raw_percentiles": {p: int(np.percentile(raw, p)) for p in (10, 25, 50, 75, 90, 99)},
             "by_length": {}, "if_truncated_at": {}}
    for L in ("short", "medium", "long"):
        sel = [m for m in meta if m["length"] == L]
        stats["by_length"][L] = {"n": len(sel), "mean_raw": int(np.mean([m["n_raw"] for m in sel]))}
    for cap in (32000, 64000, 120000, 128000, 256000, 512000, 1000000):
        c = np.minimum(raw, cap)
        stats["if_truncated_at"][cap] = {"total_tokens": int(c.sum()), "sum_sq_1e12": float((c.astype(float) ** 2).sum() / 1e12),
                                         "n_truncated": int((raw > cap).sum())}
    stats["example"] = example
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


@app.local_entrypoint()
def main(max_len: int = 120000, effort: str = "high"):
    import json

    print("RESULT_JSON " + json.dumps(prep.remote(max_len, effort)))
