"""Patch a skylight_kernels checkout to support GQA group_size=3 (Inkling 2 Mini: 24 q / 8 kv heads).

Applied only to the build copy inside the Modal image; the repo itself is untouched.
    python patch_gs3.py /path/to/skylight_kernels
"""

import pathlib
import sys

root = pathlib.Path(sys.argv[1])

# 1. score kernels: add `case 3` to DISPATCH_GROUP by cloning the `case 2` block.
for rel in ("sparse_oracle_topk_optimized/csrc/oracle_topk_score_kernel.cu",
            "sparse_oracle_topk_sink_local_optimized/csrc/oracle_topk_score_kernel_compact.cu"):
    p = root / rel
    s = p.read_text()
    start = s.index("    case 2: {")
    end = s.index("    case 4: {", start)
    block = s[start:end]
    assert "GS = 2;" in block and "case 3:" not in s, rel
    s = s[:end] + block.replace("case 2:", "case 3:").replace("GS = 2;", "GS = 3;") + s[end:]
    p.write_text(s)
    print("patched", rel)

# 2. python plan() allow-list (the FlashInfer dense dispatch already supports 3).
p = root / "original_optimized/batch_decode_with_paged_kv_cache_wrapper.py"
s = p.read_text()
old = "if group_size not in (1, 2, 4, 6, 8, 16):"
assert old in s
s = s.replace(old, "if group_size not in (1, 2, 3, 4, 6, 8, 16):")
p.write_text(s)
print("patched", p.relative_to(root))
