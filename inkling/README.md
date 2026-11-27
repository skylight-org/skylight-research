# Inkling 2 Mini: sparse attention experiments

Running thinkingmachines/Inkling-2-Mini on Modal (A100-80GB) with sparse-attention-hub and skylight_kernels.

## Main pieces
- `inkling_sah.py` - sparse-attention-hub adapter for Inkling (looped layers, learned position bias, per-token sinks).
- `inkling_fa.py` - fused Triton attention: long-context prefill (bias band + sinks) and graph-safe flash-decode.
- `inkling_ssa.py` - stratified sparse attention (PQ importance sampling, from SSA-Paper-Code). Fastest version: `SSAFusedV3`.
- `inkling_rows.py` - dense / SSA / PQCache rows sharing one prefill, decoded together in one CUDA graph.
- `inkling_static.py`, `inkling_paged.py`, `inkling_quest_triton.py`, `inkling_sparse_op.py` - earlier Quest page-selection kernels and CUDA-graph decode.
- `lb2_prep.py`, `lb2_modal.py` - LongBench v2 prep, runs, benchmarks and analysis.
- `graph_modal.py` - RULER-32k runs; `ssa_modal.py` - kernel and SSA correctness tests; `kernels_bench.py` - skylight_kernels benchmarks.
- `patch_gs3.py` - adds GQA group size 3 to skylight_kernels at build time.

## Results (A100-80GB)
SSA operator, one query per full-attention visit, vs dense FlashInfer decode:

| Context | Dense | SSA 2% | Speedup |
|---|---|---|---|
| 32k | 97 us | 98 us | 1.0x |
| 131k | 343 us | 139 us | 2.5x |
| 512k | 1300 us | 250 us | 5.2x |
| 1M | 2519 us | 396 us | 6.4x |

LongBench v2, direct answer (no reasoning), 418 of 503 questions, 120k-token truncation:

| Config | Accuracy |
|---|---|
| Dense | 47.8% |
| SSA 2% | 47.4% |
| PQCache 2% | 45.9% |

RULER-32k (104 samples): dense 95.3%, Quest 5% 88.9%, Quest 2% 82.5%, Quest 1% 75.0%.
