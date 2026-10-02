# KV cache quantization (NVFP4)

One flag on `ModelAdapterHF` picks what the KV cache is stored as:

```python
adapter = ModelAdapterHF(
    model_name="Qwen/Qwen3-4B",
    sparse_attention_config=sparse_config,   # or None, orthogonal to this
    model_kwargs={"dtype": torch.bfloat16},
    device="cuda",
    quantize_kv_cache="fake_nvfp4",          # "fake_nvfp4" | "nvfp4" | False
)
```

| `quantize_kv_cache` | Storage | Needs | Use it to |
|---|---|---|---|
| `False` | model dtype, HF's default cache | — | baseline |
| `"fake_nvfp4"` | ordinary dtype tensors rounded to the NVFP4 grid, pure torch (**the simulator**) | nothing beyond torch | measure the accuracy cost anywhere, including CPU |
| `"nvfp4"` | packed 4-bit codes + E4M3 block scales + FP32 tensor scale, via `nvidia-modelopt` | NVIDIA CUDA GPU, `nvidia-modelopt` | actually save memory |

Both backends impose NVFP4's accuracy loss; only `"nvfp4"` makes the cache
smaller. The simulator answers "what does NVFP4 cost this model?" on any
hardware; switch to `"nvfp4"` to collect the memory win.

`quantize_kv_cache=True` is rejected rather than defaulted: the backends differ
in what they need and what they give back. The value is validated when the
adapter is constructed, so a typo in a benchmark sweep fails before any model runs.

## NVFP4 in one paragraph

NVIDIA's 4-bit float format (Blackwell): each value is a 4-bit **E2M1** code
(`0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6`), every **16** consecutive values share
an 8-bit **E4M3** block scale, and the whole tensor shares one **FP32** scale:
`value ≈ code × block_scale × tensor_scale`, about **4.5 bits per element**.

## Supported models

Models whose decoder layers are **all full attention** (e.g. Llama, Qwen2, Qwen3):
every layer keeps a token-indexed key/value cache, and every layer is quantized.
Models with sliding-window, chunked or linear-attention layers are rejected
(`layers.py`): those layers have different cache layouts. Mistral works only when
its config has no sliding window (`sliding_window: null`); HF's default Mistral
config uses 4096 and is rejected.

## Cache options

Everything except the backend choice goes in `kv_quantization_kwargs`:

| Option | Default | Meaning |
|---|---|---|
| `block_size` | `16` | Elements sharing one E4M3 scale. 16 is the only value the NVFP4 format defines. |
| `residual_length` | `128` | Trailing tokens kept unquantized. |
| `axis_key` | `-1` | `-1` blocks over `head_dim` (per token); `0` blocks over tokens (per channel, **KIVI-style**, see below). |
| `axis_value` | `-1` | Same, for values. |
| `quantize_prefill` | `False` | Return quantized KV from the prefill call too. Off = prefill attends over exact KV (serving behaviour); on = every token reads quantized KV (single-pass perplexity). |

## KIVI-style layout (`axis_key=0`)

Following KIVI (Liu et al., 2024): **keys per channel, values per token, and
the most recent tokens kept exact**, here with NVFP4 numerics.

```python
adapter = ModelAdapterHF(..., quantize_kv_cache="fake_nvfp4",   # or "nvfp4" (real kernel)
                         kv_quantization_kwargs={"axis_key": 0, "residual_length": 128})
```

- **Keys, per channel:** keys are transposed so each NVFP4 block covers **16
  tokens of one channel**. A loud channel then gets its own scale instead of
  setting the scale for 15 quiet neighbours.
- **Values, per token:** blocks of 16 channels of one token (`axis_value=-1`).
- **Recent tokens exact:** the residual buffer (`residual_length`).
- **Simulated or real:** the layout is the same; the backend picks the kernel
  underneath (`fake_quantize_nvfp4` or modelopt's `NVFP4QTensor`).
- Token counts that aren't multiples of 16 are zero-padded for blocking and
  trimmed after (zeros never change a block's max).

On synthetic keys (64 channels, 3 with a large near-constant value):

| Channels | `axis_key=-1` (per token) | `axis_key=0` (per channel) |
|---|---|---|
| quiet (61) | 0.467 | 0.096 |
| outlier (3) | 0.024 | 0.078 |

On **real Qwen3-4B** (WikiText-2, first 20 × 2048-token windows, bf16 = 13.276):

| `axis_key` / `axis_value` | Perplexity | vs bf16 |
|---|---|---|
| per token / per token (default) | 13.727 | +3.4% |
| **per channel (KIVI-style)** / per token | 13.440 | **+1.2%** |
| per token / per channel | 13.546 | +2.0% |
| per channel / per channel | 13.263 | −0.1% |

Per-channel blocking **lowers perplexity** on Qwen3-4B, even though the relative
key reconstruction error is higher per channel (0.105 vs 0.091, averaged over all
36 layers). An overall Frobenius error doesn't track what attention cares about,
so judge layouts by perplexity. (The real `nvfp4` backend gives exactly the
same 13.727 and 13.440 for the first two rows.)

## Simulator numerics (`nvfp4.py`)

`fake_quantize_nvfp4` / `quantize_nvfp4` implement the format in pure torch:

- E2M1 and E4M3 rounding: round half to even (`q_e2m1(5.0) == 4.0`), saturate
  instead of overflowing, E4M3 max 448 (not 480), subnormals handled.
- Scale arithmetic in float32 (float64 flips scales near E4M3 ties).
- Exponents are read from the float32 bits (no `log2`/`exp2`), so the rounding
  step itself is exact on every device (CPU, NVIDIA, AMD, Apple).
- Block scales follow nvidia-modelopt's rule (what the real backend uses):
  `block_amax / (6 · tensor_scale)`, floored at 2⁻⁹ (the smallest E4M3
  subnormal), codes by true division. On a GPU the output is bit-identical to
  the real `nvfp4` backend.

## Verified on a Blackwell GPU (RTX PRO 6000, CC 12.0)

`tests/integration/kv_quantization/test_nvfp4_simulator_heavy.py`: **28/28 passing**, model Qwen/Qwen3-4B.

| Check | Result |
|---|---|
| E2M1/E4M3 rounding on every finite bf16 value + 16.8M random fp32 | 0 mismatches vs an independent reference and PyTorch `float8_e4m3fn` |
| Simulator vs modelopt: 12 stress inputs × 4.2M elements + real Qwen3-4B K/V (36 layers × 3 layouts = 108 tensors) | **0 mismatches** |
| 128 greedy tokens: simulator vs real | identical tokens, logit difference 0.0 (both layouts) |
| WikiText-2 perplexity: simulator vs real | **identical** (13.726882 / 13.440394) |
| KIVI layout on real `nvfp4`: 1/15/16/17/100/4097 tokens, 300 decode steps | all correct |
| 32,768-token prompt, peak GPU memory above weights | bf16 13.93 GiB → real `nvfp4` **11.06 GiB** (−2.87 GiB) |

Limits: the simulator reproduces NVFP4 *storage* (quantize → dequantize), not
FP4 tensor-core compute (attention still runs in bf16). It's bit-exact with
modelopt on a GPU; on a CPU a value exactly on a rounding tie can occasionally
round the other way (CUDA divides by a constant via its reciprocal).

## Measuring accuracy

`evaluation.py`: `perplexity(model, ids, make_cache, seq_len=2048)` evaluates
non-overlapping windows with a fresh cache each (use `quantize_prefill=True`);
`wikitext2_test_ids(tokenizer)` loads the WikiText-2 test split;
`kv_reconstruction_error(...)` gives per-layer relative KV error.

## How it hooks in

`ModelAdapterHF` is the only place a `Cache` object is constructed, so
`adapters/huggingface.py` is the only file outside this folder that changed.
When the flag names a backend, `_create_kv_cache()` returns that backend's
cache, which is passed as `past_key_values` to the context prefill. HF then
calls `cache.update(...)` in every attention layer (the cache subclasses HF's
`Cache` / `QuantizedLayer`) and carries the same object through generation. A
fresh cache is built per question.

Sparse attention works unchanged on top: both caches dequantize on read, so maskers
see ordinary dense key/value tensors (carrying the quantization error) and need no
changes. Sparse attention doesn't evict anything, so the cache still stores every
token; masks only choose which cached tokens each query reads, and the memory saving
comes from `"nvfp4"` alone. Tested on the GPU with sink + local maskers for all three
flag values (`test_adapter_with_sparse_attention`).

## Requirements

- `transformers>=5.0` for the layer-based cache API (`QuantizedLayer`); tested with 5.17.
- `"nvfp4"` only: `nvidia-modelopt` and an NVIDIA CUDA GPU (tested with
  modelopt 0.47). Nothing imports modelopt unless that backend is selected.
  Note: installing modelopt can upgrade torch; check `torch.__version__` afterwards.
- `datasets` for `wikitext2_test_ids`.

## Tests

The tests live in the repo's `tests/` folder, so a plain `pytest` (and CI) runs them:

```bash
pytest tests/unit/kv_quantization/                  # unit tests; the real-NVFP4 ones skip without CUDA + modelopt
pytest tests/integration/kv_quantization/ -v -s     # heavy suite (CUDA; modelopt + model for some parts)
python -m sparse_attention_hub.kv_quantization.WIP --backend fake_nvfp4   # end-to-end smoke script
```

| File | Needs |
|---|---|
| `tests/unit/kv_quantization/test_fake_nvfp4.py` | torch only: element grid, E4M3 saturation, tie-breaking, underflow clamp, dynamic range |
| `tests/unit/kv_quantization/test_fake_nvfp4_cache.py` | torch only: cache semantics, axis choice, backend validation and dispatch |
| `tests/unit/kv_quantization/test_adapter_flag.py` | torch only: each flag value reaches the forward pass |
| `tests/unit/kv_quantization/test_nvfp4_cache.py` | modelopt + CUDA (skips otherwise): real backend + agreement with the simulator |
| `tests/integration/kv_quantization/test_nvfp4_simulator_heavy.py` | CUDA (+ modelopt, + Qwen3-4B for real-model parts): heavy proof the simulator matches real NVFP4, incl. the KIVI-style layout. Writes a JSON report to `$KVQ_REPORT_PATH`. |
