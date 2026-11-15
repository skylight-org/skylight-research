# KV quantization experiments

Run from the repository root with the project's Python environment (Python 3.10+):

```bash
python -m pytest -q tests/integration/test_kv_quantization_sliding.py
```

Transformers is pinned to 4.57.1 because this integration uses its attention and
mask registries. The reference CPU checks use PyTorch 2.8.0, with CI jobs for Python
3.10 and 3.12. Those jobs run the category commands below with offline tiny models;
they do not certify other runtime combinations or GPU kernels.

This runs tiny randomly initialized Hugging Face Mistral models on CPU, with local
tokenizers and no model/data downloads. It checks prefill capture, fresh tokens at
buffer boundaries, native sliding caches, mask alignment, fake/packed logit parity
and cleanup. These are numerical/integration checks, not a language-quality benchmark.
The [KIVI implementation note](implementations/README.md#kivi)
describes the pinned reference codec, numerical checks and validation limits.
The [QuaRot KV implementation note](implementations/README.md#quarot-kv) describes
its accuracy-profile quantizer and original-coordinate reconstruction.
The [TurboQuant KV implementation note](implementations/README.md#turboquant-kv) describes
its MSE and QJL-corrected prod algorithms, bit budget and validation limits.
The optional after-attention hook remains deferred.

The actual call flow is:

```text
Context prefill: capture all K/V -> store -> native attention on original or prepared K/V
HF decode: RoPE -> native cache.update -> attention registry callback
  adapters/huggingface.py: custom_attention_callable (adds layer_idx, sparse_meta_data)
    research_attention/base.py: ResearchAttention.custom_attention
      sparse_attention/base.py: SparseAttention.quantize_kv (returns K, V unchanged when off)
        kv_quantization/: method.process -> shared _process_kv_history
          default/KIVI: snapshot previous reconstruction + exact arriving K/V
          cache/store.py: append the query-count suffix, independently for each layer
          policies/cache/: decide which K/V tokens leave the local window
          transforms/: forward transform
          policies/precision/: precision plan
          quantizers/: encode + pack, with per-group parameters
          QuaRot/TurboQuant: reconstruct history including arriving tokens after compression
          return the method-selected attention tensors
      align HF's suffix mask with the returned full history
      maskers + attention on the reconstructed K, V (every layer, before layer gating)
  adapters/huggingface.py: per-question finally -> SparseAttention.clear_request_state
```

## Files and extension points

Each component family follows the repository's `base.py` / `implementations/`
pattern. An implementation and its configuration live in the same file.
Package `__init__.py` files are omitted from this tree:

```text
kv_quantization/
├── base.py                         # KVQuantizationConfig, KVQuantization
├── attention_mask.py               # Native visibility on our retained history
├── numerics.py                     # Finite checks and saturating output conversion
├── registry.py
├── context.py                      # AttentionContext, tracing helper
├── implementations/                # Complete methods and their configs
│   ├── README.md                   # Method notes, references and implementation status
│   ├── kivi.py
│   ├── quarot_kv.py
│   ├── turboquant_kv.py
│   ├── kvquant_nuq.py
│   ├── kitty.py
│   └── nvfp4_kv.py
├── transforms/
│   ├── base.py
│   └── implementations/
│       ├── identity.py
│       ├── hadamard.py
│       ├── random_hadamard.py
│       ├── random_rotation.py
│       └── learned_rotation.py   # stub
├── quantizers/
│   ├── base.py
│   └── implementations/
│       ├── integer.py
│       ├── kivi.py                 # Released KIVI numerical rules
│       ├── quarot.py               # Released QuaRot accuracy-path numerics
│       ├── turboquant.py           # Sphere MSE codebooks and QJL-corrected prod
│       ├── fp8.py
│       ├── nvfp4.py
│       ├── fake_nvfp4.py
│       ├── mxfp.py
│       └── non_uniform.py
├── policies/
│   ├── precision/
│   │   ├── base.py
│   │   └── implementations/
│   │       ├── fixed.py
│   │       ├── channel_boost.py
│   │       └── outlier_token.py
│   └── cache/
│       ├── base.py
│       └── implementations/
│           ├── default.py          # DefaultCachePolicy
│           ├── sink_and_local.py   # SinkAndLocalCachePolicy, KittyCachePolicy
│           ├── immediate.py        # SinkOnlyCachePolicy, FullQuantizationCachePolicy
│           └── local_only.py       # LocalOnlyCachePolicy
├── cache/
│   ├── store.py                    # QuantizedKVStore
│   ├── state.py                    # LayerKVState, TensorCacheState
│   └── block.py                    # QuantizedTensor, StoredChunk
├── metrics/
│   └── quant_error.py
└── README.md
```

| Location | Main classes / responsibility |
|---|---|
| `base.py` | `KVQuantizationConfig` and `KVQuantization`: select and orchestrate components |
| `registry.py` | Independent config-type factories; implementations use their registration decorators |
| `context.py` | `AttentionContext` and opt-in tracing; metadata only, no KV storage |
| `transforms/` | `KVTransform`; working Identity, Hadamard, RandomHadamard and RandomRotation; LearnedRotation scaffolded |
| `quantizers/` | `KVQuantizer`, `QuantGranularity`; working INT, KIVI, Kitty, QuaRot, TurboQuant MSE/prod, FP8, FakeNVFP4, MXFP and nonuniform reference codecs; native NVFP4 scaffold |
| `policies/precision/` | `PrecisionPolicy`, `PrecisionPlan`; fixed bits, mixed-channel bits and optional exact-token preservation |
| `policies/cache/` | `CachePolicy`, `RegionConfig` (sink/local size and quantizer), and six working policies (Default, SinkAndLocal, Kitty, SinkOnly, FullQuantization, LocalOnly) |
| `cache/` | `QuantizedKVStore`, per-layer state, encoded tensor records and chunks |
| `metrics/quant_error.py` | Optional reconstruction errors on quantized regions; stored tensor bytes |
| `implementations/` | Implemented KIVI/Kitty/QuaRotKV/TurboQuantKV recipes; KVQuantNUQ and NVFP4KV declarations |

For example, `implementations/kivi.py` assembles the complete method from
components, selecting `kivi_int_quantizer()` by default and
`SinkAndLocalCachePolicyConfig` for buffering. Methods share
components instead of defining new ones each time.

Base configurations live beside their contracts in each family's `base.py`.
Concrete configurations live beside their implementation: `IntQuantizerConfig`
and `IntQuantizer` are both in `quantizers/implementations/integer.py`.
There is no central file containing every method's configuration.

Existing package-level imports remain available:

```python
from sparse_attention_hub.kv_quantization import (
    IntQuantizerConfig,
    KVQuantizationConfig,
    QuantizedKVStore,
)
```

To add a component, put its class and config in a new implementation file, apply
the appropriate registry decorator, and import/export it in the package-level
`__init__.py` so registration runs. Shared base classes and other implementations
normally do not need changes. Unit tests live directly in
`tests/unit/kv_quantization/test_*.py`; the local-model integration tests are
`tests/integration/test_kv_quantization_flow.py` and `test_kv_quantization_sliding.py`.

`SparseAttentionConfig.kv_quantization_config` turns it on. `SparseAttention` builds
the coordinator once and exposes `quantize_kv`, which each attention subclass calls
first in its `custom_attention` (`ResearchAttention` does; efficient attention is not
run by the adapter yet). `after_attention` is an unsupported extension point that
current policies do not require; F09 is deferred until a selected method needs it.
The config field is keyword-only to preserve required subclass fields, requiring
Python 3.10 or newer. Existing config calls without quantization keep the OFF path.

## What is implemented

### KIVI's portable reference path

```python
from sparse_attention_hub.kv_quantization import KIVIConfig

kv_config = KIVIConfig.with_settings(
    bits=2, group_size=32, residual_length=128, store_codes=True
)
```

Pass this as `ResearchAttentionConfig(masker_configs=[],
kv_quantization_config=kv_config)` for quantization-only experiments. The recipe
uses `IntQuantizer` with `kivi_int_quantizer()`, which reproduces
the released code at `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6` (see the next
section). Minima, ranges, scale division and rounding use the input dtype, and
scale/minimum tensors keep that dtype. Exported configuration records the
preset's settings (`compute_dtype="input"`, `parameter_storage="compute"`, ...).
Inherited `scale_dtype`/`offset_dtype` fields are unused by this preset; use a
generic `IntQuantizerConfig` for experiments with independently chosen
parameter dtypes.

### One integer quantizer with the papers' arithmetic as settings

`IntQuantizerConfig` keeps Skylight's generic arithmetic by default (FP32
statistics, conservatively rounded FP16 parameters, record format
`uniform_int_v1`). With `parameter_storage="compute"` its other settings select
the operation order and dtype of a reference implementation (record format
`uniform_int_v2`). `IntQuantizerConfig` itself holds no method presets: each
recipe file has a helper that sets them exactly as the authors' code does and
returns a plain `IntQuantizerConfig`:

```python
from sparse_attention_hub.kv_quantization import (
    kitty_int_quantizer,   # implementations/kitty.py
    kivi_int_quantizer,    # implementations/kivi.py
    quarot_int_quantizer,  # implementations/quarot_kv.py
)

kivi_int_quantizer(bits=2, granularity=PER_CHANNEL, group_size=32)   # KIVI new_pack.py
kitty_int_quantizer(bits=2, granularity=PER_CHANNEL, group_size=128) # Kitty kitty_sim (+ ChannelBoost)
quarot_int_quantizer("key_head")      # QuaRot ActQuantizer, per-head keys (FP32)
quarot_int_quantizer("value_group")   # QuaRot ActQuantizer, v_proj values
```

| Setting | Generic | `kivi_int_quantizer()` | `kitty_int_quantizer()` | `quarot_int_quantizer()` |
|---|---|---|---|---|
| `compute_dtype` | fp32 | input | input | key_head fp32, value_group input |
| `parameter_storage` | rounded | compute | compute | compute |
| `zero_point` | float | float | float | integer (`round(-min/scale)`) |
| `clip_ratio` | 1.0 | 1.0 | 1.0 | 0.95 |
| `range_includes_zero` | no | no | no | key_head yes |
| `degenerate_range` | unit_scale | zero_scale | eps (1e-4 / 1e-6) | pm_one |
| `rounding` | round_clamp | clamp_round | clamp_round | round_clamp |
| `decode_dtype` | fp32 | fp16 | compute | compute |
| `symmetric_range` | balanced | - | - | full (symmetric only) |
| `complete_groups` | no | yes | yes | yes |

These helpers build the recipes' defaults: `KIVIConfig`, `KittyConfig` and
`QuaRotKVConfig` (and their `with_settings`) build `IntQuantizer`s. The dedicated
`KIVIQuantizer`, `KittyQuantizer` and `QuaRotQuantizer` remain and are still
accepted by the recipes. `tests/unit/kv_quantization/test_integer_reference_presets.py`
checks each preset bit for bit (`torch.equal`) against the authors' pinned code
and the dedicated quantizer, alone and inside the KIVI, Kitty and QuaRot recipes
(`test_*_default_recipe_equals_the_dedicated_codec`, FAKE and PACKED).
`QuaRotKVConfig` accepts `IntQuantizerConfig` only when it equals the matching
`quarot_int_quantizer()` result. Packing (a uint8 bitstream) never changes values.

Groups must be complete: R must be a positive multiple of the key group size,
and the value group size must divide head_dim. The FP16 numerical path is checked
against the authors' encode/unpack helpers. BF16/FP32 and INT8 are portable
extensions; the released fused CUDA path supports FP16 INT2/4. The native fused
CUDA kernels are not integrated or certified by the CPU tests.

Constant groups use zero scale/code and reconstruct exactly. Nonconstant native
scale underflow and nonfinite arithmetic raise clear errors. Finite decode
overflow follows the framework's saturating reconstruction rule. These guards,
optional kept-token preservation and framework byte packing are documented
extensions to the reference. Current-call K/V stay exact while storage is updated
for the next call. FAKE and PACKED share the same reference codec.

Explicit generic integer overrides remain available for labelled experiments;
they retain their own FP32/conservative parameter arithmetic described below.
The [implementation note](implementations/README.md#kivi)
lists the exact comparison scope and tests.

### Generic codecs

Grouped uniform INT2/4/8 quantization computes FP32 statistics and stores a
floating offset (the group minimum) and a scale per group. Asymmetric encoding is
`round((x - offset) / scale)`, with `scale = (max - offset) / (2**bits - 1)`. The
optional symmetric mode uses signed `[-qmax, qmax]`, stored as unsigned codes, and
stores no offset. Codes are packed into uint8: four INT2 or two INT4 entries per byte.
This is a reference PyTorch implementation, not an optimized GPU kernel.

Scales and offsets are stored in `KVQuantizerConfig.scale_dtype` / `offset_dtype`
(`ParamDType.FP32`, `FP16` or `BF16`; default FP16 for both, set separately for keys
and values). They are rounded to that dtype *before* the codes are computed, so
decoding uses exactly the stored values. Rounding is conservative: offsets round down
and scales round up by at most one step of the dtype, so the stored range always covers
the group and nothing is clipped; a coarser dtype only widens the range. bf16 has few
mantissa bits, so offsets far from zero cost more precision than with fp16. FP16 halves
the parameter overhead of FP32: INT2 with groups of 32 costs 2 + 32/32 = 3 bits per
value instead of 4. A parameter beyond the dtype's range raises a clear error.

`PER_TOKEN` groups channels; `PER_CHANNEL` groups tokens. `PER_BLOCK` groups
consecutive channels within one token, and `group_size` must divide head_dim, so a
block never spans two tokens.
Incomplete groups repeat the last entry for statistics and discard padding on
decode. For paper reproduction, cache-policy boundaries must match that paper's
grouping instead of relying on this baseline padding convention.

`OutlierTokenPolicy` is an optional generic experiment: it selects whole tokens
to preserve in their original precision. The coordinator saves those originals
and forwards `PrecisionPlan.keep_tokens` to a supporting codec. The codec excludes
kept positions from its own group statistics and scaling, leaving disposable
placeholders at the same positions. The coordinator restores originals after the
inverse transform. Raw codec dequantization alone does not restore them.
FAKE chunks release the separate kept-value/index tensors after merging their
originals into the reconstruction. PACKED chunks retain them for future decoding.
All-kept groups use safe dummy parameters; kept tokens and padding do not widen
another group's range. Codecs must explicitly support this contract or reject it.
This policy is not automatically enabled for named paper recipes: a recipe must
select behavior matching its intended paper/reference.

For `NonUniformQuantizerConfig(mode="calibrated")`, prepare all required layer/K/V
tables with `calibrate()` or `load()` before encoding starts. The first successful
encoding freezes the codec for its lifetime; later table changes raise an error.
Request cleanup does not thaw it. Use a fresh quantizer and cache for another
table configuration. The `tables` property provides read-only inspection copies,
and `tables_frozen` exposes the state. Saved table files can be written after the
freeze and loaded into a fresh instance. Encoded records include a table fingerprint:
a decoder must have matching table contents, bit width and channel count. Decoding
a matching record freezes a fresh receiver too. Invalid file/sample updates leave
previously prepared tables intact. Gaussian mode does not use calibrated tables.

Reconstruction rejects NaN/infinity and clamps finite values outside the output
dtype's finite range before conversion. The same rule applies after inverse
rotation, when restoring the original model dtype. Newly captured K/V are checked
even if they stay in an uncompressed buffer. Coding parameters too large for
their selected storage dtype still raise; choose wider parameter dtypes rather
than clipping the encoding parameters. This does not add sample-level benchmark
recovery or general attention/logit validation.

Each history (keys, and separately values) has three regions:

```text
[ sink: first tokens ][ middle: older history ][ local: newest tokens ]
```

The cache policy sets the sink and local regions independently for keys and values.
Each is a `RegionConfig(tokens, quantizer)`, and `quantizer=None` keeps it in its
original precision. The middle region uses `KVQuantizationConfig.key_quantizer` /
`value_quantizer`.

```python
DefaultCachePolicyConfig(
    key_sink=RegionConfig(4, IntQuantizerConfig(bits=8)),    # first 4 key tokens in INT8
    value_sink=RegionConfig(4),                              # first 4 value tokens in FP
    key_local=RegionConfig(128, IntQuantizerConfig(bits=4)), # newest 128 key tokens in INT4
    value_local=RegionConfig(64),                            # newest 64 value tokens in FP
)
```

Every policy shares those region settings and differs only in *when* local tokens
move to the middle:

| Policy (config) | Rule | Full-precision tail |
|---|---|---|
| `DefaultCachePolicy` (default) | move the oldest `*_block_tokens` once local holds `local.tokens + block` | `local` to `local + block - 1` |
| `SinkAndLocalCachePolicy` (KIVI's buffering) | keys: buffer `key_local.tokens` (R), then move whole multiples of R; values: exact rolling window | keys 0 to R-1, values exactly R |
| `KittyCachePolicy` (Kitty's buffering) | 32 sinks; keys: once more than R (128) wait, full pages move, each page its own chunk; values: exact rolling window | keys 1 to R, values exactly R, after the sinks |
| `SinkOnlyCachePolicy` | sinks only; every other token is quantized on arrival | sinks only |
| `FullQuantizationCachePolicy` | no sinks, no local window; everything is quantized on arrival | none |
| `LocalOnlyCachePolicy` | no sinks; an exact local window (blocks of 1 by default) | exactly `local.tokens` |

To add a policy, subclass `CachePolicyConfig` (for any new settings) and `CachePolicy`,
implement `next_chunk_tokens` and `min_move_tokens`, and register it with
`@CachePolicyRegistry.register(...)`. Optionally override `page_tokens` to store the
moved tokens as several chunks of that size (Kitty does this so that precision
policies such as `ChannelBoost` decide per 128-token page). A per-channel middle quantizer is rejected when a
policy would move fewer tokens than its `group_size` at once, because a one-token
per-channel group reconstructs exactly and would silently skip quantization. Tokens that are already
old when they arrive (for example, the context imported on the first call) go straight
to the middle quantizer. A token that sat in a *quantized* local window is re-quantized
from its stored local copy when it moves; the original value is never consulted. A
local quantizer must be per-token or per-block, because local tokens arrive one at a
time. Separately for K and V, the counters satisfy:

```text
seen_tokens = sink_tokens + compressed_tokens + recent_tokens
```

Local tokens can temporarily exceed the requested minimum by less than one
block. Stored compressed chunks persist without re-encoding. Reconstructed K/V
are temporary, in original coordinates, shape, dtype and device. FAKE stores
reconstructed chunks; PACKED stores codes and decodes on read. Both use the same
encoder/decoder, rather than two different rounding implementations.

## Transforms

Outer transforms rotate each key/value vector along `head_dim` before quantization
and rotates it back after decoding. Rotations compute in FP32; the inverse
restores the original dtype. Hadamard forward can also cast to the input dtype
with preserve_input_dtype=True, as selected by QuaRot.

| Transform (config) | Rotation | Used by |
|---|---|---|
| `IdentityTransform` | none (default) | most methods |
| `HadamardTransform` | normalized Hadamard `H`; `head_dim` must be a power of 2 | QuaRot KV (preserve_input_dtype=True) |
| `RandomHadamardTransform` (`seed`) | `diag(s) @ H` with random signs `s` | Generic experiments |
| `RandomRotation` (`seed`) | random orthogonal matrix (QR of a Gaussian) | TurboQuant |

The random transforms derive reproducible rotations per (layer, keys/values)
from `seed`. TurboQuant selects identity outer transforms and reuses RandomRotation
internally, normalizing original vectors before rotation and computing QJL
residuals in original coordinates. Rotation spreads channel magnitudes; it does
not guarantee lower error for every individual vector.

## Intentionally pending

Learned rotations, the native NVFP4 codec, coordinator-level calibration, and the
KVQuantNUQ and NVFP4KV paper recipes still raise
`NotImplementedError` when selected. Their classes, configs and registry entries
are ready for focused implementations. Generic FP8/FakeNVFP4/MXFP/nonuniform
codecs, ChannelBoost and OutlierTokenPolicy are implemented; their existence does
not establish paper-reference parity. KIVI's portable reference path is implemented
and numerically checked. QuaRot KV also has a portable, source-checked
accuracy-profile implementation; see its implementation note for scope.
KIVI's shared buffer is `SinkAndLocalCachePolicy`.
Kitty follows the authors' accuracy simulation (per-token value window, page-wise
key boost); see `implementations/README.md`.
KVQuant requires calibrated codebooks and per-element
outliers; keys are quantized post-RoPE (no RoPE inversion). TurboQuantKV implements
both uniform MSE and QJL-corrected prod profiles; fractional-bit outlier profiles
are not reproduced.

Mixed precision is an explicit `PrecisionPlan` extension. The working codec
rejects per-entry assignments until their packing/layout is implemented.
`QuantizedTensor` reserves format, codebook, global-scale, norm and outlier fields;
future encodings must account for all owned metadata in storage measurements.

## Scope and lifecycle

Standard causal self-attention supports full-history and native rolling HF caches
when prefill is captured. Our store retains its own full history independently.
Static padded caches, beam reordering, rollback, batch shape changes and eviction
from our own stored history need separate contracts. Rewritten past tokens cannot
be detected from length alone.

Each independent question has one metadata dictionary and one store. The
coordinator and fixed numerical parameters can be reused, while histories live
only in metadata. The HF adapter clears the extra store in `finally`, including
failed generations. Direct callback callers must call `clear_request_state` too.

The adapter captures context K/V during prefill while delegating the computation
to native dense attention, then marks subsequent calls as decode. The capture
prevents HF sliding eviction from losing context before our store sees it.
Direct HF callers should mark metadata `kv_quantization_phase='prefill'` or
`'decode'` and capture prefill before decoding with a rolling cache.

`KVQuantization.process` requires `num_new_tokens=queries.shape[-2]` for self-attention.
It prepares `restored previous history + original new K/V` before appending and
compressing for the next call. `num_new_tokens=0` is the explicit completed-history
inspection mode for codec tests. The store's total token count is independent of
HF's retained tensor length. Its full history is retained; native causal/sliding
and padding visibility is preserved by mask alignment.

With `quantize_prefill=False`, direct prefill returns original K/V while still
capturing storage. With True, earlier cached chunks can be reconstructed during
chunked prefill. Default/KIVI processing keeps current-call tokens exact; QuaRot
and TurboQuant override process() to reconstruct those tokens too. Adapter
capture-only prefill uses native attention with the method-prepared K/V when
quantize_prefill=True and original K/V otherwise. These settings alone do not
establish paper fidelity.

HF's original full-precision cache remains allocated. Store bytes describe only
additional owned tensor storage (not Python objects or temporary decoded K/V).
This implementation demonstrates numerical effects and packed storage, not total
inference memory savings or native low-bit attention speedups. Optional local
K/V error differs from the existing sparse-vs-dense output error and from final
benchmark quality. Diagnostics should be disabled for timing comparisons.

## Verification

```bash
python -m pytest -q -m unit tests/unit/kv_quantization
python -m pytest -q -m integration tests/integration/test_kv_quantization*.py tests/integration/test_kivi_reference_flow.py
```

## Benchmark diagnostics and saved configuration

Set `KVQuantizationConfig(error_metric=True)` to request K/V reconstruction errors
and store-owned tensor bytes. The benchmark worker then enables
`kv_quantization_key_error`, `kv_quantization_value_error` and
`kv_quantization_store_bytes` in addition to its existing research-attention metrics.
Events are written to `micro_metrics.jsonl`, including partial diagnostics from a
failed job. Jobs use separate output directories; their queues are flushed before
the worker switches to the next directory.

`BenchmarkExecutor(micro_metric_logger_enabled_metrics=[])` disables all micro
metrics for timing comparisons. The quantization coordinator skips disabled error
calculations and does not issue disabled-metric logging warnings. An explicit list
selects a subset; requesting quantization metrics requires `error_metric=True`.
For example, select only `['kv_quantization_store_bytes']` to omit reconstruction
error calculation. Record limits and sampling limit saved events; use `[]` to
disable the calculations themselves.

`micro_metrics_config.json` records enabled metrics, sampling and record limits.
The benchmark's `config.json` exports Enum settings as values such as `per_token`
and `fp16`. Nested dataclass configurations include a stable `__type__` name so a
named preset can be distinguished from a generic composition. Framework settings
and method identity are preserved; these files do not replace trained-model,
dataset-revision or full software-environment provenance.

Extension order: implement one component, test its round trip and format, define
its cache/calibration policy, then enable its named method after comparing with
the reference method. A new class name alone does not establish paper fidelity.
