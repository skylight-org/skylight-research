# KV quantization method implementations

This directory contains named quantization recipes and their configurations.
Shared quantizers, transforms, precision policies and cache policies live in the
parent package. The generic baseline is configured through
[`KVQuantizationConfig`](../base.py).

| Method | Configuration / class | Source | Status |
|---|---|---|---|
| [KIVI](#kivi) | `KIVIConfig` / `KIVI` | [kivi.py](kivi.py) | Implemented portable numerical/cache path |
| [Kitty](#kitty) | `KittyConfig` / `Kitty` | [kitty.py](kitty.py) | Implemented; bit-exact with the authors' accuracy simulation |
| [QuaRot KV](#quarot-kv) | `QuaRotKVConfig` / `QuaRotKV` | [quarot_kv.py](quarot_kv.py) | Implemented portable accuracy-profile KV path |
| [TurboQuant KV](#turboquant-kv) | `TurboQuantKVConfig` / `TurboQuantKV` | [turboquant_kv.py](turboquant_kv.py) | Implemented portable MSE/prod mathematical profiles |
| [TurboQuant KV, split](#split-rotation-and-codebook) | `TurboQuantSplitKVConfig` / `TurboQuantSplitKV` | [turboquant_kv.py](turboquant_kv.py) | Implemented; rotation as a transform, codebook as a quantizer |
| KVQuant NUQ | `KVQuantNUQConfig` / `KVQuantNUQ` | [kvquant_nuq.py](kvquant_nuq.py) | Scaffold |
| NVFP4 KV | `NVFP4KVConfig` / `NVFP4KV` | [nvfp4_kv.py](nvfp4_kv.py) | Scaffold |

Scaffold recipes inherit `PendingRecipe` and raise `NotImplementedError` when
selected. Their available building blocks do not establish a complete paper
implementation. Add each method's reference, behavior, tests and limits here as
its recipe is implemented.

## KIVI

The KIVI recipe uses `IntQuantizer` with `kivi_int_quantizer()` (defined in `kivi.py`)
(the released arithmetic) and the shared request-local cache store. The generic
integer arithmetic remains available for experiments.

### Reference

The implementation follows [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI) at commit
`876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`:

- [quant/new_pack.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/quant/new_pack.py): grouped encoding and standalone unpack arithmetic.
- [models/llama_kivi.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/models/llama_kivi.py): residual-buffer behavior during prefill and single-token decoding.

The numerical policy reproduces the chosen released helpers. The paper does not
mandate the dtype of every intermediate operation.

### Implementation

| File | Responsibility |
|---|---|
| [Recipe](kivi.py) | `KIVIConfig`, default components and configuration validation |
| [Quantizer](../quantizers/implementations/integer.py) | `IntQuantizer`, configured by `kivi_int_quantizer()` in the recipe (default): native-dtype encoding, unpacking and finite-value checks |
| [Coordinator](../base.py) | Prepare attention K/V before updating and compressing the store |
| [Store](../cache/store.py) | Independent per-layer K/V histories and compressed/full-precision chunks |
| [Cache policy](../policies/cache/implementations/sink_and_local.py) | Buffered keys and rolling full-precision values |
| [Reference fixtures](../../../tests/reference/kivi) | Hash-verified authors' source, provenance, MIT license and CPU helper loader |

Defaults are INT2, group size 32 and residual length 128. Key groups span tokens
within each channel. Value groups span channels within each token.

The following excerpt is copied from the current quantizer:

```python
groups = groups.to(x.dtype)
minimum, maximum = group_min_max(groups, valid)
scale = (maximum - minimum) / (2**bits - 1)
```

Scales and minima retain the input dtype. Unlike the generic integer quantizer,
KIVI does not use directed parameter rounding through `store_param()`.
Dequantization follows the authors' standalone helper: unpacked codes become
FP16, then multiplication/addition follow the stored parameters' dtype promotion.

Keys stay in full precision until a complete residual buffer can be compressed;
values retain the newest residual window. Older compressed history is retained.
Attention receives restored previous history plus exact current-call K/V, prepared
before the persistent store is updated.

### Tests

Run from the repository root in the project's Python environment:

```bash
python -m pytest -q tests/unit/kv_quantization/test_integer_reference_presets.py
python -m pytest -q tests/integration/test_kivi_reference_flow.py
python -m pytest -q -m unit tests/unit/kv_quantization
python -m pytest -q -m integration tests/integration/test_kv_quantization*.py tests/integration/test_kivi_reference_flow.py
```

- The KIVI unit suite compares logical codes, scales, minima, reconstruction,
  storage ownership and streaming buffer transitions against pinned CPU helpers.
- The HF reference-flow suite uses tiny random Llama, Qwen3 and Mistral models to
  check callbacks, cache state, attention outputs and FAKE/PACKED agreement.
- Unit tests include independent packing/numerical regressions and recipe-scope
  checks. Integration tests cover cache/attention regressions, native and
  multi-architecture comparisons, component combinations and an offline benchmark
  worker. The deferred after-attention extension test is an expected failure.

These tests require no pretrained model weights or evaluation datasets.

### Scope and limits

- HF's unquantized cache remains allocated alongside our quantized store.
- Packed codes use Skylight's uint8 layout. Tests unpack both representations to
  compare logical codes with the authors' int32 layout.
- Complete groups are required. The key residual length must be a positive
  multiple of its group size; the value group size must divide the head dimension.
- Constant groups use zero scale/code. Nonconstant scale underflow and nonfinite
  input/parameters raise errors; finite output overflow follows the framework's
  saturation rule. These are explicit extensions around reference edge cases.
- BF16/FP32, INT8, kept-token policies and other component overrides are portable
  experiments. The authors' fused backend is a separate FP16 INT2/4 path.
- The shared flow supports batched current tokens, but reference streaming checks
  use native prefill followed by one-token calls. A batched call is not claimed
  numerically equivalent to sequential calls crossing buffer boundaries.
- CPU reference/callback checks do not establish fused CUDA equivalence,
  pretrained task quality, total GPU-memory savings or inference speed.

## Kitty

Kitty (Xia et al., MLSys 2026, [arXiv 2511.18643](https://arxiv.org/abs/2511.18643))
stores keys and values in 2 bits but gives the most important key channels of
every 128-token page 4 bits ("channel boost").

### Reference

The recipe reproduces the authors' **accuracy simulation**, which produced the
paper's accuracy tables: [Summer-Summer/Kitty](https://github.com/Summer-Summer/Kitty)
at `dfd2c07b407d6b407179359207c612ab631f3ed1`, `src/kitty_sim/utils_quant.py`
and `src/kitty_sim/kitty_simulate.py` (MIT, see [LICENSE-KITTY](../LICENSE-KITTY)).
The pinned files are vendored, hash-checked, in `tests/reference/kitty/`.

### What the method does

| Step | Kitty | Part in this package |
|---|---|---|
| First 32 tokens (keys and values) | full precision | `KittyCachePolicyConfig.key_sink` / `value_sink` |
| Keys | wait in a 128-token buffer; when the 129th key arrives, the full page is quantized | `KittyCachePolicy` (`move_full_pages_past_buffer`) |
| Key page | per channel, one group of 128 tokens; 2 bits | `kitty_int_quantizer(bits=2, PER_CHANNEL, group_size=128)` |
| Channel boost | per page and head, the top `int(0.125 * 128) = 16` channels by mean \|k\| get 4 bits (Kitty-Pro: 25%) | `ChannelBoostConfig(count_rounding="floor", score_dtype="input")`; `KittyCachePolicy.page_tokens` makes every page its own chunk |
| Values | newest 128 in full precision; older ones per token, 2 bits, group = whole head (128) | `KittyCachePolicy` value window + `kitty_int_quantizer(bits=2, group_size=128)` |
| Current call | attends to its exact K/V, quantized afterwards (`PostQuant=True`) | the shared coordinator's default flow |

Arithmetic (`kitty_int_quantizer()` in `kitty.py`) stays in the input
dtype like the reference: `scale = max(max - min, eps) / (2^b - 1)` with eps 1e-4 for
FP16/BF16, `q = round(clamp((x - min) / scale, 0, 2^b - 1))`, `x~ = q * scale + min`.

### Usage

```python
from sparse_attention_hub.kv_quantization import KittyConfig

KittyConfig()                                   # Kitty: 12.5% boosted key channels
KittyConfig.pro()                               # Kitty-Pro: 25%
KittyConfig.with_settings(boost_fraction=0)     # the paper's KIVI* baseline (sinks, no boost)
KittyConfig.with_settings(channel_selection="random")   # ablation; also "variance"
```

`ChannelBoost`, `IntQuantizer` and `KittyCachePolicy` also compose freely in a
plain `KVQuantizationConfig`. `validate_kitty` only guards the named recipe.

### Tests

- `tests/unit/kv_quantization/test_integer_reference_presets.py`: key pages (12.5/25/10/20%,
  magnitude and variance selection) and values are **bit-identical** to the
  authors' `build_promote_mask` + `fake_quant_groupwise_lastdim` in FP16 and BF16;
  the attention K/V of every prefill and decode step equal the authors'
  `KittyKVCache.update` for several prompt lengths.
- `tests/unit/kv_quantization/test_kitty_preset.py`: configuration, validation,
  page-wise storage and boosting, packed = fake storage, storage size.

### Scope and limits

- When the prompt after the sinks is an exact multiple of 128 (more than one
  page), the reference fake-quantizes the last prompt page a second time at the
  first decode step. We quantize each page once; only that page differs.
- The authors' system path (CUDA kernels, two 2-bit planes, a paged value buffer)
  is a different storage backend; packed storage here uses Skylight's bitstream
  with one width byte per group. No speed or total-memory claim.
- The paper evaluates in FP16; BF16 runs use the same operations in BF16.

## QuaRot KV

QuaRot KV uses the authors' accuracy-path quantizer with Hadamard preprocessing,
packed storage and reconstruction into the original attention coordinates.
Weights and ordinary model activations keep their existing precision.

### Reference

The reference is [spcl/QuaRot](https://github.com/spcl/QuaRot) at commit
5008669b08c1f11f9b64d52d16fddd47ca754c5a:

- [fake_quant/quant_utils.py](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/quant_utils.py):
  ActQuantizer, integer zero points, clipping and encode/decode arithmetic.
- [fake_quant/rotation_utils.py](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/rotation_utils.py):
  post-RoPE Q/K rotation and V/O projection preparation.
- [fake_quant/main.py](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/main.py):
  head-wise key and grouped value quantizer installation.
- [Paper](https://arxiv.org/pdf/2404.00456): experimental setup and KV-only ablation.

ActQuantizer's arithmetic is reproduced by `IntQuantizer` configured with
`quarot_int_quantizer()` (in [quarot_kv.py](quarot_kv.py)), checked bit for bit
against the pinned source. There is no vendor package. Apache-2.0 attribution/license is retained in
[LICENSE-QUAROT](../LICENSE-QUAROT) and packaged with the library.

### Defaults and numerical behavior

- Asymmetric INT4 K/V, groups of 128 channels and clipping ratio 0.95.
- Key groups cover one token's whole KV head; value groups cover channels within
  a token/head. Groups must be complete. Key group size must equal head_dim;
  value group size must divide it. Other sizes are explicit experimental profiles.
- The existing HadamardTransform uses preserve_input_dtype=True. It computes the
  rotation in FP32 and casts back before quantization. Its default False remains
  unchanged for existing generic configurations.
- Both previous and arriving tokens are quantized before attention. There is no
  exact residual or sink region. Quantized prefill is enabled by default.
- Bits 2, 3, 4 and 8 are supported independently for K/V. Compact unsigned codes
  use format quarot_accuracy_int_v1 and a continuous bitstream; 3-bit codes really
  occupy ceil(code_count * 3 / 8) payload bytes.
- FP16 is the primary profile. FP32 helper checks and BF16 experiments are also
  supported; those do not establish CUDA-backend parity.

Parameters follow the actual upstream call paths. With FP16 inputs, head-wise
keys use FP32 scales/zero points, while grouped values use FP16 parameters.
The difference comes from the authors' zero-including key range calculation.
The explicit parameter_dtype="upstream_path" configuration records this rule;
inherited scale_dtype/offset_dtype retain unused defaults, as with KIVI.
Independent parameter-dtype overrides are rejected.

The upstream CLI defaults differ from the paper's asymmetric G128/0.95 profile;
the named recipe selects those settings explicitly. Symmetric experiments use
the authors' full signed range (INT4 includes -8), with reversible unsigned
code bias in our storage.

All-zero groups keep the reference handling. Undefined nonzero constant
asymmetric value groups, scale underflow and nonfinite inputs/parameters raise
errors before invalid codes can be stored. Finite reconstruction overflow follows
the framework's saturation rule. These guards are documented extensions around
undefined reference cases; no epsilon is silently added to the quantizer.

### Implementation and call flow

| File | Responsibility |
|---|---|
| [Recipe](quarot_kv.py) | Configuration, validation, process override selecting current-token reconstruction |
| [Quantizer](../quantizers/implementations/integer.py) | `IntQuantizer` with `quarot_int_quantizer()`: the authors' numerics, native arithmetic dtypes and compact parameters |
| [Hadamard transform](../transforms/implementations/hadamard.py) | Shared matrix/inverse with optional forward dtype preservation |
| [Packing](../quantizers/common.py) | Cross-byte bitstream packing/unpacking; existing codecs retain their layout |
| [Coordinator](../base.py) | Shared _process_kv_history helper; default process preserves KIVI's exact-current ordering |
| [HF adapter](../../adapters/huggingface.py) | Feed prepared prefill K/V to native attention when requested |
| [Reference fixtures](../../../tests/reference/quarot) | Hash-verified authors' sources, provenance, Apache license and independent loader |

~~~text
HF post-RoPE K/V
  -> identify arriving tokens
  -> Hadamard -> input-dtype cast -> reference quantizer -> store
  -> decode history including arriving tokens -> inverse Hadamard
  -> existing attention with unchanged queries
~~~

Returning decoded rotated keys multiplied by H.T is algebraically equivalent
to rotating queries in the dot product. Returning decoded values multiplied by
H.T similarly moves the inverse output rotation before attention. Thus
ResearchAttention and its maskers require no rotation-specific edits.

This changes floating-point operation order: the authors rotate Q explicitly
and incorporate V rotations into projection weights. Our original-coordinate
reconstruction is not claimed bit-identical to that full model preparation.
The codec (`quarot_int_quantizer()`) is checked exactly on the same rotated input.

Prefill keeps the native mask format when token positions/lengths are unchanged.
For longer independent history, eager/SDPA prefill uses the existing mask alignment
helper. Other native backends explicitly reject that longer-history case.
Storage retention and native sliding visibility remain separate.

### Usage and tests

~~~python
from sparse_attention_hub.kv_quantization import QuaRotKVConfig
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig

config = ResearchAttentionConfig(
    masker_configs=[],
    kv_quantization_config=QuaRotKVConfig(store_codes=True),
)
~~~

The default requires head_dim=128. For another head dimension, set the group
sizes explicitly, e.g. `key_quantizer=quarot_int_quantizer("key_head",
group_size=head_dim)` (keys use one group per head) and a matching
`quarot_int_quantizer("value_group", group_size=...)`. Setting store_codes=False selects FAKE storage.
Setting quantize_prefill=False is an explicit native-prefill experiment.

~~~bash
python tutorials/py_examples/06_quarot_kv_demo.py --device cpu
python tutorials/py_examples/06_quarot_kv_demo.py --device cpu --bits 3
python -m pytest -q tests/unit/kv_quantization/test_integer_reference_presets.py tests/unit/kv_quantization/test_quarot_preset.py tests/unit/kv_quantization/test_bitstream_packing.py
python -m pytest -q -m integration tests/integration/test_kv_quantization_quarot_flow.py
python -m pytest -q -m unit tests/unit/kv_quantization
python -m pytest -q -m integration tests/integration/test_kv_quantization*.py tests/integration/test_kivi_reference_flow.py
~~~

The demo constructs a tiny random Llama with head_dim=128 and identical forced
tokens/weights for native, FAKE and PACKED runs. It prints cache counts and owned
store bytes and checks FAKE/PACKED logits. It downloads no models or datasets.

Unit tests check exact logical codes, parameter values/dtypes and reconstruction
against the pinned source, including signed/asymmetric paths and 3-bit packing.
Integration tests exercise Llama, Qwen3 and Mistral with GQA, chunked prefill,
multi-token decoding, dtype-aware independent reconstruction, actual eager/SDPA
prefill, cleanup and an offline benchmark worker. Source codec checks are exact;
independent matrix/attention checks allow explicitly bounded floating-point
roundoff. FAKE/PACKED logits must agree exactly.

Validation of this implementation on CPU used Python 3.10, PyTorch 2.8.0 and
Transformers 4.57.1:

- The full focused command below passed **1,650 tests**, with **one existing
  expected failure** for the deferred after-attention extension. It includes
  KIVI, generic quantization, adapter and benchmark regression checks.
- The 3-bit and 4-bit offline demos both produced identical FAKE/PACKED logits.
- Black, isort and flake8 passed on the changed Python files. The six changed
  implementation modules passed focused mypy checks and the configured pylint
  score threshold.
- A built wheel was extracted and imported independently of the checkout; its
  QuaRot codec/recipe, packaged Apache license, packed prefill and request cleanup
  passed a smoke test.

~~~bash
python -m pytest -q tests/unit/kv_quantization tests/integration/test_kv_quantization*.py tests/integration/test_kivi_reference_flow.py tests/unit/benchmark/test_config_serialization.py tests/unit/benchmark/test_quantization_test_categories.py tests/unit/benchmark/test_supported_runtime.py tests/unit/adapters
~~~

### Scope and remaining work

- This is KV-only QuaRot with original-coordinate reconstruction. Full-model
  norm fusion, residual/FFN rotations, weight quantization/GPTQ and ordinary
  activation quantization are not implemented here.
- The separate upstream MultiLayerPagedKVCache4Bit uses different asymmetric
  numerics, a range floor and native prefill. This accuracy-profile recipe does
  not claim to reproduce that cache or its fused CUDA attention.
- The current transform uses the existing matrix implementation. A future
  optional fast-Hadamard backend should be checked against the upstream pinned
  submodule revision 4ea722e434e3d4f2a14522341959ebdbe62be2de.
- Our quantized store coexists with the native unquantized HF cache. Report
  owned-store bytes separately from model memory and dense reconstruction
  temporaries. The current tests establish no overall GPU-memory or speed gain.
- GPU kernel parity and pretrained task-quality evaluation remain unverified.


## TurboQuant KV

This implements the uniform whole-head MSE and prod algorithms from
[arXiv:2504.19874v1](https://arxiv.org/html/2504.19874v1), Algorithms 1/2,
Sections 2.2, 3.1 and 3.2. It is an independent implementation of those
mathematical algorithms. No author-released TurboQuant code was available in
the sources reviewed. Community code from 0xSero/turboquant is not copied or
used as an official reference. The separate authors' QJL repository is not a
TurboQuant cache implementation.

### Algorithms and numerical policy

For a row vector x in one token's KV head, the codec stores its FP32 norm rho
and uses u=x/rho (zero vectors have an explicit zero reconstruction):

~~~text
MSE:
  rotated = u @ R
  indices = nearest_level(rotated, levels)
  coarse = levels[indices] @ R.T
  reconstructed = rho * coarse

prod with b total nominal bits:
  coarse = MSE reconstruction of u using b-1 bits
  residual = u - coarse                     # original coordinates
  gamma = norm(residual)
  signs = sign(residual @ S.T)               # iid Gaussian S, independent of R
  correction = sqrt(pi/2) / head_dim * gamma * (signs @ S)
  reconstructed = rho * (coarse + correction)
~~~

Prod's 1-bit profile uses a zero coarse estimate and no coarse index stream.
It never calls a zero-bit packing helper. Zero sign projections select +1.
Codebook midpoint ties select the lower code. The result is not renormalized.
QJL's unbiased dot-product property does not establish unbiased softmax or
unbiased final model outputs.

The default table is Lloyd-Max for the finite-dimensional sphere-coordinate
density proportional to (1-z²)^((d-3)/2). Codebooks are independent of samples
and keyed by head dimension and coarse bit width. A CPU FP64 Gauss-Legendre
solver uses the z=sin(theta) change of variables, order 96, a 1e-11 convergence
threshold and at most 2000 iterations; nonconvergence raises an error. The
optional codebook_distribution="gaussian_limit" is a labelled experiment.

Runtime normalization, rotation, residuals and reconstruction use FP32; outputs
are cast to the input FP16/BF16/FP32 dtype. A scaled norm calculation avoids
squaring raw large/small magnitudes. Nonfinite inputs/parameters/arithmetic raise
errors; finite output-dtype overflow uses the framework's saturating cast.
These dtype, solver, tie and edge rules are our explicit numerical contract,
not claims about unpublished author kernels. Exported configurations include
reference="arxiv:2504.19874v1:algorithms-1-2" and a numerics_version.

The codec reuses RandomRotation's seeded FP64 Gaussian QR construction, with
FP32 stored matrices. It creates a separate iid Gaussian QJL matrix when needed.
Matrices remain fixed per codec/layer/K-or-V/head dimension, are independent of
input samples and do not consume the global PyTorch RNG. Inspection returns
copies. Records carry table/matrix fingerprints; incompatible decoders reject
rather than silently interpreting codes with different resources. Fingerprints
also detect RNG/QR differences across runtime versions when moving records.

### Outlier-channel split (2.5 / 3.5 bits)

The paper's KV results use fractional bit-widths: it splits each head's channels
into outliers and the rest and runs two independent TurboQuant instances.
`TurboQuantKVConfig.outlier_split(2.5)` / `outlier_split(3.5)` build this:

| Preset | Outlier channels | Outlier bits | Other bits | Average |
|---|---|---|---|---|
| `outlier_split(2.5)` | 64 of 128 | 3 | 2 | (64·3 + 64·2) / 128 = 2.5 |
| `outlier_split(3.5)` | 64 of 128 | 4 | 3 | (64·4 + 64·3) / 128 = 3.5 |

The paper's own 2.5-bit example (32 channels at 3 bits + 96 at 2 bits) works
out to 288 / 128 = 2.25 bits, and it gives no 3.5-bit ratio, so both splits are
our choice of a true average.

~~~text
first chunk of a request (prefill), per layer, K/V and head:
  ChannelBoost(freeze="prefill") -> top channels by mean |x| -> channel_bits
later chunks (decode tokens): the same channel_bits object is reused

TurboQuantQuantizer.quantize(x, plan with channel_bits):
  outlier channels -> norm -> own rotation -> codebook(k dims, high bits) [+ QJL]
  other channels   -> norm -> own rotation -> codebook(d-k dims, low bits) [+ QJL]
  record: two code streams, norms [..., 2], outlier_indices [batch, heads, k]
dequantize: decode both groups, scatter back to the original channel order
~~~

- Each group gets its own seeded rotation and QJL matrix, so equal-sized groups
  (64 / 64) never share one. Without `channel_bits` the codec is unchanged.
- The split happens before rotation; `TurboQuantSplitKV` (rotation as a
  transform over all channels) does not support it.
- The outlier index is shared by every record of a request, and
  `QuantizedKVStore.stats()` counts it once per (layer, K/V).

### Files and flow

| File | Responsibility |
|---|---|
| [Recipe](turboquant_kv.py) | TurboQuantKVConfig, named-profile validation and current-token reconstruction |
| [Codec](../quantizers/implementations/turboquant.py) | Sphere codebooks, norm/rotation, MSE indices, QJL correction, record validation and shared-resource byte accounting |
| [Encoded record](../cache/block.py) | Existing norms plus optional qjl_codes/residual_norms, included in storage_bytes() |
| [Independent references](../../../tests/reference/turboquant) | NumPy FP64 algebra, Python bitstream decoder, analytic/Beta moment checks and fixed vectors |
| [Demo](../../../tutorials/py_examples/07_turboquant_kv_demo.py) | Offline random-model native/FAKE/PACKED flow and memory breakdown |

~~~text
HF K/V -> identify arriving suffix -> append to request-local history
  -> identity outer transform
  -> codec: normalize -> rotate -> encode MSE / optional QJL residual
  -> packed store -> decode complete history, including arriving tokens
  -> original-coordinate K/V -> existing attention and unchanged queries
~~~

TurboQuantKV.process() selects reconstruct_current=True on the shared
_process_kv_history() helper. FullQuantizationCachePolicy keeps no exact sinks
or residual region. Quantized prefill is on by default; an explicit
quantize_prefill=False experiment still captures/compresses the context for
subsequent calls. Completed history is not requantized. Our cache retains full
history while the existing adapter/attention mask alignment preserves native
sliding/padding visibility. ResearchAttention, HF's cache class and its query
handling need no additional edits for TurboQuant.

### Split rotation and codebook

`TurboQuantSplitKVConfig` runs the same algorithm with its two stages as
separate components:

~~~text
HF K/V -> RandomRotation transform (y = x R, FP32)
  -> TurboQuantCodebookQuantizer: normalize -> codebook index / optional QJL residual
  -> packed store -> decode -> inverse transform (R^T) -> original-coordinate K/V
~~~

With the same seed, `RandomRotation` builds the same matrices as the bundled
codec, so MSE indices agree except where FP32 rounding of normalize-then-rotate
vs rotate-then-normalize moves a coordinate across a codebook boundary. For
`prod`, the residual is in rotated coordinates, so the effective QJL projection
is `S R` instead of `S`; both are i.i.d. Gaussian, so the estimator is the same
(an injected `S R` reproduces the bundled signs). The codebook is only right for
randomly rotated input, so the recipe requires `RandomRotationConfig` in both
transform slots. Tests: `tests/unit/kv_quantization/test_turboquant_split.py`.
The bundled `TurboQuantKVConfig` is unchanged.

### Storage and configuration

Both K/V default to 3-bit MSE with packed storage. Whole-head dimensions >=2
are supported, including non-powers of two. K/V may have different dimensions
and independently configured bit widths/variants. Precision policies that
assign mixed-channel bits or exact-token regions are rejected for this named
profile. Inherited group_size/scale_dtype/offset_dtype settings are unused and
must retain their defaults; actual vectors span the entire KV head and norms
are FP32.

~~~python
from sparse_attention_hub.kv_quantization import (
    TurboQuantKVConfig, TurboQuantQuantizerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig

config = ResearchAttentionConfig(
    masker_configs=[],
    kv_quantization_config=TurboQuantKVConfig(
        key_quantizer=TurboQuantQuantizerConfig(bits=3, variant="prod"),
        value_quantizer=TurboQuantQuantizerConfig(bits=3, variant="mse"),
        store_codes=True,
    ),
)
~~~

Set store_codes=False for FAKE reconstructed storage with the same encoder and
decoder. Packed request payloads contain:

- MSE: ceil(N*b/8) index bytes and one FP32 original norm per vector.
- Prod: ceil(N*(b-1)/8) coarse bytes (absent at b=1), ceil(N/8) sign bytes,
  and two FP32 norms per vector.

N is the number of scalar entries in a chunk. Thus nominal b excludes norm
metadata: for complete heads, effective payload bits/channel are b+32/d for
MSE and b+64/d for prod, plus each stream's final-byte padding. At d=128,
3-bit MSE costs 3.25 bits/channel and 3-bit prod costs 3.5. This latter total
is not evidence of reproducing the paper's fractional-bit outlier profile.

store_bytes counts owned request tensor payloads, including both norm fields
and packed signs. shared_state_bytes() separately counts owned codebook,
rotation and projection tensor allocations on CPU/device without alias double
counting. It excludes Python objects and inspection copies owned by callers.
The demo prints both scopes and packed payload components. Neither measurement
includes HF's native cache, weights or dense reconstruction temporaries.

### Verification and remaining scope

~~~bash
python tutorials/py_examples/07_turboquant_kv_demo.py --device cpu
python tutorials/py_examples/07_turboquant_kv_demo.py --device cpu --key-variant prod
python -m pytest -q tests/unit/kv_quantization/test_turboquant*.py
python -m pytest -q tests/integration/test_kv_quantization_turboquant_flow.py tests/integration/test_kv_quantization_benchmark_pipeline.py
~~~

Unit checks use analytic uniform/Gamma cases, independent incomplete-Beta
conditional moments, higher quadrature order, exact logical indices/signs,
independent reconstruction, QJL query/projection algebra, fixed-seed statistical
unbiasedness checks, byte counts, serialization, corrupted records and numerical
edge cases. NumPy FP64 formulas are checked separately from FP32 runtime
roundoff. Reference tests use SciPy, already a dependency of the project's
scikit-learn requirement; the runtime codec only requires PyTorch/NumPy.

Tiny random Llama, Qwen3 and Mistral checks cover FP16/BF16/FP32, GQA,
full/chunked prefill, single/multi-token decode, MSE/MSE and prod/MSE, native
sliding input, eager/SDPA prefill, request isolation and cleanup after invalid
decode. The real offline benchmark worker covers execution, diagnostics and
configuration/results export. FAKE/PACKED logits must match exactly on the
same backend. Codecs are checked independently; these local-model checks are
not a pretrained task-quality benchmark.

Validation on CPU with Python 3.10, PyTorch 2.8.0 and Transformers 4.57.1:

- The focused regression command below passed **1,877 tests**, with **one
  existing expected failure** for the deferred after-attention extension. It
  includes KIVI, QuaRot, generic codecs, adapters and benchmark checks.
- Both 3-bit MSE/MSE and prod/MSE demos produced identical FAKE/PACKED logits.
- Black, isort and flake8 passed on 14 Python files changed for this method.
  Four implementation modules passed focused mypy checks; their pylint score
  was 9.36/10, above the configured threshold.
- A wheel was built, extracted and imported outside the checkout. Its MSE/prod
  packed prefill, exports and request cleanup passed an independent smoke run.

~~~bash
python -m pytest -q tests/unit/kv_quantization tests/integration/test_kv_quantization*.py tests/integration/test_kivi_reference_flow.py tests/unit/benchmark/test_config_serialization.py tests/unit/benchmark/test_quantization_test_categories.py tests/unit/benchmark/test_supported_runtime.py tests/unit/adapters
~~~

Remaining work before claiming published benchmark reproduction:

- Resolve the v1 Section 4.3 outlier-selection, split lifetime and fractional-bit
  settings from an authoritative implementation or clarified final version.
  The printed 32-of-128 allocation example also has a bit-average arithmetic
  inconsistency. We do not guess that missing profile or relabel our uniform one.
- Evaluate pretrained LongBench/needle quality with pinned model/data settings.
- Validate GPU precision/backend behavior and any future fused implementation.
- Native low-bit attention, entropy coding and fused kernels are not implemented.
  This path retains HF's unquantized cache and reconstructs dense K/V, so it makes
  no claim of total GPU-memory savings or native quantized-attention speed.
