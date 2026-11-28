"""Reject invalid numbers and saturate finite reconstruction at dtype boundaries."""

from dataclasses import replace

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    AttentionContext,
    FakeNVFP4QuantizerConfig,
    FP8QuantizerConfig,
    FullQuantizationCachePolicyConfig,
    HadamardTransformConfig,
    IdentityTransformConfig,
    IntQuantizerConfig,
    KVQuantization,
    KVQuantizationConfig,
    KVQuantizer,
    KVTransform,
    MXFPQuantizerConfig,
    NonUniformQuantizerConfig,
    ParamDType,
    PrecisionPlan,
    QuantGranularity,
    RandomHadamardTransformConfig,
    RandomRotationConfig,
)
from sparse_attention_hub.kv_quantization.numerics import finite_cast
from sparse_attention_hub.kv_quantization.quantizers.common import store_param

pytestmark = pytest.mark.unit
CONTEXT = AttentionContext(0, "key", 0)
CODECS = [
    IntQuantizerConfig(bits=2),
    FP8QuantizerConfig(),
    FakeNVFP4QuantizerConfig(),
    MXFPQuantizerConfig(),
    NonUniformQuantizerConfig(),
]
ROTATIONS = [
    HadamardTransformConfig(),
    RandomHadamardTransformConfig(seed=3),
    RandomRotationConfig(seed=1),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_finite_overshoots_saturate_both_signs_without_changing_input(dtype):
    limit = torch.finfo(dtype).max
    x = torch.tensor(
        [-2 * limit, -limit, -1.25, 0.0, 1.25, limit, 2 * limit], dtype=torch.float64
    )
    before = x.clone()
    result = finite_cast(x, dtype)
    expected = torch.tensor(
        [-limit, -limit, -1.25, 0.0, 1.25, limit, limit], dtype=dtype
    )
    assert torch.equal(result, expected) and torch.isfinite(result).all()
    assert result.dtype == dtype and result.device == x.device
    assert torch.equal(x, before)


def test_bfloat16_source_uses_exact_fp16_bounds_before_casting():
    # BF16 rounds the FP16 limit 65504 to 65536; clamp in FP32 instead.
    x = torch.tensor([-65536.0, 65536.0], dtype=torch.bfloat16)
    assert finite_cast(x, torch.float16).tolist() == [-65504.0, 65504.0]


def test_widening_conversion_does_not_try_to_represent_large_bounds_in_the_source():
    x = torch.tensor([-65504.0, 65504.0], dtype=torch.float16)
    assert torch.equal(finite_cast(x, torch.float64), x.double())


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_in_range_conversion_matches_normal_casting(dtype):
    x = torch.tensor([-100.0, -0.125, -0.0, 0.0, 0.125, 100.0], dtype=torch.float64)
    result = finite_cast(x, dtype)
    assert torch.equal(result, x.to(dtype))
    assert torch.equal(torch.signbit(result), torch.signbit(x.to(dtype)))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_reconstruction_is_rejected_instead_of_clamped(invalid):
    with pytest.raises(ValueError, match="NaN or infinity"):
        finite_cast(torch.tensor([invalid]), torch.float16)


def test_finite_cast_rejects_nonfloating_data_and_destination():
    with pytest.raises(ValueError, match="floating point"):
        finite_cast(torch.tensor([1]), torch.float16)
    with pytest.raises(ValueError, match="floating point"):
        finite_cast(torch.tensor([1.0]), torch.int32)


@pytest.mark.parametrize(
    "config,raw_max",
    [
        (IntQuantizerConfig(bits=2), 65536.0),
        (FP8QuantizerConfig(), 65520.0),
    ],
)
def test_original_overflow_reconstructions_now_saturate_exactly(config, raw_max):
    x = torch.tensor([-65504.0, 65504.0] * 16, dtype=torch.float16).view(1, 1, 1, 32)
    codec = KVQuantizer.create_from_config(config)
    encoded = codec.quantize(x)
    raw = codec.dequantize(replace(encoded, dtype=torch.float32))
    assert raw.max().item() == raw_max
    assert raw.max() > torch.finfo(torch.float16).max
    assert torch.equal(codec.dequantize(encoded), x)


@pytest.mark.parametrize(
    "config",
    [
        IntQuantizerConfig(bits=2),
        IntQuantizerConfig(bits=4),
        IntQuantizerConfig(bits=8),
        IntQuantizerConfig(bits=2, symmetric=True),
        IntQuantizerConfig(bits=4, symmetric=True),
        FP8QuantizerConfig(),
        FP8QuantizerConfig(format="E5M2"),
    ],
)
@pytest.mark.parametrize("pattern", ["extrema", "zero", "positive", "negative", "tiny"])
def test_fp16_edges_stay_finite_for_integer_and_fp8_codecs(config, pattern):
    patterns = {
        "extrema": [-65504.0, 65504.0] * 16,
        "zero": [0.0] * 32,
        "positive": [5.0] * 32,
        "negative": [-5.0] * 32,
        "tiny": [
            -torch.finfo(torch.float16).smallest_normal / 32,
            torch.finfo(torch.float16).smallest_normal / 32,
        ]
        * 16,
    }
    x = torch.tensor(patterns[pattern], dtype=torch.float16).view(1, 1, 1, 32)
    result = KVQuantizer.create_from_config(config).fake_quantize(x)
    assert result.shape == x.shape and result.dtype == x.dtype
    assert torch.isfinite(result).all()
    assert result.abs().max() <= torch.finfo(x.dtype).max


@pytest.mark.parametrize("config", CODECS)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_codecs_reject_invalid_input_even_when_tokens_are_kept(config, invalid):
    x = torch.ones(1, 1, 1, 32)
    x[..., 0] = invalid
    with pytest.raises(ValueError, match="NaN or infinity"):
        KVQuantizer.create_from_config(config).quantize(
            x,
            PrecisionPlan(
                bits=config.bits,
                keep_tokens=torch.ones(1, 1, 1, dtype=torch.bool),
            ),
        )


@pytest.mark.parametrize(
    "config,field,invalid",
    [
        (IntQuantizerConfig(bits=2), "scales", float("nan")),
        (FP8QuantizerConfig(), "scales", float("inf")),
        (FakeNVFP4QuantizerConfig(), "global_scale", float("nan")),
        (MXFPQuantizerConfig(), "scales", 255),
        (NonUniformQuantizerConfig(), "norms", float("inf")),
    ],
)
def test_nonfinite_decoded_values_are_not_hidden_by_saturation(config, field, invalid):
    codec = KVQuantizer.create_from_config(config)
    encoded = codec.quantize(torch.ones(1, 1, 1, 32))
    getattr(encoded, field).fill_(invalid)
    with pytest.raises(ValueError, match="NaN or infinity"):
        codec.dequantize(encoded)


def test_nan_fp8_codes_are_rejected():
    codec = KVQuantizer.create_from_config(FP8QuantizerConfig())
    encoded = codec.quantize(torch.ones(1, 1, 1, 32))
    encoded.codes.fill_(0x7F)  # E4M3 NaN bit pattern.
    with pytest.raises(ValueError, match="NaN or infinity"):
        codec.dequantize(encoded)


def test_calibrated_levels_outside_output_range_saturate_without_changing_tables():
    codec = KVQuantizer.create_from_config(NonUniformQuantizerConfig(mode="calibrated"))
    samples = torch.tensor([-100000.0, -1.0, 1.0, 100000.0])[:, None].repeat(1, 32)
    codec.calibrate({(0, "key"): samples})
    tables = codec.tables[(0, "key")]
    x = torch.tensor([-65504.0, 65504.0] * 16, dtype=torch.float16).view(1, 1, 1, 32)
    result = codec.fake_quantize(x, context=CONTEXT)
    assert torch.equal(result, x)
    assert torch.equal(codec.tables[(0, "key")], tables)


@pytest.mark.parametrize("config", ROTATIONS)
def test_inverse_rotation_saturates_at_the_original_dtype_boundary(config):
    transform = KVTransform.create_from_config(config)
    original = torch.full((1, 1, 1, 32), 65504.0, dtype=torch.float16)
    rotated, state = transform.forward(original, CONTEXT)
    restored = transform.inverse(rotated * 2, CONTEXT, state)
    assert torch.equal(restored, original)
    assert restored.dtype == torch.float16 and torch.isfinite(restored).all()


@pytest.mark.parametrize("config", [IdentityTransformConfig(), *ROTATIONS])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_inverse_transform_rejects_nonfinite_values(config, invalid):
    transform = KVTransform.create_from_config(config)
    x = torch.ones(1, 1, 1, 32, dtype=torch.float16)
    rotated, state = transform.forward(x, CONTEXT)
    rotated = rotated.clone()
    rotated[..., 0] = invalid
    with pytest.raises(ValueError, match="NaN or infinity"):
        transform.inverse(rotated, CONTEXT, state)


def test_internal_fp32_inverse_overflow_is_an_error_not_a_saturated_value():
    transform = KVTransform.create_from_config(HadamardTransformConfig())
    with pytest.raises(ValueError, match="NaN or infinity"):
        transform.inverse(torch.full((1, 1, 1, 32), 1e38), CONTEXT, torch.float16)


@pytest.mark.parametrize("config", [IdentityTransformConfig(), *ROTATIONS])
def test_fake_and_packed_extreme_kv_agree_after_inverse_conversion(config):
    x = torch.randn(1, 1, 8, 32, generator=torch.Generator().manual_seed(8)) * 30000
    x = x.clamp(-65504, 65504).half()
    codec = IntQuantizerConfig(
        bits=2, scale_dtype=ParamDType.FP32, offset_dtype=ParamDType.FP32
    )
    results = []
    for store_codes in (False, True):
        quant = KVQuantization(
            KVQuantizationConfig(
                key_quantizer=codec,
                value_quantizer=codec,
                key_transform=config,
                value_transform=config,
                cache_policy=FullQuantizationCachePolicyConfig(),
                store_codes=store_codes,
            )
        )
        k, v = quant.process(0, x, x, {}, num_new_tokens=0)
        for restored in (k, v):
            assert restored.dtype == x.dtype and restored.shape == x.shape
            assert torch.isfinite(restored).all()
        # Seeded rotations deliberately differ for keys and values. Compare each
        # kind across storage modes, rather than expecting K and V to be equal.
        results.append((k, v))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


@pytest.mark.parametrize("kind", ["key", "value"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_invalid_current_kv_is_rejected_before_it_enters_an_uncompressed_cache(
    kind, invalid
):
    quant = KVQuantization(
        KVQuantizationConfig()
    )  # Large FP local region: no codec yet.
    keys, values = torch.ones(1, 1, 2, 32), torch.ones(1, 1, 2, 32)
    (keys if kind == "key" else values)[..., 0] = invalid
    metadata = {}
    with pytest.raises(ValueError, match="NaN or infinity"):
        quant.process(0, keys, values, metadata, num_new_tokens=2)
    assert metadata[CACHE_KEY].layers == {}


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_invalid_parameter_values_raise_before_rounding(invalid):
    with pytest.raises(ValueError, match="NaN or infinity"):
        store_param(torch.tensor([invalid]), ParamDType.FP16)


def test_unrepresentable_coding_parameters_still_raise_instead_of_being_clipped():
    with pytest.raises(ValueError, match="scale exceeds"):
        store_param(torch.tensor([100000.0]), ParamDType.FP16)


def test_mixed_channel_bit_reconstruction_saturates_like_fixed_widths():
    codec = KVQuantizer.create_from_config(
        IntQuantizerConfig(
            bits=2,
            group_size=4,
            granularity=QuantGranularity.PER_CHANNEL,
        )
    )
    x = (
        torch.tensor([-65504.0, 65504.0] * 4, dtype=torch.float16)
        .view(1, 1, 8, 1)
        .expand(1, 1, 8, 3)
    )
    result = codec.fake_quantize(
        x, PrecisionPlan(bits=2, channel_bits=torch.tensor([[[2, 4, 8]]]))
    )
    assert torch.equal(result, x) and torch.isfinite(result).all()
