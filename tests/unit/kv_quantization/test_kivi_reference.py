"""Golden-code/parameter/reconstruction checks against the unchanged authors' source."""

from dataclasses import replace

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    kivi_int_quantizer,
    CACHE_KEY,
    KIVI_REFERENCE_REVISION,
    IntQuantizer,
    IntQuantizerConfig,
    KIVIConfig,
    KIVIQuantizer,
    KIVIQuantizerConfig,
    KVQuantization,
    KVQuantizer,
    OutlierTokenPolicyConfig,
    ParamDType,
    PrecisionPlan,
    QuantGranularity,
)
from tests.reference.kivi import OfficialKIVICache, decode, encode, official_functions

pytestmark = pytest.mark.unit


def logical_codes(record):
    """Decode the framework's byte layout independently using Python integers."""
    values = [
        (byte >> shift) & (2**record.bits - 1)
        for byte in record.codes.tolist()
        for shift in range(0, 8, record.bits)
    ][: record.metadata["code_count"]]
    b, h, t, d = record.shape
    result = torch.tensor(values, dtype=torch.int16)
    if record.axis == 2:
        return result.reshape(b, h, d, t).transpose(2, 3)
    return result.reshape(record.shape)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("granularity", list(QuantGranularity))
def test_exact_codes_parameters_and_reconstruction_match_official_functions(
    dtype, bits, group_size, granularity
):
    x = torch.randn(2, 3, 64, 128, generator=torch.Generator().manual_seed(124)).to(
        dtype
    )[..., ::2]
    original = x.clone()
    kind = "key" if granularity == QuantGranularity.PER_CHANNEL else "value"
    expected = encode(x, kind, bits, group_size)
    codec = KIVIQuantizer(
        KIVIQuantizerConfig(bits=bits, group_size=group_size, granularity=granularity)
    )
    actual = codec.quantize(x)
    reference_codes = official_functions()["unpack_tensor"](
        expected[0], bits, 2 if kind == "key" else 3
    )
    assert torch.equal(logical_codes(actual), reference_codes)
    scales, minima = actual.scales, actual.zero_points
    if kind == "key":
        scales = scales.permute(0, 1, 3, 4, 2)
        minima = minima.permute(0, 1, 3, 4, 2)
    else:
        scales, minima = scales.reshape_as(expected[1]), minima.reshape_as(expected[2])
    assert scales.dtype == minima.dtype == dtype
    assert torch.equal(scales, expected[1]) and torch.equal(minima, expected[2])
    assert torch.equal(
        codec.dequantize(actual), decode(expected, kind, bits, group_size, dtype=dtype)
    )
    assert actual.metadata["reference_revision"] == KIVI_REFERENCE_REVISION
    assert (
        actual.storage_bytes()
        == x.numel() * bits // 8 + 2 * scales.numel() * x.element_size()
    )
    assert torch.equal(x, original)


def test_default_recipe_selects_the_kivi_preset_and_keeps_other_codecs_available():
    quant = KVQuantization.create_from_config(KIVIConfig())
    assert type(quant.key_quantizer) is IntQuantizer
    assert type(quant.value_quantizer) is IntQuantizer
    assert quant.config.value_quantizer == kivi_int_quantizer()
    dedicated = KVQuantization.create_from_config(
        KIVIConfig(
            key_quantizer=KIVIQuantizerConfig(
                bits=2, granularity=QuantGranularity.PER_CHANNEL
            ),
            value_quantizer=KIVIQuantizerConfig(bits=2),
        )
    )
    assert type(dedicated.key_quantizer) is KIVIQuantizer
    x = torch.randn(2, 3, 32, 64, generator=torch.Generator().manual_seed(424)).half()
    reference = decode(encode(x, "value"), "value", dtype=x.dtype)
    assert torch.equal(quant.value_quantizer.fake_quantize(x), reference)
    generic = IntQuantizer(IntQuantizerConfig(bits=2)).fake_quantize(x)
    assert not torch.equal(generic, reference)
    explicit = KVQuantization.create_from_config(
        KIVIConfig(
            key_quantizer=IntQuantizerConfig(
                bits=2, granularity=QuantGranularity.PER_CHANNEL
            ),
            value_quantizer=IntQuantizerConfig(bits=2),
        )
    )
    assert type(explicit.key_quantizer) is IntQuantizer


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("prefill", [31, 32, 33, 65])
def test_prefill_and_decode_match_the_released_buffer_order(
    dtype, bits, packed, prefill
):
    """Independent reference state covers R-1/R/R+1, long prefill and two flushes."""
    generator = torch.Generator().manual_seed(123)
    keys = torch.randn(2, 2, prefill + 34, 32, generator=generator).to(dtype)
    values = torch.randn(2, 2, prefill + 34, 32, generator=generator).to(dtype)
    coordinator = KVQuantization.create_from_config(
        KIVIConfig.with_settings(
            bits=bits, group_size=16, residual_length=32, store_codes=packed
        )
    )
    reference = OfficialKIVICache(bits, 16, 32)
    metadata = {}
    expected = reference.prefill(keys[..., :prefill, :], values[..., :prefill, :])
    actual = coordinator.process(
        0, *expected, metadata, num_new_tokens=prefill, phase="prefill"
    )
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    for end in range(prefill + 1, keys.shape[-2] + 1):
        expected = reference.step(
            keys[..., end - 1 : end, :], values[..., end - 1 : end, :]
        )
        actual = coordinator.process(
            0, keys[..., :end, :], values[..., :end, :], metadata, num_new_tokens=1
        )
        assert all(torch.equal(a, b) for a, b in zip(actual, expected)), end
        state = metadata[CACHE_KEY].layers[0]
        assert state.key.recent_tokens == end % 32
        assert state.value.recent_tokens == min(end, 32)
        stored = metadata[CACHE_KEY].get(0, coordinator._decode_chunk)
        assert torch.equal(stored[0], reference.get("key"))
        assert torch.equal(stored[1], reference.get("value"))
    coordinator.clear_request(metadata)
    assert CACHE_KEY not in metadata


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("constant", [0.0, 3.25, -3.25])
def test_constant_groups_have_defined_zero_codes_and_exact_reconstruction(
    dtype, constant
):
    x = torch.full((1, 2, 32, 32), constant, dtype=dtype)
    codec = KIVIQuantizer(KIVIQuantizerConfig())
    record = codec.quantize(x)
    assert not record.codes.any() and not record.scales.any()
    assert torch.equal(codec.dequantize(record), x)


@pytest.mark.parametrize("packed", [False, True])
def test_optional_whole_token_preservation_still_excludes_and_restores_originals(
    packed,
):
    x = torch.randn(1, 2, 64, 32, generator=torch.Generator().manual_seed(818)).half()
    x[..., 5, :] = 1000
    quant = KVQuantization.create_from_config(
        KIVIConfig.with_settings(
            residual_length=32,
            store_codes=packed,
            precision_policy=OutlierTokenPolicyConfig(threshold=3, max_fraction=0.125),
        )
    )
    metadata = {}
    quant.process(0, x, x, metadata, num_new_tokens=64, phase="prefill")
    stored = metadata[CACHE_KEY].get(0, quant._decode_chunk)
    assert all(torch.equal(t[..., 5, :], x[..., 5, :]) for t in stored)
    assert all(torch.isfinite(t).all() for t in stored)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"symmetric": True},
        {"bits": 3},
        {"bits": True},
        {"group_size": 0},
        {"group_size": True},
        {"scale_dtype": ParamDType.FP32},
        {"offset_dtype": ParamDType.BF16},
    ],
)
def test_reference_configuration_rejects_incompatible_options(kwargs):
    with pytest.raises(ValueError):
        KVQuantizer.create_from_config(KIVIQuantizerConfig(**kwargs))


@pytest.mark.parametrize(
    "granularity", [QuantGranularity.PER_CHANNEL, QuantGranularity.PER_TOKEN]
)
def test_incomplete_groups_are_rejected(granularity):
    codec = KIVIQuantizer(KIVIQuantizerConfig(group_size=4, granularity=granularity))
    with pytest.raises(ValueError, match="complete groups"):
        codec.quantize(torch.ones(1, 1, 5, 5))


def test_nonconstant_scale_underflow_is_an_actionable_error():
    x = torch.zeros(1, 1, 1, 32, dtype=torch.float16)
    x[..., 0] = torch.finfo(torch.float16).smallest_normal * 2**-10
    with pytest.raises(ValueError, match="underflow"):
        KIVIQuantizer(KIVIQuantizerConfig()).quantize(x)


def test_invalid_input_and_unsupported_plan_are_rejected():
    codec = KIVIQuantizer(KIVIQuantizerConfig())
    with pytest.raises(ValueError, match="NaN"):
        codec.quantize(torch.full((1, 1, 1, 32), float("nan")))
    with pytest.raises(ValueError, match="FP16, BF16 or FP32"):
        codec.quantize(torch.ones(1, 1, 1, 32, dtype=torch.float64))
    with pytest.raises(NotImplementedError, match="per-channel bits"):
        codec.quantize(
            torch.ones(1, 1, 1, 32),
            PrecisionPlan(bits=2, channel_bits=torch.ones(1, 1, 32)),
        )


def test_reconstruction_clamps_finite_overflow_and_rejects_invalid_parameters():
    codec = KIVIQuantizer(KIVIQuantizerConfig())
    record = codec.quantize(torch.zeros(1, 1, 1, 32, dtype=torch.float16))
    record.codes.fill_(255)
    record.scales.fill_(8)
    record.zero_points.fill_(65504)
    assert torch.equal(
        codec.dequantize(record), torch.full(record.shape, 65504, dtype=torch.float16)
    )
    record.scales.fill_(float("nan"))
    with pytest.raises(ValueError, match="NaN or infinity"):
        codec.dequantize(record)
    with pytest.raises(ValueError, match="Unsupported KIVI"):
        codec.dequantize(replace(record, format_id="uniform_int_v1"))


@pytest.mark.parametrize(
    "group_size,residual_length", [(0, 128), (True, 128), (32, 0), (32, True), (32, -1)]
)
def test_invalid_recipe_sizes_fail_before_division_or_cache_mutation(
    group_size, residual_length
):
    with pytest.raises(ValueError, match="positive integer"):
        KVQuantization.create_from_config(
            KIVIConfig.with_settings(
                group_size=group_size, residual_length=residual_length
            )
        )
