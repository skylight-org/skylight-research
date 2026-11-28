"""Generic kept-token exclusion: numerical oracles and codec/coordinator checks."""

from dataclasses import replace

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    DefaultCachePolicyConfig,
    FakeNVFP4QuantizerConfig,
    FP8QuantizerConfig,
    HadamardTransformConfig,
    IntQuantizer,
    IntQuantizerConfig,
    KVQuantization,
    KVQuantizationConfig,
    KVQuantizer,
    MXFPQuantizerConfig,
    NonUniformQuantizerConfig,
    OutlierTokenPolicyConfig,
    ParamDType,
    PrecisionPlan,
    QuantGranularity,
    RegionConfig,
)
from sparse_attention_hub.kv_quantization.context import AttentionContext
from sparse_attention_hub.kv_quantization.quantizers.common import check_input

pytestmark = pytest.mark.unit
CHANNEL = QuantGranularity.PER_CHANNEL
TOKEN = QuantGranularity.PER_TOKEN
BLOCK = QuantGranularity.PER_BLOCK
CONTEXT = AttentionContext(0, "key", 0, "decode")


def whole_chunk_policy(tokens):
    # Flush a complete input chunk, compatible with per-channel group sizes.
    return DefaultCachePolicyConfig(
        key_local=RegionConfig(0),
        value_local=RegionConfig(0),
        key_block_tokens=tokens,
        value_block_tokens=tokens,
    )


CODECS = [
    pytest.param(
        IntQuantizerConfig(bits=2, group_size=4, granularity=CHANNEL), id="int2-channel"
    ),
    pytest.param(
        IntQuantizerConfig(bits=4, group_size=4, granularity=TOKEN), id="int4-token"
    ),
    pytest.param(
        IntQuantizerConfig(bits=8, group_size=4, granularity=BLOCK), id="int8-block"
    ),
    pytest.param(
        IntQuantizerConfig(bits=2, group_size=4, granularity=CHANNEL, symmetric=True),
        id="int2-symmetric",
    ),
    pytest.param(
        FP8QuantizerConfig(group_size=4, granularity=CHANNEL), id="fp8-e4m3-channel"
    ),
    pytest.param(FP8QuantizerConfig(group_size=4, format="E5M2"), id="fp8-e5m2-token"),
    pytest.param(FakeNVFP4QuantizerConfig(granularity=CHANNEL), id="nvfp4-channel"),
    pytest.param(FakeNVFP4QuantizerConfig(), id="nvfp4-token"),
    pytest.param(MXFPQuantizerConfig(granularity=CHANNEL), id="mxfp4-channel"),
    pytest.param(MXFPQuantizerConfig(granularity=BLOCK), id="mxfp4-block"),
    pytest.param(
        MXFPQuantizerConfig(bits=8, format="E4M3", granularity=CHANNEL),
        id="mxfp8-e4m3-channel",
    ),
    pytest.param(MXFPQuantizerConfig(bits=8, format="E5M2"), id="mxfp8-e5m2-token"),
    pytest.param(NonUniformQuantizerConfig(), id="nonuniform-gaussian"),
    pytest.param(
        NonUniformQuantizerConfig(mode="calibrated"), id="nonuniform-calibrated"
    ),
]


def make_quantizer(config):
    quantizer = KVQuantizer.create_from_config(config)
    if isinstance(config, NonUniformQuantizerConfig) and config.mode == "calibrated":
        # Four deterministic calibration vectors; encoding keeps this table fixed.
        samples = torch.tensor([-2.0, -0.5, 0.5, 2.0]).repeat(32, 1).T
        quantizer.calibrate({(0, "key"): samples})
    return quantizer


def inputs():
    x = torch.randn(2, 2, 35, 32, generator=torch.Generator().manual_seed(71))
    keep = torch.zeros(x.shape[:3], dtype=torch.bool)
    keep[..., [0, 3, 4, 17, 34]] = True
    keep[0, 0, :4] = True  # Entire numerical group, for INT/FP8.
    keep[1, 0, 32:] = True  # Entire partial last group, including padding.
    keep[1, 1] = True  # Entire head; other heads must remain independent.
    return x, keep


@pytest.mark.parametrize("config", CODECS)
def test_preserved_outlier_magnitude_cannot_change_the_encoding(config):
    """Fixed mask: kept values cannot affect any code, scale or norm."""
    quantizer = make_quantizer(config)
    x, keep = inputs()
    changed = x.clone()
    x[keep] = 10_000
    changed[keep] = -1_000_000
    original = x.clone()
    plan = PrecisionPlan(bits=config.bits, keep_tokens=keep)
    a = quantizer.quantize(x, plan, context=CONTEXT)
    b = quantizer.quantize(changed, plan, context=CONTEXT)
    for field in ("codes", "scales", "zero_points", "norms", "global_scale"):
        left, right = getattr(a, field), getattr(b, field)
        assert (left is None) == (right is None)
        if left is not None:
            assert torch.equal(left, right), field
            assert torch.isfinite(left).all(), field
    restored = quantizer.dequantize(a)
    assert restored.shape == x.shape and torch.isfinite(restored).all()
    assert torch.equal(restored[~keep], quantizer.dequantize(b)[~keep])
    assert torch.equal(x, original)


@pytest.mark.parametrize("config", CODECS)
def test_an_empty_exclusion_mask_preserves_the_normal_numerics(config):
    quantizer = make_quantizer(config)
    x, keep = inputs()
    plan = PrecisionPlan(bits=config.bits, keep_tokens=torch.zeros_like(keep))
    plain = quantizer.quantize(x, context=CONTEXT)
    excluded = quantizer.quantize(x, plan, context=CONTEXT)
    assert torch.equal(quantizer.dequantize(plain), quantizer.dequantize(excluded))
    for field in ("scales", "zero_points", "norms", "global_scale"):
        left, right = getattr(plain, field), getattr(excluded, field)
        if left is not None:
            assert torch.equal(left, right), field


@pytest.mark.parametrize("config", CODECS)
def test_all_kept_tokens_restore_exactly_in_fake_and_packed_storage(config):
    x = torch.full((1, 2, 35, 32), 100_000.0)
    outputs = []
    for store_codes in (False, True):
        quant = KVQuantization(
            KVQuantizationConfig(
                key_quantizer=config,
                value_quantizer=config,
                cache_policy=whole_chunk_policy(35),
                precision_policy=OutlierTokenPolicyConfig(
                    threshold=0.01, max_fraction=1
                ),
                store_codes=store_codes,
            )
        )
        if (
            isinstance(config, NonUniformQuantizerConfig)
            and config.mode == "calibrated"
        ):
            table = torch.tensor([-2.0, -0.5, 0.5, 2.0]).repeat(32, 1)
            quant.key_quantizer.calibrate({(0, "key"): table.T})
            quant.value_quantizer.calibrate({(0, "value"): table.T})
        meta = {}
        k, v = quant.process(0, x, x, meta, num_new_tokens=0)
        assert torch.equal(k, x) and torch.equal(v, x)
        chunk = meta[CACHE_KEY].layers[0].key.chunks[0]
        if store_codes:
            assert chunk.kept_values.shape[0] == 70
        else:
            assert chunk.kept_values is None and chunk.kept_index is None
        outputs.append((k, v))
    assert torch.equal(outputs[0][0], outputs[1][0])


@pytest.mark.parametrize("store_codes", [False, True])
def test_original_cross_group_failure_has_independent_expected_values(store_codes):
    x = torch.tensor([100.0, 100.0, 100.0, 100.0, 9999.0, 1.0, 2.0, 3.0]).view(
        1, 1, 8, 1
    )
    codec = IntQuantizerConfig(
        bits=2,
        group_size=4,
        granularity=CHANNEL,
        scale_dtype=ParamDType.FP32,
        offset_dtype=ParamDType.FP32,
    )
    quant = KVQuantization(
        KVQuantizationConfig(
            key_quantizer=codec,
            value_quantizer=codec,
            cache_policy=whole_chunk_policy(8),
            precision_policy=OutlierTokenPolicyConfig(threshold=2, max_fraction=1 / 8),
            store_codes=store_codes,
        )
    )
    k, v = quant.process(0, x, x, {}, num_new_tokens=0)
    expected = x.clone()
    expected[..., 6, :] = 7 / 3
    assert torch.allclose(k, expected, atol=1e-6, rtol=0)
    assert torch.equal(k, v)
    assert k[..., 4, :].item() == 9999


@pytest.mark.parametrize("survivors", [[1.0, 2.0, 3.0], [-3.0, -2.0, -1.0]])
def test_dummy_zero_does_not_widen_a_one_sided_integer_range(survivors):
    x = torch.tensor([9999.0, *survivors]).view(1, 1, 4, 1)
    keep = torch.tensor([[[True, False, False, False]]])
    codec = IntQuantizer(
        IntQuantizerConfig(
            bits=2,
            group_size=4,
            granularity=CHANNEL,
            scale_dtype=ParamDType.FP32,
            offset_dtype=ParamDType.FP32,
        )
    )
    encoded = codec.quantize(x, PrecisionPlan(bits=2, keep_tokens=keep))
    assert encoded.zero_points.item() == min(survivors)
    assert encoded.scales.item() == pytest.approx(2 / 3)


@pytest.mark.parametrize("symmetric", [False, True])
def test_mixed_channel_widths_honor_exclusion_and_preserve_normal_channels(symmetric):
    codec = IntQuantizer(
        IntQuantizerConfig(
            bits=2,
            group_size=4,
            granularity=CHANNEL,
            symmetric=symmetric,
        )
    )
    x, keep = inputs()
    changed = x.clone()
    changed[keep] = 1_000_000
    widths = torch.tensor([2, 4, 8, 2] * 8).expand(2, 2, 32)
    plan = PrecisionPlan(bits=2, channel_bits=widths, keep_tokens=keep)
    a, b = codec.quantize(x, plan), codec.quantize(changed, plan)
    assert torch.equal(a.codes, b.codes)
    assert torch.equal(a.scales, b.scales)
    assert torch.equal(codec.dequantize(a)[~keep], codec.dequantize(b)[~keep])
    # An independent fixed-width call provides the per-channel numerical oracle.
    mixed = codec.dequantize(a)
    for width in (2, 4, 8):
        reference_codec = IntQuantizer(replace(codec.config, bits=width))
        reference = reference_codec.fake_quantize(
            x, replace(plan, bits=width, channel_bits=None)
        )
        channels = widths == width
        assert torch.equal(
            mixed.movedim(-1, -2)[channels], reference.movedim(-1, -2)[channels]
        )


@pytest.mark.parametrize("store_codes", [False, True])
def test_exact_tokens_are_restored_in_original_coordinates_after_rotation(store_codes):
    x = torch.randn(1, 1, 8, 32, generator=torch.Generator().manual_seed(13))
    x[..., 4, :] *= 1000
    quant = KVQuantization(
        KVQuantizationConfig(
            key_transform=HadamardTransformConfig(),
            cache_policy=whole_chunk_policy(8),
            precision_policy=OutlierTokenPolicyConfig(threshold=2, max_fraction=1 / 8),
            store_codes=store_codes,
        )
    )
    result, _ = quant.process(0, x, x, {}, num_new_tokens=0)
    assert torch.equal(result[..., 4, :], x[..., 4, :])


@pytest.mark.parametrize(
    "mask",
    [
        torch.zeros(1, 1, 8),
        torch.zeros(1, 1, 8, dtype=torch.uint8),
        torch.zeros(1, 1, 7, dtype=torch.bool),
        torch.zeros(8, dtype=torch.bool),
    ],
)
def test_invalid_exclusion_masks_fail_before_encoding(mask):
    with pytest.raises(ValueError, match="boolean with shape"):
        IntQuantizer(IntQuantizerConfig()).quantize(
            torch.ones(1, 1, 8, 32),
            PrecisionPlan(bits=4, keep_tokens=mask),
        )


def test_mask_device_must_match_the_input():
    with pytest.raises(ValueError, match="input device"):
        IntQuantizer(IntQuantizerConfig()).quantize(
            torch.ones(1, 1, 8, 32),
            PrecisionPlan(
                bits=4,
                keep_tokens=torch.zeros(1, 1, 8, dtype=torch.bool, device="meta"),
            ),
        )


def test_codecs_must_explicitly_opt_into_the_generic_exclusion_contract():
    with pytest.raises(NotImplementedError, match="keep_tokens"):
        check_input(
            torch.ones(1, 1, 8, 32),
            PrecisionPlan(
                bits=4,
                keep_tokens=torch.zeros(1, 1, 8, dtype=torch.bool),
            ),
        )
