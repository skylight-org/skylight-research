"""IntQuantizer presets vs the authors' pinned code, bit for bit.

``kivi_int_quantizer()``, ``kitty_int_quantizer()`` and ``quarot_int_quantizer()``
(defined in the recipe files) must reproduce KIVI (new_pack.py), Kitty
(kitty_sim) and QuaRot (ActQuantizer) exactly.
"""

from dataclasses import replace

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    kivi_int_quantizer,
    kitty_int_quantizer,
    quarot_int_quantizer,
    CACHE_KEY,
    ChannelBoost,
    IntQuantizer,
    IntQuantizerConfig,
    Kitty,
    KittyConfig,
    KIVIConfig,
    KVQuantization,
    QuantGranularity,
    QuaRotKVConfig,
)
from sparse_attention_hub.kv_quantization.implementations.kitty import (
    kitty_channel_boost,
)
from sparse_attention_hub.kv_quantization.quantizers.common import unpack_bitstream
from tests.reference.kitty import official_functions as kitty_official
from tests.reference.kivi import OfficialKIVICache, decode, encode
from tests.reference.quarot import encode as quarot_encode

pytestmark = pytest.mark.unit

PER_TOKEN = QuantGranularity.PER_TOKEN
PER_CHANNEL = QuantGranularity.PER_CHANNEL
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def sample(shape, dtype, seed=0):
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(seed))
    x[..., [3, 40 % shape[-1]]] *= 12.0  # outlier channels
    return x.to(dtype)


def logical_codes(record):
    """Unsigned codes in [batch, heads, tokens, channels] order."""
    codes = unpack_bitstream(record.codes, record.bits, record.metadata["code_count"])
    b, h, t, d = record.shape
    if record.axis == 2:
        return codes.reshape(b, h, d, t).transpose(2, 3)
    return codes.reshape(record.shape)


# ---------------------------------------------------------------- KIVI


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("kind", ["key", "value"])
def test_kivi_preset_matches_official_functions(dtype, bits, group_size, kind):
    x = sample((2, 3, 64, 64), dtype, seed=124)
    original = x.clone()
    granularity = PER_CHANNEL if kind == "key" else PER_TOKEN
    expected = encode(x, kind, bits, group_size)
    codec = IntQuantizer(
        kivi_int_quantizer(bits=bits, granularity=granularity, group_size=group_size)
    )
    actual = codec.quantize(x)
    from tests.reference.kivi import official_functions

    reference_codes = official_functions()["unpack_tensor"](
        expected[0], bits, 2 if kind == "key" else 3
    )
    assert torch.equal(logical_codes(actual).to(reference_codes.dtype), reference_codes)
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
    assert torch.equal(x, original)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("prefill", [31, 32, 33, 65])
def test_kivi_preset_in_the_recipe_matches_the_released_buffer_order(
    dtype, bits, packed, prefill
):
    generator = torch.Generator().manual_seed(123)
    keys = torch.randn(2, 2, prefill + 34, 32, generator=generator).to(dtype)
    values = torch.randn(2, 2, prefill + 34, 32, generator=generator).to(dtype)
    config = replace(
        KIVIConfig.with_settings(
            bits=bits, group_size=16, residual_length=32, store_codes=packed
        ),
        key_quantizer=kivi_int_quantizer(bits, PER_CHANNEL, 16),
        value_quantizer=kivi_int_quantizer(bits, PER_TOKEN, 16),
    )
    coordinator = KVQuantization.create_from_config(config)
    assert type(coordinator.key_quantizer) is IntQuantizer
    reference = OfficialKIVICache(bits, 16, 32)
    metadata: dict = {}
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


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("constant", [0.0, 3.25, -3.25])
def test_kivi_preset_constant_groups_decode_exactly(dtype, constant):
    x = torch.full((1, 2, 32, 32), constant, dtype=dtype)
    codec = IntQuantizer(kivi_int_quantizer())
    record = codec.quantize(x)
    assert not record.codes.any() and not record.scales.any()
    assert torch.equal(codec.dequantize(record), x)


# ---------------------------------------------------------------- Kitty

PAGE = 128


def kitty_reference_key_page(x, ratio, selection=1):
    ref = kitty_official()
    page = x.transpose(2, 3).contiguous()  # (B, nh, D, T), as the reference stores it
    mask = ref["build_promote_mask"](page, ratio, selection)
    out = ref["fake_quant_groupwise_lastdim"](page, 128, 2, mask, 4)
    return out.transpose(2, 3).contiguous(), mask


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ratio", [0.1, 0.125, 0.2, 0.25])
@pytest.mark.parametrize("selection, name", [(1, "magnitude"), (2, "variance")])
def test_kitty_preset_key_page_matches_reference(dtype, ratio, selection, name):
    x = sample((1, 2, PAGE, 128), dtype)
    x[..., 90] += 5.0
    expected, mask = kitty_reference_key_page(x, ratio, selection)
    plan = ChannelBoost(kitty_channel_boost(ratio, name)).bits_for(
        0, x, torch.arange(PAGE), tensor_kind="key", default_bits=2
    )
    assert torch.equal(plan.channel_bits == 4, mask)
    codec = IntQuantizer(kitty_int_quantizer(granularity=PER_CHANNEL))
    record = codec.quantize(x, plan)
    assert torch.equal(codec.dequantize(record), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bits", [1, 2, 3, 4, 8])
def test_kitty_preset_values_match_reference(dtype, bits):
    v = sample((1, 2, 300, 128), dtype, seed=3)
    expected = kitty_official()["fake_quant_groupwise_lastdim"](v, 128, bits)
    codec = IntQuantizer(kitty_int_quantizer(bits=bits))
    assert torch.equal(codec.fake_quantize(v), expected)


def test_kitty_preset_constant_groups_use_the_reference_range_floor():
    x = torch.full((1, 1, PAGE, 128), 0.75, dtype=torch.float16)
    expected, _ = kitty_reference_key_page(x, 0.125)
    plan = ChannelBoost(kitty_channel_boost(0.125)).bits_for(
        0, x, torch.arange(PAGE), tensor_kind="key", default_bits=2
    )
    codec = IntQuantizer(kitty_int_quantizer(granularity=PER_CHANNEL))
    assert torch.equal(codec.dequantize(codec.quantize(x, plan)), expected)


def kitty_reference_run(ratio, k, v, prompt):
    ref = kitty_official()
    cache = ref["KittyKVCache"](
        ref["KittyKVCacheConfig"](
            sink_length=32,
            buffer_length=PAGE,
            group_size=PAGE,
            kbits=2,
            vbits=2,
            promote_ratio=ratio,
            promote_bit=4,
            channel_selection=1,
        )
    )
    outputs = [cache.update(k[..., :prompt, :], v[..., :prompt, :], 0)]
    for t in range(prompt, k.shape[-2]):
        outputs.append(cache.update(k[..., t : t + 1, :], v[..., t : t + 1, :], 0))
    return outputs


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ratio", [0.125, 0.25])
@pytest.mark.parametrize("after_sink, decode_steps", [(1, 140), (129, 140), (300, 230)])
def test_kitty_preset_in_the_recipe_matches_reference_kv_cache(
    dtype, ratio, after_sink, decode_steps
):
    prompt = 32 + after_sink
    k, v = sample((1, 2, prompt + decode_steps, 128), dtype, seed=after_sink), sample(
        (1, 2, prompt + decode_steps, 128), dtype, seed=after_sink + 1
    )
    expected = kitty_reference_run(ratio, k, v, prompt)
    config = KittyConfig.with_settings(
        boost_fraction=ratio,
        key_quantizer=kitty_int_quantizer(granularity=PER_CHANNEL),
        value_quantizer=kitty_int_quantizer(),
    )
    kvq = Kitty(config)
    assert type(kvq.key_quantizer) is IntQuantizer
    meta: dict = {}
    actual = [
        kvq.process(
            0, k[..., :prompt, :], v[..., :prompt, :], meta,
            num_new_tokens=prompt, phase="prefill",
        )
    ]
    for t in range(prompt + 1, k.shape[-2] + 1):
        actual.append(kvq.process(0, k[..., :t, :], v[..., :t, :], meta, num_new_tokens=1))
    assert len(actual) == len(expected)
    for step, ((ek, ev), (ak, av)) in enumerate(zip(expected, actual)):
        assert torch.equal(ak, ek), f"keys differ at step {step}"
        assert torch.equal(av, ev), f"values differ at step {step}"
    assert CACHE_KEY in meta


# ---------------------------------------------------------------- QuaRot


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize(
    "mode,group", [("key_head", 128), ("value_group", 128), ("value_group", 32)]
)
@pytest.mark.parametrize("clip", [0.95, 1.0])
def test_quarot_preset_matches_official(dtype, bits, symmetric, mode, group, clip):
    x = torch.randn(2, 3, 4, 256, generator=torch.Generator().manual_seed(824)).to(
        dtype
    )[..., ::2]
    original = x.clone()
    expected = quarot_encode(
        x, mode=mode, group_size=group, bits=bits, symmetric=symmetric, clip_ratio=clip
    )
    codec = IntQuantizer(
        quarot_int_quantizer(
            mode, bits=bits, group_size=group, symmetric=symmetric, clip_ratio=clip
        )
    )
    actual = codec.quantize(x)
    bias = 2 ** (bits - 1) if symmetric else 0
    assert torch.equal(
        logical_codes(actual).to(torch.int64) - bias,
        expected.logical_codes.to(torch.int64),
    )
    assert actual.scales.dtype == expected.compact_scales.dtype
    assert torch.equal(actual.scales, expected.compact_scales)
    if symmetric:
        assert actual.zero_points is None
    else:
        assert actual.zero_points.dtype == expected.compact_zero_points.dtype
        assert torch.equal(actual.zero_points, expected.compact_zero_points)
    assert torch.equal(codec.dequantize(actual), expected.reconstructed)
    assert torch.equal(x, original)


@pytest.mark.parametrize("mode", ["key_head", "value_group"])
@pytest.mark.parametrize("constant", [0.0, 3.25, -3.25])
def test_quarot_preset_zero_and_constant_groups(mode, constant):
    x = torch.full((1, 2, 3, 128), constant, dtype=torch.float16)
    codec = IntQuantizer(quarot_int_quantizer(mode))
    if mode == "value_group" and constant != 0:
        # The reference divides by a zero scale here (infinite zero point).
        with pytest.raises(ValueError, match="nonpositive"):
            codec.quantize(x)
    else:
        assert torch.equal(
            codec.fake_quantize(x), quarot_encode(x, mode=mode).reconstructed
        )


def test_quarot_preset_parameter_dtypes_follow_the_upstream_paths():
    x = torch.linspace(-1, 2, 128).half().reshape(1, 1, 1, 128)
    key = IntQuantizer(quarot_int_quantizer("key_head")).quantize(x)
    value = IntQuantizer(quarot_int_quantizer("value_group")).quantize(x)
    assert key.scales.dtype == key.zero_points.dtype == torch.float32
    assert value.scales.dtype == value.zero_points.dtype == torch.float16


def test_quarot_recipe_rejects_int_settings_that_are_not_quarot():
    with pytest.raises(ValueError, match="quarot"):
        KVQuantization.create_from_config(
            QuaRotKVConfig(key_quantizer=quarot_int_quantizer("value_group"))
        )
    with pytest.raises(ValueError, match="quarot"):
        KVQuantization.create_from_config(
            QuaRotKVConfig(value_quantizer=kivi_int_quantizer(bits=4, group_size=128))
        )


# ---------------------------------------------------------------- Generic


def test_generic_default_is_unchanged():
    x = sample((1, 2, 32, 64), torch.float16)
    record = IntQuantizer(IntQuantizerConfig(bits=4)).quantize(x)
    assert record.format_id == "uniform_int_v1"
    assert record.scales.dtype == torch.float16


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"clip_ratio": 0.9}, "parameter_storage"),
        ({"compute_dtype": "input"}, "parameter_storage"),
        ({"parameter_storage": "compute", "compute_dtype": "fp64"}, "compute_dtype"),
        ({"parameter_storage": "compute", "clip_ratio": 0.0}, "clip_ratio"),
        (
            {"parameter_storage": "compute", "zero_point": "integer",
             "degenerate_range": "zero_scale"},
            "nonzero scale",
        ),
        (
            {"parameter_storage": "compute", "symmetric": True,
             "degenerate_range": "eps"},
            "Symmetric",
        ),
        ({"parameter_storage": "compute", "symmetric_range": "full"}, "symmetric"),
        ({"parameter_storage": "compute", "range_eps": 1e-3}, "range_eps"),
        ({"parameter_storage": "compute", "bits": 9}, "1 to 8"),
    ],
)
def test_invalid_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        IntQuantizer(IntQuantizerConfig(**kwargs))


def test_complete_groups_rejects_partial_groups():
    codec = IntQuantizer(kivi_int_quantizer(granularity=PER_CHANNEL))
    with pytest.raises(ValueError, match="complete_groups"):
        codec.quantize(sample((1, 1, 40, 32), torch.float16))
