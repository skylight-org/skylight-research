"""Kitty parity against the authors' pinned accuracy simulation (Summer-Summer/Kitty)."""

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    ChannelBoost,
    Kitty,
    KittyConfig,
    KittyQuantizer,
    KittyQuantizerConfig,
    QuantGranularity,
)
from sparse_attention_hub.kv_quantization.implementations.kitty import (
    kitty_channel_boost,
)
from tests.reference.kitty import official_functions

pytestmark = pytest.mark.unit

PER_CHANNEL = QuantGranularity.PER_CHANNEL
DTYPES = [torch.float16, torch.bfloat16]
SINK, PAGE = 32, 128


def kv(tokens: int, dtype: torch.dtype, seed: int = 0, heads: int = 2):
    generator = torch.Generator().manual_seed(seed)
    k = torch.randn(1, heads, tokens, 128, generator=generator)
    k[..., [3, 40, 77]] *= 12.0  # outlier key channels, as after RoPE
    k[..., 90] += 5.0
    v = torch.randn(1, heads, tokens, 128, generator=generator)
    return k.to(dtype), v.to(dtype)


def reference_key_page(x, ratio, selection=1):
    ref = official_functions()
    page = x.transpose(2, 3).contiguous()  # (B, nh, D, T), as the reference stores it
    mask = ref["build_promote_mask"](page, ratio, selection)
    out = ref["fake_quant_groupwise_lastdim"](page, 128, 2, mask, 4)
    return out.transpose(2, 3).contiguous(), mask


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ratio", [0.1, 0.125, 0.2, 0.25])
@pytest.mark.parametrize("selection, name", [(1, "magnitude"), (2, "variance")])
def test_key_page_matches_reference_bit_for_bit(dtype, ratio, selection, name):
    x, _ = kv(PAGE, dtype)
    expected, mask = reference_key_page(x, ratio, selection)
    policy = ChannelBoost(kitty_channel_boost(ratio, name))
    plan = policy.bits_for(0, x, torch.arange(PAGE), tensor_kind="key", default_bits=2)
    assert torch.equal(plan.channel_bits == 4, mask)
    quantizer = KittyQuantizer(
        KittyQuantizerConfig(bits=2, granularity=PER_CHANNEL, group_size=128)
    )
    assert torch.equal(quantizer.dequantize(quantizer.quantize(x, plan)), expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_values_match_reference_bit_for_bit(dtype):
    _, v = kv(300, dtype)
    expected = official_functions()["fake_quant_groupwise_lastdim"](v, 128, 2)
    quantizer = KittyQuantizer(KittyQuantizerConfig(bits=2, group_size=128))
    assert torch.equal(quantizer.dequantize(quantizer.quantize(v)), expected)


def test_constant_groups_use_the_reference_range_floor():
    x = torch.full((1, 1, PAGE, 128), 0.75, dtype=torch.float16)
    expected, mask = reference_key_page(x, 0.125)
    policy = ChannelBoost(kitty_channel_boost(0.125))
    plan = policy.bits_for(0, x, torch.arange(PAGE), tensor_kind="key", default_bits=2)
    quantizer = KittyQuantizer(
        KittyQuantizerConfig(bits=2, granularity=PER_CHANNEL, group_size=128)
    )
    assert torch.equal(quantizer.dequantize(quantizer.quantize(x, plan)), expected)


def reference_run(ratio, k, v, prompt):
    """Attention K/V the authors' KittyKVCache returns for a prompt, then decode."""
    ref = official_functions()
    cache = ref["KittyKVCache"](
        ref["KittyKVCacheConfig"](
            sink_length=SINK,
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


def our_run(config, k, v, prompt):
    kvq = Kitty(config)
    meta: dict = {}
    outputs = [
        kvq.process(
            0, k[..., :prompt, :], v[..., :prompt, :], meta,
            num_new_tokens=prompt, phase="prefill",
        )
    ]
    for t in range(prompt + 1, k.shape[-2] + 1):
        outputs.append(
            kvq.process(0, k[..., :t, :], v[..., :t, :], meta, num_new_tokens=1)
        )
    return outputs


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ratio", [0.125, 0.25])
@pytest.mark.parametrize(
    # Prompt lengths after the sinks: short, exactly one page, one past a page,
    # and pages plus a remainder. Decoding then crosses further page boundaries.
    "after_sink, decode",
    [(1, 140), (100, 160), (128, 140), (129, 140), (300, 230)],
)
def test_prefill_and_decode_match_reference_kv_cache(dtype, ratio, after_sink, decode):
    prompt = SINK + after_sink
    k, v = kv(prompt + decode, dtype, seed=after_sink)
    expected = reference_run(ratio, k, v, prompt)
    actual = our_run(KittyConfig.with_settings(boost_fraction=ratio), k, v, prompt)
    assert len(actual) == len(expected)
    for step, ((ek, ev), (ak, av)) in enumerate(zip(expected, actual)):
        assert torch.equal(ak, ek), f"keys differ at step {step}"
        assert torch.equal(av, ev), f"values differ at step {step}"


@pytest.mark.parametrize("store_codes", [False, True])
def test_whole_page_prompt_differs_only_where_the_reference_requantizes(store_codes):
    """A prompt that fills exactly two pages after the sinks: the reference
    quantizes both pages, then at the first decode step fake-quantizes the second
    page *again* (its slice [-129:-1]). We quantize each page once."""
    prompt = SINK + 2 * PAGE
    k, v = kv(prompt + 20, torch.float16, seed=7)
    expected = reference_run(0.125, k, v, prompt)
    actual = our_run(
        KittyConfig.with_settings(store_codes=store_codes), k, v, prompt
    )
    second_page = slice(SINK + PAGE, SINK + 2 * PAGE)
    for step, ((ek, ev), (ak, av)) in enumerate(zip(expected, actual)):
        assert torch.equal(av, ev)
        if step <= 1:
            assert torch.equal(ak, ek)
            continue
        outside = torch.ones(ak.shape[-2], dtype=torch.bool)
        outside[second_page] = False
        assert torch.equal(ak[..., outside, :], ek[..., outside, :])
        # Our second page is the reference's single (prefill) quantization.
        assert torch.equal(ak[..., second_page, :], expected[1][0][..., second_page, :])
