"""Kitty recipe: configuration, validation, page-wise boosting and storage."""

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    kitty_int_quantizer,
    CACHE_KEY,
    ChannelBoostConfig,
    FixedPrecisionConfig,
    IntQuantizerConfig,
    Kitty,
    KittyCachePolicyConfig,
    KittyConfig,
    KittyQuantizer,
    KittyQuantizerConfig,
    KVQuantization,
    KVQuantizationConfig,
    QuantGranularity,
    RegionConfig,
    SinkAndLocalCachePolicyConfig,
)

pytestmark = pytest.mark.unit

PER_CHANNEL = QuantGranularity.PER_CHANNEL


def kv(tokens: int, seed: int = 0, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(seed)
    k = torch.randn(1, 2, tokens, 128, generator=generator)
    k[..., [5, 60]] *= 10.0
    v = torch.randn(1, 2, tokens, 128, generator=generator)
    return k.to(dtype), v.to(dtype)


def stored_keys(meta, layer=0):
    return meta[CACHE_KEY].layers[layer].key


class TestConfig:
    def test_defaults_follow_the_paper(self):
        config = KittyConfig()
        assert config.key_quantizer == kitty_int_quantizer(
            bits=2, granularity=PER_CHANNEL, group_size=128
        )
        assert config.value_quantizer == kitty_int_quantizer(bits=2, group_size=128)
        boost = config.precision_policy
        assert isinstance(boost, ChannelBoostConfig)
        assert (boost.fraction, boost.boosted_bits, boost.kinds) == (0.125, 4, ("key",))
        assert (boost.count_rounding, boost.score_dtype) == ("floor", "input")
        policy = config.cache_policy
        assert policy.key_sink.tokens == policy.value_sink.tokens == 32
        assert policy.key_local.tokens == policy.value_local.tokens == 128
        assert config.with_settings() == config

    def test_pro_boosts_a_quarter_of_the_channels(self):
        assert KittyConfig.pro().precision_policy.fraction == 0.25

    def test_zero_boost_is_the_kivi_star_baseline(self):
        config = KittyConfig.with_settings(boost_fraction=0)
        assert isinstance(config.precision_policy, FixedPrecisionConfig)
        assert type(KVQuantization.create_from_config(config)) is Kitty

    def test_settings_reach_every_part(self):
        config = KittyConfig.with_settings(
            boost_fraction=0.25, bits=4, sink_length=16, buffer_length=256,
            group_size=64, channel_selection="variance", store_codes=True,
        )
        assert config.key_quantizer.bits == config.value_quantizer.bits == 4
        assert config.key_quantizer.group_size == 64
        assert config.cache_policy.key_sink.tokens == 16
        assert config.cache_policy.value_local.tokens == 256
        assert config.precision_policy.selection == "variance"
        assert config.store_codes

    def test_registry_selects_the_recipe(self):
        assert type(KVQuantization.create_from_config(KittyConfig())) is Kitty

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"cache_policy": SinkAndLocalCachePolicyConfig()}, "KittyCachePolicyConfig"),
            ({"key_quantizer": KittyQuantizerConfig(bits=2)}, "per channel"),
            (
                {"value_quantizer": KittyQuantizerConfig(granularity=PER_CHANNEL)},
                "values per token",
            ),
            (
                {"precision_policy": ChannelBoostConfig(kinds=("key", "value"))},
                "key channels only",
            ),
            (
                {
                    "cache_policy": KittyCachePolicyConfig(
                        key_local=RegionConfig(96)
                    )
                },
                "multiple of the key group size",
            ),
        ],
    )
    def test_rejects_non_kitty_structure(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            KVQuantization.create_from_config(KittyConfig(**kwargs))

    def test_generic_int_quantizer_is_an_allowed_variant(self):
        config = KittyConfig(
            key_quantizer=IntQuantizerConfig(
                bits=2, granularity=PER_CHANNEL, group_size=128
            ),
            value_quantizer=IntQuantizerConfig(bits=2, group_size=128),
        )
        kvq = KVQuantization.create_from_config(config)
        k, v = kv(32 + 300)
        out_k, _ = kvq.process(0, k, v, {}, num_new_tokens=0)
        assert out_k.shape == k.shape


class TestPages:
    def test_prompt_keys_are_stored_page_by_page(self):
        kvq = Kitty(KittyConfig())
        meta: dict = {}
        k, v = kv(32 + 300)
        kvq.process(0, k, v, meta, num_new_tokens=k.shape[-2], phase="prefill")
        keys = stored_keys(meta)
        assert [c.token_count for c in keys.sink] == [32]
        assert [c.token_count for c in keys.chunks] == [128, 128]
        assert [c.start_token for c in keys.chunks] == [32, 160]
        assert keys.recent_tokens == 44

    def test_each_page_chooses_its_own_boosted_channels(self):
        kvq = Kitty(KittyConfig(store_codes=True))
        meta: dict = {}
        k, v = kv(32 + 256)
        k[..., 32:160, 100] *= 50.0  # channel 100 dominates only the first page
        k[..., 160:, 100] *= 0.01
        kvq.process(0, k, v, meta, num_new_tokens=k.shape[-2], phase="prefill")
        first, second = stored_keys(meta).chunks
        bits = [c.quantized.group_bits.reshape(1, 2, 128) for c in (first, second)]
        assert (bits[0][..., 100] == 4).all()
        assert not (bits[1][..., 100] == 4).any()
        assert (bits[0] == 4).sum(-1).tolist() == [[16, 16]]

    def test_keys_keep_one_to_128_tokens_in_fp_while_decoding(self):
        kvq = Kitty(KittyConfig())
        meta: dict = {}
        k, v = kv(32 + 400)
        for t in range(1, k.shape[-2] + 1):
            kvq.process(0, k[..., :t, :], v[..., :t, :], meta, num_new_tokens=1)
            after_sink = max(0, t - 32)
            keys = stored_keys(meta)
            expected = after_sink if after_sink <= 128 else (after_sink - 1) % 128 + 1
            assert keys.recent_tokens == expected
            assert all(c.token_count == 128 for c in keys.chunks)
            values = meta[CACHE_KEY].layers[0].value
            assert values.recent_tokens == min(after_sink, 128)


class TestStorage:
    def test_fake_and_packed_storage_agree(self):
        k, v = kv(32 + 300)
        outs = []
        for store_codes in (False, True):
            kvq = Kitty(KittyConfig(store_codes=store_codes))
            meta: dict = {}
            kvq.process(0, k[..., :200, :], v[..., :200, :], meta, num_new_tokens=200)
            for t in range(201, k.shape[-2] + 1):
                out = kvq.process(0, k[..., :t, :], v[..., :t, :], meta, num_new_tokens=1)
            outs.append(out)
        assert torch.equal(outs[0][0], outs[1][0])
        assert torch.equal(outs[0][1], outs[1][1])

    def test_packed_key_page_size(self):
        # 2 heads x 128 channels x 128 tokens: 16 channels per head at 4 bits.
        quantizer = KittyQuantizer(
            KittyQuantizerConfig(bits=2, granularity=PER_CHANNEL, group_size=128)
        )
        from sparse_attention_hub.kv_quantization import ChannelBoost
        from sparse_attention_hub.kv_quantization.implementations.kitty import (
            kitty_channel_boost,
        )

        k, _ = kv(128)
        plan = ChannelBoost(kitty_channel_boost(0.125)).bits_for(
            0, k, torch.arange(128), tensor_kind="key", default_bits=2
        )
        record = quantizer.quantize(k, plan)
        code_bytes = 2 * (16 * 128 * 4 + 112 * 128 * 2) // 8
        assert record.codes.numel() == code_bytes
        # Scale and minimum per channel and page (bf16), one width byte per group.
        assert record.storage_bytes() == code_bytes + 2 * 128 * (2 + 2) + 2 * 128

    def test_rejects_incomplete_groups(self):
        quantizer = KittyQuantizer(
            KittyQuantizerConfig(bits=2, granularity=PER_CHANNEL, group_size=128)
        )
        with pytest.raises(ValueError, match="complete groups"):
            quantizer.quantize(kv(100)[0])

    @pytest.mark.parametrize(
        "kwargs",
        [{"bits": 0}, {"symmetric": True}, {"range_eps": 0.0}, {"group_size": 0}],
    )
    def test_invalid_quantizer_settings(self, kwargs):
        with pytest.raises(ValueError):
            KittyQuantizer(KittyQuantizerConfig(**kwargs))


def test_generic_coordinator_accepts_kitty_parts():
    """The parts compose without the named recipe (e.g. for ablations)."""
    config = KVQuantizationConfig(
        key_quantizer=KittyConfig().key_quantizer,
        value_quantizer=KittyConfig().value_quantizer,
        precision_policy=KittyConfig().precision_policy,
        cache_policy=KittyCachePolicyConfig(),
    )
    k, v = kv(32 + 300)
    results = []
    for kvq in (Kitty(KittyConfig()), KVQuantization(config)):
        meta: dict = {}
        kvq.process(0, k[..., :-1, :], v[..., :-1, :], meta, num_new_tokens=0)
        results.append(kvq.process(0, k, v, meta, num_new_tokens=1))
    (ek, ev), (ak, av) = results
    assert not torch.equal(ek, k)  # history really was quantized
    assert torch.equal(ek, ak) and torch.equal(ev, av)
