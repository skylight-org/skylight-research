"""Unit tests for precision policies (Fixed, ChannelBoost, OutlierToken) and the
quantizer / coordinator support they rely on (mixed bits, kept tokens)."""

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    ChannelBoost,
    ChannelBoostConfig,
    DefaultCachePolicyConfig,
    FakeNVFP4QuantizerConfig,
    FixedPrecision,
    FixedPrecisionConfig,
    FP8QuantizerConfig,
    IntQuantizer,
    IntQuantizerConfig,
    KVQuantization,
    KVQuantizationConfig,
    OutlierTokenPolicy,
    OutlierTokenPolicyConfig,
    PrecisionPlan,
    PrecisionPolicy,
    PrecisionPolicyRegistry,
    QuantGranularity,
    RegionConfig,
)

pytestmark = pytest.mark.unit

PER_CHANNEL_INT2 = IntQuantizerConfig(
    bits=2, granularity=QuantGranularity.PER_CHANNEL, group_size=32
)
PER_TOKEN_INT2 = IntQuantizerConfig(bits=2, group_size=32)


def keys(tokens: int = 128, channels: int = 64, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 2, tokens, channels, generator=generator)
    x[..., [3, 17, 40]] *= 10.0  # a few large channels, as real keys have
    return x


def all_middle(**overrides) -> DefaultCachePolicyConfig:
    """No sink and no local window: every token goes to the middle region."""
    return DefaultCachePolicyConfig(
        key_local=RegionConfig(0),
        value_local=RegionConfig(0),
        key_block_tokens=32,
        value_block_tokens=32,
        **overrides,
    )


def coordinator(
    policy, key_quantizer=PER_CHANNEL_INT2, value_quantizer=PER_TOKEN_INT2, **kw
):
    return KVQuantization.create_from_config(
        KVQuantizationConfig(
            key_quantizer=key_quantizer,
            value_quantizer=value_quantizer,
            precision_policy=policy,
            cache_policy=all_middle(),
            **kw,
        )
    )


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / a.norm()).item()


class TestRegistry:
    @pytest.mark.parametrize(
        "config, cls",
        [
            (FixedPrecisionConfig(), FixedPrecision),
            (ChannelBoostConfig(), ChannelBoost),
            (OutlierTokenPolicyConfig(), OutlierTokenPolicy),
        ],
    )
    def test_each_config_selects_its_policy(self, config, cls):
        assert type(PrecisionPolicyRegistry.create(config)) is cls
        assert cls.__bases__ == (PrecisionPolicy,)


class TestMixedBitInt:
    def test_each_channel_uses_its_own_width(self):
        x = keys()
        channel_bits = torch.full((1, 2, 64), 2)
        channel_bits[..., :8] = 4
        quantizer = IntQuantizer(PER_CHANNEL_INT2)
        restored = quantizer.dequantize(
            quantizer.quantize(x, PrecisionPlan(bits=2, channel_bits=channel_bits))
        )
        int4 = IntQuantizer(
            IntQuantizerConfig(
                bits=4, granularity=QuantGranularity.PER_CHANNEL, group_size=32
            )
        ).fake_quantize(x)
        int2 = quantizer.fake_quantize(x)
        assert torch.equal(restored[..., :8], int4[..., :8])
        assert torch.equal(restored[..., 8:], int2[..., 8:])

    def test_storage_counts_each_width(self):
        x = keys()  # 2 heads x 64 channels x 128 tokens
        channel_bits = torch.full((1, 2, 64), 2)
        channel_bits[..., :8] = 4
        encoded = IntQuantizer(PER_CHANNEL_INT2).quantize(
            x, PrecisionPlan(bits=2, channel_bits=channel_bits)
        )
        codes = 2 * 128 * (8 * 4 + 56 * 2) // 8
        groups = 2 * 64 * (128 // 32)
        assert encoded.storage_bytes() == codes + groups * (2 + 2 + 1)

    def test_heads_may_boost_different_channels(self):
        x = keys()
        channel_bits = torch.full((1, 2, 64), 2)
        channel_bits[0, 0, 0] = 8
        channel_bits[0, 1, 5] = 4
        quantizer = IntQuantizer(PER_CHANNEL_INT2)
        restored = quantizer.fake_quantize(
            x, PrecisionPlan(bits=2, channel_bits=channel_bits)
        )
        assert restored.shape == x.shape

    def test_needs_per_channel_granularity(self):
        with pytest.raises(ValueError, match="PER_CHANNEL"):
            IntQuantizer(PER_TOKEN_INT2).quantize(
                keys(), PrecisionPlan(bits=2, channel_bits=torch.full((1, 2, 64), 4))
            )

    def test_other_quantizers_reject_channel_bits(self):
        kvq = KVQuantization.create_from_config(
            KVQuantizationConfig(key_quantizer=FP8QuantizerConfig())
        )
        with pytest.raises(NotImplementedError, match="per-channel bits"):
            kvq.key_quantizer.quantize(
                keys(), PrecisionPlan(bits=8, channel_bits=torch.full((1, 2, 64), 8))
            )


class TestChannelBoost:
    def test_boosts_the_largest_channels_of_keys_only(self):
        policy = ChannelBoost(ChannelBoostConfig(fraction=3 / 64, boosted_bits=4))
        x = keys()
        plan = policy.bits_for(
            0, x, torch.arange(128), tensor_kind="key", default_bits=2
        )
        boosted = (plan.channel_bits == 4).nonzero()[:, -1].reshape(2, 3).sort().values
        assert boosted.tolist() == [[3, 17, 40], [3, 17, 40]]
        value_plan = policy.bits_for(
            0, x, torch.arange(128), tensor_kind="value", default_bits=2
        )
        assert value_plan.channel_bits is None

    def test_fraction_rounds_to_a_channel_count(self):
        policy = ChannelBoost(ChannelBoostConfig(fraction=0.125))
        plan = policy.bits_for(
            0, keys(), torch.arange(128), tensor_kind="key", default_bits=2
        )
        assert (plan.channel_bits == 4).sum(dim=-1).tolist() == [[8, 8]]

    def test_lowers_error_inside_kv_quantization(self):
        x, v = keys(), keys(seed=1)
        fixed, _ = coordinator(FixedPrecisionConfig()).process(
            0, x, v, {}, num_new_tokens=0
        )
        boosted, _ = coordinator(ChannelBoostConfig()).process(
            0, x, v, {}, num_new_tokens=0
        )
        assert rel_error(x, boosted) < 0.8 * rel_error(x, fixed)

    def test_fake_equals_packed(self):
        x, v = keys(), keys(seed=1)
        fake = coordinator(ChannelBoostConfig()).process(0, x, v, {}, num_new_tokens=0)
        packed = coordinator(ChannelBoostConfig(), store_codes=True).process(
            0, x, v, {}, num_new_tokens=0
        )
        assert torch.equal(fake[0], packed[0]) and torch.equal(fake[1], packed[1])

    def test_rejects_quantizers_that_cannot_mix_bits(self):
        with pytest.raises(ValueError, match="PER_CHANNEL"):
            coordinator(ChannelBoostConfig(), key_quantizer=PER_TOKEN_INT2)
        with pytest.raises(ValueError, match="IntQuantizer"):
            coordinator(ChannelBoostConfig(), key_quantizer=FakeNVFP4QuantizerConfig())

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"fraction": 0.0},
            {"boosted_bits": 5},
            {"kinds": ("query",)},
            {"freeze": "always"},
        ],
    )
    def test_invalid_settings(self, kwargs):
        with pytest.raises(ValueError):
            ChannelBoost(ChannelBoostConfig(**kwargs))


def with_outlier_tokens(seed: int = 0) -> torch.Tensor:
    x = torch.randn(1, 2, 128, 64, generator=torch.Generator().manual_seed(seed))
    x[:, 0, [10, 70]] *= 30.0  # two huge tokens in head 0
    x[:, 1, 99] *= 30.0  # one in head 1
    return x


class TestOutlierTokenPolicy:
    def test_flags_only_the_huge_tokens(self):
        policy = OutlierTokenPolicy(OutlierTokenPolicyConfig())
        plan = policy.bits_for(
            0,
            with_outlier_tokens(),
            torch.arange(128),
            tensor_kind="key",
            default_bits=2,
        )
        assert plan.keep_tokens[0, 0].nonzero().flatten().tolist() == [10, 70]
        assert plan.keep_tokens[0, 1].nonzero().flatten().tolist() == [99]

    def test_no_outliers_means_a_plain_plan(self):
        policy = OutlierTokenPolicy(OutlierTokenPolicyConfig())
        plan = policy.bits_for(
            0, keys(), torch.arange(128), tensor_kind="key", default_bits=2
        )
        assert plan.keep_tokens is None

    def test_max_fraction_caps_the_kept_tokens(self):
        policy = OutlierTokenPolicy(OutlierTokenPolicyConfig(max_fraction=1 / 128))
        plan = policy.bits_for(
            0,
            with_outlier_tokens(),
            torch.arange(128),
            tensor_kind="key",
            default_bits=2,
        )
        assert plan.keep_tokens.sum(dim=-1).tolist() == [[1, 1]]

    def test_kept_tokens_come_back_exact_and_others_improve(self):
        x, v = with_outlier_tokens(), keys(seed=1)
        policy = OutlierTokenPolicyConfig(kinds=("key",))
        quantizer = IntQuantizerConfig(bits=2, group_size=32)
        out, _ = coordinator(policy, key_quantizer=quantizer).process(
            0, x, v, {}, num_new_tokens=0
        )
        plain, _ = coordinator(FixedPrecisionConfig(), key_quantizer=quantizer).process(
            0, x, v, {}, num_new_tokens=0
        )
        assert torch.equal(out[0, 0, [10, 70]], x[0, 0, [10, 70]])
        assert torch.equal(out[0, 1, 99], x[0, 1, 99])
        assert rel_error(x, out) < 0.5 * rel_error(x, plain)

    def test_kept_tokens_do_not_widen_per_channel_groups(self):
        """With per-channel groups, a kept token must not set its group's range."""
        x, v = with_outlier_tokens(), keys(seed=1)
        out, _ = coordinator(OutlierTokenPolicyConfig(kinds=("key",))).process(
            0, x, v, {}, num_new_tokens=0
        )
        plain, _ = coordinator(FixedPrecisionConfig()).process(
            0, x, v, {}, num_new_tokens=0
        )
        others = torch.ones(128, dtype=torch.bool)
        others[[10, 70, 99]] = False
        assert rel_error(x[..., others, :], out[..., others, :]) < 0.5 * rel_error(
            x[..., others, :], plain[..., others, :]
        )

    @pytest.mark.parametrize(
        "quantizer",
        [PER_CHANNEL_INT2, FP8QuantizerConfig(), FakeNVFP4QuantizerConfig()],
    )
    def test_fake_equals_packed_and_storage_counts_kept_tokens(self, quantizer):
        x, v = with_outlier_tokens(), keys(seed=1)
        policy = OutlierTokenPolicyConfig(kinds=("key",))
        fake = coordinator(policy, key_quantizer=quantizer).process(
            0, x, v, {}, num_new_tokens=0
        )
        meta: dict = {}
        packed = coordinator(policy, key_quantizer=quantizer, store_codes=True).process(
            0, x, v, meta, num_new_tokens=0
        )
        assert torch.equal(fake[0], packed[0])
        chunks = meta[CACHE_KEY].layers[0].key.chunks
        kept = sum(
            0 if c.kept_values is None else c.kept_values.shape[0] for c in chunks
        )
        assert kept == 3  # tokens 10 and 70 of head 0, token 99 of head 1
        kept_bytes = sum(
            c.kept_values.numel() * c.kept_values.element_size()
            + c.kept_index.numel() * 4
            for c in chunks
            if c.kept_values is not None
        )
        quantized_bytes = sum(c.quantized.storage_bytes() for c in chunks)
        stored = meta[CACHE_KEY].layers[0].value.chunks
        value_bytes = sum(c.quantized.storage_bytes() for c in stored)
        assert (
            meta[CACHE_KEY].stats()["store_bytes"]
            == quantized_bytes + kept_bytes + value_bytes
        )

    @pytest.mark.parametrize(
        "kwargs", [{"threshold": 0.0}, {"max_fraction": 1.5}, {"kinds": ("query",)}]
    )
    def test_invalid_settings(self, kwargs):
        with pytest.raises(ValueError):
            OutlierTokenPolicy(OutlierTokenPolicyConfig(**kwargs))

    def test_empty_keep_mask_preserves_quantized_values(self):
        quantizer = IntQuantizer(PER_TOKEN_INT2)
        x = keys()
        with_mask = quantizer.fake_quantize(
            x,
            PrecisionPlan(bits=2, keep_tokens=torch.zeros(1, 2, 128, dtype=torch.bool)),
        )
        assert torch.equal(with_mask, quantizer.fake_quantize(x))
