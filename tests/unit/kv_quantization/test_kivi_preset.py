"""Unit tests for the KIVI preset."""

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    kivi_int_quantizer,
    CACHE_KEY,
    KIVI,
    FixedPrecision,
    FP8QuantizerConfig,
    IdentityTransform,
    IntQuantizer,
    IntQuantizerConfig,
    KittyCachePolicyConfig,
    KIVIConfig,
    KVQuantization,
    KVQuantizationConfig,
    OutlierTokenPolicyConfig,
    ParamDType,
    QuantGranularity,
    RandomHadamardTransformConfig,
    RegionConfig,
    SinkAndLocalCachePolicy,
    SinkAndLocalCachePolicyConfig,
)
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig,
)

pytestmark = pytest.mark.unit

PER_CHANNEL = QuantGranularity.PER_CHANNEL


def history(tokens: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 2, tokens, 64, generator=generator)
    x[..., [3, 40]] *= 10.0  # outlier channels, as real keys have
    return x


def run(kvq: KVQuantization, k: torch.Tensor, v: torch.Tensor, steps):
    meta: dict = {}
    for t in steps:
        out = kvq.process(0, k[..., :t, :], v[..., :t, :], meta, num_new_tokens=0)
    return out, meta


class TestConstruction:
    def test_config_builds_the_kivi_class(self):
        kvq = KVQuantization.create_from_config(KIVIConfig())
        assert type(kvq) is KIVI
        assert isinstance(kvq, KVQuantization)
        assert isinstance(kvq.cache_policy, SinkAndLocalCachePolicy)
        assert isinstance(kvq.precision_policy, FixedPrecision)
        assert isinstance(kvq.key_transform, IdentityTransform)
        assert type(kvq.key_quantizer) is IntQuantizer
        assert type(kvq.value_quantizer) is IntQuantizer
        assert kvq.config.key_quantizer == kivi_int_quantizer(
            granularity=PER_CHANNEL
        )
        assert kvq.config.value_quantizer == kivi_int_quantizer()

    def test_defaults_are_kivi_2(self):
        config = KIVIConfig()
        assert (config.key_quantizer.bits, config.value_quantizer.bits) == (2, 2)
        assert QuantGranularity(config.key_quantizer.granularity) == PER_CHANNEL
        assert (
            QuantGranularity(config.value_quantizer.granularity)
            == QuantGranularity.PER_TOKEN
        )
        assert (
            config.key_quantizer.group_size == config.value_quantizer.group_size == 32
        )
        assert (
            not config.key_quantizer.symmetric and not config.value_quantizer.symmetric
        )
        assert config.key_quantizer.scale_dtype == ParamDType.FP16
        assert (
            config.cache_policy.key_local.tokens
            == config.cache_policy.value_local.tokens
            == 128
        )
        assert config.cache_policy.key_sink.tokens == 0

    def test_through_research_attention(self):
        attention = ResearchAttention.create_from_config(
            ResearchAttentionConfig(
                masker_configs=[], kv_quantization_config=KIVIConfig()
            )
        )
        assert type(attention.kv_quantization) is KIVI


class TestNumerics:
    def test_same_numbers_as_the_equivalent_generic_config(self):
        """The recipe wires the reference codec and shared buffering correctly."""
        k, v = history(300), history(300, seed=1)
        generic = KVQuantization.create_from_config(
            KVQuantizationConfig(
                key_quantizer=kivi_int_quantizer(bits=2, granularity=PER_CHANNEL),
                value_quantizer=kivi_int_quantizer(bits=2),
                cache_policy=SinkAndLocalCachePolicyConfig(),
            )
        )
        steps = [200, *range(201, 301)]
        (kivi_k, kivi_v), _ = run(
            KVQuantization.create_from_config(KIVIConfig()), k, v, steps
        )
        (gen_k, gen_v), _ = run(generic, k, v, steps)
        assert torch.equal(kivi_k, gen_k) and torch.equal(kivi_v, gen_v)

    def test_schedule_keys_flush_every_r_values_roll(self):
        k, v = history(300), history(300, seed=1)
        kvq = KVQuantization.create_from_config(KIVIConfig())
        meta: dict = {}
        for t in range(1, 301):
            kvq.process(0, k[..., :t, :], v[..., :t, :], meta, num_new_tokens=0)
            stats = meta[CACHE_KEY].stats()["layers"][0]
            assert stats["key_recent_tokens"] == t % 128
            assert stats["value_recent_tokens"] == min(t, 128)

    def test_full_precision_tails_are_exact(self):
        k, v = history(300), history(300, seed=1)
        (out_k, out_v), _ = run(
            KVQuantization.create_from_config(KIVIConfig()), k, v, [300]
        )
        assert torch.equal(out_k[..., 256:, :], k[..., 256:, :])  # 300 % 128 = 44 keys
        assert torch.equal(out_v[..., -128:, :], v[..., -128:, :])
        assert not torch.equal(out_k[..., :256, :], k[..., :256, :])

    def test_fake_equals_packed_and_packed_is_small(self):
        k, v = history(512), history(512, seed=1)
        fake, fake_meta = run(
            KVQuantization.create_from_config(KIVIConfig()), k, v, [512]
        )
        packed, packed_meta = run(
            KVQuantization.create_from_config(KIVIConfig(store_codes=True)), k, v, [512]
        )
        assert torch.equal(fake[0], packed[0]) and torch.equal(fake[1], packed[1])
        fake_bytes = fake_meta[CACHE_KEY].stats()["store_bytes"]
        packed_bytes = packed_meta[CACHE_KEY].stats()["store_bytes"]
        assert packed_bytes < 0.5 * fake_bytes

    def test_per_channel_keys_beat_per_token_on_outlier_channels(self):
        """KIVI's reason for per-channel keys."""
        k, v = history(512), history(512, seed=1)
        (kivi_k, _), _ = run(
            KVQuantization.create_from_config(KIVIConfig()), k, v, [512]
        )
        per_token = KIVIConfig(
            key_quantizer=IntQuantizerConfig(bits=2),
            cache_policy=SinkAndLocalCachePolicyConfig(),
        )
        with pytest.raises(ValueError, match="per channel"):
            KVQuantization.create_from_config(per_token)
        generic = KVQuantization.create_from_config(
            KVQuantizationConfig(
                key_quantizer=IntQuantizerConfig(bits=2),
                value_quantizer=IntQuantizerConfig(bits=2),
                cache_policy=SinkAndLocalCachePolicyConfig(),
            )
        )
        (token_k, _), _ = run(generic, k, v, [512])

        def error(a):
            return ((a - k).norm() / k.norm()).item()

        assert error(kivi_k) < error(token_k)


class TestVariants:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_with_settings(self, bits):
        config = KIVIConfig.with_settings(bits=bits, group_size=64, residual_length=256)
        assert config.key_quantizer.bits == config.value_quantizer.bits == bits
        assert config.key_quantizer.group_size == 64
        assert config.cache_policy.key_local.tokens == 256
        assert type(KVQuantization.create_from_config(config)) is KIVI

    def test_more_bits_less_error(self):
        k, v = history(512), history(512, seed=1)
        errors = []
        for bits in (2, 4, 8):
            kvq = KVQuantization.create_from_config(KIVIConfig.with_settings(bits=bits))
            (out_k, _), _ = run(kvq, k, v, [512])
            errors.append(((out_k - k).norm() / k.norm()).item())
        assert errors[0] > errors[1] > errors[2]

    def test_research_knobs_are_allowed(self):
        """Only KIVI's defining choices are checked; everything else stays free."""
        overrides = KIVIConfig.with_settings(
            key_transform=RandomHadamardTransformConfig(),
            precision_policy=OutlierTokenPolicyConfig(),
            store_codes=True,
            error_metric=True,
        )
        fp32_scales_and_sinks = KIVIConfig(
            key_quantizer=IntQuantizerConfig(
                bits=2, granularity=PER_CHANNEL, scale_dtype=ParamDType.FP32
            ),
            cache_policy=SinkAndLocalCachePolicyConfig(key_sink=RegionConfig(4)),
        )
        for config in (overrides, fp32_scales_and_sinks):
            assert type(KVQuantization.create_from_config(config)) is KIVI


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"key_quantizer": FP8QuantizerConfig()}, "integer"),
            (
                {
                    "value_quantizer": IntQuantizerConfig(
                        bits=2, granularity=PER_CHANNEL
                    ),
                },
                "values per token",
            ),
            (
                {"cache_policy": KittyCachePolicyConfig()},
                "SinkAndLocalCachePolicyConfig",
            ),
        ],
    )
    def test_rejects_non_kivi_structure(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            KVQuantization.create_from_config(KIVIConfig(**kwargs))

    def test_residual_must_be_a_multiple_of_the_key_group(self):
        with pytest.raises(ValueError, match="multiple of the key group size"):
            KVQuantization.create_from_config(
                KIVIConfig.with_settings(group_size=48, residual_length=128)
            )
        assert (
            type(
                KVQuantization.create_from_config(
                    KIVIConfig.with_settings(group_size=64)
                )
            )
            is KIVI
        )
