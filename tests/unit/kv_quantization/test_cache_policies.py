"""Unit tests for the cache policies: when local tokens move to the quantized middle."""

from typing import Any, Dict

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    CachePolicy,
    CachePolicyRegistry,
    DefaultCachePolicy,
    DefaultCachePolicyConfig,
    FullQuantizationCachePolicy,
    FullQuantizationCachePolicyConfig,
    IntQuantizerConfig,
    KittyCachePolicy,
    KittyCachePolicyConfig,
    KVQuantization,
    KVQuantizationConfig,
    LocalOnlyCachePolicy,
    LocalOnlyCachePolicyConfig,
    QuantGranularity,
    RegionConfig,
    SinkAndLocalCachePolicy,
    SinkAndLocalCachePolicyConfig,
    SinkOnlyCachePolicy,
    SinkOnlyCachePolicyConfig,
)

pytestmark = pytest.mark.unit

HEADS = 2
DIM = 32
PER_TOKEN = IntQuantizerConfig(bits=2, group_size=DIM)
PER_CHANNEL = IntQuantizerConfig(
    bits=2, granularity=QuantGranularity.PER_CHANNEL, group_size=32
)


def coordinator(cache_policy, key_quantizer=PER_TOKEN, value_quantizer=PER_TOKEN):
    return KVQuantization.create_from_config(
        KVQuantizationConfig(
            key_quantizer=key_quantizer,
            value_quantizer=value_quantizer,
            cache_policy=cache_policy,
        )
    )


def history(tokens: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, HEADS, tokens, DIM)


def step_stats(kvq: KVQuantization, tokens: int):
    """Feed one token per call and yield (t, outputs, layer stats) after each call."""
    k, v = history(tokens), history(tokens, seed=1)
    meta: Dict[str, Any] = {}
    for t in range(1, tokens + 1):
        out_k, out_v = kvq.process(
            0, k[..., :t, :], v[..., :t, :], meta, num_new_tokens=0
        )
        yield t, (k, v, out_k, out_v), meta[CACHE_KEY].stats()["layers"][0]


class TestRegistry:
    @pytest.mark.parametrize(
        "config, cls",
        [
            (DefaultCachePolicyConfig(), DefaultCachePolicy),
            (SinkAndLocalCachePolicyConfig(), SinkAndLocalCachePolicy),
            (KittyCachePolicyConfig(), KittyCachePolicy),
            (SinkOnlyCachePolicyConfig(), SinkOnlyCachePolicy),
            (FullQuantizationCachePolicyConfig(), FullQuantizationCachePolicy),
            (LocalOnlyCachePolicyConfig(), LocalOnlyCachePolicy),
        ],
    )
    def test_each_config_selects_its_policy(self, config, cls):
        assert type(CachePolicyRegistry.create(config)) is cls

    def test_default_is_the_default(self):
        assert isinstance(KVQuantizationConfig().cache_policy, DefaultCachePolicyConfig)
        assert isinstance(
            KVQuantization.create_from_config(KVQuantizationConfig()).cache_policy,
            DefaultCachePolicy,
        )


class TestDefault:
    def test_tail_stays_between_window_and_window_plus_block(self):
        kvq = coordinator(
            DefaultCachePolicyConfig(
                key_local=RegionConfig(16),
                value_local=RegionConfig(16),
                key_block_tokens=8,
                value_block_tokens=8,
            )
        )
        for t, _, stats in step_stats(kvq, 80):
            assert stats["key_recent_tokens"] == (t if t < 24 else 16 + (t - 16) % 8)

    def test_per_channel_needs_blocks_of_a_full_group(self):
        coordinator(DefaultCachePolicyConfig(key_block_tokens=32), PER_CHANNEL)
        with pytest.raises(ValueError, match="group_size"):
            coordinator(DefaultCachePolicyConfig(key_block_tokens=16), PER_CHANNEL)


class TestSinkAndLocalAndKitty:
    def test_rejects_per_channel_values(self):
        # Values leave the rolling window one at a time.
        with pytest.raises(ValueError, match="value tokens 1 at a time"):
            coordinator(SinkAndLocalCachePolicyConfig(), PER_CHANNEL, PER_CHANNEL)

    def test_rejects_a_key_buffer_smaller_than_a_group(self):
        with pytest.raises(ValueError, match="group_size"):
            coordinator(
                SinkAndLocalCachePolicyConfig(key_local=RegionConfig(16)), PER_CHANNEL
            )

    def test_zero_key_buffer_quantizes_keys_immediately(self):
        kvq = coordinator(SinkAndLocalCachePolicyConfig(key_local=RegionConfig(0)))
        for t, _, stats in step_stats(kvq, 20):
            assert stats["key_recent_tokens"] == 0
            assert stats["value_recent_tokens"] == min(t, 128)

    def test_kitty_defaults(self):
        config = KittyCachePolicyConfig()
        assert config.key_sink.tokens == config.value_sink.tokens == 32
        assert config.key_local.tokens == config.value_local.tokens == 128

    def test_policies_are_siblings(self):
        for cls in (
            DefaultCachePolicy,
            SinkAndLocalCachePolicy,
            KittyCachePolicy,
            SinkOnlyCachePolicy,
            FullQuantizationCachePolicy,
            LocalOnlyCachePolicy,
        ):
            assert cls.__bases__ == (CachePolicy,)


class TestSinkOnly:
    def test_everything_but_the_sink_is_quantized_on_arrival(self):
        kvq = coordinator(SinkOnlyCachePolicyConfig())
        for t, (k, _, out_k, _), stats in step_stats(kvq, 20):
            assert stats["key_sink_tokens"] == min(t, 4)
            assert stats["key_recent_tokens"] == stats["value_recent_tokens"] == 0
            assert stats["key_encoded_tokens"] == max(0, t - 4)
            assert torch.equal(out_k[..., : min(t, 4), :], k[..., : min(t, 4), :])

    def test_quantized_sink(self):
        kvq = coordinator(
            SinkOnlyCachePolicyConfig(
                key_sink=RegionConfig(4, PER_TOKEN),
                value_sink=RegionConfig(4, PER_TOKEN),
            )
        )
        *_, (_, _, stats) = step_stats(kvq, 10)
        assert stats["key_encoded_tokens"] == stats["value_encoded_tokens"] == 10

    def test_rejects_a_local_window(self):
        with pytest.raises(ValueError, match="no local region"):
            coordinator(SinkOnlyCachePolicyConfig(key_local=RegionConfig(8)))

    def test_rejects_per_channel(self):
        with pytest.raises(ValueError, match="group_size"):
            coordinator(SinkOnlyCachePolicyConfig(), PER_CHANNEL)


class TestFullQuantization:
    def test_nothing_stays_in_full_precision(self):
        kvq = coordinator(FullQuantizationCachePolicyConfig())
        for t, (k, v, out_k, out_v), stats in step_stats(kvq, 20):
            assert stats["key_encoded_tokens"] == stats["value_encoded_tokens"] == t
            assert stats["key_recent_tokens"] == stats["key_sink_tokens"] == 0
        assert not torch.equal(out_k, k) and not torch.equal(out_v, v)

    def test_rejects_sinks(self):
        with pytest.raises(ValueError, match="no sink region"):
            coordinator(FullQuantizationCachePolicyConfig(key_sink=RegionConfig(2)))


class TestLocalOnly:
    def test_window_is_exact(self):
        kvq = coordinator(
            LocalOnlyCachePolicyConfig(
                key_local=RegionConfig(16), value_local=RegionConfig(8)
            )
        )
        for t, (k, _, out_k, _), stats in step_stats(kvq, 40):
            assert stats["key_recent_tokens"] == min(t, 16)
            assert stats["value_recent_tokens"] == min(t, 8)
            assert stats["key_sink_tokens"] == 0
            local = min(t, 16)
            assert torch.equal(out_k[..., -local:, :], k[..., t - local : t, :])

    def test_rejects_sinks(self):
        with pytest.raises(ValueError, match="no sink region"):
            coordinator(LocalOnlyCachePolicyConfig(key_sink=RegionConfig(2)))

    def test_per_channel_needs_larger_blocks(self):
        with pytest.raises(ValueError, match="group_size"):
            coordinator(LocalOnlyCachePolicyConfig(), PER_CHANNEL)
        coordinator(LocalOnlyCachePolicyConfig(key_block_tokens=32), PER_CHANNEL)
