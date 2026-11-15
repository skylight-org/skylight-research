"""Unit tests for the configurable sink / middle / local regions of the KV store.

Per-token quantizers are used throughout so that each token's reconstruction is
independent of how tokens are chunked, which makes the expected values exact.
"""

from typing import Optional

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    DefaultCachePolicyConfig,
    IntQuantizer,
    IntQuantizerConfig,
    KittyConfig,
    KIVIConfig,
    KVQuantization,
    KVQuantizationConfig,
    QuantGranularity,
    RegionConfig,
)

pytestmark = pytest.mark.unit

HEADS = 2
DIM = 32
INT2 = IntQuantizerConfig(bits=2, group_size=DIM)
INT4 = IntQuantizerConfig(bits=4, group_size=DIM)
INT8 = IntQuantizerConfig(bits=8, group_size=DIM)


def fq(config: IntQuantizerConfig, x: torch.Tensor) -> torch.Tensor:
    """Quantize then dequantize, exactly as the store does for one chunk."""
    return IntQuantizer(config).fake_quantize(x)


def history(tokens: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, HEADS, tokens, DIM)


def make(
    *,
    key_sink: RegionConfig = RegionConfig(),
    value_sink: RegionConfig = RegionConfig(),
    key_local: RegionConfig = RegionConfig(tokens=16),
    value_local: RegionConfig = RegionConfig(tokens=16),
    block: int = 16,
    store_codes: bool = False,
    middle: IntQuantizerConfig = INT2,
) -> KVQuantization:
    return KVQuantization.create_from_config(
        KVQuantizationConfig(
            key_quantizer=middle,
            value_quantizer=middle,
            cache_policy=DefaultCachePolicyConfig(
                key_sink=key_sink,
                value_sink=value_sink,
                key_local=key_local,
                value_local=value_local,
                key_block_tokens=block,
                value_block_tokens=block,
            ),
            store_codes=store_codes,
        )
    )


def feed(kvq: KVQuantization, k: torch.Tensor, v: torch.Tensor, steps, meta=None):
    """Inspect growing completed histories after storage/compression (no queries)."""
    meta = {} if meta is None else meta
    out_k: Optional[torch.Tensor] = None
    out_v: Optional[torch.Tensor] = None
    for t in steps:
        out_k, out_v = kvq.process(
            0, k[..., :t, :], v[..., :t, :], meta, num_new_tokens=0
        )
    return out_k, out_v, meta


def layer_stats(meta):
    return meta[CACHE_KEY].stats()["layers"][0]


class TestDefaults:
    def test_sink_and_local_stay_exact_by_default(self):
        k, v = history(64), history(64, seed=1)
        kvq = make(key_sink=RegionConfig(tokens=2), value_sink=RegionConfig(tokens=2))
        out_k, out_v, meta = feed(kvq, k, v, [64])
        stats = layer_stats(meta)
        local = stats["key_recent_tokens"]
        assert torch.equal(out_k[..., :2, :], k[..., :2, :])
        assert torch.equal(out_k[..., -local:, :], k[..., -local:, :])
        assert torch.equal(out_v[..., -local:, :], v[..., -local:, :])
        assert not torch.equal(out_k[..., 2:-local, :], k[..., 2:-local, :])

    def test_keys_and_values_can_have_different_sizes(self):
        k, v = history(80), history(80, seed=1)
        kvq = make(
            key_sink=RegionConfig(tokens=2),
            value_sink=RegionConfig(tokens=4),
            key_local=RegionConfig(tokens=16),
            value_local=RegionConfig(tokens=32),
        )
        _, _, meta = feed(kvq, k, v, [80])
        stats = layer_stats(meta)
        assert stats["key_sink_tokens"] == 2 and stats["value_sink_tokens"] == 4
        # 78 key tokens after the sink: move 16-token blocks while >= 16 + 16 remain.
        assert stats["key_recent_tokens"] == 30
        # 76 value tokens after the sink: move while >= 32 + 16 remain.
        assert stats["value_recent_tokens"] == 44


class TestQuantizedSink:
    def test_sink_uses_its_own_quantizer(self):
        k, v = history(64), history(64, seed=1)
        kvq = make(key_sink=RegionConfig(4, INT8), value_sink=RegionConfig(4))
        out_k, out_v, _ = feed(kvq, k, v, [64])
        assert torch.equal(out_k[..., :4, :], fq(INT8, k[..., :4, :]))
        assert not torch.equal(out_k[..., :4, :], k[..., :4, :])
        assert torch.equal(out_v[..., :4, :], v[..., :4, :])

    def test_sink_filled_across_calls(self):
        k, v = history(40), history(40, seed=1)
        kvq = make(key_sink=RegionConfig(4, INT8))
        out_k, _, meta = feed(kvq, k, v, [2, 3, 40])
        assert layer_stats(meta)["key_sink_tokens"] == 4
        assert torch.equal(out_k[..., :4, :], fq(INT8, k[..., :4, :]))


class TestQuantizedLocal:
    def test_local_uses_its_own_quantizer_and_middle_its_own(self):
        k, v = history(64), history(64, seed=1)
        kvq = make(key_local=RegionConfig(16, INT4))
        out_k, _, meta = feed(kvq, k, v, [64])
        local = layer_stats(meta)["key_recent_tokens"]
        assert torch.equal(out_k[..., -local:, :], fq(INT4, k[..., -local:, :]))
        # Tokens that were already old on arrival go straight to the middle quantizer.
        assert torch.equal(out_k[..., :-local, :], fq(INT2, k[..., :-local, :]))

    def test_moving_to_middle_requantizes_the_local_copy(self):
        """Option A: a token that sat in a quantized local window is re-quantized
        from its local copy, not from the original value."""
        k, v = history(64), history(64, seed=1)
        # Coarse local (int2), fine middle (int8), so the two options differ.
        kvq = make(key_local=RegionConfig(16, INT2), middle=INT8)
        # Tokens 32..47 enter the local window at t=48, then leave it at t=64.
        out_k, _, meta = feed(kvq, k, v, [48, *range(49, 65)])
        assert layer_stats(meta)["key_recent_tokens"] == 16
        option_a = fq(INT8, fq(INT2, k[..., 32:48, :]))
        option_b = fq(INT8, k[..., 32:48, :])
        assert not torch.equal(option_a, option_b)
        assert torch.equal(out_k[..., 32:48, :], option_a)
        assert torch.equal(out_k[..., 48:64, :], fq(INT2, k[..., 48:64, :]))

    def test_partially_moved_local_chunk_is_requantized_from_its_copy(self):
        k, v = history(32), history(32, seed=1)
        kvq = make(key_local=RegionConfig(16, INT2), middle=INT8)
        # t=20: 20 tokens stay local (one int2 chunk). t=32: the oldest 16 move out,
        # splitting that chunk; its last 4 tokens are re-stored from the int2 copy.
        out_k, _, _ = feed(kvq, k, v, [20, 32])
        once = fq(INT2, k[..., :20, :])
        assert not torch.equal(fq(INT8, once[..., :16, :]), fq(INT8, k[..., :16, :]))
        assert torch.equal(out_k[..., :16, :], fq(INT8, once[..., :16, :]))
        assert torch.equal(out_k[..., 16:20, :], fq(INT2, once[..., 16:, :]))
        assert torch.equal(out_k[..., 20:32, :], fq(INT2, k[..., 20:32, :]))

    def test_per_channel_local_quantizer_is_rejected(self):
        per_channel = IntQuantizerConfig(
            bits=4, granularity=QuantGranularity.PER_CHANNEL
        )
        with pytest.raises(ValueError, match="per-token or per-block"):
            make(key_local=RegionConfig(16, per_channel))


class TestWholeHistory:
    def test_everything_quantized_when_local_is_empty(self):
        k, v = history(40), history(40, seed=1)
        kvq = make(key_local=RegionConfig(0), value_local=RegionConfig(0), block=1)
        _, _, meta = feed(kvq, k, v, [40])
        stats = layer_stats(meta)
        assert stats["key_encoded_tokens"] == 40
        assert stats["value_encoded_tokens"] == 40

    def test_encoded_tokens_count_every_quantized_region(self):
        k, v = history(64), history(64, seed=1)
        kvq = make(key_sink=RegionConfig(4, INT8), key_local=RegionConfig(16, INT4))
        _, _, meta = feed(kvq, k, v, [64])
        stats = layer_stats(meta)
        assert stats["key_encoded_tokens"] == 64
        assert stats["value_encoded_tokens"] == stats["value_compressed_tokens"]

    def test_fake_and_packed_agree_with_quantized_regions(self):
        k, v = history(80), history(80, seed=1)
        regions = dict(
            key_sink=RegionConfig(4, INT8),
            value_sink=RegionConfig(2, INT4),
            key_local=RegionConfig(16, INT4),
            value_local=RegionConfig(24, INT8),
        )
        steps = [40, *range(41, 81)]
        fake_k, fake_v, _ = feed(make(**regions), k, v, steps)
        packed_k, packed_v, _ = feed(make(**regions, store_codes=True), k, v, steps)
        assert torch.equal(fake_k, packed_k)
        assert torch.equal(fake_v, packed_v)


class TestMethodSchedulesAreSettings:
    """KIVI's and Kitty's buffering are configurations of the generic policy."""

    @staticmethod
    def coordinator(method_config: KVQuantizationConfig) -> KVQuantization:
        # Reuse only the presets' buffering parts (no precision policy).
        return KVQuantization.create_from_config(
            KVQuantizationConfig(
                key_quantizer=method_config.key_quantizer,
                value_quantizer=method_config.value_quantizer,
                cache_policy=method_config.cache_policy,
            )
        )

    def test_kivi_keys_flush_every_128_and_values_roll(self):
        kvq = self.coordinator(KIVIConfig())
        k, v = history(300), history(300, seed=1)
        meta: dict = {}
        for t in range(1, 301):
            kvq.process(0, k[..., :t, :], v[..., :t, :], meta, num_new_tokens=0)
            stats = layer_stats(meta)
            assert stats["key_recent_tokens"] == t % 128
            assert stats["key_compressed_tokens"] == t - t % 128
            assert stats["value_recent_tokens"] == min(t, 128)

    def test_kivi_prompt_import_keeps_the_remainder_in_fp(self):
        kvq = self.coordinator(KIVIConfig())
        _, _, meta = feed(kvq, history(300), history(300, seed=1), [300])
        stats = layer_stats(meta)
        assert stats["key_recent_tokens"] == 300 % 128
        assert stats["value_recent_tokens"] == 128

    def test_kitty_sinks_key_pages_and_value_window(self):
        # Kitty's value group is the whole head; these test heads have DIM channels.
        kvq = self.coordinator(KittyConfig.with_settings(group_size=DIM))
        k, v = history(330), history(330, seed=1)
        meta: dict = {}
        for t in range(1, 331):
            kvq.process(0, k[..., :t, :], v[..., :t, :], meta, num_new_tokens=0)
            stats = layer_stats(meta)
            after_sink = max(0, t - 32)
            assert stats["key_sink_tokens"] == stats["value_sink_tokens"] == min(t, 32)
            # Kitty flushes a key page when the 129th key arrives: 1..128 stay FP.
            expected = after_sink if after_sink <= 128 else (after_sink - 1) % 128 + 1
            assert stats["key_recent_tokens"] == expected
            assert stats["value_recent_tokens"] == min(after_sink, 128)
