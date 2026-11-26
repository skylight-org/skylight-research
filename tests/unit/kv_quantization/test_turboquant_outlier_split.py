"""TurboQuant outlier-channel split (paper's 2.5/3.5-bit KV setup)."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    AttentionContext,
    ChannelBoost,
    ChannelBoostConfig,
    IntQuantizerConfig,
    KVQuantization,
    PrecisionPlan,
    RegionConfig,
    SinkAndLocalCachePolicyConfig,
    TurboQuantKV,
    TurboQuantKVConfig,
    TurboQuantQuantizer,
    TurboQuantQuantizerConfig,
    TurboQuantSplitKVConfig,
)
from tests.reference.turboquant import encode, unpack

pytestmark = pytest.mark.unit
CONTEXT = AttentionContext(1, "key", 0)


def random_channel_bits(batch, heads, dimension, count, low, high, seed=5):
    generator = torch.Generator().manual_seed(seed)
    bits = torch.full((batch, heads, dimension), low, dtype=torch.int64)
    for b in range(batch):
        for h in range(heads):
            chosen = torch.randperm(dimension, generator=generator)[:count]
            bits[b, h, chosen] = high
    return bits


def reference_split(codec, x, channel_bits, high, context=CONTEXT):
    """Two independent paper codecs on the gathered channels, scattered back."""
    expected = np.zeros(tuple(x.shape))
    batch, heads, _, dimension = x.shape
    for b in range(batch):
        for h in range(heads):
            mask = (channel_bits[b, h] == high).numpy()
            for name, bits, channels in (
                ("outlier", high, np.flatnonzero(mask)),
                ("normal", codec.config.bits, np.flatnonzero(~mask)),
            ):
                coarse = bits - (codec.config.variant == "prod")
                state = codec._resources(
                    context, len(channels), torch.device("cpu"), name, coarse
                )
                part = x[b : b + 1, h : h + 1][..., torch.from_numpy(channels)]
                result = encode(
                    part,
                    {
                        "rotation": state.rotation,
                        "codebook": state.levels,
                        "projection": state.projection,
                    },
                    variant=codec.config.variant,
                    bits=bits,
                )
                expected[b, h][..., channels] = result.reconstructed[0, 0]
    return expected


@pytest.mark.parametrize("variant", ["mse", "prod"])
@pytest.mark.parametrize("low,high,count", [(2, 3, 32), (3, 4, 64)])
def test_split_equals_two_independent_turboquant_codecs(variant, low, high, count):
    codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=low, variant=variant))
    x = torch.randn(2, 2, 8, 128, generator=torch.Generator().manual_seed(11))
    channel_bits = random_channel_bits(2, 2, 128, count, low, high)
    record = codec.quantize(x, PrecisionPlan(low, channel_bits=channel_bits), context=CONTEXT)
    assert record.format_id == f"turboquant_{variant}_outlier_v1"
    assert record.norms.shape == (2, 2, 8, 2) and record.norms.dtype == torch.float32
    assert record.outlier_indices.dtype == torch.int16
    assert record.outlier_indices.shape == (2, 2, count)
    np.testing.assert_allclose(
        codec.dequantize(record).double().numpy(),
        reference_split(codec, x, channel_bits, high),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize(
    "average,low,high,count", [(2.25, 2, 3, 32), (2.5, 2, 3, 64), (3.5, 3, 4, 64)]
)
def test_code_streams_average_the_named_bit_width(average, low, high, count):
    codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=low))
    x = torch.randn(1, 2, 8, 128, generator=torch.Generator().manual_seed(3))
    channel_bits = random_channel_bits(1, 2, 128, count, low, high)
    record = codec.quantize(x, PrecisionPlan(low, channel_bits=channel_bits), context=CONTEXT)
    assert record.codes.numel() * 8 == x.numel() * average
    vectors = x.shape[0] * x.shape[1] * x.shape[2]
    outlier = unpack(record.codes[: record.metadata["outlier_code_bytes"]], high, vectors * count)
    assert outlier.max() < 2**high


def test_equal_sized_groups_use_different_rotations():
    codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=3))
    outlier = codec._resources(CONTEXT, 64, torch.device("cpu"), "outlier", 4)
    normal = codec._resources(CONTEXT, 64, torch.device("cpu"), "normal", 3)
    whole = codec._resources(CONTEXT, 64, torch.device("cpu"))
    assert not torch.equal(outlier.rotation, normal.rotation)
    assert not torch.equal(outlier.rotation, whole.rotation)
    eye = torch.eye(64)
    torch.testing.assert_close(outlier.rotation @ outlier.rotation.T, eye, atol=1e-5, rtol=0)


def test_without_channel_bits_the_whole_vector_codec_is_unchanged():
    codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=3))
    x = torch.randn(1, 2, 4, 128, generator=torch.Generator().manual_seed(2))
    record = codec.quantize(x, context=CONTEXT)
    assert record.format_id == "turboquant_mse_v1" and record.outlier_indices is None
    state = codec.state_for(CONTEXT, 128)
    np.testing.assert_allclose(
        codec.dequantize(record).double().numpy(),
        encode(x, state, bits=3).reconstructed,
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize(
    "channel_bits",
    [
        torch.full((1, 2, 128), 2, dtype=torch.int64),  # one width only
        random_channel_bits(1, 2, 128, 32, 1, 3),  # low width is not config.bits
        random_channel_bits(1, 2, 128, 32, 2, 8),  # wider than TurboQuant supports
        random_channel_bits(1, 2, 128, 1, 2, 3),  # outlier group too small
    ],
)
def test_invalid_split_plans_are_rejected(channel_bits):
    codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=2))
    x = torch.randn(1, 2, 4, 128)
    with pytest.raises(ValueError):
        codec.quantize(x, PrecisionPlan(2, channel_bits=channel_bits), context=CONTEXT)


def test_unequal_outlier_counts_per_head_are_rejected():
    codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=2))
    channel_bits = random_channel_bits(1, 2, 128, 32, 2, 3)
    channel_bits[0, 1, (channel_bits[0, 1] == 2).nonzero()[0]] = 3
    with pytest.raises(ValueError, match="same number"):
        codec.quantize(
            torch.randn(1, 2, 4, 128),
            PrecisionPlan(2, channel_bits=channel_bits),
            context=CONTEXT,
        )


@pytest.mark.parametrize("field", ["groups", "outlier_code_bytes", "indices"])
def test_tampered_split_records_are_rejected(field):
    codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=2))
    channel_bits = random_channel_bits(1, 2, 128, 32, 2, 3)
    record = codec.quantize(
        torch.randn(1, 2, 4, 128), PrecisionPlan(2, channel_bits=channel_bits), context=CONTEXT
    )
    if field == "groups":
        record.metadata["groups"]["outlier"]["rotation_id"] = "x"
    elif field == "outlier_code_bytes":
        record.metadata["outlier_code_bytes"] = "3"
    else:
        record.outlier_indices = record.outlier_indices.flip(-1).contiguous()
    with pytest.raises(ValueError):
        codec.dequantize(record)


class TestFrozenChannelBoost:
    def policy(self):
        return ChannelBoost(
            ChannelBoostConfig(fraction=0.25, boosted_bits=3, kinds=("key",), freeze="prefill")
        )

    def test_decode_chunks_reuse_the_prefill_choice(self):
        policy = self.policy()
        prefill = torch.randn(1, 2, 16, 32, generator=torch.Generator().manual_seed(0))
        first = policy.bits_for(0, prefill, torch.arange(16), tensor_kind="key", default_bits=2)
        token = torch.randn(1, 2, 1, 32) * 100
        later = policy.bits_for(0, token, torch.arange(16, 17), tensor_kind="key", default_bits=2)
        assert later.channel_bits is first.channel_bits
        expected = prefill.abs().mean(dim=-2).topk(8, dim=-1).indices
        chosen = (first.channel_bits == 3).nonzero()[:, -1].reshape(1, 2, 8)
        assert torch.equal(chosen, expected.sort(dim=-1).values)

    def test_a_new_request_chooses_again(self):
        policy = self.policy()
        first = policy.bits_for(0, torch.randn(1, 2, 8, 32), torch.arange(8), tensor_kind="key", default_bits=2)
        again = policy.bits_for(0, torch.randn(1, 2, 8, 32), torch.arange(8), tensor_kind="key", default_bits=2)
        assert again.channel_bits is not first.channel_bits

    def test_layers_and_kinds_are_independent(self):
        policy = ChannelBoost(ChannelBoostConfig(fraction=0.25, boosted_bits=3, kinds=("key", "value"), freeze="prefill"))
        x = torch.randn(1, 2, 8, 32)
        a = policy.bits_for(0, x, torch.arange(8), tensor_kind="key", default_bits=2)
        b = policy.bits_for(1, x, torch.arange(8), tensor_kind="key", default_bits=2)
        c = policy.bits_for(0, x, torch.arange(8), tensor_kind="value", default_bits=2)
        assert a.channel_bits is not b.channel_bits and a.channel_bits is not c.channel_bits

    def test_default_still_rescores_every_chunk(self):
        policy = ChannelBoost(ChannelBoostConfig(fraction=0.25, boosted_bits=4))
        a = policy.bits_for(0, torch.randn(1, 2, 8, 32), torch.arange(8), tensor_kind="key", default_bits=2)
        b = policy.bits_for(0, torch.randn(1, 2, 1, 32), torch.arange(8, 9), tensor_kind="key", default_bits=2)
        assert a.channel_bits is not b.channel_bits


class TestOutlierSplitProfile:
    def test_presets(self):
        for average, (low, high, fraction) in {2.5: (2, 3, 0.5), 3.5: (3, 4, 0.5)}.items():
            config = TurboQuantKVConfig.outlier_split(average)
            assert config.key_quantizer.bits == low and config.value_quantizer.bits == low
            policy = config.precision_policy
            assert (policy.boosted_bits, policy.fraction) == (high, fraction)
            assert policy.kinds == ("key", "value") and policy.freeze == "prefill"
            assert type(KVQuantization.create_from_config(config)) is TurboQuantKV
        assert TurboQuantKVConfig.outlier_split(2.5, "prod").key_quantizer.variant == "prod"
        with pytest.raises(ValueError):
            TurboQuantKVConfig.outlier_split(3.0)

    def test_unfrozen_or_split_variant_profiles_are_rejected(self):
        config = TurboQuantKVConfig.outlier_split(2.5)
        with pytest.raises(ValueError, match="freeze"):
            KVQuantization.create_from_config(
                replace(config, precision_policy=replace(config.precision_policy, freeze="none"))
            )
        with pytest.raises(ValueError):
            KVQuantization.create_from_config(
                TurboQuantSplitKVConfig(precision_policy=config.precision_policy)
            )

    def test_channel_boost_still_rejects_per_token_int(self):
        from sparse_attention_hub.kv_quantization import ChannelBoost as Policy

        with pytest.raises(ValueError, match="PER_CHANNEL"):
            Policy(ChannelBoostConfig()).validate_quantizers(IntQuantizerConfig(bits=2), IntQuantizerConfig(bits=2))

    @pytest.mark.parametrize("packed", [True, False])
    @pytest.mark.parametrize("variant", ["mse", "prod"])
    def test_streaming_flow_matches_one_fixed_choice(self, packed, variant):
        config = TurboQuantKVConfig.outlier_split(2.5, variant, store_codes=packed)
        method = KVQuantization.create_from_config(config)
        generator = torch.Generator().manual_seed(42)
        keys, values = (torch.randn(1, 2, 9, 16, generator=generator) for _ in range(2))
        metadata = {}
        for end, count in [(4, 4), (5, 1), (8, 3), (9, 1)]:
            result = method.process(
                0,
                keys[..., :end, :],
                values[..., :end, :],
                metadata,
                num_new_tokens=count,
                phase="prefill" if end == 4 else "decode",
            )
        for actual, original, kind in zip(result, (keys, values), ("key", "value")):
            codec = getattr(method, f"{kind}_quantizer")
            scores = original[..., :4, :].abs().mean(dim=-2)
            channel_bits = torch.full((1, 2, 16), 2, dtype=torch.int64)
            channel_bits.scatter_(-1, scores.topk(8, dim=-1).indices, 3)
            expected = reference_split(codec, original, channel_bits, 3, AttentionContext(0, kind, 0))
            np.testing.assert_allclose(actual.double().numpy(), expected, rtol=3e-5, atol=3e-5)
        if packed:
            store = metadata[CACHE_KEY]
            chunks = store.layers[0].key.chunks
            assert len({id(c.quantized.outlier_indices) for c in chunks}) == 1
            index_bytes = chunks[0].quantized.outlier_indices.numel() * 2
            payload = sum(
                c.quantized.storage_bytes()
                for kind in ("key", "value")
                for c in getattr(store.layers[0], kind).chunks
            )
            repeats = sum(len(getattr(store.layers[0], k).chunks) - 1 for k in ("key", "value"))
            assert store.stats()["store_bytes"] == payload - repeats * index_bytes


@pytest.mark.parametrize("packed", [True, False])
def test_kivi_window_keeps_recent_tokens_exact(packed):
    # Keys wait in a buffer of 3 (moved in whole buffers); values keep a window of 3.
    window = SinkAndLocalCachePolicyConfig(
        key_local=RegionConfig(tokens=3), value_local=RegionConfig(tokens=3)
    )
    config = TurboQuantKVConfig.outlier_split(
        2.5, store_codes=packed, cache_policy=window, quantize_prefill=True
    )
    method = KVQuantization.create_from_config(config)
    generator = torch.Generator().manual_seed(9)
    keys, values = (torch.randn(1, 2, 7, 16, generator=generator) for _ in range(2))
    result = method.process(0, keys, values, {}, num_new_tokens=7, phase="prefill")
    # 7 keys: 6 quantized (two full buffers), 1 exact; 7 values: 4 quantized, 3 exact.
    for actual, original, kind, moved in zip(
        result, (keys, values), ("key", "value"), (6, 4)
    ):
        torch.testing.assert_close(actual[..., moved:, :], original[..., moved:, :])
        codec = getattr(method, f"{kind}_quantizer")
        scores = original[..., :moved, :].abs().mean(dim=-2)
        channel_bits = torch.full((1, 2, 16), 2, dtype=torch.int64)
        channel_bits.scatter_(-1, scores.topk(8, dim=-1).indices, 3)
        expected = reference_split(
            codec, original[..., :moved, :], channel_bits, 3, AttentionContext(0, kind, 0)
        )
        np.testing.assert_allclose(
            actual[..., :moved, :].double().numpy(), expected, rtol=3e-5, atol=3e-5
        )


def test_other_cache_policies_are_still_rejected():
    from sparse_attention_hub.kv_quantization import DefaultCachePolicyConfig

    with pytest.raises(ValueError, match="SinkAndLocal"):
        KVQuantization.create_from_config(
            TurboQuantKVConfig(cache_policy=DefaultCachePolicyConfig())
        )


class TestKeyCentering:
    """center=("key",): remove the per-channel prefill mean before TurboQuant."""

    def offset_keys(self, tokens=64, seed=4):
        generator = torch.Generator().manual_seed(seed)
        offset = torch.randn(1, 2, 1, 128, generator=generator) * 20
        return offset + torch.randn(1, 2, tokens, 128, generator=generator)

    def test_centered_record_is_the_plain_codec_on_x_minus_mean(self):
        x = self.offset_keys()
        plain = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=2))
        centered = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=2, center=("key",)))
        record = centered.quantize(x, context=CONTEXT)
        mean = x.mean(dim=-2, keepdim=True)
        assert record.metadata["centered"] is True and record.dtype == x.dtype
        torch.testing.assert_close(record.zero_points, mean)
        expected = plain.dequantize(plain.quantize(x - mean, context=CONTEXT)) + mean
        torch.testing.assert_close(centered.dequantize(record), expected, rtol=1e-5, atol=1e-4)

    def test_centering_removes_the_shared_offset_error(self):
        x = self.offset_keys()
        plain = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=2))
        centered = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=2, center=("key",)))
        token_part = x - x.mean(dim=-2, keepdim=True)
        error = {
            name: (codec.dequantize(codec.quantize(x, context=CONTEXT)) - x).norm()
            / token_part.norm()
            for name, codec in (("plain", plain), ("centered", centered))
        }
        assert error["plain"] > 5 * error["centered"]
        assert error["centered"] < 0.4  # about the 2-bit relative error, 0.34

    def test_mean_is_frozen_at_prefill_and_retaken_for_a_new_request(self):
        codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=3, center=("key",)))
        x = self.offset_keys()
        first = codec.quantize(x[..., :60, :], context=AttentionContext(1, "key", 0))
        later = codec.quantize(x[..., 60:, :] + 100, context=AttentionContext(1, "key", 60))
        assert later.zero_points is first.zero_points
        again = codec.quantize(x[..., :8, :], context=AttentionContext(1, "key", 0))
        assert again.zero_points is not first.zero_points
        other_layer = codec.quantize(x[..., 60:, :], context=AttentionContext(2, "key", 60))
        torch.testing.assert_close(other_layer.zero_points, x[..., 60:, :].mean(-2, keepdim=True))

    def test_split_plans_are_centered_too(self):
        x = self.offset_keys()
        channel_bits = random_channel_bits(1, 2, 128, 64, 2, 3)
        plan = PrecisionPlan(2, channel_bits=channel_bits)
        plain = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=2))
        centered = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=2, center=("key",)))
        record = centered.quantize(x, plan, context=CONTEXT)
        assert record.format_id == "turboquant_mse_outlier_v1"
        mean = x.mean(dim=-2, keepdim=True)
        expected = plain.dequantize(plain.quantize(x - mean, plan, context=CONTEXT)) + mean
        torch.testing.assert_close(centered.dequantize(record), expected, rtol=1e-5, atol=1e-4)

    def test_kinds_outside_center_are_unchanged(self):
        codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=2, center=("key",)))
        plain = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=2))
        x = self.offset_keys()
        value = AttentionContext(1, "value", 0)
        record = codec.quantize(x, context=value)
        assert record.zero_points is None and "centered" not in record.metadata
        torch.testing.assert_close(
            codec.dequantize(record), plain.dequantize(plain.quantize(x, context=value))
        )

    @pytest.mark.parametrize("center", [("query",), ("key", "key"), ["key"]])
    def test_invalid_center_is_rejected(self, center):
        with pytest.raises(ValueError, match="center"):
            TurboQuantQuantizer(TurboQuantQuantizerConfig(center=center))

    @pytest.mark.parametrize("tamper", ["drop_mean", "drop_flag", "shape", "plain_codec"])
    def test_tampered_centered_records_are_rejected(self, tamper):
        codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=2, center=("key",)))
        record = codec.quantize(self.offset_keys(), context=CONTEXT)
        if tamper == "drop_mean":
            record.zero_points = None
        elif tamper == "drop_flag":
            del record.metadata["centered"]
        elif tamper == "shape":
            record.zero_points = record.zero_points[..., :64]
        else:
            codec = TurboQuantQuantizer(TurboQuantQuantizerConfig(bits=2))
        with pytest.raises(ValueError):
            codec.dequantize(record)

    def test_preset_centers_keys_only_and_selects_by_variance(self):
        config = TurboQuantKVConfig.outlier_split(2.5, center_keys=True)
        assert config.key_quantizer.center == ("key",)
        assert config.value_quantizer.center == ()
        assert config.precision_policy.selection == "variance"
        assert TurboQuantKVConfig.outlier_split(2.5).key_quantizer.center == ()

    def test_store_counts_the_shared_mean_once(self):
        config = TurboQuantKVConfig.outlier_split(2.5, center_keys=True, store_codes=True)
        method = KVQuantization.create_from_config(config)
        keys = self.offset_keys(tokens=9)[..., :16]
        values = torch.randn(1, 2, 9, 16)
        metadata = {}
        for end, count in [(4, 4), (5, 1), (9, 4)]:
            result = method.process(
                0, keys[..., :end, :], values[..., :end, :], metadata,
                num_new_tokens=count, phase="prefill" if end == 4 else "decode",
            )
        assert result[0].shape == keys.shape
        store = metadata[CACHE_KEY]
        chunks = store.layers[0].key.chunks
        assert len({id(c.quantized.zero_points) for c in chunks}) == 1
        mean_bytes = chunks[0].quantized.zero_points.numel() * 4
        index_bytes = chunks[0].quantized.outlier_indices.numel() * 2
        payload = sum(
            c.quantized.storage_bytes()
            for kind in ("key", "value")
            for c in getattr(store.layers[0], kind).chunks
        )
        repeats = sum(len(getattr(store.layers[0], k).chunks) - 1 for k in ("key", "value"))
        key_repeats = len(chunks) - 1
        assert store.stats()["store_bytes"] == (
            payload - repeats * index_bytes - key_repeats * mean_bytes
        )
