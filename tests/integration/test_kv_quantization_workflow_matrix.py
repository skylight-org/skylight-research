"""Cross-component contracts, using multiple layers, heads and streamed arrivals."""

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    DefaultCachePolicyConfig,
    FakeNVFP4QuantizerConfig,
    FP8QuantizerConfig,
    FullQuantizationCachePolicyConfig,
    HadamardTransformConfig,
    IdentityTransformConfig,
    IntQuantizerConfig,
    KittyCachePolicyConfig,
    KVQuantization,
    KVQuantizationConfig,
    LocalOnlyCachePolicyConfig,
    MXFPQuantizerConfig,
    NonUniformQuantizerConfig,
    OutlierTokenPolicyConfig,
    RandomHadamardTransformConfig,
    RandomRotationConfig,
    RegionConfig,
    SinkAndLocalCachePolicyConfig,
    SinkOnlyCachePolicyConfig,
)

pytestmark = pytest.mark.integration

CODECS = [
    pytest.param(IntQuantizerConfig(bits=2), id="int2"),
    pytest.param(IntQuantizerConfig(bits=4), id="int4"),
    pytest.param(IntQuantizerConfig(bits=8), id="int8"),
    pytest.param(FP8QuantizerConfig(), id="fp8-e4m3"),
    pytest.param(FP8QuantizerConfig(format="E5M2"), id="fp8-e5m2"),
    pytest.param(FakeNVFP4QuantizerConfig(), id="nvfp4-reference"),
    pytest.param(MXFPQuantizerConfig(), id="mxfp4"),
    pytest.param(MXFPQuantizerConfig(bits=8, format="E4M3"), id="mxfp8-e4m3"),
    pytest.param(MXFPQuantizerConfig(bits=8, format="E5M2"), id="mxfp8-e5m2"),
    pytest.param(NonUniformQuantizerConfig(), id="gaussian-nuq"),
    pytest.param(NonUniformQuantizerConfig(mode="calibrated"), id="calibrated-nuq"),
]
TRANSFORMS = [
    pytest.param(IdentityTransformConfig(), id="identity"),
    pytest.param(HadamardTransformConfig(), id="hadamard"),
    pytest.param(RandomHadamardTransformConfig(seed=47), id="random-hadamard"),
    pytest.param(RandomRotationConfig(seed=47), id="random-rotation"),
]
POLICIES = [
    pytest.param(
        DefaultCachePolicyConfig(
            key_sink=RegionConfig(3),
            value_sink=RegionConfig(1),
            key_local=RegionConfig(3),
            value_local=RegionConfig(2),
            key_block_tokens=4,
            value_block_tokens=8,
        ),
        id="default",
    ),
    pytest.param(
        SinkAndLocalCachePolicyConfig(
            key_local=RegionConfig(8), value_local=RegionConfig(6)
        ),
        id="buffer-and-window",
    ),
    pytest.param(KittyCachePolicyConfig(), id="kitty-buffering"),
    pytest.param(SinkOnlyCachePolicyConfig(), id="sinks-only"),
    pytest.param(FullQuantizationCachePolicyConfig(), id="full-compression"),
    pytest.param(
        LocalOnlyCachePolicyConfig(
            key_local=RegionConfig(3), value_local=RegionConfig(5)
        ),
        id="local-only",
    ),
]


def prepare_calibration(coordinator, *, channels=32):
    """Prepare every layer/kind before first use, through the supported codec API."""
    for kind in ("key", "value"):
        codec = getattr(coordinator, f"{kind}_quantizer")
        if (
            isinstance(codec.config, NonUniformQuantizerConfig)
            and codec.config.mode == "calibrated"
        ):
            samples = torch.randn(
                96, channels, generator=torch.Generator().manual_seed(73)
            )
            codec.calibrate({(layer, kind): samples for layer in (0, 1)})


def assert_regions_cover_history(store, layer):
    state = store.layers[layer]
    for kind in ("key", "value"):
        chunks = getattr(state, kind).all_chunks()
        end = 0
        for chunk in chunks:
            assert chunk.start_token == end and chunk.token_count > 0
            end += chunk.token_count
        assert end == state.seen_tokens


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("policy", POLICIES)
def test_streaming_full_and_sliding_inputs_match_in_fake_and_packed_modes(
    codec, transform, policy
):
    """Future calls must read our stored past, even when HF supplies a poisoned prefix.

    Four independent requests receive identical new tensors: full/suffix HF input
    crossed with FAKE/PACKED storage. Their returned tensors and persistent stores
    must agree. The current suffix must remain bit-exact after compression flushes.
    """
    coordinators, metadata = {}, {}
    for packed in (False, True):
        for suffix in (False, True):
            key = packed, suffix
            coordinators[key] = KVQuantization.create_from_config(
                KVQuantizationConfig(
                    key_transform=transform,
                    value_transform=transform,
                    key_quantizer=codec,
                    value_quantizer=codec,
                    cache_policy=policy,
                    store_codes=packed,
                )
            )
            prepare_calibration(coordinators[key])
            metadata[key] = {}
    generator = torch.Generator().manual_seed(911)
    total = 0
    # Kitty's 32 sinks and 128 key-buffer boundary must both be crossed.
    try:
        for count in (35, 2, 29, 1, 32, 64, 1):
            total += count
            for layer in (0, 1):
                # Strided input exercises ownership/layout across all components.
                new_k = torch.randn(2, 2, count, 64, generator=generator)[..., ::2]
                new_v = torch.randn(2, 2, count, 64, generator=generator)[..., ::2]
                results, snapshots = [], []
                for key, coordinator in coordinators.items():
                    _, suffix = key
                    history = 0 if total == count else total - count
                    prefix = min(3, history) if suffix else history
                    # Old HF content must never overwrite previously captured KV.
                    incoming_k = torch.cat(
                        [torch.full((2, 2, prefix, 32), 71.0), new_k], dim=-2
                    )
                    incoming_v = torch.cat(
                        [torch.full((2, 2, prefix, 32), -53.0), new_v], dim=-2
                    )
                    original_k, original_v = incoming_k.clone(), incoming_v.clone()
                    output = coordinator.process(
                        layer,
                        incoming_k,
                        incoming_v,
                        metadata[key],
                        num_new_tokens=count,
                        phase="prefill" if total == count else "decode",
                    )
                    store = metadata[key][CACHE_KEY]
                    assert store.seen_tokens(layer) == total
                    assert_regions_cover_history(store, layer)
                    for tensor, original in zip(output, (new_k, new_v)):
                        assert tensor.shape == (2, 2, total, 32)
                        assert torch.equal(tensor[..., -count:, :], original)
                        assert torch.isfinite(tensor).all()
                    assert torch.equal(incoming_k, original_k)
                    assert torch.equal(incoming_v, original_v)
                    # Future input mutation must not alter any owned cache tensor.
                    incoming_k.fill_(999)
                    incoming_v.fill_(-999)
                    results.append(output)
                    snapshots.append(store.get(layer, coordinator._decode_chunk))
                for result in results[1:]:
                    assert all(torch.equal(a, b) for a, b in zip(results[0], result))
                for snapshot in snapshots[1:]:
                    assert all(
                        torch.equal(a, b) for a, b in zip(snapshots[0], snapshot)
                    )
        for key, coordinator in coordinators.items():
            store = metadata[key][CACHE_KEY]
            assert set(store.layers) == {0, 1}
            assert all(s.key.compressed_tokens > 0 for s in store.layers.values())
            coordinator.clear_request(metadata[key])
            assert CACHE_KEY not in metadata[key] and not store.layers
    finally:
        for key, coordinator in coordinators.items():
            coordinator.clear_request(metadata[key])


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("packed", [False, True])
def test_quantized_edge_regions_and_outlier_policy_are_composable(codec, packed):
    """Sink/local codecs plus a distinct middle codec; expire partial local chunks."""
    config = KVQuantizationConfig(
        key_quantizer=codec,
        value_quantizer=codec,
        precision_policy=OutlierTokenPolicyConfig(threshold=2, max_fraction=0.5),
        key_transform=RandomHadamardTransformConfig(seed=3),
        value_transform=RandomRotationConfig(seed=9),
        cache_policy=DefaultCachePolicyConfig(
            key_sink=RegionConfig(2, IntQuantizerConfig(bits=8)),
            value_sink=RegionConfig(1, FP8QuantizerConfig()),
            key_local=RegionConfig(3, IntQuantizerConfig(bits=4)),
            value_local=RegionConfig(4, IntQuantizerConfig(bits=8)),
            key_block_tokens=4,
            value_block_tokens=4,
        ),
        store_codes=packed,
    )
    coordinator = KVQuantization.create_from_config(config)
    prepare_calibration(coordinator)
    metadata = {}
    generator = torch.Generator().manual_seed(111)
    for count in (11, 1, 3, 2):
        x = torch.randn(2, 2, count, 32, generator=generator)
        x[..., 0, :] *= 30
        k, v = coordinator.process(0, x, x + 1, metadata, num_new_tokens=count)
        assert torch.equal(k[..., -count:, :], x)
        assert torch.equal(v[..., -count:, :], x + 1)
        assert_regions_cover_history(metadata[CACHE_KEY], 0)
        stored = metadata[CACHE_KEY].get(0, coordinator._decode_chunk)
        assert all(torch.isfinite(t).all() for t in stored)
    coordinator.clear_request(metadata)
