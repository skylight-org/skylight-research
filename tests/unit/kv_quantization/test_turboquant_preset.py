"""Whole-head TurboQuant cache flow and ownership, independent of HF internals."""

from dataclasses import replace

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    AttentionContext,
    DefaultCachePolicyConfig,
    HadamardTransformConfig,
    IntQuantizerConfig,
    KVQuantization,
    OutlierTokenPolicyConfig,
    TurboQuantKV,
    TurboQuantKVConfig,
    TurboQuantQuantizer,
    TurboQuantQuantizerConfig,
)
from tests.reference.turboquant import encode

pytestmark = pytest.mark.unit


def make_config(*, packed=True, variant="mse", quantize_prefill=True):
    return TurboQuantKVConfig(
        key_quantizer=TurboQuantQuantizerConfig(bits=3, variant=variant),
        value_quantizer=TurboQuantQuantizerConfig(bits=2),
        store_codes=packed,
        quantize_prefill=quantize_prefill,
    )


def inputs():
    generator = torch.Generator().manual_seed(603)
    return tuple(torch.randn(2, 2, 9, 16, generator=generator) for _ in range(2))


def reference(method, tensor, kind):
    codec = getattr(method, f"{kind}_quantizer")
    state = codec.state_for(AttentionContext(0, kind, 0), tensor.shape[-1])
    return torch.tensor(
        encode(
            tensor, state, bits=codec.config.bits, variant=codec.config.variant
        ).reconstructed,
        dtype=tensor.dtype,
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("variant", ["mse", "prod"])
def test_registry_and_streaming_flow_reconstruct_all_tokens(packed, variant):
    method = KVQuantization.create_from_config(
        make_config(packed=packed, variant=variant)
    )
    assert (
        type(method) is TurboQuantKV
        and type(method.key_quantizer) is TurboQuantQuantizer
    )
    keys, values = inputs()
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
            torch.testing.assert_close(
                actual,
                reference(method, original[..., :end, :], kind),
                rtol=3e-6,
                atol=3e-6,
            )
            assert not torch.equal(
                actual[..., -count:, :], original[..., end - count : end, :]
            )
        state = metadata[CACHE_KEY].layers[0]
        assert state.seen_tokens == end
        assert state.key.recent_tokens == state.value.recent_tokens == 0
        assert state.key.sink_tokens == state.value.sink_tokens == 0
    chunk = metadata[CACHE_KEY].layers[0].key.chunks[0]
    assert chunk.full_precision is None
    assert (chunk.quantized is not None) is packed
    store = metadata[CACHE_KEY]
    method.clear_request(metadata)
    assert not store.layers and CACHE_KEY not in metadata


@pytest.mark.parametrize("variant", ["mse", "prod"])
def test_fake_and_packed_storage_are_identical_and_count_all_request_tensors(variant):
    keys, values = inputs()
    outputs, sizes = [], []
    for packed in (False, True):
        method = KVQuantization.create_from_config(
            make_config(packed=packed, variant=variant)
        )
        metadata = {}
        outputs.append(method.process(0, keys, values, metadata, num_new_tokens=9))
        sizes.append(metadata[CACHE_KEY].stats()["store_bytes"])
    assert all(torch.equal(a, b) for a, b in zip(*outputs))
    vectors, channels = 2 * 2 * 9, 16
    key_coarse = 3 - (variant == "prod")
    expected = (vectors * channels * key_coarse + 7) // 8 + 4 * vectors
    if variant == "prod":
        expected += (vectors * channels + 7) // 8 + 4 * vectors
    expected += (vectors * channels * 2 + 7) // 8 + 4 * vectors
    assert sizes[1] == expected < sizes[0] == keys.numel() * keys.element_size() * 2


def test_native_prefill_option_still_stores_quantized_context():
    keys, values = inputs()
    method = KVQuantization.create_from_config(make_config(quantize_prefill=False))
    metadata = {}
    result = method.process(
        0,
        keys[..., :4, :],
        values[..., :4, :],
        metadata,
        num_new_tokens=4,
        phase="prefill",
    )
    assert torch.equal(result[0], keys[..., :4, :])
    assert metadata[CACHE_KEY].layers[0].key.compressed_tokens == 4
    result = method.process(
        0, keys[..., :5, :], values[..., :5, :], metadata, num_new_tokens=1
    )
    torch.testing.assert_close(
        result[0], reference(method, keys[..., :5, :], "key"), rtol=3e-6, atol=3e-6
    )


def test_future_tokens_and_hf_eviction_do_not_change_our_completed_history():
    keys, values = inputs()
    changed = keys.clone()
    changed[..., -1, :] *= 100
    earlier = []
    for tensor in (keys, changed):
        method = KVQuantization.create_from_config(make_config(variant="prod"))
        earlier.append(
            method.process(0, tensor, values, {}, num_new_tokens=9)[0][..., :-1, :]
        )
    assert torch.equal(*earlier)
    method = KVQuantization.create_from_config(make_config())
    metadata = {}
    method.process(0, keys[..., :8, :], values[..., :8, :], metadata, num_new_tokens=8)
    # HF exposes only its current suffix, while our policy retains all history.
    actual = method.process(
        0, keys[..., -2:, :], values[..., -2:, :], metadata, num_new_tokens=1
    )
    torch.testing.assert_close(
        actual[0], reference(method, keys, "key"), rtol=3e-6, atol=3e-6
    )


def test_store_and_returned_tensors_own_their_data_and_requests_are_isolated():
    keys, values = inputs()
    method = KVQuantization.create_from_config(make_config(variant="prod"))
    metadata = {}
    actual = method.process(0, keys, values, metadata, num_new_tokens=9)
    saved = metadata[CACHE_KEY].get(0, method._decode_chunk)[0].clone()
    actual[0].zero_()
    keys.zero_()
    assert torch.equal(metadata[CACHE_KEY].get(0, method._decode_chunk)[0], saved)
    other = {}
    method.process(0, keys, values, other, num_new_tokens=9)
    assert other[CACHE_KEY] is not metadata[CACHE_KEY]
    assert not other[CACHE_KEY].get(0, method._decode_chunk)[0].any()


@pytest.mark.parametrize(
    "changes",
    [
        {"key_transform": HadamardTransformConfig()},
        {"value_quantizer": IntQuantizerConfig()},
        {"cache_policy": DefaultCachePolicyConfig()},
        {"precision_policy": OutlierTokenPolicyConfig()},
        {"store_codes": 1},
        {"quantize_prefill": 1},
    ],
)
def test_invalid_named_components_fail_at_selection(changes):
    with pytest.raises(ValueError):
        KVQuantization.create_from_config(replace(make_config(), **changes))


def test_invalid_layout_is_rejected_before_creating_request_state():
    method = KVQuantization.create_from_config(make_config())
    for keys, values in (
        (torch.ones(1, 1, 2, 1), torch.ones(1, 1, 2, 1)),
        (torch.ones(1, 1, 2, 4).double(), torch.ones(1, 1, 2, 4)),
        (torch.ones(1, 1, 2, 4), torch.ones(1, 2, 2, 4)),
    ):
        metadata = {}
        with pytest.raises(ValueError):
            method.process(0, keys, values, metadata, num_new_tokens=2)
        assert not metadata
