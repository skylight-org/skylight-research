"""QuaRot's immediate current-token reconstruction and request-local storage."""

from dataclasses import replace

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    DefaultCachePolicyConfig,
    HadamardTransformConfig,
    IntQuantizer,
    IntQuantizerConfig,
    KIVIConfig,
    KVQuantization,
    OutlierTokenPolicyConfig,
    QuaRotKV,
    QuaRotKVConfig,
    quarot_int_quantizer,
)
from tests.reference.quarot import reconstruct

pytestmark = pytest.mark.unit


def small_config(*, packed=True, quantize_prefill=True):
    return QuaRotKVConfig(
        key_quantizer=quarot_int_quantizer("key_head", group_size=16),
        value_quantizer=quarot_int_quantizer("value_group", group_size=8),
        store_codes=packed,
        quantize_prefill=quantize_prefill,
    )


def inputs():
    generator = torch.Generator().manual_seed(412)
    return tuple(torch.randn(2, 2, 9, 16, generator=generator).half() for _ in range(2))


def expected(keys, values):
    return (
        reconstruct(keys, mode="key_head", group_size=16),
        reconstruct(values, mode="value_group", group_size=8),
    )


@pytest.mark.parametrize("packed", [False, True])
def test_registry_flow_uses_reconstructed_prefill_and_current_tokens(packed):
    method = KVQuantization.create_from_config(small_config(packed=packed))
    assert type(method) is QuaRotKV
    assert type(method.key_quantizer) is IntQuantizer
    keys, values = inputs()
    metadata = {}
    for end, count in [(4, 4), (5, 1), (8, 3), (9, 1)]:
        actual = method.process(
            0,
            keys[..., :end, :],
            values[..., :end, :],
            metadata,
            num_new_tokens=count,
            phase="prefill" if end == 4 else "decode",
        )
        target = expected(keys[..., :end, :], values[..., :end, :])
        assert all(torch.equal(a, b) for a, b in zip(actual, target))
        assert not torch.equal(
            actual[0][..., -count:, :], keys[..., end - count : end, :]
        )
        layer = metadata[CACHE_KEY].layers[0]
        assert layer.seen_tokens == end
        assert layer.key.recent_tokens == layer.value.recent_tokens == 0
        assert layer.key.sink_tokens == layer.value.sink_tokens == 0
    store = metadata[CACHE_KEY]
    if packed:
        k, v = store.layers[0].key.chunks[0], store.layers[0].value.chunks[0]
        assert k.quantized is not None and v.quantized is not None
        assert k.full_precision is None and k.dequantized is None
    method.clear_request(metadata)
    assert not store.layers and CACHE_KEY not in metadata


def test_fake_and_packed_storage_agree_and_have_distinct_byte_counts():
    keys, values = inputs()
    outputs, byte_counts = [], []
    for packed in (False, True):
        method = KVQuantization.create_from_config(small_config(packed=packed))
        metadata = {}
        outputs.append(
            method.process(0, keys, values, metadata, num_new_tokens=9, phase="prefill")
        )
        byte_counts.append(metadata[CACHE_KEY].stats()["store_bytes"])
    assert all(torch.equal(a, b) for a, b in zip(*outputs))
    assert byte_counts[1] < byte_counts[0]


def test_explicit_native_prefill_still_compresses_for_the_next_decode():
    keys, values = inputs()
    method = KVQuantization.create_from_config(small_config(quantize_prefill=False))
    metadata = {}
    actual = method.process(
        0,
        keys[..., :4, :],
        values[..., :4, :],
        metadata,
        num_new_tokens=4,
        phase="prefill",
    )
    assert torch.equal(actual[0], keys[..., :4, :])
    assert metadata[CACHE_KEY].layers[0].key.compressed_tokens == 4
    actual = method.process(
        0, keys[..., :5, :], values[..., :5, :], metadata, num_new_tokens=1
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(actual, expected(keys[..., :5, :], values[..., :5, :]))
    )


def test_kivi_inherits_base_process_and_preserves_its_arriving_tokens():
    assert (
        "process" not in type(KVQuantization.create_from_config(KIVIConfig())).__dict__
    )
    keys, values = inputs()
    method = KVQuantization.create_from_config(
        KIVIConfig.with_settings(group_size=8, residual_length=16)
    )
    actual = method.process(0, keys, values, {}, num_new_tokens=9, phase="prefill")
    assert torch.equal(actual[0], keys) and torch.equal(actual[1], values)


@pytest.mark.parametrize(
    "changes",
    [
        {"key_transform": HadamardTransformConfig()},
        {"key_quantizer": IntQuantizerConfig(group_size=16)},
        {"key_quantizer": quarot_int_quantizer("value_group", group_size=16)},
        {"cache_policy": DefaultCachePolicyConfig()},
        {"precision_policy": OutlierTokenPolicyConfig()},
        {"quantize_prefill": 1},
    ],
)
def test_invalid_named_recipe_components_are_rejected(changes):
    with pytest.raises(ValueError):
        KVQuantization.create_from_config(replace(small_config(), **changes))


def test_invalid_layout_is_rejected_before_request_state_is_created():
    method = KVQuantization.create_from_config(small_config())
    for keys in (
        torch.randn(1, 2, 3, 15),
        torch.randn(1, 2, 3, 32),
        torch.randn(1, 2, 3, 16, dtype=torch.float64),
    ):
        metadata = {}
        with pytest.raises(ValueError):
            method.process(0, keys, keys.clone(), metadata, num_new_tokens=3)
        assert not metadata


def test_altering_future_tokens_does_not_change_earlier_reconstructions():
    keys, values = inputs()
    changed = keys.clone()
    changed[..., -1, :] *= 100
    outputs = []
    for k in (keys, changed):
        method = KVQuantization.create_from_config(small_config())
        outputs.append(
            method.process(0, k, values, {}, num_new_tokens=9, phase="prefill")
        )
    assert torch.equal(outputs[0][0][..., :-1, :], outputs[1][0][..., :-1, :])
