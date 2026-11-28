"""Store-owned tensor bytes and exact-token lifetime in FAKE/PACKED modes."""

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    DefaultCachePolicyConfig,
    FullQuantizationCachePolicyConfig,
    HadamardTransformConfig,
    IntQuantizerConfig,
    KVQuantization,
    KVQuantizationConfig,
    OutlierTokenPolicyConfig,
    QuantGranularity,
    RegionConfig,
    TurboQuantKVConfig,
    TurboQuantQuantizerConfig,
)

pytestmark = pytest.mark.unit


def chunks(store):
    for layer in store.layers.values():
        for state in (layer.key, layer.value):
            yield from state.all_chunks()


def allocated_tensor_bytes(store):
    """Independent oracle: count distinct backing allocations in tensor payloads.

    Matches the store's documented scope: payloads and per-record quantizer
    parameters, excluding shared components, context/transform state and temporaries.
    """
    allocations = {}
    for chunk in chunks(store):
        tensors = [
            chunk.full_precision,
            chunk.dequantized,
            chunk.kept_index,
            chunk.kept_values,
        ]
        if chunk.quantized is not None:
            tensors.extend(vars(chunk.quantized).values())
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                continue
            storage = tensor.untyped_storage()
            allocations[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(allocations.values())


def coordinator(store_codes, *, rotate=False):
    settings = {
        "store_codes": store_codes,
        "precision_policy": OutlierTokenPolicyConfig(threshold=2, max_fraction=0.25),
        "key_quantizer": IntQuantizerConfig(
            bits=2,
            group_size=4,
            granularity=QuantGranularity.PER_CHANNEL,
        ),
        "cache_policy": DefaultCachePolicyConfig(
            key_local=RegionConfig(0),
            value_local=RegionConfig(0),
            key_block_tokens=8,
            value_block_tokens=8,
        ),
    }
    if rotate:
        settings["key_transform"] = HadamardTransformConfig()
    return KVQuantization(KVQuantizationConfig(**settings))


@pytest.mark.parametrize("store_codes", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_outliers", [False, True])
def test_count_matches_allocations_and_kept_tokens_remain_exact(
    store_codes,
    dtype,
    with_outliers,
):
    quant = coordinator(store_codes, rotate=True)
    keys = torch.ones(2, 2, 8, 32, dtype=dtype)
    if with_outliers:
        keys[0, 0, 3] = 100
        keys[1, 1, 5] = 200
    values = keys / 2
    metadata = {}
    restored_k, restored_v = quant.process(
        0,
        keys,
        values,
        metadata,
        num_new_tokens=0,
    )
    store = metadata[CACHE_KEY]
    assert store.stats()["store_bytes"] == allocated_tensor_bytes(store)
    if not store_codes:
        assert store.stats()["store_bytes"] == 2 * keys.numel() * keys.element_size()
    kept_tokens = 0
    for chunk in chunks(store):
        if store_codes and with_outliers:
            assert chunk.kept_index is not None and chunk.kept_values is not None
            kept_tokens += chunk.kept_index.numel()
        else:
            assert chunk.kept_index is None and chunk.kept_values is None
    assert kept_tokens == (4 if store_codes and with_outliers else 0)
    if with_outliers:
        assert torch.equal(restored_k[0, 0, 3], keys[0, 0, 3])
        assert torch.equal(restored_k[1, 1, 5], keys[1, 1, 5])
        assert torch.equal(restored_v[0, 0, 3], values[0, 0, 3])
        assert torch.equal(restored_v[1, 1, 5], values[1, 1, 5])
    # Stored tensors own their originals, independently of caller buffers.
    expected_k, expected_v = restored_k.clone(), restored_v.clone()
    keys.zero_()
    values.zero_()
    again_k, again_v = store.get(0, quant._decode_chunk)
    assert torch.equal(again_k, expected_k) and torch.equal(again_v, expected_v)


def test_original_fake_undercount_example_now_owns_exactly_2048_bytes():
    quant = KVQuantization(
        KVQuantizationConfig(
            precision_policy=OutlierTokenPolicyConfig(
                threshold=2,
                max_fraction=0.25,
                kinds=("key",),
            ),
            cache_policy=FullQuantizationCachePolicyConfig(),
        )
    )
    x = torch.ones(1, 1, 8, 32)
    x[..., 3, :] = 100
    metadata = {}
    keys, _ = quant.process(0, x, x, metadata, num_new_tokens=0)
    store = metadata[CACHE_KEY]
    assert store.stats()["store_bytes"] == allocated_tensor_bytes(store) == 2048
    assert torch.equal(keys[..., 3, :], x[..., 3, :])
    for chunk in chunks(store):
        assert chunk.kept_index is None and chunk.kept_values is None


def test_fake_kept_tokens_survive_later_attention_and_storage_updates():
    quant = coordinator(False)
    x = torch.ones(1, 1, 16, 32)
    x[..., 3, :] = 100
    x[..., 12, :] = 200
    metadata = {}
    quant.process(0, x[..., :8, :], x[..., :8, :], metadata, num_new_tokens=8)
    keys, values = quant.process(0, x, x, metadata, num_new_tokens=8)
    assert torch.equal(keys[..., 3, :], x[..., 3, :])
    assert torch.equal(values[..., 3, :], x[..., 3, :])
    assert torch.equal(keys[..., 8:, :], x[..., 8:, :])
    store = metadata[CACHE_KEY]
    stored_keys, stored_values = store.get(0, quant._decode_chunk)
    assert torch.equal(stored_keys[..., [3, 12], :], x[..., [3, 12], :])
    assert torch.equal(stored_values[..., [3, 12], :], x[..., [3, 12], :])
    assert store.stats()["store_bytes"] == allocated_tensor_bytes(store) == 4096
    for chunk in chunks(store):
        assert chunk.kept_index is None and chunk.kept_values is None


@pytest.mark.parametrize("variant", ["mse", "prod"])
@pytest.mark.parametrize("packed", [False, True])
def test_turboquant_payload_allocation_count_includes_norms_and_sketches(
    variant, packed
):
    method = KVQuantization.create_from_config(
        TurboQuantKVConfig(
            key_quantizer=TurboQuantQuantizerConfig(variant=variant),
            store_codes=packed,
        )
    )
    generator = torch.Generator().manual_seed(972)
    metadata = {}
    for layer, length in ((0, 5), (1, 5), (0, 1), (1, 1)):
        x = torch.randn(2, 3, length, 17, generator=generator).half()
        method.process(layer, x, x.clone(), metadata, num_new_tokens=length)
        store = metadata[CACHE_KEY]
        assert store.stats()["store_bytes"] == allocated_tensor_bytes(store)
        if packed:
            for chunk in chunks(store):
                record = chunk.quantized
                assert record.norms is not None
                assert (record.qjl_codes is not None) is (
                    variant == "prod" and chunk.context.tensor_kind == "key"
                )
