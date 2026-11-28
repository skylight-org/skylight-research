"""F01/F02: exact current-call tokens and history independent of HF cache length."""

from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, MistralConfig

from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    DefaultCachePolicyConfig,
    FullQuantizationCachePolicyConfig,
    IntQuantizerConfig,
    KIVIConfig,
    KVQuantization,
    KVQuantizationConfig,
    QuantGranularity,
    RegionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig,
)

pytestmark = pytest.mark.unit


def history(tokens):
    return torch.randn(2, 2, tokens, 8, generator=torch.Generator().manual_seed(31))


def config(store_codes):
    return KVQuantizationConfig(
        key_quantizer=IntQuantizerConfig(
            bits=2, group_size=4, granularity=QuantGranularity.PER_CHANNEL
        ),
        value_quantizer=IntQuantizerConfig(bits=2, group_size=4),
        cache_policy=DefaultCachePolicyConfig(
            key_local=RegionConfig(0),
            value_local=RegionConfig(0),
            key_block_tokens=4,
            value_block_tokens=4,
        ),
        store_codes=store_codes,
    )


@pytest.mark.parametrize("store_codes", [False, True])
@pytest.mark.parametrize("new_count", [1, 2, 8])
def test_current_tokens_are_exact_and_the_prepared_view_survives_compression(
    store_codes, new_count
):
    quant = KVQuantization(config(store_codes))
    x = history(8 + new_count)
    meta = {}
    quant.process(0, x[..., :8, :], x[..., :8, :], meta, num_new_tokens=0)
    previous, _ = meta[CACHE_KEY].get(0, quant._decode_chunk)
    keys, values = quant.process(0, x, x, meta, num_new_tokens=new_count)
    assert torch.equal(keys[..., :8, :], previous)
    assert torch.equal(keys[..., -new_count:, :], x[..., -new_count:, :])
    assert torch.equal(values[..., -new_count:, :], x[..., -new_count:, :])
    snapshot = keys.clone()
    quant.process(0, x, x, meta, num_new_tokens=0)
    assert torch.equal(keys, snapshot)  # no aliases into mutable cache state
    assert meta[CACHE_KEY].seen_tokens(0) == 8 + new_count


@pytest.mark.parametrize("store_codes", [False, True])
def test_kivi_full_key_buffer_flush_keeps_both_new_tokens_exact(store_codes):
    quant = KVQuantization.create_from_config(
        KIVIConfig.with_settings(
            group_size=4,
            residual_length=8,
            store_codes=store_codes,
        )
    )
    x = history(8)
    meta = {}
    quant.process(0, x[..., :6, :], x[..., :6, :], meta, num_new_tokens=6)
    keys, values = quant.process(0, x, x, meta, num_new_tokens=2)
    assert torch.equal(keys, x) and torch.equal(values, x)
    assert meta[CACHE_KEY].layers[0].key.recent_tokens == 0
    stored, _ = meta[CACHE_KEY].get(0, quant._decode_chunk)
    assert not torch.equal(stored[..., -2:, :], x[..., -2:, :])


@pytest.mark.parametrize("store_codes", [False, True])
def test_new_sink_and_local_tokens_are_exact_even_when_their_store_is_quantized(
    store_codes,
):
    codec = IntQuantizerConfig(bits=2, group_size=4)
    quant = KVQuantization(
        KVQuantizationConfig(
            key_quantizer=codec,
            value_quantizer=codec,
            cache_policy=DefaultCachePolicyConfig(
                key_sink=RegionConfig(2, codec),
                value_sink=RegionConfig(2, codec),
                key_local=RegionConfig(4, codec),
                value_local=RegionConfig(4, codec),
                key_block_tokens=4,
                value_block_tokens=4,
            ),
            store_codes=store_codes,
        )
    )
    x = history(6)
    meta = {}
    keys, values = quant.process(0, x, x, meta, num_new_tokens=6)
    assert torch.equal(keys, x) and torch.equal(values, x)
    stored, _ = meta[CACHE_KEY].get(0, quant._decode_chunk)
    assert not torch.equal(stored, x)


@pytest.mark.parametrize("store_codes", [False, True])
def test_full_quantization_policy_stores_codes_but_returns_exact_current_tokens(
    store_codes,
):
    quant = KVQuantization(
        KVQuantizationConfig(
            cache_policy=FullQuantizationCachePolicyConfig(),
            store_codes=store_codes,
        )
    )
    x = history(3)
    meta = {}
    k, v = quant.process(0, x, x, meta, num_new_tokens=3)
    assert torch.equal(k, x) and torch.equal(v, x)
    assert meta[CACHE_KEY].layers[0].key.compressed_tokens == 3
    stored, _ = meta[CACHE_KEY].get(0, quant._decode_chunk)
    assert not torch.equal(stored, x)


@pytest.mark.parametrize("store_codes", [False, True])
def test_sliding_native_cache_keeps_new_suffix_and_our_entire_history(store_codes):
    native = DynamicCache(config=MistralConfig(num_hidden_layers=1, sliding_window=4))
    quant = KVQuantization(config(store_codes))
    x = history(16)
    meta = {}
    previous_end = 0
    for end in (9, 10, 12, 16):
        count = end - previous_end
        keys, values = native.update(
            x[..., previous_end:end, :], x[..., previous_end:end, :], 0
        )
        restored_k, restored_v = quant.process(
            0, keys, values, meta, num_new_tokens=count
        )
        assert restored_k.shape[-2] == end
        assert torch.equal(restored_k[..., -count:, :], x[..., previous_end:end, :])
        assert torch.equal(restored_v[..., -count:, :], x[..., previous_end:end, :])
        assert meta[CACHE_KEY].seen_tokens(0) == end
        previous_end = end
    assert native.layers[0].keys.shape[-2] == 3
    assert meta[CACHE_KEY].layers[0].seen_tokens == 16
    positions = [c.start_token for c in meta[CACHE_KEY].layers[0].key.all_chunks()]
    assert positions == sorted(set(positions))


def test_existing_stored_history_is_not_replaced_with_original_hf_values():
    quant = KVQuantization(config(True))
    x = history(10)
    meta = {}
    quant.process(0, x[..., :8, :], x[..., :8, :], meta, num_new_tokens=0)
    old, _ = meta[CACHE_KEY].get(0, quant._decode_chunk)
    incoming = x.clone()
    incoming[..., :8, :] += 100  # native prefix is deliberately not our stored state
    keys, _ = quant.process(0, incoming, incoming, meta, num_new_tokens=2)
    assert torch.equal(keys[..., :8, :], old)
    assert torch.equal(keys[..., -2:, :], incoming[..., -2:, :])


@pytest.mark.parametrize("invalid", [-1, 5, True, 1.5, None])
def test_explicit_new_token_count_is_validated(invalid):
    x = history(4)
    with pytest.raises(ValueError, match="num_new_tokens"):
        KVQuantization(config(True)).process(0, x, x, {}, num_new_tokens=invalid)


def test_late_sliding_callback_requires_capturing_the_context_first():
    attention = ResearchAttention.create_from_config(
        ResearchAttentionConfig(masker_configs=[], kv_quantization_config=config(True))
    )
    x = history(4)
    module = SimpleNamespace(training=False, layer_idx=0, config=SimpleNamespace())
    with pytest.raises(ValueError, match="Capture prefill"):
        attention.custom_attention(
            module,
            x[..., -1:, :],
            x,
            x,
            None,
            0.5,
            0.0,
            sparse_meta_data={},
            layer_idx=0,
            sliding_window=4,
        )


def test_prefill_capture_is_layer_local_and_keeps_native_attention_inputs():
    quant = KVQuantization(config(True))
    x = history(12)
    meta = {}
    for layer in (0, 1):
        k, v = quant.process(layer, x, x, meta, num_new_tokens=12, phase="prefill")
        assert k is x and v is x
        assert meta[CACHE_KEY].seen_tokens(layer) == 12
    assert set(meta[CACHE_KEY].layers) == {0, 1}


@pytest.mark.parametrize("quantize_prefill", [False, True])
def test_chunked_prefill_flag_controls_previous_history_but_new_tokens_stay_exact(
    quantize_prefill,
):
    from dataclasses import replace

    quant = KVQuantization(replace(config(True), quantize_prefill=quantize_prefill))
    x = history(10)
    meta = {}
    first, _ = quant.process(
        0, x[..., :8, :], x[..., :8, :], meta, num_new_tokens=8, phase="prefill"
    )
    assert torch.equal(first, x[..., :8, :])
    previous, _ = meta[CACHE_KEY].get(0, quant._decode_chunk)
    assert not torch.equal(previous, x[..., :8, :])
    result, _ = quant.process(0, x, x, meta, num_new_tokens=2, phase="prefill")
    assert torch.equal(result[..., -2:, :], x[..., -2:, :])
    expected = previous if quantize_prefill else x[..., :8, :]
    assert torch.equal(result[..., :8, :], expected)


@pytest.mark.parametrize("store_codes", [False, True])
def test_native_sliding_reference_metrics_compare_only_available_originals(
    store_codes, monkeypatch
):
    from dataclasses import replace

    from sparse_attention_hub.metric_logging import MicroMetricLogger

    events = []
    monkeypatch.setattr(
        MicroMetricLogger,
        "is_metric_enabled",
        lambda self, name: name.startswith("kv_quantization_"),
    )
    monkeypatch.setattr(
        MicroMetricLogger,
        "log",
        lambda self, name, value, metadata: events.append((name, value, metadata)),
    )
    quant = KVQuantization(replace(config(store_codes), error_metric=True))
    native = DynamicCache(config=MistralConfig(num_hidden_layers=1, sliding_window=4))
    x = history(12)
    meta = {}
    for start, end in ((0, 8), (8, 9), (9, 12)):
        keys, values = native.update(x[..., start:end, :], x[..., start:end, :], 0)
        quant.process(0, keys, values, meta, num_new_tokens=end - start)
    assert events
    assert all(torch.isfinite(torch.tensor(value)) for _, value, _ in events)
    assert all(info.get("tokens", 0) <= 6 for _, _, info in events[-3:])
