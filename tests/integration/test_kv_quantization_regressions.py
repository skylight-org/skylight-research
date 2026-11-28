"""Regression checks for cache ownership, HF callbacks and attention flow."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, MistralConfig

from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    CachePolicyRegistry,
    DefaultCachePolicy,
    DefaultCachePolicyConfig,
    FullQuantizationCachePolicyConfig,
    IntQuantizerConfig,
    KIVIConfig,
    KVQuantization,
    KVQuantizationConfig,
    OutlierTokenPolicyConfig,
    QuantGranularity,
    RegionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig,
)

pytestmark = pytest.mark.integration


def attention(config=None):
    return ResearchAttention.create_from_config(
        ResearchAttentionConfig(masker_configs=[], kv_quantization_config=config)
    )


def middle_config(**overrides):
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
        **overrides,
    )


def call_attention(implementation, q, k, v, metadata=None):
    length = q.shape[-2]
    mask = torch.zeros(1, 1, length, k.shape[-2])
    offset = k.shape[-2] - length
    future = torch.arange(k.shape[-2])[None, :] > (
        torch.arange(length)[:, None] + offset
    )
    mask.masked_fill_(future, float("-inf"))
    module = SimpleNamespace(training=False, layer_idx=0, config=SimpleNamespace())
    return implementation.custom_attention(
        module,
        q,
        k,
        v,
        mask,
        0.5,
        0.0,
        sparse_meta_data={} if metadata is None else metadata,
        layer_idx=0,
    )[0]


def test_request_ownership_layer_isolation_rollback_and_reset():
    quant = KVQuantization(middle_config())
    x = torch.randn(2, 2, 12, 8, generator=torch.Generator().manual_seed(17))
    meta = {}
    for layer in (0, 3):
        quant.process(layer, x, x, meta, num_new_tokens=0)
    store = meta[CACHE_KEY]
    assert set(store.layers) == {0, 3}
    first = store.layers[0].key.chunks[0]
    quant.process(0, x, x, meta, num_new_tokens=0)
    assert store.layers[0].key.chunks[0] is first
    with pytest.raises(ValueError, match="shrank"):
        quant.process(0, x[..., :-1, :], x[..., :-1, :], meta, num_new_tokens=0)
    with pytest.raises(ValueError, match="another quantization"):
        KVQuantization(middle_config()).process(0, x, x, meta, num_new_tokens=0)
    with pytest.raises(ValueError, match="dtype or device"):
        quant.process(0, x.half(), x.half(), meta, num_new_tokens=0)
    quant.clear_request(meta)
    assert CACHE_KEY not in meta and store.layers == {}


def test_direct_coordinator_prefill_option_works():
    x = torch.randn(1, 1, 4, 4)
    q = KVQuantization(middle_config())
    meta = {}
    k, v = q.process(0, x, x, meta, phase="prefill", num_new_tokens=0)
    assert k is x and v is x
    assert meta[CACHE_KEY].seen_tokens(0) == 4


def test_batched_quantized_attention_preserves_causality():
    q = torch.ones(1, 1, 4, 4)
    k = torch.arange(4.0).reshape(1, 1, 4, 1).expand(1, 1, 4, 4).clone()
    v = k + 1
    future_changed = k.clone()
    future_changed[..., 3, :] = 100
    plain = attention()
    assert torch.equal(
        call_attention(plain, q, k, v)[:, :3],
        call_attention(plain, q, future_changed, v)[:, :3],
    )
    quantized = attention(middle_config())
    before = call_attention(quantized, q, k, v)
    after = call_attention(quantized, q, future_changed, v)
    torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=0)


def test_rolling_hf_cache_tracks_new_tokens_with_independent_retention():
    native = DynamicCache(config=MistralConfig(num_hidden_layers=1, sliding_window=4))
    quant = KVQuantization(KVQuantizationConfig())
    meta = {}
    for token in range(5):
        current = torch.full((1, 1, 1, 32), float(token))
        keys, values = native.update(current, current, 0)
        restored, _ = quant.process(0, keys, values, meta, num_new_tokens=1)
    assert restored[0, 0, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert torch.equal(restored[..., -keys.shape[-2] :, :], keys)
    assert meta[CACHE_KEY].seen_tokens(0) == 5


def test_kivi_flush_boundary_matches_released_reference_current_step():
    quant = KVQuantization.create_from_config(
        KIVIConfig.with_settings(group_size=4, residual_length=8)
    )
    x = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(1))
    meta = {}
    quant.process(0, x[..., :7, :], x[..., :7, :], meta, num_new_tokens=0)
    current_keys, _ = quant.process(0, x, x, meta, num_new_tokens=1)
    # In released llama_kivi.py, QK with the full buffer is computed before flush.
    assert torch.equal(current_keys, x)


def test_attention_hook_honors_prefill_metadata():
    x = torch.randn(1, 1, 4, 4)
    meta = {"kv_quantization_phase": "prefill"}
    k, v = attention(middle_config()).quantize_kv(
        x, x, meta, num_new_tokens=4, layer_idx=0
    )
    assert k is x and v is x
    assert meta[CACHE_KEY].seen_tokens(0) == 4


def test_fake_store_bytes_counts_every_owned_kept_token_tensor():
    quant = KVQuantization(
        KVQuantizationConfig(
            precision_policy=OutlierTokenPolicyConfig(
                threshold=2, max_fraction=0.25, kinds=("key",)
            ),
            cache_policy=FullQuantizationCachePolicyConfig(),
        )
    )
    x = torch.ones(1, 1, 8, 32)
    x[..., 3, :] = 100
    meta = {}
    quant.process(0, x, x, meta, num_new_tokens=0)
    store = meta[CACHE_KEY]
    actual = 0
    for state in (store.layers[0].key, store.layers[0].value):
        for chunk in state.all_chunks():
            for tensor in (
                chunk.full_precision,
                chunk.dequantized,
                chunk.kept_index,
                chunk.kept_values,
            ):
                if tensor is not None:
                    actual += tensor.numel() * tensor.element_size()
    assert store.stats()["store_bytes"] == actual


@dataclass(frozen=True)
class AuditAfterAttentionConfig(DefaultCachePolicyConfig):
    pass


@CachePolicyRegistry.register(AuditAfterAttentionConfig)
class AuditAfterAttentionPolicy(DefaultCachePolicy):
    def next_chunk_tokens(self, tensor_kind, state, *, phase="before_attention"):
        if phase != "after_attention":
            return 0
        return super().next_chunk_tokens(tensor_kind, state, phase="before_attention")


@pytest.mark.xfail(
    strict=True,
    reason="Deferred F09: no current production policy requires post-attention compression",
)
def test_attention_calls_post_attention_compression_hook():
    config = KVQuantizationConfig(
        cache_policy=AuditAfterAttentionConfig(
            key_local=RegionConfig(0),
            value_local=RegionConfig(0),
            key_block_tokens=4,
            value_block_tokens=4,
        )
    )
    x = torch.randn(1, 1, 4, 4)
    meta = {}
    call_attention(attention(config), x, x, x, meta)
    assert meta[CACHE_KEY].layers[0].key.compressed_tokens == 4
