"""Unit tests for wiring KV quantization into SparseAttention / ResearchAttention."""

from types import SimpleNamespace

import pytest
import torch

from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    DefaultCachePolicyConfig,
    IntQuantizerConfig,
    KVQuantization,
    KVQuantizationConfig,
    QuantizedKVStore,
    RegionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import (
    get_true_attention_output,
)

pytestmark = pytest.mark.unit

Q_LEN = 4
K_LEN = 64
HEADS = 2
DIM = 32


def _kv_config(store_codes: bool = False) -> KVQuantizationConfig:
    """Small blocks so a 64-token history has compressed chunks plus a bf16 tail."""
    return KVQuantizationConfig(
        key_quantizer=IntQuantizerConfig(bits=2),
        value_quantizer=IntQuantizerConfig(bits=2),
        cache_policy=DefaultCachePolicyConfig(
            key_sink=RegionConfig(tokens=2),
            value_sink=RegionConfig(tokens=2),
            key_local=RegionConfig(tokens=16),
            value_local=RegionConfig(tokens=16),
            key_block_tokens=16,
            value_block_tokens=16,
        ),
        store_codes=store_codes,
    )


def _attention(kv_config=None, maskers=None, apply_to_layer_types=None):
    config = ResearchAttentionConfig(
        masker_configs=[] if maskers is None else maskers,
        apply_to_layer_types=apply_to_layer_types,
        kv_quantization_config=kv_config,
    )
    return ResearchAttention.create_from_config(config)


def _module(layer_idx=0, layer_types=None):
    """Stand-in for a HF attention module: needs .config, .layer_idx and .training."""
    return SimpleNamespace(
        config=SimpleNamespace(layer_types=layer_types),
        layer_idx=layer_idx,
        training=False,
    )


def _tensors():
    torch.manual_seed(0)
    return (
        torch.randn((1, HEADS, Q_LEN, DIM)),
        torch.randn((1, HEADS, K_LEN, DIM)),
        torch.randn((1, HEADS, K_LEN, DIM)),
    )


def _causal_mask():
    offset = K_LEN - Q_LEN
    rows = torch.arange(Q_LEN).unsqueeze(1) + offset
    cols = torch.arange(K_LEN).unsqueeze(0)
    mask = torch.zeros(1, 1, Q_LEN, K_LEN)
    mask.masked_fill_((cols > rows)[None, None], torch.finfo(torch.float32).min)
    return mask


def _run(attention, q, k, v, meta, layer_idx=0, module=None):
    return attention.custom_attention(
        module=module if module is not None else _module(layer_idx),
        queries=q,
        keys=k,
        values=v,
        attention_mask=_causal_mask(),
        scaling=1.0 / DIM**0.5,
        dropout=0.0,
        sparse_meta_data=meta,
        layer_idx=layer_idx,
    )


class TestConfigField:
    def test_default_is_none(self):
        config = ResearchAttentionConfig(masker_configs=[])
        assert config.kv_quantization_config is None

    def test_keyword_only_keeps_masker_configs_positional(self):
        kv_config = _kv_config()
        config = ResearchAttentionConfig([], kv_quantization_config=kv_config)
        assert config.masker_configs == []
        assert config.kv_quantization_config is kv_config

    def test_attention_builds_quantization_only_when_configured(self):
        assert _attention().kv_quantization is None
        assert isinstance(_attention(_kv_config()).kv_quantization, KVQuantization)


class TestOff:
    def test_quantize_kv_returns_inputs_unchanged(self):
        _, k, v = _tensors()
        meta = {}
        out_k, out_v = _attention().quantize_kv(
            k, v, meta, num_new_tokens=Q_LEN, layer_idx=0
        )
        assert out_k is k and out_v is v
        assert CACHE_KEY not in meta

    def test_output_identical_to_attention_without_the_field(self):
        q, k, v = _tensors()
        maskers = [SinkMaskerConfig(sink_size=2), LocalMaskerConfig(window_size=8)]
        plain = ResearchAttention.create_from_config(
            ResearchAttentionConfig(masker_configs=maskers)
        )
        out_plain, _ = _run(plain, q, k, v, {})
        out_off, _ = _run(_attention(maskers=maskers), q, k, v, {})
        assert torch.equal(out_plain, out_off)


class TestOn:
    def test_attention_runs_on_reconstructed_kv(self):
        q, k, v = _tensors()
        meta = {}
        out, _ = _run(_attention(_kv_config()), q, k, v, meta)

        # Same history through a separate coordinator gives the K̂, V̂ attention used.
        k_hat, v_hat = KVQuantization.create_from_config(_kv_config()).process(
            0, k, v, {}, num_new_tokens=0
        )
        expected, _ = _run(_attention(), q, k_hat, v_hat, {})
        assert torch.equal(out, expected)

        unquantized, _ = _run(_attention(), q, k, v, {})
        assert not torch.allclose(out, unquantized)

    def test_store_lives_in_sparse_meta_data(self):
        q, k, v = _tensors()
        meta = {}
        _run(_attention(_kv_config()), q, k, v, meta, layer_idx=3)
        store = meta[CACHE_KEY]
        assert isinstance(store, QuantizedKVStore)
        layer = store.stats()["layers"][3]
        assert layer["seen_tokens"] == K_LEN
        assert layer["key_sink_tokens"] == 2
        assert layer["key_compressed_tokens"] > 0
        assert (
            layer["key_sink_tokens"]
            + layer["key_compressed_tokens"]
            + layer["key_recent_tokens"]
            == K_LEN
        )

    def test_hf_inputs_are_not_modified(self):
        q, k, v = _tensors()
        k_before, v_before = k.clone(), v.clone()
        _run(_attention(_kv_config()), q, k, v, {})
        assert torch.equal(k, k_before) and torch.equal(v, v_before)

    def test_fake_and_packed_give_identical_output(self):
        q, k, v = _tensors()
        fake, _ = _run(_attention(_kv_config(store_codes=False)), q, k, v, {})
        packed, _ = _run(_attention(_kv_config(store_codes=True)), q, k, v, {})
        assert torch.equal(fake, packed)

    def test_layer_excluded_by_gating_is_still_quantized(self):
        q, k, v = _tensors()
        attention = _attention(_kv_config(), apply_to_layer_types=("full_attention",))
        module = _module(0, layer_types=["sliding_attention"])
        meta = {}
        out, _ = _run(attention, q, k, v, meta, module=module)

        k_hat, v_hat = meta[CACHE_KEY].get(0, attention.kv_quantization._decode_chunk)
        expected, _ = get_true_attention_output(
            module, q, k_hat, v_hat, _causal_mask(), 1.0 / DIM**0.5, 0.0
        )
        assert torch.equal(out, expected)

    def test_missing_layer_idx_raises(self):
        _, k, v = _tensors()
        with pytest.raises(KeyError):
            _attention(_kv_config()).quantize_kv(k, v, {}, num_new_tokens=Q_LEN)


class TestRequestCleanup:
    def test_clear_request_state_removes_the_store(self):
        q, k, v = _tensors()
        attention = _attention(_kv_config())
        meta = {}
        _run(attention, q, k, v, meta)
        assert CACHE_KEY in meta
        attention.clear_request_state(meta)
        assert CACHE_KEY not in meta

    def test_clear_request_state_is_a_no_op_when_off(self):
        meta = {"other": 1}
        _attention().clear_request_state(meta)
        assert meta == {"other": 1}
