"""Offline, teacher-forced logits checks across three actual HF architectures."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sparse_attention_hub.adapters.huggingface import ModelAdapterHF
from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    DefaultCachePolicyConfig,
    FakeNVFP4QuantizerConfig,
    FP8QuantizerConfig,
    IntQuantizerConfig,
    KVQuantizationConfig,
    MXFPQuantizerConfig,
    NonUniformQuantizerConfig,
    RegionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig,
)
from tests.integration.test_kv_quantization_flow import (
    tiny_model_dir as shared_tiny_model_dir,
)

tiny_model_dir = shared_tiny_model_dir
pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "config_class,model_class",
    [
        (LlamaConfig, LlamaForCausalLM),
        (Qwen3Config, Qwen3ForCausalLM),
        (MistralConfig, MistralForCausalLM),
    ],
)
@pytest.mark.parametrize(
    "codec",
    [
        IntQuantizerConfig(bits=2),
        FP8QuantizerConfig(),
        FakeNVFP4QuantizerConfig(),
        MXFPQuantizerConfig(),
        NonUniformQuantizerConfig(),
    ],
)
def test_teacher_forced_logits_match_fake_and_packed_across_hf_models(
    config_class, model_class, codec
):
    results = {}
    for mode in ("off", "fake", "packed"):
        torch.manual_seed(29)
        config = config_class(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            max_position_embeddings=64,
            sliding_window=None,
        )
        config._attn_implementation = "eager"
        model = model_class(config).eval()
        quantization = (
            None
            if mode == "off"
            else KVQuantizationConfig(
                key_quantizer=codec,
                value_quantizer=codec,
                cache_policy=DefaultCachePolicyConfig(
                    key_local=RegionConfig(2),
                    value_local=RegionConfig(2),
                    key_block_tokens=4,
                    value_block_tokens=4,
                ),
                store_codes=mode == "packed",
            )
        )
        research = ResearchAttention.create_from_config(
            ResearchAttentionConfig(
                masker_configs=[],
                kv_quantization_config=quantization,
            )
        )
        # Use the production adapter callback; no adapter model loading/download.
        callback = ModelAdapterHF.get_custom_attention_function(
            SimpleNamespace(), research
        )
        name = f"skylight_audit_{uuid4().hex}"
        ALL_ATTENTION_FUNCTIONS.register(name, callback)
        ALL_MASK_ATTENTION_FUNCTIONS.register(name, eager_mask)
        meta = {}
        try:
            with torch.no_grad():
                cache = model(
                    torch.tensor([[1, 2, 3, 4, 5, 6, 7]]), use_cache=True
                ).past_key_values
                model.config._attn_implementation = name
                logits = []
                for token in (8, 9, 10):
                    outputs = model(
                        torch.tensor([[token]]),
                        past_key_values=cache,
                        use_cache=True,
                        sparse_meta_data=meta,
                    )
                    cache = outputs.past_key_values
                    logits.append(outputs.logits.detach())
            results[mode] = torch.cat(logits, dim=1)
            assert cache.get_seq_length() == 10
            assert all(layer.keys.dtype == torch.float32 for layer in cache.layers)
            if mode == "off":
                assert CACHE_KEY not in meta
            else:
                assert set(meta[CACHE_KEY].layers) == {0, 1}
                assert meta[CACHE_KEY].layers[0].key.compressed_tokens > 0
        finally:
            research.clear_request_state(meta)
            model.config._attn_implementation = "eager"
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
        assert CACHE_KEY not in meta
    assert torch.equal(results["fake"], results["packed"])
    assert torch.isfinite(results["fake"]).all()
    assert not torch.equal(results["off"], results["fake"])


@pytest.mark.parametrize("failure_phase", ["prefill", "decode"])
def test_adapter_clears_store_and_restores_attention_after_generation_error(
    tiny_model_dir, monkeypatch, failure_phase
):
    adapter = ModelAdapterHF(
        model_name=tiny_model_dir,
        device="cpu",
        sparse_attention_config=ResearchAttentionConfig(
            masker_configs=[], kv_quantization_config=KVQuantizationConfig()
        ),
    )
    original_impl = adapter.model.config._attn_implementation
    quantization = adapter.sparse_attention.kv_quantization
    original_process = quantization.process
    captured = {}

    def fail_after_store(layer_idx, keys, values, metadata, **kwargs):
        prepared = original_process(layer_idx, keys, values, metadata, **kwargs)
        if kwargs.get("phase", "decode") != failure_phase:
            return prepared
        captured["store"] = metadata[CACHE_KEY]
        captured["metadata"] = metadata
        raise RuntimeError("audit: deliberate failure after storing KV")

    monkeypatch.setattr(quantization, "process", fail_after_store)
    from sparse_attention_hub.adapters import Request

    try:
        with pytest.raises(RuntimeError, match="deliberate failure"):
            adapter.process_request(
                Request(context="w1 w2 w3", questions="w4 w5", answer_prefix=""),
                generation_kwargs={"max_new_tokens": 3},
                request_kwargs={},
            )
        assert captured["store"].layers == {}
        assert CACHE_KEY not in captured["metadata"]
        assert adapter.model.config._attn_implementation == original_impl
        assert (
            ALL_MASK_ATTENTION_FUNCTIONS[adapter._registered_attention_name]
            is eager_mask
        )
    finally:
        adapter._cleanup_attention_registration()
