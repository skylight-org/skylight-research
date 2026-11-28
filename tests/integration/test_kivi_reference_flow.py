"""Real HF callback traces compared with the pinned authors' CPU numerical helpers."""

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
from sparse_attention_hub.kv_quantization import CACHE_KEY, KIVIConfig
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig,
)
from tests.reference.kivi import OfficialKIVICache

pytestmark = pytest.mark.integration


def reference_attention(query, keys, values, mask, scale):
    """Independent softmax attention over the reference reconstruction, with GQA."""
    copies = query.shape[1] // keys.shape[1]
    keys = keys.repeat_interleave(copies, dim=1).float()
    values = values.repeat_interleave(copies, dim=1).float()
    scores = (query.float() @ keys.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores + mask[..., : keys.shape[-2]].float()
    return (
        (torch.softmax(scores, dim=-1) @ values)
        .to(query.dtype)
        .transpose(1, 2)
        .contiguous()
    )


@pytest.mark.parametrize(
    "config_class,model_class",
    [
        (LlamaConfig, LlamaForCausalLM),
        (Qwen3Config, Qwen3ForCausalLM),
        (MistralConfig, MistralForCausalLM),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("prefill", [31, 65])
def test_hf_kivi_codes_state_and_attention_follow_reference_and_storage_modes_agree(
    config_class, model_class, dtype, bits, prefill
):
    torch.manual_seed(806)
    config = config_class(
        vocab_size=96,
        hidden_size=128,
        intermediate_size=160,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        sliding_window=None,
        max_position_embeddings=192,
    )
    config._attn_implementation = "eager"
    model = model_class(config).to(dtype=dtype).eval()
    results = []
    for packed in (False, True):
        research = ResearchAttention.create_from_config(
            ResearchAttentionConfig(
                masker_configs=[],
                kv_quantization_config=KIVIConfig.with_settings(
                    bits=bits, group_size=16, residual_length=32, store_codes=packed
                ),
            )
        )
        quant = research.kv_quantization
        original_process = quant.process
        references = {}
        prepared = {}
        phases = []

        def check_process(layer, keys, values, metadata, **kwargs):
            if kwargs["phase"] == "prefill":
                references[layer] = OfficialKIVICache(bits, 16, 32)
                expected = references[layer].prefill(keys, values)
            else:
                assert kwargs["num_new_tokens"] == 1
                expected = references[layer].step(
                    keys[..., -1:, :], values[..., -1:, :]
                )
            actual = original_process(layer, keys, values, metadata, **kwargs)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
            stored = metadata[CACHE_KEY].get(layer, quant._decode_chunk)
            assert torch.equal(stored[0], references[layer].get("key"))
            assert torch.equal(stored[1], references[layer].get("value"))
            prepared[layer] = expected
            phases.append((layer, kwargs["phase"]))
            return actual

        quant.process = check_process
        callback = ModelAdapterHF.get_custom_attention_function(
            SimpleNamespace(), research
        )

        def check_callback(
            module,
            queries,
            keys,
            values,
            attention_mask,
            scaling,
            dropout=0.0,
            **kwargs,
        ):
            actual = callback(
                module,
                queries,
                keys,
                values,
                attention_mask,
                scaling,
                dropout,
                **kwargs,
            )
            if "kv_quantization_capture_only" not in kwargs["sparse_meta_data"]:
                expected_k, expected_v = prepared[module.layer_idx]
                expected = reference_attention(
                    queries, expected_k, expected_v, attention_mask, scaling
                )
                torch.testing.assert_close(
                    actual[0],
                    expected,
                    rtol=3e-6 if dtype == torch.float32 else 2 * torch.finfo(dtype).eps,
                    atol=3e-7 if dtype == torch.float32 else 2e-6,
                )
            return actual

        name = "kivi_reference_" + uuid4().hex
        ALL_ATTENTION_FUNCTIONS.register(name, check_callback)
        ALL_MASK_ATTENTION_FUNCTIONS.register(name, eager_mask)
        metadata = {
            "kv_quantization_phase": "prefill",
            "kv_quantization_capture_only": {
                id(module): "eager"
                for module in model.modules()
                if hasattr(module, "layer_idx")
            },
        }
        model.config._attn_implementation = name
        try:
            with torch.no_grad():
                output = model(
                    torch.arange(1, prefill + 1).unsqueeze(0),
                    use_cache=True,
                    sparse_meta_data=metadata,
                )
                cache = output.past_key_values
                logits = [output.logits]
                metadata.pop("kv_quantization_capture_only")
                metadata["kv_quantization_phase"] = "decode"
                for token in (70, 71, 72, 73):
                    output = model(
                        torch.tensor([[token]]),
                        past_key_values=cache,
                        use_cache=True,
                        sparse_meta_data=metadata,
                    )
                    cache = output.past_key_values
                    logits.append(output.logits)
            assert phases.count((0, "prefill")) == phases.count((1, "prefill")) == 1
            assert phases.count((0, "decode")) == phases.count((1, "decode")) == 4
            results.append(torch.cat(logits, dim=1).detach())
            store = metadata[CACHE_KEY]
            research.clear_request_state(metadata)
            assert not store.layers and CACHE_KEY not in metadata
        finally:
            research.clear_request_state(metadata)
            model.config._attn_implementation = "eager"
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
    assert torch.equal(results[0], results[1])
