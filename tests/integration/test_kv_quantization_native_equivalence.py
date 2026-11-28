"""Separate lossless-cache correctness from attention-backend rounding."""

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
    KVQuantizationConfig,
    RegionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "config_class,model_class,window",
    [
        (LlamaConfig, LlamaForCausalLM, None),
        (Qwen3Config, Qwen3ForCausalLM, None),
        (MistralConfig, MistralForCausalLM, None),
        (MistralConfig, MistralForCausalLM, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_lossless_owned_history_matches_native_prefill_and_multitoken_decode(
    config_class, model_class, window, dtype, record_property
):
    torch.manual_seed(719)
    config = config_class(
        vocab_size=48,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=window,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    model = model_class(config).to(dtype=dtype).eval()
    # A long FP sink makes this a lossless control of the shared integration.
    research = ResearchAttention.create_from_config(
        ResearchAttentionConfig(
            masker_configs=[],
            kv_quantization_config=KVQuantizationConfig(
                cache_policy=DefaultCachePolicyConfig(
                    key_sink=RegionConfig(128), value_sink=RegionConfig(128)
                )
            ),
        )
    )
    research_off = ResearchAttention.create_from_config(
        ResearchAttentionConfig(masker_configs=[])
    )
    name = "preflight_lossless_" + uuid4().hex
    ALL_MASK_ATTENTION_FUNCTIONS.register(name, eager_mask)
    results = {}
    try:
        for mode in ("native", "research-off", "owned-cache"):
            metadata = {}
            native = mode == "native"
            selected = research if mode == "owned-cache" else research_off
            callback = ModelAdapterHF.get_custom_attention_function(
                SimpleNamespace(), selected
            )
            ALL_ATTENTION_FUNCTIONS.register(name, callback)
            model.config._attn_implementation = "eager" if native else name
            cache = None
            logits = []
            with torch.no_grad():
                for step, tokens in enumerate(
                    ([1, 2, 3, 4, 5, 6, 7], [8, 9, 10], [11], [12, 13])
                ):
                    metadata["kv_quantization_phase"] = (
                        "prefill" if step == 0 else "decode"
                    )
                    if step == 0:
                        if mode == "owned-cache":
                            metadata["kv_quantization_capture_only"] = {
                                id(module): "eager"
                                for module in model.modules()
                                if hasattr(module, "layer_idx")
                            }
                            model.config._attn_implementation = name
                        else:
                            model.config._attn_implementation = "eager"
                    else:
                        model.config._attn_implementation = "eager" if native else name
                    kwargs = {} if native else {"sparse_meta_data": metadata}
                    output = model(
                        torch.tensor([tokens]),
                        past_key_values=cache,
                        use_cache=True,
                        **kwargs,
                    )
                    metadata.pop("kv_quantization_capture_only", None)
                    cache = output.past_key_values
                    logits.append(output.logits.detach())
            results[mode] = torch.cat(logits, dim=1)
            if mode == "owned-cache":
                assert set(metadata[CACHE_KEY].layers) == {0, 1}
                assert all(
                    state.seen_tokens == 13
                    for state in metadata[CACHE_KEY].layers.values()
                )
                research.clear_request_state(metadata)
                assert CACHE_KEY not in metadata
        native_difference = (
            (results["research-off"].float() - results["native"].float())
            .abs()
            .max()
            .item()
        )
        cache_difference = (
            (results["owned-cache"].float() - results["research-off"].float())
            .abs()
            .max()
            .item()
        )
        record_property("native_vs_research_max_abs", native_difference)
        record_property("lossless_cache_vs_research_max_abs", cache_difference)
        # Full attention keeps identical layouts. Longer retained sliding history
        # can change FP32 reduction order despite identical visible values.
        atol = (
            0
            if window is None
            else {torch.float32: 3e-7, torch.float16: 5e-4, torch.bfloat16: 4e-3}[dtype]
        )
        torch.testing.assert_close(
            results["owned-cache"], results["research-off"], rtol=0, atol=atol
        )
    finally:
        model.config._attn_implementation = "eager"
        ALL_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
        ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
