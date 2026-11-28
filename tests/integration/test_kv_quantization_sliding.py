"""Long prefill capture and F02 integration with a real sliding Mistral model."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch
from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sparse_attention_hub.adapters import ModelAdapterHF, Request
from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    DefaultCachePolicyConfig,
    FakeNVFP4QuantizerConfig,
    FP8QuantizerConfig,
    IntQuantizerConfig,
    KIVIConfig,
    KVQuantizationConfig,
    MXFPQuantizerConfig,
    NonUniformQuantizerConfig,
    RegionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig,
)
from tests.integration.test_kv_quantization_flow import CONTEXT, QUESTION
from tests.integration.test_kv_quantization_flow import (
    tiny_model_dir as shared_tiny_model_dir,
)

tiny_model_dir = shared_tiny_model_dir
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sliding_model_dir(tmp_path_factory, tiny_model_dir):
    path = tmp_path_factory.mktemp("sliding_mistral")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tiny_model_dir)
    tokenizer.save_pretrained(path)
    torch.manual_seed(37)
    config = MistralConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        sliding_window=4,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
    )
    MistralForCausalLM(config).save_pretrained(path)
    return str(path)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("store_codes", [False, True])
def test_adapter_captures_all_prefill_and_keeps_history_after_native_eviction(
    sliding_model_dir, implementation, store_codes, monkeypatch
):
    adapter = ModelAdapterHF(
        model_name=sliding_model_dir,
        device="cpu",
        model_kwargs={"attn_implementation": implementation},
        sparse_attention_config=ResearchAttentionConfig(
            masker_configs=[],
            kv_quantization_config=KVQuantizationConfig(
                key_quantizer=IntQuantizerConfig(bits=2),
                value_quantizer=IntQuantizerConfig(bits=2),
                cache_policy=DefaultCachePolicyConfig(
                    key_local=RegionConfig(2),
                    value_local=RegionConfig(2),
                    key_block_tokens=2,
                    value_block_tokens=2,
                ),
                store_codes=store_codes,
            ),
        ),
    )
    ids = adapter.tokenizer.encode(CONTEXT, return_tensors="pt")
    with torch.no_grad():
        expected_prefill = adapter.model(ids, use_cache=True).logits
    original_forward = adapter.model.forward
    observed = []
    retained = []
    original_clear = adapter.sparse_attention.clear_request_state

    def forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        metadata = kwargs.get("sparse_meta_data", {})
        if CACHE_KEY in metadata:
            store = metadata[CACHE_KEY]
            seen = {layer.seen_tokens for layer in store.layers.values()}
            assert seen == {output.past_key_values.get_seq_length()}
            assert all(
                layer.keys.shape[-2] == 3 for layer in output.past_key_values.layers
            )
            observed.append((metadata.get("kv_quantization_phase"), next(iter(seen))))
            if metadata.get("kv_quantization_phase") == "prefill":
                assert (
                    ALL_MASK_ATTENTION_FUNCTIONS[adapter._registered_attention_name]
                    is ALL_MASK_ATTENTION_FUNCTIONS[implementation]
                )
                torch.testing.assert_close(
                    output.logits, expected_prefill, rtol=1e-6, atol=1e-7
                )
        return output

    def clear(metadata):
        retained.append((metadata, metadata[CACHE_KEY]))
        original_clear(metadata)

    monkeypatch.setattr(adapter.model, "forward", forward)
    monkeypatch.setattr(adapter.sparse_attention, "clear_request_state", clear)
    try:
        adapter.process_request(
            Request(context=CONTEXT, questions=[QUESTION, QUESTION], answer_prefix=""),
            generation_kwargs={"max_new_tokens": 4},
            request_kwargs={},
        )
        starts = [count for phase, count in observed if phase == "prefill"]
        assert starts == [ids.shape[-1], ids.shape[-1]]
        assert starts[0] > 4
        assert any(count > starts[0] for phase, count in observed if phase == "decode")
        assert len(retained) == 2 and retained[0][1] is not retained[1][1]
        assert all(
            CACHE_KEY not in meta and not store.layers for meta, store in retained
        )
        assert adapter.model.config._attn_implementation == implementation
        assert (
            ALL_MASK_ATTENTION_FUNCTIONS[adapter._registered_attention_name]
            is eager_mask
        )
    finally:
        adapter._cleanup_attention_registration()


@pytest.mark.parametrize("store_codes", [False, True])
def test_real_hf_batched_question_is_causal_at_kivi_flush_boundary(store_codes):
    torch.manual_seed(43)
    model = MistralForCausalLM(
        MistralConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            sliding_window=None,
        )
    ).eval()
    model.config._attn_implementation = "eager"
    research = ResearchAttention.create_from_config(
        ResearchAttentionConfig(
            masker_configs=[],
            kv_quantization_config=KIVIConfig.with_settings(
                group_size=4,
                residual_length=8,
                store_codes=store_codes,
            ),
        )
    )
    callback = ModelAdapterHF.get_custom_attention_function(SimpleNamespace(), research)
    name = f"causal_flush_{uuid4().hex}"
    ALL_ATTENTION_FUNCTIONS.register(name, callback)
    ALL_MASK_ATTENTION_FUNCTIONS.register(name, eager_mask)
    results = []
    try:
        with torch.no_grad():
            for last_token in (8, 29):
                model.config._attn_implementation = "eager"
                cache = model(
                    torch.tensor([[1, 2, 3, 4, 5, 6]]), use_cache=True
                ).past_key_values
                model.config._attn_implementation = name
                metadata = {}
                outputs = model(
                    torch.tensor([[7, last_token]]),
                    past_key_values=cache,
                    use_cache=True,
                    sparse_meta_data=metadata,
                )
                results.append(outputs.logits[:, 0, :])
                assert metadata[CACHE_KEY].layers[0].key.recent_tokens == 0
                research.clear_request_state(metadata)
        assert torch.equal(results[0], results[1])
    finally:
        model.config._attn_implementation = "eager"
        ALL_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
        ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)


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
def test_sliding_hf_teacher_forced_logits_match_fake_and_packed(codec):
    results = {}
    for packed in (False, True):
        torch.manual_seed(47)
        model = MistralForCausalLM(
            MistralConfig(
                vocab_size=32,
                hidden_size=64,
                intermediate_size=96,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                sliding_window=4,
            )
        ).eval()
        research = ResearchAttention.create_from_config(
            ResearchAttentionConfig(
                masker_configs=[],
                kv_quantization_config=KVQuantizationConfig(
                    key_quantizer=codec,
                    value_quantizer=codec,
                    cache_policy=DefaultCachePolicyConfig(
                        key_local=RegionConfig(0),
                        value_local=RegionConfig(0),
                        key_block_tokens=2,
                        value_block_tokens=2,
                    ),
                    store_codes=packed,
                ),
            )
        )
        name = f"sliding_parity_{uuid4().hex}"
        ALL_ATTENTION_FUNCTIONS.register(
            name,
            ModelAdapterHF.get_custom_attention_function(SimpleNamespace(), research),
        )
        ALL_MASK_ATTENTION_FUNCTIONS.register(name, eager_mask)
        model.config._attn_implementation = name
        metadata = {"kv_quantization_phase": "prefill"}
        try:
            with torch.no_grad():
                output = model(
                    torch.tensor([list(range(1, 13))]),
                    use_cache=True,
                    sparse_meta_data=metadata,
                )
                cache = output.past_key_values
                metadata["kv_quantization_phase"] = "decode"
                logits = []
                for tokens in ([13, 14], [15], [16, 17, 18]):
                    output = model(
                        torch.tensor([tokens]),
                        use_cache=True,
                        past_key_values=cache,
                        sparse_meta_data=metadata,
                    )
                    cache = output.past_key_values
                    logits.append(output.logits)
            results[packed] = torch.cat(logits, dim=1)
            assert cache.layers[0].keys.shape[-2] == 3
            assert metadata[CACHE_KEY].seen_tokens(0) == 18
        finally:
            research.clear_request_state(metadata)
            model.config._attn_implementation = "eager"
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
    assert torch.equal(results[False], results[True])


def test_sliding_mask_preserves_left_padding_for_two_sequences():
    results = []
    for quantized in (False, True):
        torch.manual_seed(53)
        model = MistralForCausalLM(
            MistralConfig(
                vocab_size=32,
                hidden_size=64,
                intermediate_size=96,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                sliding_window=4,
            )
        ).eval()
        model.config._attn_implementation = "eager"
        # All these short sequences remain full precision in our local window,
        # isolating mask/position handling from quantization error.
        research = ResearchAttention.create_from_config(
            ResearchAttentionConfig(
                masker_configs=[], kv_quantization_config=KVQuantizationConfig()
            )
        )
        name = f"sliding_padding_{uuid4().hex}"
        ALL_ATTENTION_FUNCTIONS.register(
            name,
            ModelAdapterHF.get_custom_attention_function(SimpleNamespace(), research),
        )
        ALL_MASK_ATTENTION_FUNCTIONS.register(name, eager_mask)
        metadata = {"kv_quantization_phase": "prefill"}
        inputs = torch.tensor([[0, 0, 0, 0, 1, 2], [0, 0, 3, 4, 5, 6]])
        padding = inputs != 0
        try:
            if quantized:
                model.config._attn_implementation = name
            with torch.no_grad():
                output = model(
                    inputs,
                    attention_mask=padding,
                    use_cache=True,
                    sparse_meta_data=metadata,
                )
                metadata["kv_quantization_phase"] = "decode"
                output = model(
                    torch.tensor([[10, 11], [12, 13]]),
                    attention_mask=torch.cat(
                        [padding, torch.ones(2, 2, dtype=torch.bool)], -1
                    ),
                    past_key_values=output.past_key_values,
                    use_cache=True,
                    sparse_meta_data=metadata,
                )
            results.append(output.logits)
            if quantized:
                assert metadata[CACHE_KEY].seen_tokens(0) == 8
        finally:
            research.clear_request_state(metadata)
            model.config._attn_implementation = "eager"
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
    torch.testing.assert_close(results[0], results[1], rtol=1e-5, atol=1e-6)
