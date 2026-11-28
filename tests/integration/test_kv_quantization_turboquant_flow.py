"""Real HF callbacks checked against independent TurboQuant Algorithms 1/2."""

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
    AttentionContext,
    TurboQuantKVConfig,
    TurboQuantQuantizerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig,
)
from tests.reference.turboquant import encode

pytestmark = pytest.mark.integration


def method_config(*, packed=True, variant="mse", quantize_prefill=True):
    return TurboQuantKVConfig(
        key_quantizer=TurboQuantQuantizerConfig(variant=variant),
        store_codes=packed,
        quantize_prefill=quantize_prefill,
    )


def reconstruct(codec, x, layer, kind):
    state = codec.state_for(AttentionContext(layer, kind, 0), x.shape[-1])
    result = encode(x, state, variant=codec.config.variant, bits=codec.config.bits)
    return torch.tensor(result.reconstructed, dtype=x.dtype, device=x.device)


def research_for(config):
    return ResearchAttention.create_from_config(
        ResearchAttentionConfig(masker_configs=[], kv_quantization_config=config)
    )


def reference_attention(query, keys, values, mask, scaling, *, native_eager=False):
    copies = query.shape[1] // keys.shape[1]
    k = keys.repeat_interleave(copies, dim=1)
    v = values.repeat_interleave(copies, dim=1)
    q = query if native_eager else query.float()
    if not native_eager:
        k, v = k.float(), v.float()
    scores = q @ k.transpose(-1, -2) * scaling
    if mask is not None:
        scores = scores + mask[..., : keys.shape[-2]].to(scores.dtype)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
    if native_eager:
        weights = weights.to(query.dtype)
    return (weights @ v).to(query.dtype).transpose(1, 2).contiguous()


@pytest.mark.parametrize(
    "config_class,model_class,window",
    [
        (LlamaConfig, LlamaForCausalLM, None),
        (Qwen3Config, Qwen3ForCausalLM, None),
        (MistralConfig, MistralForCausalLM, None),
        (MistralConfig, MistralForCausalLM, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("variant", ["mse", "prod"])
@pytest.mark.parametrize("chunked", [False, True])
def test_real_hf_prefill_decode_and_gqa_match_reference(
    config_class, model_class, window, dtype, variant, chunked
):
    torch.manual_seed(509)
    dimension = 32  # Small random models exercise routing, not paper benchmark quality.
    config = config_class(
        vocab_size=96,
        hidden_size=128,
        intermediate_size=160,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=dimension,
        sliding_window=window,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = model_class(config).to(dtype=dtype).eval()
    logits_by_mode = []
    for packed in (False, True):
        research = research_for(method_config(packed=packed, variant=variant))
        method = research.kv_quantization
        original_process = method.process
        expected_by_layer, history, phases = {}, {}, []

        def check_process(layer, keys, values, metadata, **kwargs):
            count = kwargs["num_new_tokens"]
            incoming = (keys[..., -count:, :], values[..., -count:, :])
            restored = tuple(
                reconstruct(codec, x, layer, kind)
                for x, codec, kind in zip(
                    incoming,
                    (method.key_quantizer, method.value_quantizer),
                    ("key", "value"),
                )
            )
            previous = history.get(layer, tuple(x[..., :0, :] for x in restored))
            expected = tuple(
                torch.cat([a, b], dim=-2) for a, b in zip(previous, restored)
            )
            history[layer] = expected
            actual = original_process(layer, keys, values, metadata, **kwargs)
            for a, b in zip(actual, expected):
                # Codec codes/parameters are checked exactly in the unit oracle.
                # These tensors additionally cross two FP32 matrix products;
                # native strides can change accumulation order near cancellation.
                roundoff = (
                    4
                    * dimension
                    * torch.finfo(torch.float32).eps
                    * max(1.0, b.abs().max().item())
                )
                torch.testing.assert_close(
                    a, b, rtol=2 * torch.finfo(dtype).eps, atol=roundoff
                )
            assert not torch.equal(actual[0][..., -count:, :], incoming[0])
            state = metadata[CACHE_KEY].layers[layer]
            assert state.key.recent_tokens == state.value.recent_tokens == 0
            assert state.seen_tokens == expected[0].shape[-2]
            expected_by_layer[layer] = actual
            phases.append(kwargs["phase"])
            return actual

        method.process = check_process
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
            output = callback(
                module,
                queries,
                keys,
                values,
                attention_mask,
                scaling,
                dropout,
                **kwargs,
            )
            expected_keys, expected_values = expected_by_layer[module.layer_idx]
            reference_mask = attention_mask
            offset = expected_keys.shape[-2] - keys.shape[-2]
            if offset:
                if reference_mask is None:
                    kpos = torch.arange(expected_keys.shape[-2])
                    qpos = torch.arange(
                        expected_keys.shape[-2] - queries.shape[-2],
                        expected_keys.shape[-2],
                    )
                    visible = (kpos[None, :] >= offset) & (
                        kpos[None, :] <= qpos[:, None]
                    )
                    if window is not None:
                        visible &= kpos[None, :] > qpos[:, None] - window
                    reference_mask = torch.zeros_like(
                        visible, dtype=queries.dtype
                    ).masked_fill(~visible, float("-inf"))[None, None]
                else:
                    prefix = reference_mask.new_full(
                        (*reference_mask.shape[:-1], offset), float("-inf")
                    )
                    reference_mask = torch.cat([prefix, reference_mask], dim=-1)
            expected = reference_attention(
                queries,
                expected_keys,
                expected_values,
                reference_mask,
                scaling,
                native_eager="kv_quantization_capture_only"
                in kwargs["sparse_meta_data"],
            )
            atol = {torch.float32: 5e-7, torch.float16: 5e-5, torch.bfloat16: 5e-4}[
                dtype
            ]
            rtol = 3e-5 if dtype == torch.float32 else 2 * torch.finfo(dtype).eps
            torch.testing.assert_close(output[0], expected, rtol=rtol, atol=atol)
            return output

        name = "turboquant_flow_" + uuid4().hex
        ALL_ATTENTION_FUNCTIONS.register(name, check_callback)
        ALL_MASK_ATTENTION_FUNCTIONS.register(name, eager_mask)
        model.config._attn_implementation = name
        metadata = {
            "kv_quantization_phase": "prefill",
            "kv_quantization_capture_only": {
                id(module): "eager"
                for module in model.modules()
                if hasattr(module, "layer_idx")
            },
        }
        try:
            with torch.no_grad():
                pieces = (
                    [
                        torch.tensor([[1, 2, 3], [4, 5, 6]]),
                        torch.tensor([[7, 8, 9], [10, 11, 12]]),
                    ]
                    if chunked
                    else [torch.tensor([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]])]
                )
                cache, logits = None, []
                for piece in pieces:
                    result = model(
                        piece,
                        past_key_values=cache,
                        use_cache=True,
                        sparse_meta_data=metadata,
                    )
                    cache = result.past_key_values
                    logits.append(result.logits)
                metadata.pop("kv_quantization_capture_only")
                metadata["kv_quantization_phase"] = "decode"
                for piece in (
                    torch.tensor([[13, 14], [15, 16]]),
                    torch.tensor([[17], [18]]),
                ):
                    result = model(
                        piece,
                        past_key_values=cache,
                        use_cache=True,
                        sparse_meta_data=metadata,
                    )
                    cache = result.past_key_values
                    logits.append(result.logits)
            assert phases.count("prefill") == (4 if chunked else 2)
            assert phases.count("decode") == 4
            logits_by_mode.append(torch.cat(logits, dim=1))
            store = metadata[CACHE_KEY]
            research.clear_request_state(metadata)
            assert CACHE_KEY not in metadata and not store.layers
        finally:
            research.clear_request_state(metadata)
            model.config._attn_implementation = "eager"
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
    assert torch.equal(*logits_by_mode)


@pytest.mark.parametrize("native", ["eager", "sdpa"])
@pytest.mark.parametrize("quantize_prefill", [False, True])
@pytest.mark.parametrize("variant", ["mse", "prod"])
def test_dimension_128_prefill_and_decode_use_selected_native_backend(
    native, quantize_prefill, variant
):
    torch.manual_seed(712)
    config = LlamaConfig(
        vocab_size=48,
        hidden_size=256,
        intermediate_size=320,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).half().eval()
    research = research_for(
        method_config(variant=variant, quantize_prefill=quantize_prefill)
    )
    method = research.kv_quantization
    callback = ModelAdapterHF.get_custom_attention_function(SimpleNamespace(), research)
    name = "turboquant_native_" + uuid4().hex
    observed = []

    def inspect(module, query, keys, values, mask, scaling, dropout=0.0, **kwargs):
        out = callback(module, query, keys, values, mask, scaling, dropout, **kwargs)
        phase = kwargs["sparse_meta_data"]["kv_quantization_phase"]
        if phase == "prefill" and not quantize_prefill:
            prepared = (keys, values)
        else:
            prepared = kwargs["sparse_meta_data"][CACHE_KEY].get(
                0, method._decode_chunk
            )
        k, v = (x.repeat_interleave(2, dim=1) for x in prepared)
        if phase == "prefill" and native == "sdpa":
            target = (
                torch.nn.functional.scaled_dot_product_attention(
                    query,
                    k,
                    v,
                    attn_mask=mask,
                    scale=scaling,
                    is_causal=mask is None and query.shape[-2] > 1,
                )
                .transpose(1, 2)
                .contiguous()
            )
        else:
            target = reference_attention(
                query, *prepared, mask, scaling, native_eager=phase == "prefill"
            )
        torch.testing.assert_close(
            out[0], target, rtol=2 * torch.finfo(torch.float16).eps, atol=5e-5
        )
        observed.append(phase)
        return out

    ALL_ATTENTION_FUNCTIONS.register(name, inspect)
    ALL_MASK_ATTENTION_FUNCTIONS.register(name, eager_mask)
    model.config._attn_implementation = name
    metadata = {
        "kv_quantization_phase": "prefill",
        "kv_quantization_capture_only": {
            id(module): native
            for module in model.modules()
            if hasattr(module, "layer_idx")
        },
    }
    try:
        with torch.no_grad():
            output = model(
                torch.tensor([[1, 2, 3, 4]]), use_cache=True, sparse_meta_data=metadata
            )
            metadata.pop("kv_quantization_capture_only")
            metadata["kv_quantization_phase"] = "decode"
            output = model(
                torch.tensor([[5]]),
                past_key_values=output.past_key_values,
                use_cache=True,
                sparse_meta_data=metadata,
            )
        assert torch.isfinite(output.logits).all() and observed == ["prefill", "decode"]
    finally:
        research.clear_request_state(metadata)
        ALL_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
        ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
