"""Real HF attention callbacks checked against independent QuaRot reconstruction."""

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
    QuaRotKVConfig,
    quarot_int_quantizer,
)
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig,
)
from tests.reference.quarot import normalized_matrix, reconstruct

pytestmark = pytest.mark.integration


def config_for_dimension(dimension, *, packed=True, quantize_prefill=True, bits=4):
    return QuaRotKVConfig(
        key_quantizer=quarot_int_quantizer("key_head", group_size=dimension, bits=bits),
        value_quantizer=quarot_int_quantizer(
            "value_group", group_size=dimension, bits=bits
        ),
        store_codes=packed,
        quantize_prefill=quantize_prefill,
    )


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
@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("chunked", [False, True])
def test_real_hf_prefill_decode_and_gqa_match_reference(
    config_class, model_class, window, dtype, bits, chunked
):
    torch.manual_seed(509)
    dimension = 32  # Explicit small-head flow profile, not the paper's G128 run.
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
        research = research_for(
            config_for_dimension(dimension, packed=packed, bits=bits)
        )
        method = research.kv_quantization
        original_process = method.process
        expected_by_layer, history, phases = {}, {}, []

        def check_process(layer, keys, values, metadata, **kwargs):
            count = kwargs["num_new_tokens"]
            incoming = (keys[..., -count:, :], values[..., -count:, :])
            restored = tuple(
                reconstruct(x, mode=mode, group_size=dimension, bits=bits)
                for x, mode in zip(incoming, ("key_head", "value_group"))
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
                    2
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
            expected_by_layer[layer] = expected
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

        name = "quarot_flow_" + uuid4().hex
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
def test_paper_head_dimension_128_prefill_reaches_native_attention(
    native, quantize_prefill
):
    torch.manual_seed(512)
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=48,
                hidden_size=256,
                intermediate_size=320,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=128,
                max_position_embeddings=32,
            )
        )
        .half()
        .eval()
    )
    research = research_for(QuaRotKVConfig(quantize_prefill=quantize_prefill))
    callback = ModelAdapterHF.get_custom_attention_function(SimpleNamespace(), research)
    name = "quarot_paper_" + uuid4().hex
    captured = []

    def inspect(
        module, queries, keys, values, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        original = keys.clone()
        out = callback(
            module, queries, keys, values, attention_mask, scaling, dropout, **kwargs
        )
        expected_keys, expected_values = keys, values
        if quantize_prefill:
            expected_keys = reconstruct(keys, mode="key_head", group_size=128)
            expected_values = reconstruct(values, mode="value_group", group_size=128)
        if native == "sdpa":
            # Verify routing against the selected external backend itself, rather
            # than imposing the eager or FP32-softmax kernel's rounding on SDPA.
            copies = queries.shape[1] // expected_keys.shape[1]
            expected_output = (
                torch.nn.functional.scaled_dot_product_attention(
                    queries,
                    expected_keys.repeat_interleave(copies, dim=1),
                    expected_values.repeat_interleave(copies, dim=1),
                    attn_mask=attention_mask,
                    scale=scaling,
                    is_causal=attention_mask is None and queries.shape[-2] > 1,
                )
                .transpose(1, 2)
                .contiguous()
            )
        else:
            expected_output = reference_attention(
                queries,
                expected_keys,
                expected_values,
                attention_mask,
                scaling,
                native_eager=True,
            )
        torch.testing.assert_close(
            out[0], expected_output, rtol=2 * torch.finfo(torch.float16).eps, atol=5e-5
        )
        captured.append(
            (
                original,
                kwargs["sparse_meta_data"][CACHE_KEY].get(
                    0, research.kv_quantization._decode_chunk
                )[0],
            )
        )
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
        assert torch.isfinite(output.logits).all() and len(captured) == 1
        assert not torch.equal(*captured[0])
    finally:
        research.clear_request_state(metadata)
        ALL_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
        ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)


@pytest.mark.parametrize("quantize_prefill", [False, True])
def test_equal_length_prefill_keeps_native_padding_mask_format(quantize_prefill):
    research = research_for(config_for_dimension(16, quantize_prefill=quantize_prefill))
    callback = ModelAdapterHF.get_custom_attention_function(SimpleNamespace(), research)
    module = SimpleNamespace(layer_idx=0)
    keys, values = torch.randn(1, 1, 3, 16), torch.randn(1, 1, 3, 16)
    queries = torch.randn(1, 2, 3, 16)
    mask = torch.tensor(
        [[1, 1, 0]]
    )  # Native 2D padding format, no CUDA backend needed.
    observed = []

    def native(module, q, k, v, attention_mask, **kwargs):
        assert q is queries and attention_mask is mask
        assert torch.equal(k, keys) is (not quantize_prefill)
        observed.append(True)
        return q.transpose(1, 2), None

    name = "quarot_mask_" + uuid4().hex
    ALL_ATTENTION_FUNCTIONS.register(name, native)
    metadata = {
        "kv_quantization_phase": "prefill",
        "kv_quantization_capture_only": {id(module): name},
    }
    try:
        callback(module, queries, keys, values, mask, 1.0, sparse_meta_data=metadata)
        assert observed == [True]
    finally:
        research.clear_request_state(metadata)
        ALL_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)


@pytest.mark.parametrize("supported", [False, True])
def test_longer_reconstructed_prefill_history_aligns_or_rejects_backend(
    monkeypatch, supported
):
    research = research_for(config_for_dimension(16))
    callback = ModelAdapterHF.get_custom_attention_function(SimpleNamespace(), research)
    module = SimpleNamespace(layer_idx=0)
    keys, values = torch.randn(1, 1, 4, 16), torch.randn(1, 1, 4, 16)
    metadata = {"kv_quantization_phase": "prefill"}
    research.kv_quantization.process(
        0,
        keys[..., :3, :],
        values[..., :3, :],
        metadata,
        num_new_tokens=3,
        phase="prefill",
    )
    name = "sdpa" if supported else "unsupported_quarot_backend"
    metadata["kv_quantization_capture_only"] = {id(module): name}
    observed = []

    def native(module, q, k, v, attention_mask, **kwargs):
        assert k.shape[-2] == 4
        assert torch.isneginf(attention_mask[..., :2]).all()
        assert not torch.isneginf(attention_mask[..., 2:]).any()
        observed.append(True)
        return q.transpose(1, 2), None

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, name, native)
    try:
        args = (
            module,
            torch.randn(1, 2, 1, 16),
            keys[..., -2:, :],
            values[..., -2:, :],
            torch.zeros(1, 1, 1, 2),
            1.0,
        )
        if supported:
            callback(*args, sparse_meta_data=metadata)
            assert observed == [True]
        else:
            with pytest.raises(NotImplementedError, match="eager or sdpa"):
                callback(*args, sparse_meta_data=metadata)
    finally:
        research.clear_request_state(metadata)


def test_inverse_rotation_attention_identity_in_float64():
    generator = torch.Generator().manual_seed(335)
    matrix = normalized_matrix(8).sign().double() / 8**0.5
    q, kr, vr = (
        torch.randn(2, 5, 8, generator=generator, dtype=torch.float64) for _ in range(3)
    )
    scores_original = q @ (kr @ matrix.T).transpose(-1, -2)
    scores_rotated = (q @ matrix) @ kr.transpose(-1, -2)
    torch.testing.assert_close(scores_original, scores_rotated, rtol=1e-12, atol=1e-12)
    a = torch.softmax(scores_original, dim=-1)
    torch.testing.assert_close(
        a @ (vr @ matrix.T), (a @ vr) @ matrix.T, rtol=1e-12, atol=1e-12
    )
