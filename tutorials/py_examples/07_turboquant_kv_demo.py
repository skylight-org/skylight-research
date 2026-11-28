"""Offline TurboQuant KV flow on a tiny random Llama; no downloads or datasets.

Run from the repository root:
    python tutorials/py_examples/07_turboquant_kv_demo.py --device cpu

Uses identical forced tokens and weights for native, FAKE and PACKED runs.
This checks the cache/attention flow, not pretrained language-model quality.
"""

import argparse
from types import SimpleNamespace
from uuid import uuid4

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sparse_attention_hub.adapters.huggingface import ModelAdapterHF
from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    TurboQuantKVConfig,
    TurboQuantQuantizerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttention,
    ResearchAttentionConfig,
)


def run_case(model, *, packed, key_bits, value_bits, key_variant):
    """Register the normal adapter callback, run prefill/decode and release it."""
    research = None
    metadata = {}
    name = None
    original_backend = model.config._attn_implementation
    label = "native" if packed is None else "PACKED" if packed else "FAKE"
    if packed is not None:
        research = ResearchAttention.create_from_config(
            ResearchAttentionConfig(
                masker_configs=[],
                kv_quantization_config=TurboQuantKVConfig(
                    key_quantizer=TurboQuantQuantizerConfig(
                        bits=key_bits, variant=key_variant
                    ),
                    value_quantizer=TurboQuantQuantizerConfig(bits=value_bits),
                    store_codes=packed,
                ),
            )
        )
        callback = ModelAdapterHF.get_custom_attention_function(
            SimpleNamespace(), research
        )
        name = "turboquant_demo_" + uuid4().hex
        ALL_ATTENTION_FUNCTIONS.register(name, callback)
        ALL_MASK_ATTENTION_FUNCTIONS.register(name, eager_mask)
        metadata = {
            "kv_quantization_phase": "prefill",
            "kv_quantization_capture_only": {
                id(module): original_backend
                for module in model.modules()
                if hasattr(module, "layer_idx")
            },
        }
        model.config._attn_implementation = name
    logits = []
    try:
        with torch.no_grad():
            cache = None
            for index, tokens in enumerate(([1, 2, 3, 4, 5, 6, 7, 8], [9], [10])):
                if index:
                    metadata.pop("kv_quantization_capture_only", None)
                    metadata["kv_quantization_phase"] = "decode"
                forwarding = {"sparse_meta_data": metadata} if research else {}
                result = model(
                    torch.tensor([tokens], device=model.device),
                    past_key_values=cache,
                    use_cache=True,
                    **forwarding,
                )
                cache = result.past_key_values
                logits.append(result.logits.detach().clone())
                phase = "prefill" if index == 0 else "decode"
                if research:
                    stats = metadata[CACHE_KEY].stats()
                    first_layer = stats["layers"][0]
                    print(
                        f"{label:6} {phase:7} tokens={first_layer['seen_tokens']:2} "
                        f"owned_store_bytes={stats['store_bytes']} "
                        f"shared_codec_bytes={sum(codec.shared_state_bytes() for codec in (research.kv_quantization.key_quantizer, research.kv_quantization.value_quantizer))}"
                    )
                    if packed:
                        totals = dict(
                            indices=0, original_norms=0, signs=0, residual_norms=0
                        )
                        for layer in metadata[CACHE_KEY].layers.values():
                            for state in (layer.key, layer.value):
                                for chunk in state.all_chunks():
                                    record = chunk.quantized
                                    for label_name, tensor in (
                                        ("indices", record.codes),
                                        ("original_norms", record.norms),
                                        ("signs", record.qjl_codes),
                                        ("residual_norms", record.residual_norms),
                                    ):
                                        if tensor is not None:
                                            totals[label_name] += (
                                                tensor.numel() * tensor.element_size()
                                            )
                        print(f"       packed payload bytes: {totals}")
                else:
                    print(
                        f"{label:6} {phase:7} native_cache_tokens={cache.get_seq_length()}"
                    )
        return torch.cat(logits, dim=1)
    finally:
        if research is not None:
            research.clear_request_state(metadata)
            assert CACHE_KEY not in metadata
        model.config._attn_implementation = original_backend
        if name is not None:
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--key-bits", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--value-bits", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--key-variant", choices=("mse", "prod"), default="mse")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    print(
        f"TurboQuant keys={args.key_variant}/{args.key_bits} nominal bits, "
        f"values=mse/{args.value_bits} nominal bits, distribution=sphere_beta"
    )
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=256,
        intermediate_size=384,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = (
        LlamaForCausalLM(config)
        .to(device=args.device, dtype=getattr(torch, args.dtype))
        .eval()
    )
    results = [
        run_case(
            model,
            packed=mode,
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            key_variant=args.key_variant,
        )
        for mode in (None, False, True)
    ]
    assert all(torch.isfinite(logits).all() for logits in results)
    assert torch.equal(results[1], results[2]), "FAKE and PACKED logits differ"
    print("FAKE/PACKED logits: identical")
    print(
        f"Max logit difference from native: {(results[0]-results[2]).abs().max().item():.6g}"
    )
    print(
        "Store bytes include packed indices and FP32 norms (plus QJL signs/residual norms in prod). "
        "Shared codec bytes count owned rotation/projection/table tensors on CPU and device. "
        "Both exclude Python metadata, the native HF cache, model weights and dense temporaries."
    )


if __name__ == "__main__":
    main()
