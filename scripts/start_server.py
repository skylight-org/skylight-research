#!/usr/bin/env python3

import argparse
import sys
import types
from pathlib import Path

import torch
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sparse_attention_hub.adapters import ModelAdapterHF  # noqa: E402
from sparse_attention_hub.sparse_attention.research_attention import (  # noqa: E402
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (  # noqa: E402
    LocalMaskerConfig,
    PQCacheConfig,
    SinkMaskerConfig,
)

app = Flask(__name__)
adapter = None

CONFIGS = {
    "dense": None,
    "pqcache": ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),
            LocalMaskerConfig(window_size=128),
            PQCacheConfig(
                heavy_size=0.1,
                pq_group_factor=2,
                pq_bits=6,
                kmeans_iter=10,
                init_offset=128,
                metric="euclidean",
            ),
        ]
    ),
}


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    global adapter
    data = request.json or {}
    messages = data.get("messages", [])

    prompt = adapter.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = adapter.tokenizer(prompt, return_tensors="pt").to(adapter.device)

    with torch.no_grad():
        context_mgr = (
            adapter.enable_sparse_mode()
            if adapter._sparse_attention_available
            else torch.no_grad()
        )
        with context_mgr:
            outputs = adapter.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=data.get("max_tokens", 1024),
                do_sample=data.get("temperature", 0.0) > 0,
                temperature=data.get("temperature", 0.0),
                pad_token_id=adapter.tokenizer.pad_token_id,
                eos_token_id=adapter.tokenizer.eos_token_id,
                **(
                    {"sparse_meta_data": {}}
                    if adapter._sparse_attention_available
                    else {}
                ),
            )

    text = adapter.tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()

    return jsonify(
        {
            "id": "chatcmpl-sparse",
            "object": "chat.completion",
            "model": data.get("model", "sparse-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
    )


def main():
    global adapter
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("port", type=int, default=4000)
    parser.add_argument(
        "--config", type=str, default="dense", choices=list(CONFIGS.keys())
    )
    args = parser.parse_args()

    adapter = ModelAdapterHF(
        model_name=args.model,
        sparse_attention_config=CONFIGS[args.config],
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": "cuda:0",
            "attn_implementation": "sdpa",
        },
    )

    model_dev = next(adapter.model.parameters()).device
    adapter.device = model_dev

    # Patch _validate_model_kwargs to an empty function
    def _empty_validate_model_kwargs(self, model_kwargs: dict) -> None:
        """Empty function to bypass model kwargs validation."""
        pass

    adapter.model._validate_model_kwargs = types.MethodType(
        _empty_validate_model_kwargs, adapter.model
    )

    _orig_forward = adapter.model.forward

    def _wrapped_forward(*args, **kwargs):
        new_args = tuple(
            a.to(model_dev) if isinstance(a, torch.Tensor) else a for a in args
        )
        new_kwargs = {
            k: v.to(model_dev) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return _orig_forward(*new_args, **new_kwargs)

    adapter.model.forward = _wrapped_forward

    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
