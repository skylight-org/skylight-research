"""End-to-end check of the quantize_kv_cache flag against a real model.

Runs one request through ModelAdapterHF twice, with quantization off and on, and
prints both answers plus peak memory. Needs a downloadable model, so it is a
script rather than a collected test. ``nvfp4`` additionally needs a CUDA GPU and
nvidia-modelopt; ``fake_nvfp4`` runs anywhere, including CPU, and shows the
accuracy cost without the memory saving.

Example:
    ::

        conda activate sparse_attention_hub

        # accuracy only, runs anywhere
        python -m sparse_attention_hub.kv_quantization.WIP \\
            --backend fake_nvfp4 --model Qwen/Qwen3-4B

        # the real thing, per-channel keys
        python -m sparse_attention_hub.kv_quantization.WIP \\
            --backend nvfp4 --axis-key 0
"""

import argparse
from typing import Any, Dict, Optional, Tuple, Union

import torch

from sparse_attention_hub.adapters import ModelAdapterHF, Request
from sparse_attention_hub.kv_quantization import (
    BACKENDS,
    FAKE_NVFP4,
    NVFP4,
    describe_backends,
)
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
)

CONTEXT: str = (
    "The Amazon rainforest produces roughly 20 percent of the world's oxygen. "
    "It spans nine countries, with about 60 percent of it inside Brazil. "
    "The Amazon river discharges more water than any other river on Earth. "
) * 40
QUESTION: str = "Which country contains most of the Amazon rainforest?"


def _run(
    model_name: str,
    quantize: Union[bool, str],
    sparse: bool,
    device: str,
    kv_quantization_kwargs: Dict[str, Any],
    max_new_tokens: int,
) -> Tuple[str, float]:
    """Process one request and return the answer and peak GPU memory in GB.

    Peak memory is reported as 0 on CPU, where torch does not track it.
    """
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sparse_config: Optional[ResearchAttentionConfig] = None
    if sparse:
        sparse_config = ResearchAttentionConfig(
            masker_configs=[
                SinkMaskerConfig(sink_size=128),
                LocalMaskerConfig(window_size=256),
            ]
        )

    adapter: ModelAdapterHF = ModelAdapterHF(
        model_name=model_name,
        sparse_attention_config=sparse_config,
        model_kwargs={"dtype": torch.bfloat16},
        device=device,
        quantize_kv_cache=quantize,
        kv_quantization_kwargs=kv_quantization_kwargs,
    )
    response = adapter.process_request(
        request=Request(context=CONTEXT, questions=QUESTION, answer_prefix="Answer: "),
        generation_kwargs={"max_new_tokens": max_new_tokens},
        request_kwargs={"max_context_length": 8192},
    )

    peak_gb: float = 0.0
    if device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
    del adapter
    if device == "cuda":
        torch.cuda.empty_cache()
    return str(response.responses).strip(), peak_gb


def main() -> None:
    """Parse arguments and print the baseline/quantized comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--backend",
        default=NVFP4,
        choices=list(BACKENDS),
        help="; ".join(f"{name}: {text}" for name, text in describe_backends().items()),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--axis-key",
        type=int,
        default=-1,
        choices=[-1, 0],
        help="-1 blocks keys per token, 0 per channel (KIVI-style)",
    )
    parser.add_argument("--axis-value", type=int, default=-1, choices=[-1, 0])
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="also enable a sink+local sparse attention config",
    )
    args = parser.parse_args()

    cuda_available: bool = torch.cuda.is_available()
    if args.backend == NVFP4 and not cuda_available:
        raise SystemExit(
            "The nvfp4 backend needs a GPU: modelopt's NVFP4 packing is "
            "CUDA-only. Pass --backend fake_nvfp4 to measure accuracy on CPU."
        )
    device: str = "cuda" if cuda_available else "cpu"

    kv_quantization_kwargs: Dict[str, Any] = {
        "axis_key": args.axis_key,
        "axis_value": args.axis_value,
        "residual_length": args.residual_length,
    }

    baseline_answer, baseline_gb = _run(
        args.model,
        False,
        args.sparse,
        device,
        kv_quantization_kwargs,
        args.max_new_tokens,
    )
    quantized_answer, quantized_gb = _run(
        args.model,
        args.backend,
        args.sparse,
        device,
        kv_quantization_kwargs,
        args.max_new_tokens,
    )

    print("\n=== quantize_kv_cache=False ===")
    print(f"answer     : {baseline_answer}")
    print(f"peak memory: {baseline_gb:.2f} GB")
    print(f"\n=== quantize_kv_cache={args.backend!r} ===")
    print(f"answer     : {quantized_answer}")
    print(f"peak memory: {quantized_gb:.2f} GB")
    if args.backend == FAKE_NVFP4:
        print("(simulated NVFP4 stores nothing packed, so expect no memory saving)")
    print(
        f"\nkey axis {args.axis_key}, value axis {args.axis_value}, "
        f"residual {args.residual_length}, device {device}"
    )
    print(f"answers match: {baseline_answer == quantized_answer}")


if __name__ == "__main__":
    main()
