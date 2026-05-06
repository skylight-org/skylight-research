#!/usr/bin/env python3
"""Run AIME benchmarks (dense or sparse) via the local Hugging Face adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow `python scripts/run_aime.py` from repo root without PYTHONPATH tweaks.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark import create_benchmark_instance  # noqa: E402
from benchmark.aime_common import MAX_NEW_TOKENS_CAP  # noqa: E402
from sparse_attention_hub.adapters.huggingface import ModelAdapterHF  # noqa: E402
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger  # noqa: E402
from sparse_attention_hub.sparse_attention.research_attention import (  # noqa: E402
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (  # noqa: E402
    LocalMaskerConfig,
    OracleTopKConfig,
    SinkMaskerConfig,
)


def build_sparse_config(name: str, heavy_size: float = 0.2) -> ResearchAttentionConfig | None:
    if name == "dense":
        return None

    if name == "streaming":
        return ResearchAttentionConfig(
            masker_configs=[
                SinkMaskerConfig(sink_size=128),
                LocalMaskerConfig(window_size=128),
            ]
        )

    if name == "oracle_topk":
        return ResearchAttentionConfig(
            masker_configs=[
                SinkMaskerConfig(sink_size=128),
                LocalMaskerConfig(window_size=128),
                OracleTopKConfig(heavy_size=heavy_size, search_space={}),
            ]
        )

    raise ValueError(f"Unknown sparse mode: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AIME benchmark in sparse-attention-hub")
    parser.add_argument(
        "--dataset",
        choices=["aime2024", "aime2025", "aime2026"],
        default="aime2025",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--mode", choices=["dense", "streaming", "oracle_topk"], default="dense")
    parser.add_argument("--heavy-size", type=float, default=0.2, help="Heavy tokens size ratio for oracle_topk")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS_CAP,
        help=f"Generation cap (default: {MAX_NEW_TOKENS_CAP}, aligned with AIME row cap).",
    )
    parser.add_argument("--max-context-length", type=int, default=32768)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--result-dir", default="results/aime")
    parser.add_argument(
        "--micro-metrics-dir",
        default=None,
        help=(
            "Directory for micro-metric logs. Defaults to "
            "<result-dir>/<dataset>/<mode>/micro_metrics for aime2026."
        ),
    )
    args = parser.parse_args()

    sparse_config = build_sparse_config(args.mode, args.heavy_size)

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if "cuda" in args.device else torch.float32,
    }

    adapter = ModelAdapterHF(
        model_name=args.model,
        sparse_attention_config=sparse_config,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
        device=args.device,
        hybrid=("qwen3" in args.model.lower()),
    )

    benchmark = create_benchmark_instance(args.dataset, [args.dataset])

    request_kwargs: dict = {
        "max_context_length": args.max_context_length,
    }
    if args.max_requests is not None:
        request_kwargs["max_requests"] = args.max_requests

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    }

    out_dir = Path(args.result_dir) / args.dataset / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # For AIME2026, enable micro-metric logging by default. It is mainly useful for
    # sparse runs (e.g. research_attention_density/output_error), but we still create
    # the log sink for consistency across 2026 runs.
    micro_dir: Path | None = None
    if args.dataset == "aime2026" or args.micro_metrics_dir is not None:
        micro_dir = (
            Path(args.micro_metrics_dir)
            if args.micro_metrics_dir is not None
            else out_dir / "micro_metrics"
        )
        micro_dir.mkdir(parents=True, exist_ok=True)
        metric_logger: MicroMetricLogger = MicroMetricLogger()
        metric_logger.configure_logging(
            log_path=str(micro_dir),
            enabled_metrics="all",
        )

    metrics = benchmark.run_benchmark(
        adapter=adapter,
        result_dir=str(out_dir),
        generation_kwargs=generation_kwargs,
        request_kwargs=request_kwargs,
    )

    if micro_dir is not None:
        # Flush pending queue and ensure the canonical output file exists.
        metric_logger = MicroMetricLogger()
        metric_logger.flush()
        (micro_dir / "micro_metrics.jsonl").touch(exist_ok=True)

    print("\n=== Final Metrics ===")
    print(metrics)
    print(f"\nSaved results to: {out_dir}")
    if micro_dir is not None:
        print(f"Saved micro metrics to: {micro_dir / 'micro_metrics.jsonl'}")


if __name__ == "__main__":
    main()
