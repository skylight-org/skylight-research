"""AIME2026 benchmark implementation for mathematical reasoning evaluation."""

import os
from typing import Any, Dict, List

import pandas as pd

from ..aime_common import MAX_NEW_TOKENS_CAP
from ..base import Benchmark
from ..benchmark_registry import register_benchmark
from .calculate_metrics import calculate_metrics


@register_benchmark("aime2026")
class AIME2026(Benchmark):
    """AIME2026 benchmark for evaluating mathematical reasoning."""

    all_datasets: List[str] = ["aime2026"]
    benchmark_name: str = "aime2026"
    huggingface_dataset_id: str = os.environ.get(
        "AIME2026_DATASET_ID", "MathArena/aime_2026"
    )

    def _load_datasets(self) -> pd.DataFrame:
        print("Loading AIME2026 dataset")
        try:
            from datasets import load_dataset

            # MathArena/aime_2026 currently exposes `train` only; keep a robust fallback.
            try:
                dataset = load_dataset(self.huggingface_dataset_id, split="test")
            except Exception:
                ds_dict = load_dataset(self.huggingface_dataset_id)
                split_name = "train" if "train" in ds_dict else next(iter(ds_dict.keys()))
                dataset = ds_dict[split_name]
            df = dataset.to_pandas()
            # Normalize to benchmark-required schema.
            if "context" not in df.columns:
                if "problem" not in df.columns or "answer" not in df.columns:
                    raise ValueError(
                        "AIME2026 dataset must contain either benchmark columns "
                        "(`context`, `question`, `answer_prefix`, `answer`) or raw "
                        "columns (`problem`, `answer`)."
                    )
                df["context"] = (
                    "Solve the following AIME (American Invitational Mathematics Examination) problem.\n\n"
                    + "Problem: "
                    + df["problem"].astype(str)
                    + "\n\nInstructions:\n"
                    + "- The answer should be an integer between 0 and 999\n"
                    + "- You must wrap your final answer in \\boxed{...} format"
                )
                df["question"] = "What is the answer to this problem?"
                df["answer_prefix"] = ""
                df["answer"] = df["answer"].map(lambda x: [str(x)])
            else:
                if "question" not in df.columns:
                    df["question"] = "What is the answer to this problem?"
                if "answer_prefix" not in df.columns:
                    df["answer_prefix"] = ""
                # Keep evaluator-compatible list-like references.
                df["answer"] = df["answer"].map(
                    lambda x: x if isinstance(x, (list, tuple)) else [str(x)]
                )

            df["task"] = "aime2026"
            # Keep row-level cap in sync with CLI cap behavior in Benchmark._process_all_requests.
            df["max_new_tokens"] = MAX_NEW_TOKENS_CAP
            df = df[
                [
                    "context",
                    "question",
                    "answer_prefix",
                    "answer",
                    "task",
                    "max_new_tokens",
                ]
            ]
            print(f"  ✓ Loaded {len(df)} AIME2026 problems")
            return df
        except Exception as e:
            raise Exception(
                f"Failed to load AIME2026 dataset '{self.huggingface_dataset_id}': {e}"
            )

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        metrics: Dict[str, Any] = calculate_metrics(results_df)
        overall_metrics: Dict[str, Any] = {
            "overall_score": round(metrics["exact_match"], 4),
            "exact_match": round(metrics["exact_match"], 4),
            "extraction_rate": round(metrics["extraction_rate"], 4),
            "boxed_format_rate": round(metrics["boxed_format_rate"], 4),
            "total_problems": metrics["total_problems"],
            "task_scores": {
                "aime2026": {
                    "exact_match": round(metrics["exact_match"], 4),
                    "extraction_rate": round(metrics["extraction_rate"], 4),
                    "boxed_format_rate": round(metrics["boxed_format_rate"], 4),
                }
            },
            "summary": {"total_tasks": 1, "total_samples": len(results_df)},
        }

        print(
            f"  ✓ AIME2026 Exact Match: {metrics['exact_match']:.3f} "
            f"({metrics['exact_match']*100:.1f}%)"
        )
        print(
            f"  ✓ Extraction Rate: {metrics['extraction_rate']:.3f} "
            f"({metrics['extraction_rate']*100:.1f}%)"
        )
        print(
            f"  ✓ Boxed Format Rate: {metrics['boxed_format_rate']:.3f} "
            f"({metrics['boxed_format_rate']*100:.1f}%)"
        )
        return overall_metrics

