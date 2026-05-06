"""AIME2025 benchmark implementation for mathematical reasoning evaluation."""

from typing import Dict, Any, List
import pandas as pd

from ..aime_common import MAX_NEW_TOKENS_CAP
from ..base import Benchmark
from ..benchmark_registry import register_benchmark
from .calculate_metrics import calculate_metrics


@register_benchmark("aime2025")
class AIME2025(Benchmark):
    """AIME2025 benchmark for evaluating mathematical reasoning.

    AIME2025 is a benchmark for evaluating the ability of large language models to solve
    mathematical competition problems. It contains problems from the American Invitational
    Mathematics Examination (AIME) 2025.

    The benchmark evaluates mathematical reasoning capabilities:
    - Problem solving with numerical answers (0-999)
    - Step-by-step mathematical reasoning
    - Answer extraction from \boxed{...} format

    Example:
        >>> aime2025 = AIME2025()
        >>> results = aime2025.run_benchmark(adapter, result_dir="/path/to/results")
        >>> print(f"Accuracy: {results}")
    """

    # AIME2025 has a single dataset
    all_datasets: List[str] = ["aime2025"]
    
    benchmark_name: str = "aime2025"
    huggingface_dataset_id: str = "xAlg-AI/att-hub-aime2025"

    def _load_datasets(self) -> pd.DataFrame:
        """Load AIME2025 dataset.
        
        AIME2025 uses a single dataset with all problems.
        
        Returns:
            pandas DataFrame with all AIME2025 problems.
        """
        print(f"Loading AIME2025 dataset")
        
        try:
            from datasets import load_dataset
            dataset = load_dataset(self.huggingface_dataset_id, split="test")
            df = dataset.to_pandas()
            df["task"] = "aime2025"  # Ensure task column exists
            # Hub parquet may ship an older per-row cap; align with MAX_NEW_TOKENS_CAP so
            # ``run_benchmark``'s min(user, row) does not silently cap long generations.
            df["max_new_tokens"] = MAX_NEW_TOKENS_CAP
            print(f"  ✓ Loaded {len(df)} AIME2025 problems")
            return df
        except Exception as e:
            raise Exception(f"Failed to load AIME2025 dataset: {str(e)}")

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics for AIME2025 results.

        Args:
            results_df: DataFrame containing benchmark results with columns:
                - predicted_answer: Model's predicted answer
                - answer: Ground truth answers

        Returns:
            Dictionary containing computed metrics:
            - exact_match: Overall accuracy
            - extraction_rate: Rate of successful answer extraction
            - boxed_format_rate: Rate of using \boxed{} format
        """
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        # Use the calculate_metrics function from HashAttention evaluation
        metrics: Dict[str, Any] = calculate_metrics(results_df)
        
        # Format the results for consistency with other benchmarks
        overall_metrics: Dict[str, Any] = {
            "overall_score": round(metrics["exact_match"], 4),
            "exact_match": round(metrics["exact_match"], 4),
            "extraction_rate": round(metrics["extraction_rate"], 4),
            "boxed_format_rate": round(metrics["boxed_format_rate"], 4),
            "total_problems": metrics["total_problems"],
            "task_scores": {
                "aime2025": {
                    "exact_match": round(metrics["exact_match"], 4),
                    "extraction_rate": round(metrics["extraction_rate"], 4),
                    "boxed_format_rate": round(metrics["boxed_format_rate"], 4)
                }
            },
            "summary": {
                "total_tasks": 1,
                "total_samples": len(results_df)
            }
        }
        
        print(f"  ✓ AIME2025 Exact Match: {metrics['exact_match']:.3f} ({metrics['exact_match']*100:.1f}%)")
        print(f"  ✓ Extraction Rate: {metrics['extraction_rate']:.3f} ({metrics['extraction_rate']*100:.1f}%)")
        print(f"  ✓ Boxed Format Rate: {metrics['boxed_format_rate']:.3f} ({metrics['boxed_format_rate']*100:.1f}%)")
        
        return overall_metrics 