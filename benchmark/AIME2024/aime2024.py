"""AIME2024 benchmark implementation for mathematical reasoning evaluation."""

from typing import Dict, Any, List
import pandas as pd

from ..aime_common import MAX_NEW_TOKENS_CAP
from ..base import Benchmark
from ..benchmark_registry import register_benchmark
from .calculate_metrics import calculate_metrics


@register_benchmark("aime2024")
class AIME2024(Benchmark):
    """AIME2024 benchmark for evaluating mathematical reasoning.

    AIME2024 is a benchmark for evaluating the ability of large language models to solve
    mathematical competition problems. It contains problems from the American Invitational
    Mathematics Examination (AIME) 2024.

    The benchmark evaluates mathematical reasoning capabilities:
    - Problem solving with numerical answers (0-999)
    - Step-by-step mathematical reasoning
    - Answer extraction from \boxed{...} format

    Example:
        >>> aime2024 = AIME2024()
        >>> results = aime2024.run_benchmark(adapter, result_dir="/path/to/results")
        >>> print(f"Accuracy: {results}")
    """

    # AIME2024 has a single dataset
    all_datasets: List[str] = ["aime2024"]
    
    benchmark_name: str = "aime2024"
    huggingface_dataset_id: str = "xAlg-AI/att-hub-aime2024"

    def _load_datasets(self) -> pd.DataFrame:
        """Load AIME2024 dataset.
        
        AIME2024 uses a single dataset with all problems.
        
        Returns:
            pandas DataFrame with all AIME2024 problems.
        """
        print(f"Loading AIME2024 dataset")
        
        try:
            from datasets import load_dataset
            dataset = load_dataset(self.huggingface_dataset_id, split="test")
            df = dataset.to_pandas()
            df["task"] = "aime2024"  # Ensure task column exists
            df["max_new_tokens"] = MAX_NEW_TOKENS_CAP
            print(f"  ✓ Loaded {len(df)} AIME2024 problems")
            return df
        except Exception as e:
            raise Exception(f"Failed to load AIME2024 dataset: {str(e)}")

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics for AIME2024 results.

        Args:
            results_df: DataFrame containing benchmark results with columns:
                - predicted_answer: Model's predicted answer
                - answer: Ground truth answers

        Returns:
            Dictionary containing computed metrics:
            - accuracy: Overall accuracy
            - extraction_success_rate: Rate of successful answer extraction
            - detailed_results: Individual problem results
        """
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        # Use the calculate_metrics function from HashAttention evaluation
        metrics: Dict[str, Any] = calculate_metrics(results_df)
        
        # Format the results for consistency with other benchmarks
        overall_metrics: Dict[str, Any] = {
            "overall_score": round(metrics["accuracy"], 4),
            "accuracy": round(metrics["accuracy"], 4),
            "extraction_success_rate": round(metrics["extraction_success_rate"], 4),
            "correct_answers": metrics["correct_answers"],
            "total_problems": metrics["total_problems"],
            "extraction_failures": metrics["extraction_failures"],
            "task_scores": {
                "aime2024": {
                    "accuracy": round(metrics["accuracy"], 4),
                    "extraction_success_rate": round(metrics["extraction_success_rate"], 4)
                }
            },
            "summary": {
                "total_tasks": 1,
                "total_samples": len(results_df)
            }
        }
        
        print(f"  ✓ AIME2024 Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"  ✓ Extraction Success Rate: {metrics['extraction_success_rate']:.3f} ({metrics['extraction_success_rate']*100:.1f}%)")
        
        return overall_metrics 