"""Ruler benchmark implementation for long context evaluation."""

from typing import Any, Dict, List

import pandas as pd

from ..base import Benchmark
from ..benchmark_registry import register_benchmark
from .calculate_metrics import calculate_metrics
from .prepare_dataframe import prepare_dataframe


@register_benchmark("ruler64k")
class Ruler64K(Benchmark):
    """Ruler benchmark for evaluating long context understanding."""

    # All available Ruler datasets (context lengths)
    all_datasets: List[str] = [
        "cwe",
        "fwe",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multiquery",
        "niah_multivalue",
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "qa_1",
        "qa_2",
        "vt",
    ]

    benchmark_name: str = "ruler64k"
    huggingface_dataset_id: str = "SaylorTwift/RULER-65536-llama-3.2-tokenizer"

    def _load_datasets(self) -> pd.DataFrame:
        """Load Ruler datasets by individual splits.

        Unlike ruler16k / ruler32k, the upstream dataset is *raw*: every row
        holds a single ``input`` string and the dataset exposes one ``default``
        config whose splits are the RULER task names.  ``prepare_dataframe``
        re-applies the official RULER anchors to recover context / question /
        answer_prefix at load time.

        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.
        """
        print(f"Loading Ruler datasets: {self.subsets_to_run}")
        dfs = []
        failed_subsets: List[str] = []

        for subset in self.subsets_to_run:
            try:
                from datasets import load_dataset

                subset_dataset = load_dataset(self.huggingface_dataset_id, split=subset)
                # Splits the raw `input` column and stamps context_length.
                subset_df = prepare_dataframe(subset_dataset.to_pandas(), subset, 65536)
                dfs.append(subset_df)
                print(
                    f"  ✓ Loaded {len(subset_df)} samples from {subset} || context_length = 65536"
                )
            except Exception as subset_error:
                failed_subsets.append(subset)
                print(f"  ❌ Failed to load {subset}: {str(subset_error)}")
                continue

        if not dfs:
            raise Exception("No Ruler subsets could be loaded successfully")

        # Combine all subset DataFrames
        import pandas as pd

        combined_df = pd.concat(dfs, ignore_index=True)
        if failed_subsets:
            # post_run_evaluate averages over the tasks that are present, so a
            # skipped subset silently changes overall_score. Say so loudly.
            print(
                f"  ⚠️  {len(failed_subsets)} subset(s) skipped and NOT included in "
                f"the reported scores: {failed_subsets}"
            )
        print(
            f"Combined {len(combined_df)} total samples from {len(dfs)} subsets || context_length = 65536"
        )
        return combined_df

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics for Ruler results.

        Args:
            results_df: DataFrame containing benchmark results with columns:
                - task: Dataset/task name
                - predicted_answer: Model's predicted answer
                - answer: Ground truth answers
                - context_length: Context length used

        Returns:
            Dictionary containing computed metrics:
            - overall_score: Average score across all tasks
            - task_scores: Individual scores for each task
            - context_length_scores: Scores grouped by context length
        """
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        # Use the calculate_metrics function from HashAttention evaluation
        task_scores: Dict[str, Dict[str, float]] = calculate_metrics(results_df)

        # Extract string_match scores and compute overall
        all_scores: List[float] = []
        for task, scores in task_scores.items():
            if "string_match" in scores:
                all_scores.append(scores["string_match"])

        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Group by context length if available
        context_length_scores: Dict[str, float] = {}
        if "context_length" in results_df.columns:
            for context_length in results_df["context_length"].unique():
                try:
                    length_df = results_df[
                        results_df["context_length"] == context_length
                    ]
                    if len(length_df) > 0:
                        length_scores = calculate_metrics(length_df)
                        length_overall = (
                            sum(
                                score.get("string_match", 0)
                                for score in length_scores.values()
                            )
                            / len(length_scores)
                            if length_scores
                            else 0.0
                        )
                        context_length_scores[str(context_length)] = round(
                            length_overall, 2
                        )
                except Exception as e:
                    print(
                        f"Warning: Failed to compute metrics for context length {context_length}: {e}"
                    )
                    continue

        overall_metrics: Dict[str, Any] = {
            "overall_score": round(overall_score, 2),
            "task_scores": task_scores,
            "context_length_scores": context_length_scores,
            "summary": {
                "total_tasks": len(task_scores),
                "total_samples": len(results_df),
                "context_lengths": list(context_length_scores.keys()),
            },
        }

        return overall_metrics
