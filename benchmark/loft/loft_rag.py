"""LOFT RAG benchmark implementation for long-context retrieval-augmented generation.

Scope of fidelity to https://github.com/google-deepmind/loft, measured rather than
assumed (upstream sha 219f68e):

* METRICS -- faithful.  `calculate_metrics` is a transcription of upstream's
  `evaluation/utils.py` + `evaluation/rag.py`.  Scoring identical model outputs through
  this module and through upstream's own `RagEvaluation` / `MultiValueRagEvaluation`
  gives identical em / subspan_em / f1 / coverage on all five datasets and all splits.
  The multi-value routing (qampari, quest) matches upstream's `multi_value_rag` task
  type, and `subspan_em` is LOFT's primary RAG metric.

* DATA -- a re-mix of LOFT's, not LOFT's own.  This benchmark reads the third-party
  HuggingFace mirror `f20180301/loft-rag-*`, not LOFT's `download.sh` + `preprocess.py`
  output.  What the `*_32k` subsets actually contain:
    - `dev` (10 rows) IS LOFT's 32k dev split: query text and gold answers match
      `rag/<ds>/32k/dev_queries.jsonl` exactly, 10/10 on all five datasets.
    - `test` (100 rows; 60 for qampari/quest) is LOFT's **128k** test split -- every
      query matches `rag/<ds>/128k/test_queries.jsonl` -- placed against a 32k-labelled
      corpus.  LOFT ships `test_queries.jsonl` only at 128k and 1m; there is no 32k test
      split upstream.  qampari and quest keep only 60 of those 100 queries.
    - the rendered corpus holds ~1.45x the documents of LOFT's 32k corpus (e.g. nq 309
      vs 214), presumably to support the imported 128k queries.
  So `overall` below pools LOFT's 32k dev with LOFT's 128k test at a third context
  length, and 91% of the rows are the latter.  It is NOT comparable to any published
  LOFT number; use `by_split["dev"]` for the closest thing to one.
  Known mirror defect: for qampari and quest the corpus contains NONE of the gold
  documents for LOFT's dev queries (0/50 and 0/24 by qrels), so those dev scores are
  floored at 0 by the data rather than by the model.

* PROMPT -- LOFT's corpus instruction, formatting instruction, `ID | TITLE | CONTENT`
  echo format, five few-shot examples with chain-of-thought, and query separators are all
  present in the mirror and match upstream's `prompts/`.

* CONTEXT LENGTH -- the "32k" label is LOFT's, measured with Gemini's tokenizer.  These
  contexts are 42-46k tokens under Llama-3.1's tokenizer, so a `max_context_length` of
  32768 silently truncates ~25-30% of the corpus, gold passages included.  Set it from
  the tokenizer you are actually running.

* GENERATION BUDGET -- the mirror carries max_new_tokens=256 on every row; upstream
  imposes no output cap at all (inference/models.py builds GenerationConfig without
  max_output_tokens).  Note that `benchmark/base.py` takes min(caller, row), so the row
  value is a HARD ceiling that no generation_kwargs can raise.  Left at 256 deliberately:
  with the chain-of-thought restored the answer comes last, and the cap does bind on ~11%
  of rows -- but those are degenerate generations that loop on corpus text rather than
  nearly-finished answers.  Measured on Llama-3.1-8B-Instruct over four subsets, doubling
  to 512 recovered only 6 of 49 unparseable rows (1.5% of all rows) and musique recovered
  0 of 20, so the cost is not worth the fidelity gain.  Parsed answers finish by token
  238 (p99 219).
"""

from typing import Any, Dict, List

import pandas as pd

from ..base import Benchmark
from ..benchmark_registry import register_benchmark
from .calculate_metrics import calculate_metrics


@register_benchmark("loft_rag")
class LoftRag(Benchmark):
    """LOFT RAG benchmark for evaluating long-context retrieval-augmented generation.

    LOFT (Long-context Open Foundation Tasks) RAG evaluates the ability of models to
    answer questions given long retrieved contexts. This benchmark includes:

    - Single-value RAG datasets: nq, hotpotqa, musique
    - Multi-value RAG datasets: qampari, quest

    Each dataset is available in multiple context lengths: 32k, 128k, 1m.

    Metrics:
    - Single-value: EM (Exact Match), Subspan EM, F1
    - Multi-value: EM, Coverage, Subspan EM

    Reference:
        https://github.com/google-deepmind/loft

    Example:
        >>> loft_rag = LoftRag(subsets_to_run=["nq_32k", "hotpotqa_128k"])
        >>> results = loft_rag.run_benchmark(adapter, result_dir="/path/to/results")
        >>> print(f"EM score: {results['nq_32k']['em']}")
    """

    all_datasets: List[str] = [
        "nq_32k",
        "nq_128k",
        "nq_1m",
        "hotpotqa_32k",
        "hotpotqa_128k",
        "hotpotqa_1m",
        "musique_32k",
        "musique_128k",
        "musique_1m",
        "qampari_32k",
        "qampari_128k",
        "qampari_1m",
        "quest_32k",
        "quest_128k",
        "quest_1m",
    ]

    benchmark_name: str = "loft_rag"
    huggingface_dataset_id: str = "f20180301/rag"
    # LOFT's rendered prompt ends at the query; the model is expected to emit the
    # TITLE/ID reasoning step and THEN "Final Answer: [...]" (FINAL_ANSWER_FORMAT renders
    # only the few-shot examples).  Appending the prefix to the prompt primes the answer
    # and suppresses the chain-of-thought this prompt type is built around -- measured on
    # Llama-3.1-8B-Instruct at 32k, that costs +0.036 / +0.064 / +0.072 subspan_em on
    # nq / hotpotqa / qampari.  The prefix is still used for scoring.
    prompt_includes_answer_prefix: bool = False

    def _load_datasets(self) -> pd.DataFrame:
        """Load LOFT RAG datasets from HuggingFace Hub.

        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.
        """
        print(f"Loading LOFT RAG datasets: {self.subsets_to_run}")
        dfs: List[pd.DataFrame] = []

        for subset in self.subsets_to_run:
            parts: List[str] = subset.split("_")
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid subset format: {subset} (expected: dataset_length)"
                )

            length: str = parts[-1]
            dataset: str = "_".join(parts[:-1])
            hf_dataset_id: str = f"f20180301/loft-rag-{dataset}-{length}"

            from datasets import load_dataset

            dataset_dict = load_dataset(hf_dataset_id)

            subset_dfs: List[pd.DataFrame] = []
            for split_name in ["dev", "test"]:
                if split_name in dataset_dict:
                    split_df: pd.DataFrame = dataset_dict[split_name].to_pandas()
                    split_df["split"] = split_name
                    subset_dfs.append(split_df)

            if not subset_dfs:
                raise ValueError(f"No splits found for {subset} ({hf_dataset_id})")

            subset_df: pd.DataFrame = pd.concat(subset_dfs, ignore_index=True)
            subset_df["task"] = subset
            dfs.append(subset_df)
            print(f"  ✓ Loaded {len(subset_df)} samples from {subset}")

        if not dfs:
            raise ValueError("No LOFT RAG subsets could be loaded")

        combined_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} total samples from {len(dfs)} subsets")

        required_columns: List[str] = [
            "context",
            "question",
            "answers",
            "task",
            "answer_prefix",
            "max_new_tokens",
        ]
        missing_columns: List[str] = [
            col for col in required_columns if col not in combined_df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return combined_df

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics for LOFT RAG results.

        Args:
            results_df: DataFrame containing benchmark results

        Returns:
            Dictionary containing computed metrics
        """
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        task_groups = results_df.groupby("task")
        task_metrics: Dict[str, Dict[str, float]] = {}
        all_em_scores: List[float] = []
        all_subspan_em_scores: List[float] = []
        all_f1_scores: List[float] = []
        all_coverage_scores: List[float] = []

        for task_name, task_df in task_groups:
            metrics: Dict[str, Any] = calculate_metrics(task_df)

            if "error" in metrics:
                print(f"  ❌ Error evaluating {task_name}: {metrics['error']}")
                continue

            task_metrics[task_name] = metrics
            all_em_scores.append(metrics["em"])
            all_subspan_em_scores.append(metrics["subspan_em"])

            if "f1" in metrics:
                all_f1_scores.append(metrics["f1"])
            if "coverage" in metrics:
                all_coverage_scores.append(metrics["coverage"])

            metric_str: str = (
                f"EM={metrics['em']:.4f}, Subspan_EM={metrics['subspan_em']:.4f}"
            )
            if "f1" in metrics:
                metric_str += f", F1={metrics['f1']:.4f}"
            if "coverage" in metrics:
                metric_str += f", Coverage={metrics['coverage']:.4f}"
            print(f"  ✓ {task_name}: {metric_str}")

        overall_metrics: Dict[str, Any] = {
            "overall": {
                "em": (
                    float(sum(all_em_scores) / len(all_em_scores))
                    if all_em_scores
                    else 0.0
                ),
                "subspan_em": (
                    float(sum(all_subspan_em_scores) / len(all_subspan_em_scores))
                    if all_subspan_em_scores
                    else 0.0
                ),
            },
            "task_metrics": {
                task: {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
                for task, m in task_metrics.items()
            },
            "summary": {"total_tasks": len(task_metrics), "total_samples": len(results_df)},
        }

        if all_f1_scores:
            overall_metrics["overall"]["f1"] = float(
                sum(all_f1_scores) / len(all_f1_scores)
            )
        if all_coverage_scores:
            overall_metrics["overall"]["coverage"] = float(
                sum(all_coverage_scores) / len(all_coverage_scores)
            )

        overall_metrics["overall"] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in overall_metrics["overall"].items()
        }

        # `overall` above pools the mirror's dev and test splits.  Only `dev` is LOFT's
        # published benchmark (its 10 queries per dataset match LOFT's dev_queries.jsonl
        # exactly); `test` is an extra 100/60 queries present in no LOFT query file.  With
        # 91% of the pooled rows coming from `test`, the pooled number is not comparable
        # to a published LOFT result, so expose the per-split breakdown alongside it
        # rather than only the pooled figure.
        if "split" in results_df.columns:
            by_split: Dict[str, Dict[str, Any]] = {}
            for split_name, split_df in results_df.groupby("split"):
                per_task: Dict[str, Dict[str, float]] = {}
                for task_name, task_df in split_df.groupby("task"):
                    split_metrics = calculate_metrics(task_df)
                    if "error" in split_metrics:
                        continue
                    per_task[str(task_name)] = {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in split_metrics.items()
                    }
                if not per_task:
                    continue
                agg: Dict[str, Any] = {"n_samples": int(len(split_df))}
                for key in ("em", "subspan_em", "f1", "coverage"):
                    vals = [m[key] for m in per_task.values() if key in m]
                    if vals:
                        agg[key] = round(float(sum(vals) / len(vals)), 4)
                by_split[str(split_name)] = {"overall": agg, "task_metrics": per_task}
            if by_split:
                overall_metrics["by_split"] = by_split
                overall_metrics["summary"]["loft_comparable_split"] = "dev"

        return overall_metrics

