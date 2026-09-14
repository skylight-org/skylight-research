"""Tests for the LOFT RAG benchmark's evaluation.

The metric functions themselves are a transcription of upstream
(google-deepmind/loft, `evaluation/utils.py` + `evaluation/rag.py`); these tests pin the
behaviour that is specific to this repo's wrapper, and in particular that the per-split
breakdown separates LOFT's own `dev` queries from the HuggingFace mirror's extra `test`
split.  Pooling the two is not comparable to a published LOFT number, because `test` is
100 (60 for qampari/quest) queries that appear in no LOFT query file.
"""

import pandas as pd

from benchmark.loft import LoftRag
from benchmark.loft.calculate_metrics import calculate_metrics, extract_prediction


def _row(task, split, pred, answers):
    return {
        "task": task,
        "split": split,
        "predicted_answer": pred,
        "answers": answers,
        "answer_prefix": "Final Answer: ",
    }


class TestExtractPrediction:
    """Parsing matches upstream's `utils.extract_prediction` on well-formed output."""

    def test_parses_bracketed_list(self):
        assert extract_prediction("Final Answer: ['Tyrion Lannister']") == [
            "Tyrion Lannister"
        ]

    def test_parses_multi_value_list(self):
        assert extract_prediction("Final Answer: ['a', 'b', 'c']") == ["a", "b", "c"]

    def test_strips_markdown_emphasis_and_backticks(self):
        assert extract_prediction("**Final Answer:** `['x']`") == ["x"]

    def test_handles_apostrophes_inside_values(self):
        # upstream's _escape_single_quotes exists precisely for this case
        assert extract_prediction("Final Answer: ['the devil's sleep']") == [
            "the devil's sleep"
        ]

    def test_reasoning_then_answer(self):
        out = "TITLE: Judith Keppel | ID: 219\nFinal Answer: ['Judith Keppel']"
        assert extract_prediction(out) == ["Judith Keppel"]

    def test_unparseable_output_yields_no_prediction(self):
        assert extract_prediction("I could not find the answer.") == []


class TestCalculateMetrics:
    """Single-value vs multi-value routing, and the metrics each emits."""

    def test_single_value_emits_f1_not_coverage(self):
        df = pd.DataFrame([_row("nq_32k", "dev", "Final Answer: ['Paris']", ["Paris"])])
        m = calculate_metrics(df)
        assert m["em"] == 1.0 and m["subspan_em"] == 1.0
        assert "f1" in m and "coverage" not in m

    def test_multi_value_emits_coverage_not_f1(self):
        df = pd.DataFrame(
            [_row("qampari_32k", "dev", "Final Answer: ['a', 'b']", ["a", "b"])]
        )
        m = calculate_metrics(df)
        assert m["em"] == 1.0 and m["coverage"] == 1.0
        assert "coverage" in m and "f1" not in m

    def test_normalization_ignores_articles_case_and_punctuation(self):
        df = pd.DataFrame(
            [_row("nq_32k", "dev", "Final Answer: ['The Beatles!']", ["beatles"])]
        )
        assert calculate_metrics(df)["em"] == 1.0

    def test_unparseable_prediction_scores_zero(self):
        df = pd.DataFrame([_row("nq_32k", "dev", "no idea", ["Paris"])])
        m = calculate_metrics(df)
        assert m["em"] == 0.0 and m["subspan_em"] == 0.0


class TestPerSplitReporting:
    """`by_split` must isolate LOFT's dev queries from the mirror's extra test split."""

    @staticmethod
    def _mixed_df() -> pd.DataFrame:
        # dev: 2 correct of 2.  test: 0 correct of 4.  Pooling gives 1/3, which is
        # neither number and is what a reader would otherwise quote as "LOFT".
        rows = [
            _row("nq_32k", "dev", "Final Answer: ['Paris']", ["Paris"]),
            _row("nq_32k", "dev", "Final Answer: ['Rome']", ["Rome"]),
        ] + [
            _row("nq_32k", "test", "Final Answer: ['wrong']", ["Berlin"])
            for _ in range(4)
        ]
        return pd.DataFrame(rows)

    def test_by_split_separates_dev_from_test(self):
        out = LoftRag(["nq_32k"]).post_run_evaluate(self._mixed_df())
        assert "by_split" in out
        assert out["by_split"]["dev"]["overall"]["em"] == 1.0
        assert out["by_split"]["test"]["overall"]["em"] == 0.0
        assert out["by_split"]["dev"]["overall"]["n_samples"] == 2
        assert out["by_split"]["test"]["overall"]["n_samples"] == 4

    def test_pooled_overall_is_still_reported_and_differs_from_dev(self):
        out = LoftRag(["nq_32k"]).post_run_evaluate(self._mixed_df())
        # 2 of 6 correct when pooled -- distinct from both split values, which is exactly
        # why the breakdown is needed.
        assert out["overall"]["em"] == round(2 / 6, 4)
        assert out["overall"]["em"] != out["by_split"]["dev"]["overall"]["em"]

    def test_names_the_loft_comparable_split(self):
        out = LoftRag(["nq_32k"]).post_run_evaluate(self._mixed_df())
        assert out["summary"]["loft_comparable_split"] == "dev"

    def test_absent_split_column_is_tolerated(self):
        df = self._mixed_df().drop(columns=["split"])
        out = LoftRag(["nq_32k"]).post_run_evaluate(df)
        assert "by_split" not in out
        assert "overall" in out


class TestPromptAnswerPrefix:
    """LOFT's prompt ends at the query; the prefix is for scoring only."""

    def test_loft_does_not_put_answer_prefix_in_the_prompt(self):
        assert LoftRag.prompt_includes_answer_prefix is False

    def test_other_benchmarks_keep_the_default(self):
        from benchmark.base import Benchmark

        assert Benchmark.prompt_includes_answer_prefix is True

    def test_scoring_still_uses_the_dataset_answer_prefix(self):
        # Blanking the PROMPT prefix must not blank the PARSE prefix: calculate_metrics
        # reads it from the dataframe, where it is still "Final Answer: ".
        df = pd.DataFrame([_row("nq_32k", "dev", "Final Answer: ['Paris']", ["Paris"])])
        assert df["answer_prefix"].iloc[0] == "Final Answer: "
        assert calculate_metrics(df)["em"] == 1.0
