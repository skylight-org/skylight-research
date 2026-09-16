"""Tests for the LOFT RAG benchmark's evaluation.

The metric functions themselves are a transcription of upstream
(google-deepmind/loft, `evaluation/utils.py` + `evaluation/rag.py`); these tests pin the
behaviour specific to this repo's wrapper, and in particular the per-split breakdown.
BOTH splits carry LOFT's own queries against one shared corpus -- verified against
upstream's evaluation/example_predictions/rag_nq/queries.jsonl, whose 10 test qids all
appear in the mirror's `test` split with identical golds and none in `dev`.  The corpus is
selected around the TEST queries, so dev golds are largely absent from it and dev scores
are floored by the data; `test` is both the LOFT-comparable split and the larger one.
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

    def test_multi_value_emits_f1_when_a_row_fails_to_parse(self):
        # Upstream's MultiValueRagEvaluation appends f1=0.0 on an unparseable row, so its
        # aggregate carries f1 for multi-value tasks too -- absent only when all parsed.
        rows = [
            _row("quest_32k", "test", "Final Answer: ['a']", ["a"]),
            _row("quest_32k", "test", "no idea", ["b"]),
        ]
        m = calculate_metrics(pd.DataFrame(rows))
        assert m["f1"] == 0.0
        all_parsed = calculate_metrics(pd.DataFrame([rows[0]]))
        assert "f1" not in all_parsed

    def test_tuple_of_lists_is_iterated_like_upstream(self):
        # literal_eval on a "[...]"-delimited slice yields a TUPLE when the line holds
        # several lists.  Upstream returns that tuple raw and convert_to_str stringifies
        # each element, so two lists become two predictions -- verified against upstream's
        # own extract_prediction + convert_to_str, which return exactly this.
        # Wrapping the tuple whole instead scored differently in both directions.
        from benchmark.loft.calculate_metrics import extract_prediction

        assert extract_prediction("Final Answer: ['a'], ['b']", "final answer: ") == [
            "['a']",
            "['b']",
        ]

    def test_parsed_elements_are_str_converted(self):
        # Upstream's convert_to_str; fires on real rows whose list holds an int.
        from benchmark.loft.calculate_metrics import extract_prediction

        assert extract_prediction(
            "Final Answer: ['Physical', 483]", "final answer: "
        ) == [
            "Physical",
            "483",
        ]

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
        assert out["summary"]["loft_comparable_split"] == "test"

    def test_absent_split_column_is_tolerated(self):
        df = self._mixed_df().drop(columns=["split"])
        out = LoftRag(["nq_32k"]).post_run_evaluate(df)
        assert "by_split" not in out
        assert "overall" in out


class TestGenerationBudget:
    """Upstream imposes no output cap; the dataset column must not become one."""

    def test_load_datasets_drops_the_max_new_tokens_ceiling(self):
        # base.py takes min(caller, row), so leaving the mirror's 256 in place makes it a
        # ceiling no caller can raise -- it truncated ~half the sparse rows before they
        # emitted "Final Answer".  Deleting the drop() was previously invisible here.
        from unittest.mock import patch

        frame = pd.DataFrame(
            {
                "context": ["c"],
                "question": ["q"],
                "answers": [["a"]],
                "answer_prefix": ["Final Answer: "],
                "max_new_tokens": [256],
            }
        )

        class _Split:
            def to_pandas(self):
                return frame.copy()

        with patch("datasets.load_dataset", return_value={"test": _Split()}):
            out = LoftRag(["nq_32k"])._load_datasets()
        assert "max_new_tokens" not in out.columns
        for col in ("context", "question", "answers", "answer_prefix", "task"):
            assert col in out.columns


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


class TestUpstreamParityOfExtractPrediction:
    """`extract_prediction` must not be more permissive than upstream.

    Upstream (loft/utils.py:465-478) stops at the FIRST line containing both brackets,
    whether or not it parses, and has no answer-prefix fallback.  Both of this repo's
    former deviations could only turn an upstream 0 into a non-zero score.
    """

    def test_stops_at_first_bracketed_line_even_if_it_fails_to_parse(self):
        # First bracketed line is unparseable; upstream returns [] and so must we,
        # rather than scanning on and recovering the second line's answer.
        out = "Final Answer: [Randolph County, Illinois]\nFinal Answer: ['Chicago']"
        assert extract_prediction(out) == []

    def test_no_answer_prefix_fallback(self):
        # No brackets anywhere: upstream returns [], regardless of the prefix appearing.
        assert extract_prediction("Final Answer: Tyrion Lannister") == []

    def test_well_formed_output_is_unaffected(self):
        out = "TITLE: X | ID: 3\nFinal Answer: ['Tyrion Lannister']"
        assert extract_prediction(out) == ["Tyrion Lannister"]

    def test_empty_prefix_cannot_swallow_the_whole_line(self):
        # Guards the fix that was NOT taken: blanking answer_prefix used to make the
        # fallback match every line and return the entire first line as the prediction.
        out = "The answer is the following day, based on document ID 7."
        assert extract_prediction(out, "") == []


class TestCoverageDenominator:
    """Coverage averages over PARSED rows, as upstream's aggregate_metrics does."""

    def test_unparseable_rows_are_excluded_from_coverage(self):
        rows = [
            _row("qampari_32k", "dev", "Final Answer: ['a', 'b']", ["a", "b"]),
            _row("qampari_32k", "dev", "I cannot answer that.", ["c", "d"]),
        ]
        m = calculate_metrics(pd.DataFrame(rows))
        # 1 parsed row with perfect coverage -> 1.0, not 0.5.
        assert m["coverage"] == 1.0
        assert m["num_scored_for_coverage"] == 1
        # em / subspan_em still count the unparseable row as a miss.
        assert m["em"] == 0.5
        assert m["num_samples"] == 2

    def test_all_unparseable_omits_coverage_as_upstream_does(self):
        # Upstream aggregates each metric over the list it appended to, so a metric that
        # was never appended is ABSENT.  Reporting 0.0 instead would fold a value into
        # post_run_evaluate's macro average where upstream contributes nothing.
        rows = [_row("quest_32k", "dev", "no idea", ["a"])] * 2
        m = calculate_metrics(pd.DataFrame(rows))
        assert "coverage" not in m
        assert m["num_scored_for_coverage"] == 0
        assert m["em"] == 0.0 and m["subspan_em"] == 0.0

    def test_single_value_f1_still_counts_every_row(self):
        rows = [
            _row("nq_32k", "dev", "Final Answer: ['Paris']", ["Paris"]),
            _row("nq_32k", "dev", "no idea", ["Rome"]),
        ]
        m = calculate_metrics(pd.DataFrame(rows))
        assert m["f1"] == 0.5  # unparseable row contributes 0.0, denominator 2
