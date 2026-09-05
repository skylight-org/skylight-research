"""Unit tests for Ruler128K benchmark implementation."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from benchmark.ruler128k import Ruler128K
from benchmark.ruler128k.calculate_metrics import calculate_metrics
from benchmark.ruler128k.prepare_dataframe import (
    MAX_NEW_TOKENS,
    OUTPUT_COLUMNS,
    RulerSplitError,
    prepare_dataframe,
    split_context_question,
)

DATASET_ID = "SaylorTwift/RULER-131072-llama-3.2-tokenizer"
CONTEXT_LENGTH = 131072

# Minimal stand-ins for the five RULER prompt families.  Each is
# "<preamble><haystack><question><answer_prefix>" exactly as RULER emits it, so
# split_context_question has something realistic to cut.
_TEMPLATES = {
    "niah": (
        "One of the special magic numbers for pale-cactus is hidden in the "
        "following text. Make sure to memorize it.\n"
        "{haystack}\n"
        "What is the special magic number for pale-cactus mentioned in the "
        "provided text? "
        "The special magic number for pale-cactus mentioned in the provided "
        "text is"
    ),
    # RULER puts a fully worked one-shot example near the top of every vt
    # prompt (char 1079 of 244718 in the real 64k data), so vt's question
    # anchor genuinely occurs twice -- exercise the last-match rule on it.
    "vt": (
        "Memorize and track the chain(s) of variable assignment hidden in the "
        "following text.\n"
        "{haystack}\n"
        "Question: Find all variables that are assigned the value 64886 in the "
        "text above."
        "Answer: According to the chain(s) of variable assignment in the text "
        "above, 5 variables are assigned the value 64886, they are:  SGM LJP\n"
        "{haystack}\n"
        "Question: Find all variables that are assigned the value 12345 in the "
        "text above."
        "Answer: According to the chain(s) of variable assignment in the text "
        "above, 5 variables are assigned the value 12345, they are: "
    ),
    "cwe": (
        "Below is a numbered list of words. In these words, some appear more "
        "often than others. Memorize the ones that appear most often.\n"
        "Question: What are the 10 most common words in the above list?\n"
        "{haystack}\n"
        "Question: What are the 10 most common words in the above list? "
        "Answer: The top 10 words that appear most often in the list are:"
    ),
    "fwe": (
        "Read the following coded text and track the frequency of each coded "
        "word.\n"
        "{haystack}\n"
        "Question: Do not provide any explanation. Please ignore the dots "
        "'....'. What are the three most frequently appeared words in the "
        "above coded text? "
        "Answer: According to the coded text above, the three most frequently "
        "appeared words are:"
    ),
    "qa": (
        "Answer the question based on the given documents. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given documents.\n\n"
        "{haystack}\n\n"
        "Answer the question based on the given documents. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: What colour is the sky? "
        "Answer:"
    ),
}

# Families whose question anchor occurs twice -- as a context preamble
# (cwe, qa) or as a one-shot worked example (vt) -- and again as the real
# trailing question.  The split must take the LAST occurrence.
_REPEATED_ANCHOR_FAMILIES = ["cwe", "qa", "vt"]


def _raw_input(family: str, haystack: str = "HAYSTACK " * 16) -> str:
    """Build a synthetic raw RULER sample for one prompt family."""
    return _TEMPLATES[family].format(haystack=haystack)


def _raw_frame(task: str, n_rows: int = 2) -> pd.DataFrame:
    """Build a raw upstream frame (index / input / outputs / length)."""
    family = task.split("_")[0]
    return pd.DataFrame(
        {
            "index": list(range(n_rows)),
            "input": [_raw_input(family) for _ in range(n_rows)],
            "outputs": [np.array(["gold"], dtype=object) for _ in range(n_rows)],
            "length": [CONTEXT_LENGTH] * n_rows,
        }
    )


def _mock_dataset(task: str, n_rows: int = 2) -> Mock:
    """A Mock standing in for a loaded HuggingFace split."""
    dataset = Mock()
    dataset.to_pandas.return_value = _raw_frame(task, n_rows)
    return dataset


class TestRuler128KUnit:
    """Unit tests for Ruler128K class."""

    def test_ruler128k_initialization(self):
        """Test Ruler128K initialization."""
        ruler128k = Ruler128K()
        assert len(ruler128k.all_datasets) == 13
        assert ruler128k.benchmark_name == "ruler128k"
        assert ruler128k.huggingface_dataset_id == DATASET_ID

    def test_ruler128k_initialization_with_subsets(self):
        """Test Ruler128K initialization with custom subsets."""
        subsets = ["niah_single_1", "qa_1"]
        ruler128k = Ruler128K(subsets_to_run=subsets)
        assert ruler128k.subsets_to_run == subsets

    def test_ruler128k_dataset_list_complete(self):
        """Test that Ruler128K has all expected datasets."""
        ruler128k = Ruler128K()

        expected_datasets = [
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

        for dataset in expected_datasets:
            assert dataset in ruler128k.all_datasets

        # Verify exact count
        assert len(ruler128k.all_datasets) == len(expected_datasets)

    def test_ruler128k_dataset_categories(self):
        """Test that Ruler128K datasets contain expected categories."""
        ruler128k = Ruler128K()

        # Check for different task categories
        niah_tasks = [d for d in ruler128k.all_datasets if d.startswith("niah_")]
        qa_tasks = [d for d in ruler128k.all_datasets if d.startswith("qa_")]
        other_tasks = [
            d for d in ruler128k.all_datasets if not d.startswith(("niah_", "qa_"))
        ]

        assert (
            len(niah_tasks) == 8
        )  # 3 single + 3 multikey + 1 multiquery + 1 multivalue
        assert len(qa_tasks) == 2  # qa_1, qa_2
        assert len(other_tasks) == 3  # cwe, fwe, vt

    def test_ruler128k_subset_selection_valid(self):
        """Test Ruler128K subset selection with valid datasets."""
        # Test with NIAH tasks
        ruler128k = Ruler128K(subsets_to_run=["niah_single_1", "niah_multikey_1"])
        assert ruler128k.subsets_to_run == ["niah_single_1", "niah_multikey_1"]

        # Test with QA tasks
        ruler128k = Ruler128K(subsets_to_run=["qa_1", "qa_2"])
        assert ruler128k.subsets_to_run == ["qa_1", "qa_2"]

        # Test with other tasks
        ruler128k = Ruler128K(subsets_to_run=["cwe", "fwe", "vt"])
        assert ruler128k.subsets_to_run == ["cwe", "fwe", "vt"]

    def test_ruler128k_subset_selection_invalid(self):
        """Test Ruler128K subset selection with invalid datasets."""
        with pytest.raises(ValueError, match="Invalid subsets"):
            Ruler128K(subsets_to_run=["invalid_dataset"])

        with pytest.raises(ValueError, match="Invalid subsets"):
            Ruler128K(subsets_to_run=["niah_single_1", "invalid_dataset"])

        with pytest.raises(ValueError, match="Invalid subsets"):
            Ruler128K(subsets_to_run=["ruler32k_niah_single_1"])  # wrong context length

    @patch("datasets.load_dataset")
    def test_ruler128k_load_datasets_success(self, mock_load_dataset):
        """Test successful dataset loading."""
        mock_load_dataset.side_effect = [
            _mock_dataset("niah_single_1", n_rows=2),
            _mock_dataset("qa_1", n_rows=1),
        ]

        ruler128k = Ruler128K(subsets_to_run=["niah_single_1", "qa_1"])
        df = ruler128k._load_datasets()

        # Check that datasets were loaded correctly
        assert len(df) == 3  # 2 + 1 samples
        assert "context_length" in df.columns
        assert all(df["context_length"] == CONTEXT_LENGTH)

        # The raw columns must not leak into the harness schema.
        assert list(df.columns) == OUTPUT_COLUMNS
        assert "input" not in df.columns
        assert "index" not in df.columns
        assert "length" not in df.columns

        # Check mock calls: single "default" config, so no config-name argument.
        assert mock_load_dataset.call_count == 2
        mock_load_dataset.assert_any_call(DATASET_ID, split="niah_single_1")
        mock_load_dataset.assert_any_call(DATASET_ID, split="qa_1")

    @patch("datasets.load_dataset")
    def test_ruler128k_load_datasets_reconstructs_raw_input(self, mock_load_dataset):
        """The three split columns must rebuild the upstream `input` exactly."""
        raw = _raw_frame("niah_single_1", n_rows=2)
        dataset = Mock()
        dataset.to_pandas.return_value = raw.copy()
        mock_load_dataset.return_value = dataset

        df = Ruler128K(subsets_to_run=["niah_single_1"])._load_datasets()

        for original, (_, row) in zip(raw["input"], df.iterrows()):
            assert row["context"] + row["question"] + row["answer_prefix"] == original

    @patch("datasets.load_dataset")
    def test_ruler128k_load_datasets_uses_single_default_config(
        self, mock_load_dataset
    ):
        """No config-name positional arg: these repos expose one `default` config.

        Passing the subset as a config name would raise
        ``BuilderConfig 'cwe' not found`` -- but only against the real Hub, so
        pin the call shape here.
        """
        mock_load_dataset.return_value = _mock_dataset("cwe", n_rows=1)

        Ruler128K(subsets_to_run=["cwe"])._load_datasets()

        args, kwargs = mock_load_dataset.call_args
        assert args == (DATASET_ID,)
        assert kwargs == {"split": "cwe"}

    @patch("datasets.load_dataset")
    def test_ruler128k_load_datasets_partial_failure(self, mock_load_dataset):
        """Test dataset loading with some failures."""
        mock_load_dataset.side_effect = [
            _mock_dataset("niah_single_1", n_rows=1),
            Exception("Dataset not found"),
        ]

        # Use valid subsets, but mock one to fail during loading
        ruler128k = Ruler128K(subsets_to_run=["niah_single_1", "qa_1"])
        df = ruler128k._load_datasets()

        # Should succeed with partial data (only niah_single_1 loaded)
        assert len(df) == 1
        assert "context_length" in df.columns
        assert all(df["context_length"] == CONTEXT_LENGTH)

    @patch("datasets.load_dataset")
    def test_ruler128k_load_datasets_complete_failure(self, mock_load_dataset):
        """Test dataset loading with complete failure."""
        mock_load_dataset.side_effect = Exception("No datasets found")

        ruler128k = Ruler128K(subsets_to_run=["niah_single_1"])

        with pytest.raises(
            Exception, match="No Ruler subsets could be loaded successfully"
        ):
            ruler128k._load_datasets()

    def test_ruler128k_post_run_evaluate_empty_results(self):
        """Test Ruler128K evaluation with empty results."""
        ruler128k = Ruler128K()
        empty_df = pd.DataFrame()

        results = ruler128k.post_run_evaluate(empty_df)
        assert "error" in results
        assert results["error"] == "No results to evaluate"

    def test_ruler128k_post_run_evaluate_with_results(self):
        """Test Ruler128K evaluation with valid results."""
        ruler128k = Ruler128K()

        # Mock results DataFrame
        mock_results = pd.DataFrame(
            {
                "task": ["niah_single_1", "niah_single_1", "qa_1", "qa_1"],
                "predicted_answer": ["Answer 1", "Answer 2", "Answer 3", "Answer 4"],
                "answer": [["Truth 1"], ["Truth 2"], ["Truth 3"], ["Truth 4"]],
                "context_length": [CONTEXT_LENGTH] * 4,
            }
        )

        with patch("benchmark.ruler128k.ruler128k.calculate_metrics") as mock_calc:
            # Mock different scores for different tasks
            mock_calc.side_effect = [
                {
                    "niah_single_1": {"string_match": 85.0},
                    "qa_1": {"string_match": 90.0},
                },  # first call
                {
                    "niah_single_1": {"string_match": 85.0},
                    "qa_1": {"string_match": 90.0},
                },  # second call for context length
            ]

            results = ruler128k.post_run_evaluate(mock_results)

        # Check structure
        assert "overall_score" in results
        assert "task_scores" in results
        assert "context_length_scores" in results
        assert "summary" in results

        # Check overall score (average of 85.0 and 90.0)
        expected_avg = round((85.0 + 90.0) / 2, 2)
        assert results["overall_score"] == expected_avg

        # Check context length scores
        assert str(CONTEXT_LENGTH) in results["context_length_scores"]
        assert results["context_length_scores"][str(CONTEXT_LENGTH)] == expected_avg

        # Check summary
        assert results["summary"]["total_tasks"] == 2
        assert results["summary"]["total_samples"] == 4
        assert results["summary"]["context_lengths"] == [str(CONTEXT_LENGTH)]

    def test_ruler128k_post_run_evaluate_different_task_types(self):
        """Test Ruler128K evaluation with different task types (QA vs others)."""
        ruler128k = Ruler128K()

        # Mock results with QA and non-QA tasks
        mock_results = pd.DataFrame(
            {
                "task": ["qa_1", "qa_1", "niah_single_1", "cwe"],
                "predicted_answer": [
                    "QA Answer 1",
                    "QA Answer 2",
                    "NIAH Answer",
                    "CWE Answer",
                ],
                "answer": [
                    ["QA Truth 1"],
                    ["QA Truth 2"],
                    ["NIAH Truth"],
                    ["CWE Truth"],
                ],
                "context_length": [CONTEXT_LENGTH] * 4,
            }
        )

        with patch("benchmark.ruler128k.ruler128k.calculate_metrics") as mock_calc:
            mock_calc.side_effect = [
                {
                    "qa_1": {"string_match": 80.0},
                    "niah_single_1": {"string_match": 75.0},
                    "cwe": {"string_match": 85.0},
                },
                {
                    "qa_1": {"string_match": 80.0},
                    "niah_single_1": {"string_match": 75.0},
                    "cwe": {"string_match": 85.0},
                },
            ]

            results = ruler128k.post_run_evaluate(mock_results)

        # Check that all task types are included
        assert "qa_1" in results["task_scores"]
        assert "niah_single_1" in results["task_scores"]
        assert "cwe" in results["task_scores"]

        # Check overall score includes all tasks
        expected_avg = round((80.0 + 75.0 + 85.0) / 3, 2)
        assert results["overall_score"] == expected_avg

    def test_ruler128k_post_run_evaluate_no_context_length(self):
        """Test evaluation when context_length column is missing."""
        ruler128k = Ruler128K()

        # Mock results without context_length column
        mock_results = pd.DataFrame(
            {
                "task": ["niah_single_1", "qa_1"],
                "predicted_answer": ["Answer 1", "Answer 2"],
                "answer": [["Truth 1"], ["Truth 2"]],
            }
        )

        with patch("benchmark.ruler128k.ruler128k.calculate_metrics") as mock_calc:
            mock_calc.return_value = {
                "niah_single_1": {"string_match": 85.0},
                "qa_1": {"string_match": 90.0},
            }

            results = ruler128k.post_run_evaluate(mock_results)

        # Should still work but without context_length_scores
        assert "overall_score" in results
        assert "task_scores" in results
        assert results["context_length_scores"] == {}
        assert results["summary"]["context_lengths"] == []

    def test_ruler128k_post_run_evaluate_error_handling(self):
        """Test error handling during evaluation."""
        ruler128k = Ruler128K()

        mock_results = pd.DataFrame(
            {
                "task": ["niah_single_1", "qa_1"],
                "predicted_answer": ["Answer 1", "Answer 2"],
                "answer": [["Truth 1"], ["Truth 2"]],
                "context_length": [CONTEXT_LENGTH, CONTEXT_LENGTH],
            }
        )

        with patch("benchmark.ruler128k.ruler128k.calculate_metrics") as mock_calc:
            # First call succeeds, second call (for context length) fails
            successful_result = {
                "niah_single_1": {"string_match": 85.0},
                "qa_1": {"string_match": 90.0},
            }
            mock_calc.side_effect = [successful_result, Exception("Evaluation failed")]

            results = ruler128k.post_run_evaluate(mock_results)

        # Should handle errors gracefully
        assert "overall_score" in results
        assert "task_scores" in results

        # Should still compute overall score from successful first call
        expected_avg = round((85.0 + 90.0) / 2, 2)
        assert results["overall_score"] == expected_avg

        # Context length scores should be empty due to the error
        assert results["context_length_scores"] == {}

    def test_ruler128k_post_run_evaluate_missing_string_match(self):
        """Test evaluation when string_match key is missing from some results."""
        ruler128k = Ruler128K()

        mock_results = pd.DataFrame(
            {
                "task": ["niah_single_1", "qa_1"],
                "predicted_answer": ["Answer 1", "Answer 2"],
                "answer": [["Truth 1"], ["Truth 2"]],
                "context_length": [CONTEXT_LENGTH, CONTEXT_LENGTH],
            }
        )

        with patch("benchmark.ruler128k.ruler128k.calculate_metrics") as mock_calc:
            # Return results where one task is missing string_match
            mock_calc.side_effect = [
                {
                    "niah_single_1": {"string_match": 85.0},
                    "qa_1": {"other_metric": 90.0},  # Missing string_match
                },
                {
                    "niah_single_1": {"string_match": 85.0},
                    "qa_1": {"other_metric": 90.0},
                },
            ]

            results = ruler128k.post_run_evaluate(mock_results)

        # Should only include tasks with string_match in overall score
        assert results["overall_score"] == 85.0  # Only niah_single_1 score

    @patch("datasets.load_dataset")
    def test_ruler128k_context_length_specific(self, mock_load_dataset):
        """Test that Ruler128K specifically uses 131072 context length."""
        mock_load_dataset.return_value = _mock_dataset("niah_single_1", n_rows=1)

        df = Ruler128K(subsets_to_run=["niah_single_1"])._load_datasets()

        # Verify context length is specifically 131072 (not 32768 or others)
        assert all(df["context_length"] == CONTEXT_LENGTH)
        assert not any(df["context_length"] == 32768)

    def test_ruler128k_vs_ruler32k_differences(self):
        """Test that Ruler128K has distinct properties from Ruler32K."""
        ruler128k = Ruler128K()

        # Different dataset ID
        assert ruler128k.huggingface_dataset_id == DATASET_ID
        assert ruler128k.huggingface_dataset_id != "xAlg-AI/att-hub-ruler-32k"

        # Different benchmark name
        assert ruler128k.benchmark_name == "ruler128k"
        assert ruler128k.benchmark_name != "ruler32k"

        # Same dataset list (should be identical for all context lengths)
        expected_datasets = [
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
        assert ruler128k.all_datasets == expected_datasets

    def test_ruler128k_inheritance(self):
        """Test that Ruler128K properly inherits from Benchmark."""
        from benchmark.base import Benchmark

        ruler128k = Ruler128K()
        assert isinstance(ruler128k, Benchmark)

        # Should have all required Benchmark methods
        assert hasattr(ruler128k, "_load_datasets")
        assert hasattr(ruler128k, "post_run_evaluate")
        assert hasattr(ruler128k, "get_available_datasets")
        assert hasattr(ruler128k, "_validate_subsets")

    def test_ruler128k_registry_integration(self):
        """Test that Ruler128K is properly registered in the benchmark registry."""
        from benchmark.benchmark_registry import get_registered_benchmarks

        registered = get_registered_benchmarks()
        assert "ruler128k" in registered
        assert registered["ruler128k"] == Ruler128K


class TestRuler128KPrepareDataframe:
    """Tests for the load-time split of the raw upstream `input` column."""

    @pytest.mark.parametrize("family", sorted(_TEMPLATES))
    def test_split_round_trip(self, family):
        """context + question + answer_prefix must rebuild the input exactly."""
        raw = _raw_input(family)
        context, question, answer_prefix = split_context_question(raw, family)

        assert context + question + answer_prefix == raw
        assert context and question and answer_prefix

    @pytest.mark.parametrize("family", sorted(_TEMPLATES))
    def test_split_question_anchor_is_in_question_not_context(self, family):
        """The trailing question must not survive inside the compressed context."""
        from benchmark.ruler128k.prepare_dataframe import (
            ANSWER_PATTERNS,
            QUESTION_PATTERNS,
        )

        _, question, answer_prefix = split_context_question(_raw_input(family), family)
        assert QUESTION_PATTERNS[family].search(question) is not None
        assert ANSWER_PATTERNS[family].match(answer_prefix) is not None

    @pytest.mark.parametrize("family", _REPEATED_ANCHOR_FAMILIES)
    def test_split_uses_last_question_anchor(self, family):
        """cwe/qa/vt repeat the anchor; the LAST one is the boundary.

        A first-match regression would move the whole haystack into `question`
        and leave only the preamble (cwe/qa) or the one-shot example (vt) as
        context -- plausible-looking, and wrong.
        """
        from benchmark.ruler128k.prepare_dataframe import QUESTION_PATTERNS

        raw = _raw_input(family)
        assert len(list(QUESTION_PATTERNS[family].finditer(raw))) == 2

        context, question, _ = split_context_question(raw, family)
        # The preamble occurrence stays in the context...
        assert QUESTION_PATTERNS[family].search(context) is not None
        # ...and the haystack does not leak into the question.
        assert "HAYSTACK" in context
        assert "HAYSTACK" not in question

    def test_split_missing_question_anchor_raises(self):
        """A missing question anchor raises RulerSplitError, not IndexError."""
        with pytest.raises(RulerSplitError, match="question anchor"):
            split_context_question("no anchors here at all", "niah_single_1")

    def test_split_missing_answer_anchor_raises(self):
        """A missing answer anchor raises RulerSplitError, not AttributeError."""
        truncated = "What is the special magic number for pale-cactus?"
        with pytest.raises(RulerSplitError, match="answer anchor"):
            split_context_question(truncated, "niah_single_1")

    @pytest.mark.parametrize(
        "task,family",
        [
            ("niah_multikey_2", "niah"),
            ("vt", "vt"),
            ("cwe", "cwe"),
            ("fwe", "fwe"),
            ("qa_1", "qa"),
        ],
    )
    def test_prepare_dataframe_schema_and_max_new_tokens(self, task, family):
        """Output schema, task stamping and per-family generation budgets."""
        df = prepare_dataframe(_raw_frame(task), task, CONTEXT_LENGTH)

        assert list(df.columns) == OUTPUT_COLUMNS
        assert all(df["task"] == task)
        assert all(df["context_length"] == CONTEXT_LENGTH)
        assert all(df["max_new_tokens"] == MAX_NEW_TOKENS[family])

    def test_prepare_dataframe_max_new_tokens_values(self):
        """The generation budgets match RULER's own constants."""
        assert MAX_NEW_TOKENS == {
            "niah": 128,
            "vt": 30,
            "cwe": 120,
            "fwe": 50,
            "qa": 32,
        }

    def test_prepare_dataframe_answer_reaches_calculate_metrics(self):
        """`outputs` arrives as an object ndarray; the metric must accept it."""
        df = prepare_dataframe(_raw_frame("vt"), "vt", CONTEXT_LENGTH)
        assert isinstance(df["answer"].iloc[0], np.ndarray)

        df["predicted_answer"] = ["the gold answer", "nothing useful"]
        scores = calculate_metrics(df)
        assert scores["vt"]["string_match"] == pytest.approx(50.0)

    def test_prepare_dataframe_empty_input(self):
        """An empty raw frame yields an empty typed frame, not a TypeError."""
        empty = pd.DataFrame(columns=["index", "input", "outputs", "length"])
        df = prepare_dataframe(empty, "vt", CONTEXT_LENGTH)

        assert len(df) == 0
        assert list(df.columns) == OUTPUT_COLUMNS

    def test_prepare_dataframe_row_failure_is_not_silently_dropped(self):
        """One bad row fails the whole split rather than shrinking the sample."""
        raw = _raw_frame("vt", n_rows=2)
        raw.loc[1, "input"] = "this row has no anchors"

        with pytest.raises(RulerSplitError):
            prepare_dataframe(raw, "vt", CONTEXT_LENGTH)

    @patch("datasets.load_dataset")
    def test_load_datasets_reports_unsplittable_subset(self, mock_load_dataset, capsys):
        """An unsplittable subset is skipped loudly and named in the summary."""
        good = _mock_dataset("vt", n_rows=1)
        bad_frame = _raw_frame("fwe", n_rows=1)
        bad_frame.loc[0, "input"] = "this row has no anchors"
        bad = Mock()
        bad.to_pandas.return_value = bad_frame
        mock_load_dataset.side_effect = [good, bad]

        df = Ruler128K(subsets_to_run=["vt", "fwe"])._load_datasets()

        assert set(df["task"]) == {"vt"}
        captured = capsys.readouterr().out
        assert "❌" in captured
        assert "⚠️" in captured
        assert "fwe" in captured


class TestRuler128KMetricRouting:
    """The qa-vs-everything-else metric split in calculate_metrics."""

    @pytest.mark.parametrize(
        "task,expected",
        [
            # string_match_part: credit for matching ANY one gold.
            ("qa_1", 100.0),
            ("qa_2", 100.0),
            # string_match_all: credit for the FRACTION of golds matched.
            ("vt", 50.0),
            ("fwe", 50.0),
            ("cwe", 50.0),
            ("niah_multikey_2", 50.0),
            ("niah_single_1", 50.0),
        ],
    )
    def test_qa_uses_part_and_every_other_family_uses_all(self, task, expected):
        """Routing is one `task.split("_")[0] == "qa"` branch -- pin both sides.

        The two metrics only disagree on multi-gold rows, so a single gold
        cannot detect a mis-route. With two golds and one hit, `part` scores
        100 and `all` scores 50.
        """
        df = pd.DataFrame(
            {
                "task": [task],
                "answer": [np.array(["alpha", "beta"], dtype=object)],
                "predicted_answer": ["the answer is alpha"],
            }
        )

        assert calculate_metrics(df)[task]["string_match"] == expected

    def test_family_routing_uses_name_prefix_not_substring(self):
        """`niah_multiquery` must not route to qa just because it ends in 'query'."""
        df = pd.DataFrame(
            {
                "task": ["niah_multiquery"],
                "answer": [np.array(["alpha", "beta"], dtype=object)],
                "predicted_answer": ["the answer is alpha"],
            }
        )

        assert calculate_metrics(df)["niah_multiquery"]["string_match"] == 50.0
