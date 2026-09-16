"""Tests for benchmark base functionality."""

from typing import Any, Dict
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from benchmark import Benchmark


class MockBenchmark(Benchmark):
    """Mock benchmark class for testing the base functionality."""

    all_datasets = ["test_task1", "test_task2"]
    benchmark_name = "test_benchmark"
    huggingface_dataset_id = "mock/test_dataset"

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Mock evaluation that just counts samples."""
        return {"total_samples": len(results_df), "mock_score": 85.5}


@pytest.fixture
def mock_dataset_df() -> pd.DataFrame:
    """Create a mock dataset DataFrame for testing."""
    return pd.DataFrame(
        {
            "context": ["Context 1", "Context 1", "Context 2"],
            "question": ["Question 1a", "Question 1b", "Question 2"],
            "task": ["test_task1", "test_task1", "test_task2"],
            "answers": [["Answer 1a"], ["Answer 1b"], ["Answer 2"]],
            "all_classes": [[], [], []],
            "answer_prefix": ["Answer: ", "Answer: ", "Answer: "],
            "max_new_tokens": [10, 10, 10],
        }
    )


@pytest.fixture
def mock_adapter() -> Mock:
    """Create a mock adapter for testing."""
    adapter = Mock()

    def mock_process_request(request, generation_kwargs, request_kwargs):
        """Mock processing that returns responses based on questions."""
        if isinstance(request.questions, list):
            responses = [f"Response to {q}" for q in request.questions]
        else:
            responses = f"Response to {request.questions}"
        return Mock(responses=responses)

    adapter.process_request.side_effect = mock_process_request
    return adapter


class TestBenchmarkBase:
    """Test the base Benchmark class functionality."""

    def test_benchmark_initialization(self):
        """Test basic benchmark initialization."""
        benchmark = MockBenchmark()
        assert benchmark.subsets_to_run == ["test_task1", "test_task2"]
        assert benchmark.benchmark_name == "test_benchmark"
        assert benchmark.huggingface_dataset_id == "mock/test_dataset"

    def test_benchmark_initialization_with_subsets(self):
        """Test benchmark initialization with custom subsets."""
        benchmark = MockBenchmark(subsets_to_run=["test_task1"])
        assert benchmark.subsets_to_run == ["test_task1"]

    def test_benchmark_subset_validation_valid(self):
        """Test that valid subset validation works correctly."""
        benchmark = MockBenchmark(subsets_to_run=["test_task1"])
        assert benchmark.subsets_to_run == ["test_task1"]

        benchmark = MockBenchmark(subsets_to_run=["test_task1", "test_task2"])
        assert benchmark.subsets_to_run == ["test_task1", "test_task2"]

    def test_benchmark_subset_validation_invalid(self):
        """Test that invalid subset validation raises error."""
        with pytest.raises(ValueError, match="Invalid subsets"):
            MockBenchmark(subsets_to_run=["invalid_task"])

        with pytest.raises(ValueError, match="Invalid subsets"):
            MockBenchmark(subsets_to_run=["test_task1", "invalid_task"])

    def test_benchmark_initialization_missing_attributes(self):
        """Test that missing required attributes raise errors."""

        class IncompleteBenchmark(Benchmark):
            # Missing all required attributes, but implement abstract method
            def post_run_evaluate(self, results_df):
                return {}

        with pytest.raises(ValueError, match="must define all_datasets"):
            IncompleteBenchmark()

    def test_get_available_datasets(self):
        """Test getting available datasets."""
        benchmark = MockBenchmark()
        datasets = benchmark.get_available_datasets()
        assert datasets == ["test_task1", "test_task2"]

        # Ensure it returns a copy, not the original
        datasets.append("new_task")
        assert benchmark.all_datasets == ["test_task1", "test_task2"]

    def test_validate_subsets_method(self):
        """Test the _validate_subsets method directly."""
        benchmark = MockBenchmark()

        # Valid subsets should not raise
        benchmark._validate_subsets(["test_task1"])
        benchmark._validate_subsets(["test_task1", "test_task2"])

        # Invalid subsets should raise
        with pytest.raises(ValueError, match="Invalid subsets"):
            benchmark._validate_subsets(["invalid_task"])

    @patch("benchmark.base.load_dataset")
    def test_load_datasets_with_task_column(self, mock_load_dataset, mock_dataset_df):
        """Test dataset loading functionality when task column exists."""
        # Mock the datasets load_dataset function
        mock_dataset = Mock()
        mock_dataset.to_pandas.return_value = mock_dataset_df
        mock_load_dataset.return_value = mock_dataset

        benchmark = MockBenchmark(subsets_to_run=["test_task1"])
        df = benchmark._load_datasets()

        # Should filter to only test_task1
        assert len(df) == 2  # Two rows with test_task1
        assert all(df["task"] == "test_task1")
        mock_load_dataset.assert_called_once_with("mock/test_dataset", split="test")

    @patch("benchmark.base.load_dataset")
    def test_load_datasets_without_task_column(self, mock_load_dataset):
        """Test dataset loading when task column doesn't exist."""
        # Create dataset without task column
        df_without_task = pd.DataFrame(
            {
                "context": ["Context 1"],
                "question": ["Question 1"],
                "answers": [["Answer 1"]],
                "all_classes": [[]],
            }
        )

        mock_dataset = Mock()
        mock_dataset.to_pandas.return_value = df_without_task
        mock_load_dataset.return_value = mock_dataset

        benchmark = MockBenchmark()
        df = benchmark._load_datasets()

        # Should return the full dataset since no task column to filter on
        assert len(df) == 1
        assert "context" in df.columns

    @patch("benchmark.base.load_dataset")
    @pytest.mark.skip(
        reason="Skipping since we removed error handling for better error dumps"
    )
    def test_load_datasets_error_handling(self, mock_load_dataset):
        """Test dataset loading error handling."""
        mock_load_dataset.side_effect = Exception("Dataset not found")

        benchmark = MockBenchmark()
        with pytest.raises(Exception, match="Failed to load dataset mock/test_dataset"):
            benchmark._load_datasets()

    def test_validate_dataset_size_small(self, mock_dataset_df):
        """Test dataset size validation with small dataset."""
        benchmark = MockBenchmark()

        # Small dataset should not warn
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            benchmark._validate_dataset_size(mock_dataset_df)

            # Filter warnings to only UserWarnings about dataset size
            size_warnings = [
                warning
                for warning in w
                if "Repository not expected to handle large datasets"
                in str(warning.message)
            ]
        assert len(size_warnings) == 0

    def test_validate_dataset_size_large(self, mock_dataset_df):
        """Test dataset size validation with large dataset."""
        benchmark = MockBenchmark()

        # Create large dataset
        large_df = pd.concat([mock_dataset_df] * 5000)  # ~15000 rows

        with pytest.warns(
            UserWarning, match="Repository not expected to handle large datasets"
        ):
            benchmark._validate_dataset_size(large_df)

    def test_process_all_requests_multiple_questions_per_context(
        self, mock_adapter, mock_dataset_df
    ):
        """Test processing requests with multiple questions per context."""
        benchmark = MockBenchmark()
        results_df = benchmark._process_all_requests(
            mock_adapter, mock_dataset_df, {}, {}
        )

        # Check that predicted_answer column was added
        assert "predicted_answer" in results_df.columns
        assert len(results_df) == 3

        # Check that adapter was called for each unique context
        assert mock_adapter.process_request.called

        # Verify responses are assigned correctly
        for answer in results_df["predicted_answer"]:
            assert answer.startswith("Response to")

    def test_process_all_requests_single_question_response(self, mock_dataset_df):
        """Test processing when adapter returns single string response."""
        # Create adapter that returns single string for multiple questions
        single_response_adapter = Mock()
        single_response_adapter.process_request.return_value = Mock(
            responses="Single response"
        )

        benchmark = MockBenchmark()
        results_df = benchmark._process_all_requests(
            single_response_adapter, mock_dataset_df, {}, {}
        )

        # Should handle single response for multiple questions
        context1_rows = results_df[results_df["context"] == "Context 1"]
        assert len(context1_rows) == 2
        assert all(context1_rows["predicted_answer"] == "Single response")

    @pytest.mark.skip(
        reason="Skipping since we removed error handling for better error dumps"
    )
    def test_process_all_requests_error_handling(self, mock_dataset_df):
        """Test error handling during request processing."""
        # Create adapter that throws errors
        failing_adapter = Mock()
        failing_adapter.process_request.side_effect = Exception("Adapter failed")

        benchmark = MockBenchmark()
        results_df = benchmark._process_all_requests(
            failing_adapter, mock_dataset_df, {}, {}
        )

        # Should handle errors gracefully and continue processing
        assert len(results_df) == 3
        assert "predicted_answer" in results_df.columns

        # Should have empty responses for failed contexts
        assert all(answer == "" for answer in results_df["predicted_answer"])

    def test_process_all_requests_memory_cleanup(self, mock_adapter, mock_dataset_df):
        """Test that memory cleanup is called."""
        with patch("benchmark.base.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True

            benchmark = MockBenchmark()
            benchmark._process_all_requests(mock_adapter, mock_dataset_df, {}, {})

            # Should call empty_cache for each context group
            assert mock_torch.cuda.empty_cache.called


class TestGenerationKwargsIsolation:
    """`_process_all_requests` must not mutate the caller's generation_kwargs.

    It used to write the per-group minimum back into the shared dict, so the next
    iteration read the value the previous one had written and max_new_tokens ratcheted
    monotonically downward across context groups.  Benchmarks that carry per-task limits
    (longbench, infinite_bench, loogle, ruler) would silently truncate generation for
    every task after the shortest one in a multi-subset run.
    """

    @staticmethod
    def _df_with_varying_max_new_tokens() -> pd.DataFrame:
        # groupby("context") sorts by the context string, so "A" (32) is processed before
        # "B" (512): exactly the order that used to cap B at 32.
        return pd.DataFrame(
            {
                "context": ["A", "B", "C"],
                "question": ["qa", "qb", "qc"],
                "task": ["t", "t", "t"],
                "answers": [["a"], ["b"], ["c"]],
                "answer_prefix": ["Answer: "] * 3,
                "max_new_tokens": [32, 512, 256],
            }
        )

    @staticmethod
    def _recording_adapter(seen: Dict[str, int]) -> Mock:
        adapter = Mock()

        def process(request, generation_kwargs, request_kwargs):
            seen[request.context] = generation_kwargs["max_new_tokens"]
            return Mock(responses=[f"r-{q}" for q in request.questions])

        adapter.process_request.side_effect = process
        return adapter

    def test_per_group_max_new_tokens_does_not_ratchet(self):
        seen: Dict[str, int] = {}
        benchmark = MockBenchmark()
        benchmark._process_all_requests(
            self._recording_adapter(seen),
            self._df_with_varying_max_new_tokens(),
            {},
            {},
        )
        # Each group must get its OWN row value, not the running minimum.
        assert seen == {"A": 32, "B": 512, "C": 256}

    def test_caller_generation_kwargs_not_mutated(self):
        seen: Dict[str, int] = {}
        caller_kwargs: Dict[str, Any] = {"temperature": 0.0}
        benchmark = MockBenchmark()
        benchmark._process_all_requests(
            self._recording_adapter(seen),
            self._df_with_varying_max_new_tokens(),
            caller_kwargs,
            {},
        )
        # The caller's dict must come back exactly as it was handed in.
        assert caller_kwargs == {"temperature": 0.0}

    def test_explicit_cap_still_applies_to_every_group(self):
        seen: Dict[str, int] = {}
        benchmark = MockBenchmark()
        benchmark._process_all_requests(
            self._recording_adapter(seen),
            self._df_with_varying_max_new_tokens(),
            {"max_new_tokens": 64},
            {},
        )
        # An explicit budget still caps each group: min(64, row value), per group.
        assert seen == {"A": 32, "B": 64, "C": 64}

    def test_other_generation_kwargs_are_forwarded(self):
        forwarded: Dict[str, Any] = {}
        adapter = Mock()

        def process(request, generation_kwargs, request_kwargs):
            forwarded.update(generation_kwargs)
            return Mock(responses=[f"r-{q}" for q in request.questions])

        adapter.process_request.side_effect = process
        MockBenchmark()._process_all_requests(
            adapter,
            self._df_with_varying_max_new_tokens(),
            {"temperature": 0.7, "do_sample": True},
            {},
        )
        assert forwarded["temperature"] == 0.7
        assert forwarded["do_sample"] is True

    def test_absent_max_new_tokens_column_defers_to_the_caller(self):
        # A benchmark whose dataset specifies no per-row budget (LOFT, matching upstream,
        # which imposes no output cap) must let the caller's value through unchanged
        # rather than KeyError or silently cap.
        seen: Dict[str, int] = {}
        df = self._df_with_varying_max_new_tokens().drop(columns=["max_new_tokens"])
        MockBenchmark()._process_all_requests(
            self._recording_adapter(seen), df, {"max_new_tokens": 1024}, {}
        )
        assert set(seen.values()) == {1024}


class TestPromptAnswerPrefixHook:
    """The prompt fix must be pinned at the call site, not only as a class constant."""

    @staticmethod
    def _df() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "context": ["C"],
                "question": ["Q"],
                "task": ["test_task1"],
                "answers": [["A"]],
                "answer_prefix": ["Final Answer: "],
                "max_new_tokens": [16],
            }
        )

    @staticmethod
    def _capture(store: Dict[str, Any]) -> Mock:
        adapter = Mock()

        def process(request, generation_kwargs, request_kwargs):
            store["answer_prefix"] = request.answer_prefix
            return Mock(responses=["r"])

        adapter.process_request.side_effect = process
        return adapter

    def test_prompt_prefix_is_blanked_on_the_request_when_disabled(self):
        class NoPrefix(MockBenchmark):
            prompt_includes_answer_prefix = False

        store: Dict[str, Any] = {}
        NoPrefix()._process_all_requests(self._capture(store), self._df(), {}, {})
        # Deleting the base.py hook would leave "Final Answer: " here and silently
        # re-prime the prompt, suppressing LOFT's chain-of-thought step.
        assert store["answer_prefix"] == ""

    def test_prompt_prefix_is_passed_through_by_default(self):
        store: Dict[str, Any] = {}
        MockBenchmark()._process_all_requests(self._capture(store), self._df(), {}, {})
        assert store["answer_prefix"] == "Final Answer: "
