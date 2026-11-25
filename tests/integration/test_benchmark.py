"""Integration tests for benchmark functionality with real adapters and end-to-end workflows."""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from benchmark import Benchmark, LongBench
from sparse_attention_hub.adapters import ModelAdapterHF, Request, RequestResponse
from sparse_attention_hub.sparse_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
    LocalMaskerConfig,
)


class MockBenchmark(Benchmark):
    """Mock benchmark class for integration testing."""

    all_datasets = ["test_task1", "test_task2"]
    benchmark_name = "test_benchmark"
    huggingface_dataset_id = "mock/test_dataset"

    def post_run_evaluate(self, results_df: pd.DataFrame):
        """Mock evaluation that just counts samples."""
        return {"total_samples": len(results_df), "mock_score": 85.5}


@pytest.fixture
def temp_result_dir():
    """Create a temporary directory for test results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def masker_config():
    """Create a masker configuration for testing."""
    return LocalMaskerConfig(window_size=5)


@pytest.fixture
def sparse_attention_config(masker_config):
    """Create a sparse attention configuration for testing."""
    return ResearchAttentionConfig(masker_configs=[masker_config])


class TestBenchmarkAdapterIntegration:
    """Test benchmark integration with real adapter interfaces."""

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_benchmark_with_real_adapter_interface(
        self,
        mock_tokenizer_class,
        mock_model_class,
        sparse_attention_config,
        temp_result_dir,
    ):
        """Test benchmark with real ModelAdapterHF interface."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Create real adapter
        adapter = ModelAdapterHF(
            model_name="test-model",
            sparse_attention_config=sparse_attention_config,
            device="cpu",
        )

        # Mock the process_request method to avoid actual model inference
        def mock_process_request(
            request: Request,
            generation_kwargs: Dict[str, Any],
            request_kwargs: Dict[str, Any],
            **kwargs: Dict[str, Any],
        ) -> RequestResponse:
            if isinstance(request.questions, list):
                return RequestResponse(
                    responses=[
                        f"Mock response {i}" for i in range(len(request.questions))
                    ]
                )
            else:
                return RequestResponse(responses="Mock response")

        adapter.process_request = mock_process_request

        # Test with mock benchmark
        benchmark = MockBenchmark(subsets_to_run=["test_task1"])
        mock_data = pd.DataFrame(
            {
                "context": ["Test context"],
                "question": ["Test question"],
                "task": ["test_task1"],
                "answers": [["Test answer"]],
                "all_classes": [[]],
                "answer_prefix": ["Answer: "],
                "max_new_tokens": [10],
            }
        )

        with patch.object(benchmark, "_load_datasets", return_value=mock_data):
            results = benchmark.run_benchmark(adapter, temp_result_dir)

        assert "total_samples" in results
        assert "mock_score" in results

        # Verify files were created
        result_path = Path(temp_result_dir)
        assert (result_path / "raw_results.csv").exists()
        assert (result_path / "metrics.json").exists()

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_dense_only_adapter_integration(
        self, mock_tokenizer_class, mock_model_class, temp_result_dir
    ):
        """Test benchmark with dense-only adapter (no sparse attention)."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Create adapter without sparse attention
        adapter = ModelAdapterHF(
            model_name="test-model",
            sparse_attention_config=None,  # Dense-only mode
            device="cpu",
        )

        # Mock the process_request method
        def mock_process_request(
            request: Request,
            generation_kwargs: Dict[str, Any],
            request_kwargs: Dict[str, Any],
            **kwargs: Dict[str, Any],
        ) -> RequestResponse:
            return RequestResponse(
                responses=["Dense response"] * len(request.questions)
            )

        adapter.process_request = mock_process_request

        # Test with benchmark
        benchmark = MockBenchmark()
        mock_data = pd.DataFrame(
            {
                "context": ["Context 1", "Context 2"],
                "question": ["Question 1", "Question 2"],
                "task": ["test_task1", "test_task2"],
                "answers": [["Answer 1"], ["Answer 2"]],
                "all_classes": [[], []],
                "answer_prefix": ["Answer: ", "Answer: "],
                "max_new_tokens": [10, 10],
            }
        )

        with patch.object(benchmark, "_load_datasets", return_value=mock_data):
            results = benchmark.run_benchmark(adapter, temp_result_dir)

        # Should work even without sparse attention
        assert results["total_samples"] == 2
        assert results["mock_score"] == 85.5


class TestRequestResponseIntegration:
    """Test Request/RequestResponse interface compatibility."""

    def test_request_response_compatibility(self):
        """Test that Request/RequestResponse interface works correctly."""
        # Test single question
        request = Request(
            context="Test context",
            questions="Single question",
            answer_prefix="Answer: ",
        )
        assert request.context == "Test context"
        assert request.questions == "Single question"

        # Test multiple questions
        questions = ["Question 1", "Question 2"]
        request = Request(
            context="Test context", questions=questions, answer_prefix="Answer: "
        )
        assert request.questions == questions

        # Test response formats
        single_response = RequestResponse(responses="Single response")
        assert single_response.responses == "Single response"

        multi_response = RequestResponse(responses=["Response 1", "Response 2"])
        assert len(multi_response.responses) == 2


class TestEndToEndBenchmarkWorkflow:
    """Test complete end-to-end benchmark workflows."""

    def test_complete_benchmark_workflow(self, temp_result_dir):
        """Test complete benchmark workflow from start to finish."""
        # Create mock adapter
        mock_adapter = Mock()

        def mock_process_request(
            request: Request,
            generation_kwargs: Dict[str, Any],
            request_kwargs: Dict[str, Any],
            **kwargs: Dict[str, Any],
        ) -> RequestResponse:
            # Simulate context-aware responses
            context_id = "ctx1" if "Context 1" in request.context else "ctx2"
            if isinstance(request.questions, list):
                responses = [
                    f"{context_id}_response_{i}" for i in range(len(request.questions))
                ]
            else:
                responses = f"{context_id}_response_single"
            return RequestResponse(responses=responses)

        mock_adapter.process_request = mock_process_request

        # Create benchmark
        benchmark = MockBenchmark(subsets_to_run=["test_task1"])

        # Mock dataset with multiple contexts and questions
        mock_data = pd.DataFrame(
            {
                "context": ["Context 1", "Context 1", "Context 2"],
                "question": ["Q1a", "Q1b", "Q2"],
                "task": ["test_task1", "test_task1", "test_task1"],
                "answers": [["A1a"], ["A1b"], ["A2"]],
                "all_classes": [[], [], []],
                "answer_prefix": ["Answer: ", "Answer: ", "Answer: "],
                "max_new_tokens": [10, 10, 10],
            }
        )

        with patch.object(benchmark, "_load_datasets", return_value=mock_data):
            # Run complete benchmark
            results = benchmark.run_benchmark(mock_adapter, temp_result_dir)

            # Verify results structure
            assert "total_samples" in results
            assert "mock_score" in results
            assert results["total_samples"] == 3

            # Verify files were created with correct content
            result_path = Path(temp_result_dir)

            # Check CSV file
            csv_file = result_path / "raw_results.csv"
            assert csv_file.exists()
            saved_df = pd.read_csv(csv_file)
            assert len(saved_df) == 3
            assert "predicted_answer" in saved_df.columns
            assert all(saved_df["predicted_answer"].str.contains("response"))

            # Check JSON file
            json_file = result_path / "metrics.json"
            assert json_file.exists()
            import json

            with open(json_file) as f:
                saved_metrics = json.load(f)
            assert saved_metrics == results

    def test_benchmark_context_grouping_efficiency(self, temp_result_dir):
        """Test that context grouping works efficiently."""
        # Create adapter that tracks how many times it's called
        call_count = 0
        contexts_seen = set()

        def counting_process_request(
            request: Request,
            generation_kwargs: Dict[str, Any],
            request_kwargs: Dict[str, Any],
            **kwargs: Dict[str, Any],
        ) -> RequestResponse:
            nonlocal call_count, contexts_seen
            call_count += 1
            contexts_seen.add(request.context)

            num_questions = (
                len(request.questions) if isinstance(request.questions, list) else 1
            )
            return RequestResponse(
                responses=[f"Response {i}" for i in range(num_questions)]
            )

        mock_adapter = Mock()
        mock_adapter.process_request = counting_process_request

        # Create dataset with 6 questions across 2 contexts
        mock_data = pd.DataFrame(
            {
                "context": ["Context A"] * 3 + ["Context B"] * 3,
                "question": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],
                "task": ["test_task1"] * 6,
                "answers": [["A1"], ["A2"], ["A3"], ["A4"], ["A5"], ["A6"]],
                "all_classes": [[], [], [], [], [], []],
                "answer_prefix": ["Answer: "] * 6,
                "max_new_tokens": [10, 10, 10, 10, 10, 10],
            }
        )

        benchmark = MockBenchmark()

        with patch.object(benchmark, "_load_datasets", return_value=mock_data):
            benchmark.run_benchmark(mock_adapter, temp_result_dir)

            # Should only call adapter twice (once per unique context)
            assert call_count == 2
            assert len(contexts_seen) == 2
            assert "Context A" in contexts_seen
            assert "Context B" in contexts_seen


class TestLongBenchIntegration:
    """Integration tests specific to LongBench."""

    def test_longbench_with_real_calculate_metrics(self):
        """Test LongBench with real calculate_metrics functions (mocked)."""
        longbench = LongBench()

        # Create realistic LongBench data structure
        mock_results = pd.DataFrame(
            {
                "task": ["narrativeqa", "narrativeqa", "trec_e", "trec_e"],
                "predicted_answer": [
                    "The answer is Paris",
                    "The answer is London",
                    "A",
                    "B",
                ],
                "answers": [["Paris"], ["London"], ["A"], ["B"]],
                "all_classes": [[], [], ["A", "B"], ["A", "B"]],
                "length": [
                    None,
                    None,
                    3000,
                    5000,
                ],  # Only extended datasets have length
                "max_new_tokens": [10, 10, 10, 10],
            }
        )

        # Mock the calculation functions to return realistic scores
        with patch("benchmark.longbench.longbench.calculate_metrics") as mock_calc:
            with patch(
                "benchmark.longbench.longbench.calculate_metrics_e"
            ) as mock_calc_e:
                mock_calc.return_value = 75.0  # Standard dataset score
                mock_calc_e.return_value = {
                    "0-4k": 80.0,
                    "4-8k": 85.0,
                    "8k+": 70.0,
                }  # Extended scores

                results = longbench.post_run_evaluate(mock_results)

                # Verify both calculation functions were called
                assert mock_calc.called
                assert mock_calc_e.called

                # Verify structure matches expected LongBench output
                assert "task_scores" in results
                assert "standard_overall_score" in results
                assert "extended_overall_scores" in results
                assert "overall_score" in results
                assert "summary" in results


class TestErrorHandlingIntegration:
    """Test error handling in real integration scenarios."""

    @pytest.mark.skip(
        reason="Skipping since we removed error handling for better error dumps"
    )
    def test_adapter_failure_recovery(self, temp_result_dir):
        """Test benchmark recovery when adapter fails intermittently."""
        # Create adapter that fails on certain contexts
        def failing_process_request(
            request: Request,
            generation_kwargs: Dict[str, Any],
            request_kwargs: Dict[str, Any],
            **kwargs: Dict[str, Any],
        ) -> RequestResponse:
            if "fail" in request.context.lower():
                raise Exception("Simulated adapter failure")
            return RequestResponse(
                responses=["Success response"] * len(request.questions)
            )

        mock_adapter = Mock()
        mock_adapter.process_request = failing_process_request

        # Create dataset with some contexts that will cause failures
        mock_data = pd.DataFrame(
            {
                "context": ["Good context", "FAIL context", "Another good context"],
                "question": ["Q1", "Q2", "Q3"],
                "task": ["test_task1", "test_task1", "test_task1"],
                "answers": [["A1"], ["A2"], ["A3"]],
                "all_classes": [[], [], []],
                "answer_prefix": ["Answer: ", "Answer: ", "Answer: "],
                "max_new_tokens": [10, 10, 10],
            }
        )

        benchmark = MockBenchmark()

        with patch.object(benchmark, "_load_datasets", return_value=mock_data):
            # Should complete without crashing despite adapter failures
            results = benchmark.run_benchmark(mock_adapter, temp_result_dir)

            # Should still return results
            assert "total_samples" in results
            assert results["total_samples"] == 3

            # Check that CSV was created with partial results
            csv_file = Path(temp_result_dir) / "raw_results.csv"
            assert csv_file.exists()
            saved_df = pd.read_csv(csv_file)

            # Should have some successful responses and some empty/null ones for failures
            assert "Success response" in saved_df["predicted_answer"].values
            # Failed responses could be empty string or NaN depending on error handling
            failed_responses = saved_df["predicted_answer"].isna() | (
                saved_df["predicted_answer"] == ""
            )
            assert failed_responses.any(), "Should have some failed responses"

    def test_file_creation_error_handling(self):
        """Test handling when result files cannot be created."""
        mock_adapter = Mock()
        mock_adapter.process_request.return_value = RequestResponse(
            responses=["Test response"]
        )

        benchmark = MockBenchmark()
        mock_data = pd.DataFrame(
            {
                "context": ["Test"],
                "question": ["Q"],
                "task": ["test_task1"],
                "answers": [["A"]],
                "all_classes": [[]],
                "answer_prefix": ["Answer: "],
                "max_new_tokens": [10],
            }
        )

        # Patch pandas to_csv to simulate a file write failure,
        # making the test OS-independent (avoids Windows path quirks).
        with patch.object(benchmark, "_load_datasets", return_value=mock_data):
            with patch(
                "pandas.DataFrame.to_csv",
                side_effect=OSError("Permission denied: cannot write to path"),
            ):
                with pytest.raises((OSError, PermissionError, FileNotFoundError)):
                    benchmark.run_benchmark(mock_adapter, "/invalid/nonexistent/path")