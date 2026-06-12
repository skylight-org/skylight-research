"""Base classes and interfaces for benchmark evaluation."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import os
import warnings
import csv
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import torch
import sys
import dataclasses

from sparse_attention_hub.adapters.base import Request, RequestResponse, ModelAdapter
from .utils import save_dataframe_to_csv, make_serializable


class Benchmark(ABC):
    """Abstract base class for benchmark evaluation.

    This class provides a framework for running benchmarks with sparse attention models.
    Subclasses must override the class attributes and implement the post_run_evaluate method.

    Todo:
        * Add support for generation_kwargs to be passed to adapter.process_request()
        * Add support for answer_prefix handling in request processing
        * Add support for passing max_context_length to the adapter.process_request()

    Attributes:
        all_datasets: List of all available datasets for this benchmark (to be overridden)
        benchmark_name: Name identifier for the benchmark (to be overridden)
        huggingface_dataset_id: HuggingFace dataset identifier (to be overridden)
        subsets_to_run: List of dataset names to run (subset of all_datasets)

    Example:
        >>> class MyBenchmark(Benchmark):
        ...     all_datasets = ["task1", "task2"]
        ...     benchmark_name = "my_benchmark"
        ...     huggingface_dataset_id = "my_org/my_dataset"
        ...     
        ...     def post_run_evaluate(self, results_df):
        ...         return {"accuracy": 0.95}
        >>> 
        >>> benchmark = MyBenchmark(subsets_to_run=["task1"])
        >>> results = benchmark.run_benchmark(adapter, result_dir="/path/to/results", generation_kwargs={"max_new_tokens": 50}, request_kwargs={"max_context_length": 1024})
    """

    # Class attributes to be overridden in subclasses
    all_datasets: List[str] = []
    benchmark_name: str = ""
    huggingface_dataset_id: str = ""

    def __init__(self, subsets_to_run: Optional[List[str]] = None) -> None:
        """Initialize benchmark with subset of datasets to run.

        Args:
            subsets_to_run: Optional list of dataset names to run. If None, uses all_datasets.

        Raises:
            ValueError: If any subset in subsets_to_run is not in all_datasets.
        """
        if not self.all_datasets:
            raise ValueError(f"Subclass {self.__class__.__name__} must define all_datasets")
        if not self.benchmark_name:
            raise ValueError(f"Subclass {self.__class__.__name__} must define benchmark_name")
        if not self.huggingface_dataset_id:
            raise ValueError(f"Subclass {self.__class__.__name__} must define huggingface_dataset_id")

        if subsets_to_run is None:
            self.subsets_to_run = self.all_datasets.copy()
        else:
            self._validate_subsets(subsets_to_run)
            self.subsets_to_run = subsets_to_run.copy()

    def _validate_subsets(self, subsets: List[str]) -> None:
        """Validate that requested subsets exist in all_datasets.

        Args:
            subsets: List of subset names to validate.

        Raises:
            ValueError: If any subset is not in all_datasets.
        """
        invalid_subsets: set[str] = set(subsets) - set(self.all_datasets)
        if invalid_subsets:
            raise ValueError(
                f"Invalid subsets: {invalid_subsets}. "
                f"Available datasets: {self.all_datasets}"
            )

    def get_available_datasets(self) -> List[str]:
        """Return list of all available datasets for this benchmark.

        Returns:
            Copy of all_datasets list.
        """
        return self.all_datasets.copy()

    def _load_datasets(self) -> pd.DataFrame:
        """Load and combine all specified datasets into a single DataFrame.

        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.

        Raises:
            Exception: If dataset loading fails.
        """
        try:
            # Load the full dataset
            dataset = load_dataset(self.huggingface_dataset_id, split="test")
            df: pd.DataFrame = dataset.to_pandas()
            
            # Filter to only include subsets we want to run
            if "task" in df.columns:
                # Filter by task column if it exists
                df = df[df["task"].isin(self.subsets_to_run)]
            else:
                # If no task column, assume the dataset only contains our subsets
                # This is a simplified assumption based on user guidance
                pass
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to load dataset {self.huggingface_dataset_id}: {str(e)}")

    def _validate_dataset_size(self, df: pd.DataFrame) -> None:
        """Validate dataset size and warn if too large.

        Args:
            df: DataFrame to validate.
        """
        if len(df) > 10000:
            warnings.warn(
                f"Dataset has {len(df)} rows (>10K). Repository not expected to handle "
                "large datasets. If needed, request this feature.",
                UserWarning
            )

    def _process_all_requests(
        self, 
        adapter: ModelAdapter, 
        dataset_df: pd.DataFrame,
        generation_kwargs: Dict[str, Any],
        request_kwargs: Dict[str, Any]
    ) -> pd.DataFrame:
        """Process all samples through the model adapter using context grouping for efficiency.

        Args:
            adapter: Model adapter implementing ModelHubAdapterInterface.
            dataset_df: DataFrame containing the benchmark dataset.

        Returns:
            DataFrame with added 'predicted_answer' column.
        """
        max_requests = request_kwargs.get("max_requests", sys.maxsize)

        # Initialize predicted_answer column
        dataset_df = dataset_df.copy()
        dataset_df["predicted_answer"] = None

        # Truncate dataset to max_requests
        dataset_df = dataset_df.head(max_requests)
        
        
        # Group by context for efficiency (following HashAttention approach)
        df_context = dataset_df.groupby("context")
        
        # Track total questions processed
        total_questions = len(dataset_df)
        questions_processed = 0

        pbar = tqdm(df_context, desc="Processing contexts", total=dataset_df["context"].nunique())
        for idx, (context, df_group) in enumerate(pbar):
            questions: List[str] = df_group["question"].to_list()
            questions_processed += len(questions)
            # Update progress bar suffix to show number of questions
            pbar.set_postfix({
                'ctx': idx+1,
                'q': len(questions),
                'total_q': f"{questions_processed}/{total_questions}"
            })
            # Create request using current adapter interface (simplified)
            answer_prefix = df_group["answer_prefix"].iloc[0]
            request: Request = Request(context=context, questions=questions, answer_prefix=answer_prefix)
            
            # using the first record for getting max new tokens
            max_new_tokens = df_group["max_new_tokens"].iloc[0]
            param_max_new_tokens = generation_kwargs.get("max_new_tokens", sys.maxsize)
            generation_kwargs["max_new_tokens"] = min(param_max_new_tokens, max_new_tokens)
            
            # Process through adapter
            response: RequestResponse = adapter.process_request(request, generation_kwargs, request_kwargs)
            
            # Assign responses back to DataFrame
            if isinstance(response.responses, list):
                dataset_df.loc[df_group.index, "predicted_answer"] = response.responses
            else:
                # Single response case
                dataset_df.loc[df_group.index, "predicted_answer"] = [response.responses] * len(df_group)
            
            # Memory cleanup for large contexts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return dataset_df

    def run_benchmark(
        self, 
        adapter: ModelAdapter, 
        result_dir: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        request_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Main orchestration method for running complete benchmark.

        Args:
            adapter: Model adapter implementing ModelHubAdapterInterface.
            result_dir: Directory to save results.
            generation_kwargs: Parameters for model inference/generation.
            request_kwargs: Parameters for request processing (e.g., max_context_length).

        Returns:
            Dictionary containing evaluation results and metadata.
        """
        # Set default values if not provided
        if generation_kwargs is None:
            generation_kwargs = {}
        if request_kwargs is None:
            request_kwargs = {}
            
        # Validate result directory instead of creating it.
        # On Windows, a path like '/invalid/path' is root-relative (not fully qualified),
        # which is ambiguous and should be rejected early.
        result_path: Path = Path(result_dir)
        if os.name == "nt" and result_dir.startswith(("/", "\\")) and not result_path.is_absolute():
            raise FileNotFoundError(
                f"Invalid Windows path (root-relative, not fully qualified): {result_dir}"
            )

        if not result_path.exists():
            raise FileNotFoundError(f"Result directory does not exist: {result_dir}")
        if not result_path.is_dir():
            raise NotADirectoryError(f"Result path is not a directory: {result_dir}")
        # Load datasets
        print(f"Loading {self.benchmark_name} datasets: {self.subsets_to_run}")
        dataset_df: pd.DataFrame = self._load_datasets()
        print(f"Loaded {len(dataset_df)} samples")
        # Validate dataset size
        self._validate_dataset_size(dataset_df)
        
        # Process all requests through the adapter
        print("Processing requests through adapter...")
        results_df: pd.DataFrame = self._process_all_requests(adapter, dataset_df, generation_kwargs, request_kwargs)
        
        # Compute evaluation metrics
        print("Computing evaluation metrics...")
        metrics: Dict[str, Any] = self.post_run_evaluate(results_df)
        
        # Save results
        raw_results_path: Path = result_path / "raw_results.csv"
        save_dataframe_to_csv(results_df, str(raw_results_path), index=False)
        print(f"Saved raw results to {raw_results_path}")
        
        # Save metrics
        metrics_path: Path = result_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")
        
        # Save configuration parameters
        config_path: Path = result_path / "config.json"
        
        config_data = {
            "model_kwargs": make_serializable(getattr(adapter, 'model_kwargs', {})),
            "tokenizer_kwargs": make_serializable(getattr(adapter, 'tokenizer_kwargs', {})),
            "sparse_attention_config": make_serializable(getattr(adapter, 'sparse_attention_config', None)),
            "generation_kwargs": make_serializable(generation_kwargs),
            "request_kwargs": make_serializable(request_kwargs),
            "benchmark_name": self.benchmark_name,
            "subsets_to_run": self.subsets_to_run,
            "huggingface_dataset_id": self.huggingface_dataset_id
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        print(f"Saved configuration to {config_path}")
        
        return metrics

    @abstractmethod
    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics on benchmark results.

        This method must be implemented by subclasses to provide benchmark-specific
        evaluation logic.

        Args:
            results_df: DataFrame containing input data and model outputs with columns:
                - context: The input context
                - question: The input question  
                - predicted_answer: Model's predicted answer
                - Plus any other columns from the original dataset

        Returns:
            Dictionary containing computed metrics (e.g., {"accuracy": 0.95, "f1": 0.88}).
        """
        pass



