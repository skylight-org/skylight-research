"""Configuration classes and utilities for BenchmarkExecutor.

This module provides configuration dataclasses and factory functions for orchestrating
parallel benchmark execution across multiple GPUs using multiprocessing.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import multiprocessing

from sparse_attention_hub.sparse_attention.base import SparseAttentionConfig

from .base import Benchmark
from .utils import (
    sanitize_model_name,
    construct_result_directory,
    create_result_directory,
    validate_result_directory_path,
    ensure_result_directory,
    validate_gpu_availability,
    create_gpu_pool,
    acquire_gpu_from_pool,
    release_gpu_to_pool,
    validate_gpu_for_model,
    cleanup_gpu_memory
)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark execution.
    
    During stub generation, this expands to individual BenchmarkStub per subset.
    Each BenchmarkStub will have a single subset value for resumability.
    
    Attributes:
        benchmark_name: Name of the benchmark (e.g., "longbench", "mock_benchmark")
        subsets: Optional list of dataset subsets to run (e.g., ["narrativeqa", "qasper"])
                If None, runs all available subsets for the benchmark
    
    Example:
        >>> config = BenchmarkConfig("longbench", ["narrativeqa", "qasper"])
        >>> # This will create 2 BenchmarkStub instances during matrix expansion
    """
    
    benchmark_name: str
    subsets: Optional[List[str]] = None
    
    def __post_init__(self) -> None:
        """Validate benchmark configuration after initialization.
        
        Raises:
            ValueError: If benchmark_name is empty or invalid.
        """
        if not self.benchmark_name or not self.benchmark_name.strip():
            raise ValueError("benchmark_name cannot be empty")
        
        self.benchmark_name = self.benchmark_name.strip()
        
        if self.subsets is not None:
            # Remove empty strings and strip whitespace
            self.subsets = [subset.strip() for subset in self.subsets if subset.strip()]
            if not self.subsets:
                self.subsets = None
    
    def validate_with_benchmark_instance(self, benchmark: Benchmark) -> None:
        """Validate subsets against actual benchmark instance.
        
        Args:
            benchmark: Benchmark instance to validate against
            
        Raises:
            ValueError: If any subset is not available in the benchmark.
        """
        if self.subsets is not None:
            available_datasets: List[str] = benchmark.get_available_datasets()
            invalid_subsets: set[str] = set(self.subsets) - set(available_datasets)
            if invalid_subsets:
                raise ValueError(
                    f"Invalid subsets for {self.benchmark_name}: {invalid_subsets}. "
                    f"Available datasets: {available_datasets}"
                )


@dataclass
class AdapterConfig:
    """Base configuration for model adapters (sparse attention config is part of matrix).
    
    Note: sparse_attention_config is now part of the execution matrix, not base config.
    This allows testing different sparse attention configurations with the same adapter base.
    
    Attributes:
        adapter_name: Name of the adapter type (default: "huggingface")
        model_kwargs: Additional keyword arguments for model creation
        tokenizer_kwargs: Additional keyword arguments for tokenizer creation
        revision: Optional HuggingFace revision (branch, tag or commit sha) pinning both
            the model weights and the tokenizer, e.g. "stage1-step10000". Merged into
            model_kwargs/tokenizer_kwargs by the adapter, which makes it part of the
            ModelServer cache key.
    
    Example:
        >>> config = AdapterConfig(
        ...     model_kwargs={"torch_dtype": torch.bfloat16},
        ...     tokenizer_kwargs={"padding_side": "left"}
        ... )
    """
    
    adapter_name: str = "huggingface"
    model_kwargs: Optional[Dict[str, Any]] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
    revision: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Initialize default values and validate configuration."""
        if self.model_kwargs is None:
            self.model_kwargs = {}
        if self.tokenizer_kwargs is None:
            self.tokenizer_kwargs = {}
        
        if not self.adapter_name or not self.adapter_name.strip():
            raise ValueError("adapter_name cannot be empty")
        
        self.adapter_name = self.adapter_name.strip()


@dataclass
class BenchmarkStub:
    """A single benchmark execution task (model × sparse_config × benchmark × subset).
    
    This represents one atomic execution unit that will be processed by a worker process.
    Each stub corresponds to a specific combination of model, sparse attention config,
    benchmark, and subset (if applicable).
    
    Attributes:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        sparse_config_name: Human-readable name for sparse config (e.g., "dense", "streaming")
        sparse_attention_config: SparseAttentionConfig instance or None for dense
        benchmark_name: Name of the benchmark (e.g., "longbench")
        subset: Specific subset name (e.g., "narrativeqa") or None for full benchmark
        adapter_config: Base adapter configuration (without sparse attention)
        generation_kwargs: Parameters for model inference/generation
        request_kwargs: Parameters for request processing (e.g., max_context_length)
        result_dir: Full path to directory where results should be saved
    
    Example:
        >>> stub = BenchmarkStub(
        ...     model_name="microsoft/Phi-4-mini-instruct",
        ...     sparse_config_name="dense",
        ...     sparse_attention_config=None,
        ...     benchmark_name="longbench",
        ...     subset="narrativeqa",
        ...     adapter_config=adapter_config,
        ...     generation_kwargs={"max_new_tokens": 256},
        ...     request_kwargs={"max_context_length": 1024},
        ...     result_dir="/path/to/results/phi4_dense_longbench_narrativeqa"
        ... )
    """
    
    model_name: str
    sparse_config_name: str
    sparse_attention_config: Optional[SparseAttentionConfig]
    benchmark_name: str
    subset: Optional[str]
    adapter_config: AdapterConfig
    generation_kwargs: Dict[str, Any]
    request_kwargs: Dict[str, Any]
    result_dir: str
    
    def __post_init__(self) -> None:
        """Validate benchmark stub configuration."""
        if not self.model_name or not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        if not self.sparse_config_name or not self.sparse_config_name.strip():
            raise ValueError("sparse_config_name cannot be empty")
        if not self.benchmark_name or not self.benchmark_name.strip():
            raise ValueError("benchmark_name cannot be empty")
        if not self.result_dir or not self.result_dir.strip():
            raise ValueError("result_dir cannot be empty")
        
        # Trim whitespace
        self.model_name = self.model_name.strip()
        self.sparse_config_name = self.sparse_config_name.strip()
        self.benchmark_name = self.benchmark_name.strip()
        self.result_dir = self.result_dir.strip()
        
        if self.subset is not None:
            self.subset = self.subset.strip() if self.subset.strip() else None


@dataclass
class BenchmarkResult:
    """Result from a single benchmark execution.
    
    Attributes:
        stub: The benchmark stub that was executed
        metrics: Evaluation metrics returned by benchmark.run_benchmark()
        execution_time: Time taken to execute the benchmark in seconds
        success: Whether the benchmark executed successfully
        error_message: Error message if execution failed
    """
    
    stub: BenchmarkStub
    metrics: Optional[Dict[str, Any]]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class BenchmarkFailure:
    """Information about a failed benchmark execution.
    
    Attributes:
        stub: The benchmark stub that failed
        error_message: Description of the failure
        error_type: Type of error that occurred
        execution_time: Time spent before failure in seconds
    """
    
    stub: BenchmarkStub
    error_message: str
    error_type: str
    execution_time: float


@dataclass
class ExecutionProgress:
    """Tracks execution progress across all processes.
    
    Attributes:
        total_stubs: Total number of benchmark stubs to execute
        skipped_stubs: Number of stubs skipped due to existing results (resumability)
        completed_stubs: Number of stubs completed successfully
        failed_stubs: Number of stubs that failed execution
        current_executions: Mapping of GPU ID to current task description
    """
    
    total_stubs: int
    skipped_stubs: int = 0
    completed_stubs: int = 0
    failed_stubs: int = 0
    current_executions: Dict[int, str] = field(default_factory=dict)
    
    @property
    def pending_stubs(self) -> int:
        """Calculate number of stubs pending execution."""
        return self.total_stubs - self.skipped_stubs - self.completed_stubs - self.failed_stubs
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage (0-100)."""
        if self.total_stubs == 0:
            return 100.0
        return (self.completed_stubs + self.failed_stubs) / (self.total_stubs - self.skipped_stubs) * 100.0


@dataclass
class ExecutionResults:
    """Aggregated results from benchmark execution.
    
    Attributes:
        execution_summary: Summary metadata about the execution
        individual_results: List of successful benchmark results
        failed_executions: List of failed benchmark executions
        execution_time: Total execution time in seconds
        progress: Final progress statistics
    """
    
    execution_summary: Dict[str, Any]
    individual_results: List[BenchmarkResult]
    failed_executions: List[BenchmarkFailure]
    execution_time: float
    progress: ExecutionProgress


# =============================================================================
# Benchmark Registry Import (Moved to base.py)
# =============================================================================

# Import registry functions from benchmark_registry.py to maintain backward compatibility
from .benchmark_registry import (
    register_benchmark,
    get_registered_benchmarks,
    get_available_benchmark_names,
    create_benchmark_instance,
    ensure_benchmarks_loaded,
    validate_benchmark_config as _validate_benchmark_config_base,
    get_benchmark_subsets
)


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    """Validate a benchmark configuration against available benchmarks.
    
    This function checks if the benchmark exists and if specified subsets are valid.
    
    Args:
        config: BenchmarkConfig to validate
        
    Raises:
        ValueError: If benchmark or subsets are invalid
        
    Example:
        >>> config = BenchmarkConfig("longbench", ["narrativeqa"])
        >>> validate_benchmark_config(config)  # Passes validation
        >>> 
        >>> config = BenchmarkConfig("invalid_benchmark")
        >>> validate_benchmark_config(config)  # Raises ValueError
    """
    try:
        # Create temporary benchmark instance to validate
        benchmark = create_benchmark_instance(config.benchmark_name, subsets=None)
        
        # Validate subsets if specified
        config.validate_with_benchmark_instance(benchmark)
        
    except Exception as e:
        raise ValueError(f"Invalid benchmark configuration: {e}")


# =============================================================================
# Matrix Expansion and Benchmark Stub Generation Functions
# =============================================================================

def generate_benchmark_stubs(
    model_names: List[str],
    sparse_attention_configs: List[Tuple[str, Optional[SparseAttentionConfig]]],
    benchmark_configs: List[BenchmarkConfig],
    adapter_config: AdapterConfig,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    request_kwargs: Optional[Dict[str, Any]] = None,
    base_result_dir: str = "./benchmark_results"
) -> List[BenchmarkStub]:
    """Generate all benchmark stubs for the model × sparse_config × benchmark × subset matrix.
    
    Creates BenchmarkStub instances for every combination of:
    - model_names × sparse_attention_configs × benchmark_configs × subsets
    
    Args:
        model_names: List of model names to benchmark
        sparse_attention_configs: List of (name, config) tuples for sparse attention
        benchmark_configs: List of benchmark configurations
        adapter_config: Configuration for model adapter
        generation_kwargs: Optional generation parameters
        request_kwargs: Optional request processing parameters
        base_result_dir: Base directory for storing results
        
    Returns:
        List of BenchmarkStub instances representing all combinations
        
    Example:
        >>> stubs = generate_benchmark_stubs(
        ...     model_names=["model1", "model2"], 
        ...     sparse_attention_configs=[("dense", None), ("sparse", config)],
        ...     benchmark_configs=[BenchmarkConfig("longbench", ["narrativeqa"])],
        ...     adapter_config=AdapterConfig()
        ... )
        >>> len(stubs)  # 2 models × 2 configs × 1 benchmark × 1 subset = 4
        4
    """
    
    if generation_kwargs is None:
        generation_kwargs = {}
    if request_kwargs is None:
        request_kwargs = {}
        
    stubs: List[BenchmarkStub] = []
    
    # Expand the full matrix: model × sparse_config × benchmark × subset
    for model_name in model_names:
        for sparse_config_name, sparse_attention_config in sparse_attention_configs:
            for benchmark_config in benchmark_configs:
                # Handle benchmarks with explicit subsets
                if benchmark_config.subsets:
                    for subset in benchmark_config.subsets:
                        result_dir = construct_result_directory(
                            base_result_dir=base_result_dir,
                            model_name=model_name,
                            sparse_config_name=sparse_config_name,
                            benchmark_name=benchmark_config.benchmark_name,
                            subset=subset
                        )
                        
                        stub = BenchmarkStub(
                            model_name=model_name,
                            sparse_config_name=sparse_config_name,
                            sparse_attention_config=sparse_attention_config,
                            benchmark_name=benchmark_config.benchmark_name,
                            subset=subset,
                            adapter_config=adapter_config,
                            generation_kwargs=generation_kwargs.copy(),
                            request_kwargs=request_kwargs.copy(),
                            result_dir=result_dir
                        )
                        stubs.append(stub)
                        
                else:
                    # Handle benchmarks with no subsets (run full benchmark)
                    result_dir = construct_result_directory(
                        base_result_dir=base_result_dir,
                        model_name=model_name,
                        sparse_config_name=sparse_config_name,
                        benchmark_name=benchmark_config.benchmark_name,
                        subset=None
                    )
                    
                    stub = BenchmarkStub(
                        model_name=model_name,
                        sparse_config_name=sparse_config_name,
                        sparse_attention_config=sparse_attention_config,
                        benchmark_name=benchmark_config.benchmark_name,
                        subset=None,
                        adapter_config=adapter_config,
                        generation_kwargs=generation_kwargs.copy(),
                        request_kwargs=request_kwargs.copy(),
                        result_dir=result_dir
                    )
                    stubs.append(stub)
    
    return stubs


def filter_existing_results(
    stubs: List[BenchmarkStub],
    check_result_files: bool = True,
    required_files: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[List[BenchmarkStub], List[BenchmarkStub]]:
    """Filter benchmark stubs to skip already completed experiments (resumability).
    
    Checks each stub's result_dir to determine if the experiment has already been
    completed. This enables resuming interrupted benchmark runs.
    
    Args:
        stubs: List of benchmark stubs to filter
        check_result_files: Whether to check for specific result files (not just directory)
        required_files: List of filenames that must exist for completion (default: ["results.json"])
        verbose: Whether to log skipped combinations
        
    Returns:
        Tuple of (pending_stubs, completed_stubs) where:
        - pending_stubs: Stubs that need to be executed
        - completed_stubs: Stubs that have already been completed
        
    Example:
        >>> pending, completed = filter_existing_results(all_stubs)
        >>> print(f"Found {len(completed)} completed, {len(pending)} pending")
    """
    if required_files is None:
        required_files = ["raw_results.csv"]
    
    pending_stubs: List[BenchmarkStub] = []
    completed_stubs: List[BenchmarkStub] = []
    
    for stub in stubs:
        result_path = Path(stub.result_dir)
        print(f"Checking result directory: {result_path}", flush=True)
        
        # Check if result directory exists
        if not result_path.exists():
            pending_stubs.append(stub)
            continue
            
        # If not checking result files, just check directory existence
        if not check_result_files:
            completed_stubs.append(stub)
            if verbose:
                logging.info(f"Skipping completed experiment: {stub.result_dir}")
            continue
            
        # Check for required result files
        all_files_exist = True
        for required_file in required_files:
            file_path = result_path / required_file
            if not file_path.exists() or not file_path.is_file():
                all_files_exist = False
                break
                
        if all_files_exist:
            completed_stubs.append(stub)
            if verbose:
                logging.info(f"Skipping completed experiment (has {required_files}): {stub.result_dir}")
        else:
            pending_stubs.append(stub)
            if verbose:
                logging.info(f"Pending experiment (missing results): {stub.result_dir}")
    
    if verbose:
        total = len(stubs)
        completed_count = len(completed_stubs)
        pending_count = len(completed_stubs)
        logging.info(f"Resumability check: {completed_count}/{total} completed, {pending_count}/{total} pending")
    
    return pending_stubs, completed_stubs 