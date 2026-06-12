"""BenchmarkExecutor: Parallel GPU execution for benchmark matrix experiments.

This module contains the main BenchmarkExecutor class that orchestrates parallel 
execution of benchmark experiments across the model × sparse_attention_config × 
benchmark × subset matrix with dynamic GPU allocation and resumability.
"""

import logging
import multiprocessing
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from queue import Empty
from contextlib import contextmanager
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

from sparse_attention_hub.sparse_attention.base import SparseAttentionConfig

from .executor_config import (
    AdapterConfig,
    BenchmarkConfig,
    BenchmarkStub,
    ExecutionResults,
    ExecutionProgress,
    BenchmarkResult,
    BenchmarkFailure,
    filter_existing_results,
    generate_benchmark_stubs,

)
from .benchmark_registry import (
    ensure_benchmarks_loaded,
    create_benchmark_instance
)

from .utils.gpu import (
    cleanup_gpu_memory,
    release_gpu_to_pool,
    acquire_gpu_from_pool,
    create_gpu_pool,
    validate_gpu_availability,
)

# Global flag for graceful shutdown
_shutdown_requested = False


@contextmanager
def set_gpu(gpu_id: int):
    """Context manager to set the current CUDA device.
    
    Args:
        gpu_id: The GPU device ID to set as current device
        
    Yields:
        None: The context manager yields control to the with block
    """
    import torch
    torch.cuda.set_device(gpu_id)
    yield


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    logging.getLogger(__name__).warning(f"Received signal {signum}, initiating graceful shutdown...")


def _categorize_error(error: Exception, context: str = "") -> str:
    """Categorize errors for better handling and reporting.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
        
    Returns:
        Error category string for classification
    """
    error_type = type(error).__name__
    error_msg = str(error).lower()
    
    # GPU-related errors
    if any(keyword in error_msg for keyword in ["cuda", "gpu", "out of memory", "oom"]):
        if "out of memory" in error_msg or "oom" in error_msg:
            return "GPU_OUT_OF_MEMORY"
        elif "cuda" in error_msg:
            return "GPU_CUDA_ERROR"
        else:
            return "GPU_ERROR"
    
    # Model loading errors
    elif any(keyword in error_msg for keyword in ["model", "huggingface", "transformers", "pretrained"]):
        if "not found" in error_msg or "doesn't exist" in error_msg:
            return "MODEL_NOT_FOUND"
        elif "tokenizer" in error_msg:
            return "TOKENIZER_ERROR"
        else:
            return "MODEL_LOADING_ERROR"
    
    # Dataset/benchmark errors
    elif any(keyword in error_msg for keyword in ["dataset", "benchmark", "subset"]):
        return "DATASET_ERROR"
    
    # Network/connection errors
    elif any(keyword in error_msg for keyword in ["connection", "timeout", "network", "http"]):
        return "NETWORK_ERROR"
    
    # File system errors
    elif any(keyword in error_msg for keyword in ["file", "directory", "permission", "disk"]):
        return "FILESYSTEM_ERROR"
    
    # Multiprocessing errors
    elif any(keyword in error_msg for keyword in ["queue", "process", "multiprocessing"]):
        return "MULTIPROCESSING_ERROR"
    
    # Default categories
    elif isinstance(error, TimeoutError):
        return "TIMEOUT_ERROR"
    elif isinstance(error, KeyboardInterrupt):
        return "INTERRUPT_ERROR"
    elif isinstance(error, MemoryError):
        return "SYSTEM_MEMORY_ERROR"
    else:
        return "UNKNOWN_ERROR"


def _benchmark_worker(
    stub_queue: multiprocessing.Queue,
    gpu_pool: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    error_queue: multiprocessing.Queue,
    timeout_per_benchmark: float = 3600.0,
    micro_metric_logger_max_records: Optional[int] = None,
    micro_metric_logger_sampling_factor: float = 1.0
) -> None:
    """Worker function to execute benchmarks with dynamic GPU allocation.
    
    This function runs in a separate process and handles:
    1. Acquiring GPUs from the shared pool
    2. Creating model adapters with sparse attention configurations
    3. Executing benchmarks with proper error handling
    4. Releasing GPUs back to the pool
    5. Reporting results and errors
    
    Args:
        stub_queue: Queue containing BenchmarkStub instances to process
        gpu_pool: Shared queue containing available GPU IDs
        result_queue: Queue for reporting successful benchmark results
        error_queue: Queue for reporting benchmark failures
        timeout_per_benchmark: Timeout for individual benchmark execution (seconds)
        micro_metric_logger_max_records: Maximum number of metric events to log (None for unlimited)
        micro_metric_logger_sampling_factor: Probability of logging each metric event (0.0-1.0)
    """
    worker_id = os.getpid()
    logger = logging.getLogger(f"{__name__}.worker_{worker_id}")
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    
    # Track resources for cleanup
    current_gpu_id: Optional[int] = None
    adapter = None

    try:
        while True:
            # Check for shutdown signal
            if _shutdown_requested:
                logger.info(f"Worker {worker_id}: Shutdown requested, exiting")
                break
            
            # Step 1: Get next benchmark stub from queue
            try:
                stub: BenchmarkStub = stub_queue.get(timeout=1.0)  # 1 second timeout
            except Empty:
                # No more work to do
                logger.info(f"Worker {worker_id}: No more work, shutting down")
                break
            
            # Check for sentinel value (shutdown signal)
            if stub is None:
                logger.info(f"Worker {worker_id}: Received shutdown signal, exiting")
                break
            
            # Step 2: Acquire GPU from pool
            try:
                current_gpu_id = acquire_gpu_from_pool(gpu_pool, timeout=30.0)
                logger.info(f"Worker {worker_id}: Acquired GPU {current_gpu_id} for {stub.model_name}/{stub.sparse_config_name}/{stub.benchmark_name}")
            except Exception as e:
                error_category = _categorize_error(e, "GPU acquisition")
                error_msg = f"Failed to acquire GPU for {stub.model_name}/{stub.sparse_config_name}/{stub.benchmark_name}: {e}"
                logger.error(f"Worker {worker_id}: {error_msg}")
                error_queue.put(BenchmarkFailure(
                    stub=stub,
                    error_message=error_msg,
                    error_type=error_category,
                    execution_time=0.0
                ))
                continue

            # Step 3: Run the benchmark on the current GPU
            start_time = time.time()
            execution_time = 0.0  # Initialize execution_time
            
            try:
                with set_gpu(current_gpu_id):
                    logger.debug(f"Worker {worker_id}: Set CUDA device to {current_gpu_id}")
                    execution_success = False

                    # Create model adapter with sparse attention config
                    logger.info(f"Worker {worker_id}: Creating model adapter for {stub.model_name}")
                    
                    # Import here to avoid issues with multiprocessing
                    from sparse_attention_hub.adapters.huggingface import ModelAdapterHF
                    
                    adapter = ModelAdapterHF(
                        model_name=stub.model_name,
                        sparse_attention_config=stub.sparse_attention_config,
                        model_kwargs=stub.adapter_config.model_kwargs,
                        tokenizer_kwargs=stub.adapter_config.tokenizer_kwargs
                    )
                    
                    # Create benchmark instance
                    logger.info(f"Worker {worker_id}: Creating benchmark instance for {stub.benchmark_name}")
                    benchmark = create_benchmark_instance(
                        benchmark_name=stub.benchmark_name,
                        subsets=[stub.subset] if stub.subset else None
                    )
                    
                    # Execute benchmark
                    logger.info(f"Worker {worker_id}: Executing benchmark {stub.benchmark_name} on GPU {current_gpu_id}")
                    metric_logger = MicroMetricLogger()
                    metric_logger.configure_logging(
                        log_path=stub.result_dir, 
                        enabled_metrics=["research_attention_density", "research_attention_output_error"],
                        max_records=micro_metric_logger_max_records,
                        sampling_factor=micro_metric_logger_sampling_factor
                    )
                    result_dir = Path(stub.result_dir)

                    parent_dir = result_dir.parent
                    if not parent_dir.exists():
                        raise FileNotFoundError(f"Parent directory does not exist: {parent_dir}")

                    metrics = benchmark.run_benchmark(
                        adapter=adapter,
                        result_dir=stub.result_dir,
                        generation_kwargs=stub.generation_kwargs,
                        request_kwargs=stub.request_kwargs
                    )
                    metric_logger.flush()
                    
                    execution_time = time.time() - start_time
                    execution_success = True
                    
                    # Report successful result
                    result = BenchmarkResult(
                        stub=stub,
                        metrics=metrics,
                        execution_time=execution_time,
                        success=True
                    )
                    result_queue.put(result)
                    
                    logger.info(f"Worker {worker_id}: Successfully completed {stub.model_name}/{stub.sparse_config_name}/{stub.benchmark_name} in {execution_time:.2f}s")
            except (OSError, PermissionError, FileNotFoundError):
                raise
            except Exception as e:
                execution_time = time.time() - start_time  # Calculate execution time even on failure
                error_category = _categorize_error(e, "benchmark execution")
                error_msg = f"Benchmark execution failed: {e}"
                logger.error(f"Worker {worker_id}: {error_msg}")
                
                # Log detailed error information for debugging
                logger.debug(f"Worker {worker_id}: Error details: {traceback.format_exc()}")
                
                # Report failure
                failure = BenchmarkFailure(
                    stub=stub,
                    error_message=error_msg,
                    error_type=error_category,
                    execution_time=execution_time
                )
                error_queue.put(failure)
                continue
            
            finally:
                # Step 4: Cleanup and release GPU
                _cleanup_worker_resources(gpu_pool, current_gpu_id, adapter, logger, worker_id)
                current_gpu_id = None
                adapter = None
    
    except Exception as worker_error:
        logger.error(f"Worker {worker_id}: Critical worker error: {worker_error}")
        logger.debug(f"Worker {worker_id}: Critical error details: {traceback.format_exc()}")

        # Cleanup any remaining resources
        _cleanup_worker_resources(gpu_pool, current_gpu_id, adapter, logger, worker_id)

        # Report worker failure to error queue including exception class name so main process can re-raise if needed
        error_queue.put({
            "worker_id": worker_id,
            "error_type": "WORKER_CRITICAL_ERROR",
            "error_message": str(worker_error),
            "exception_class": type(worker_error).__name__,
            "exception_message": str(worker_error),
            "error_traceback": traceback.format_exc()
        })


def _cleanup_worker_resources(
    gpu_pool: multiprocessing.Queue,
    gpu_id: Optional[int],
    adapter: Any,
    logger: logging.Logger,
    worker_id: int
) -> None:
    """Clean up worker resources including GPU memory and adapter.
    
    Args:
        gpu_pool: GPU pool queue to return GPU to
        gpu_id: GPU ID to release (can be None) - this is the actual GPU ID from the pool
        adapter: Model adapter to cleanup (can be None)
        logger: Logger instance for the worker
        worker_id: Worker process ID for logging
    """
    try:
        # Clean up adapter if it exists
        if adapter is not None:
            try:
                # Force garbage collection of adapter
                del adapter
                import gc
                gc.collect()
                logger.debug(f"Worker {worker_id}: Cleaned up model adapter")
            except Exception as e:
                logger.warning(f"Worker {worker_id}: Failed to cleanup adapter: {e}")
        
        # Clean up GPU memory and release GPU
        if gpu_id is not None:
            try:
                cleanup_gpu_memory(gpu_id)
                logger.debug(f"Worker {worker_id}: Cleaned up GPU memory for GPU {gpu_id}")
            
                # Release GPU back to pool (always use the original GPU ID for the pool)
                release_gpu_to_pool(gpu_pool, gpu_id)
                logger.debug(f"Worker {worker_id}: Released GPU {gpu_id} back to pool")
                
            except Exception as cleanup_error:
                logger.warning(f"Worker {worker_id}: GPU cleanup failed for GPU {gpu_id}: {cleanup_error}")
                # Try to release GPU even if cleanup failed
                try:
                    release_gpu_to_pool(gpu_pool, gpu_id)
                except Exception:
                    logger.error(f"Worker {worker_id}: Failed to release GPU {gpu_id} to pool")
    
    except Exception as e:
        logger.error(f"Worker {worker_id}: Error during resource cleanup: {e}")


class BenchmarkExecutor:
    """Parallel benchmark executor with dynamic GPU allocation and resumability.
    
    Executes benchmark experiments across a model × sparse_attention_config × 
    benchmark × subset matrix with parallel GPU execution and automatic resumability.
    
    Features:
    - Dynamic GPU allocation from specified gpu_ids
    - Subset-level resumability (skip completed experiments)
    - Parallel execution with configurable concurrency
    - Comprehensive error handling and logging
    - Result aggregation and summary generation
    
    Example:
        >>> executor = BenchmarkExecutor(
        ...     gpu_ids=[0, 1, 2, 3],
        ...     max_concurrent_runs=4,
        ...     base_result_dir="./benchmark_results"
        ... )
        >>> 
        >>> results = executor.run_benchmark_matrix(
        ...     model_names=["model1", "model2"],
        ...     sparse_attention_configs=[("dense", None), ("sparse", config)],
        ...     benchmark_configs=[BenchmarkConfig("longbench", ["narrativeqa"])],
        ...     adapter_config=AdapterConfig()
        ... )
    """
    
    def __init__(
        self,
        gpu_ids: List[int],
        max_concurrent_runs: int,
        base_result_dir: str = "./benchmark_results",
        enable_resumability: bool = True,
        result_file_validation: bool = True,
        required_result_files: Optional[List[str]] = None,
        timeout_per_benchmark: float = 360000.0,  # 100 hours default
        verbose: bool = True,
        micro_metric_logger_max_records: Optional[int] = None,  # Maximum number of metric events to log
        micro_metric_logger_sampling_factor: float = 1.0,  # Probability of logging each metric event (0.0-1.0)
    ):
        """Initialize the BenchmarkExecutor.
        
        Args:
            gpu_ids: List of GPU device IDs to use for parallel execution
            max_concurrent_runs: Maximum number of concurrent benchmark runs
            base_result_dir: Base directory for storing benchmark results
            enable_resumability: Whether to skip already completed experiments
            result_file_validation: Whether to check for specific result files
            required_result_files: List of files required for completion check
            timeout_per_benchmark: Timeout for individual benchmark execution (seconds)
            verbose: Whether to enable verbose logging
            micro_metric_logger_max_records: Maximum number of metric events to log (None for unlimited)
            micro_metric_logger_sampling_factor: Probability of logging each metric event (0.0-1.0)
            
        Raises:
            ValueError: If configuration parameters are invalid
            RuntimeError: If GPU validation fails
        """
        # Validate and store configuration
        self.gpu_ids = self._validate_gpu_ids(gpu_ids)
        self.max_concurrent_runs = self._validate_max_concurrent_runs(max_concurrent_runs)
        self.base_result_dir = Path(base_result_dir).resolve()
        self.enable_resumability = enable_resumability
        self.result_file_validation = result_file_validation
        self.required_result_files = required_result_files or ["raw_results.csv"]
        self.timeout_per_benchmark = timeout_per_benchmark
        self.verbose = verbose
        
        # Metric logging configuration
        self.micro_metric_logger_max_records = micro_metric_logger_max_records
        self.micro_metric_logger_sampling_factor = max(0.0, min(1.0, micro_metric_logger_sampling_factor))  # Clamp to [0.0, 1.0]
        
        # Initialize logging
        self._setup_logging()
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Validate GPU availability
        self._validate_gpu_setup()
        
        # Load all available benchmarks
        ensure_benchmarks_loaded()
        

        
        self.logger.info(f"BenchmarkExecutor initialized:")
        self.logger.info(f"  GPU IDs: {self.gpu_ids}")
        self.logger.info(f"  Max concurrent runs: {self.max_concurrent_runs}")
        self.logger.info(f"  Base result directory: {self.base_result_dir}")
        self.logger.info(f"  Resumability: {'enabled' if self.enable_resumability else 'disabled'}")
        self.logger.info(f"  Metric logging - Max records: {self.micro_metric_logger_max_records if self.micro_metric_logger_max_records else 'unlimited'}")
        self.logger.info(f"  Metric logging - Sampling factor: {self.micro_metric_logger_sampling_factor}")
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        
        # Register cleanup function to run at exit
        import atexit
        atexit.register(self._cleanup_on_exit)
    
    def _cleanup_on_exit(self) -> None:
        """Cleanup function called when the process exits."""
        if hasattr(self, 'logger'):
            self.logger.info("BenchmarkExecutor cleanup on exit")
        
        # Reset global shutdown flag
        global _shutdown_requested
        _shutdown_requested = False

    def run_benchmark_matrix(
        self,
        model_names: List[str],
        sparse_attention_configs: List[Tuple[str, Optional[SparseAttentionConfig]]],
        benchmark_configs: List[BenchmarkConfig],
        adapter_config: AdapterConfig,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        request_kwargs: Optional[Dict[str, Any]] = None
    ) -> ExecutionResults:
        """Execute benchmark matrix with parallel GPU execution.
        
        Runs all combinations of model_names × sparse_attention_configs × 
        benchmark_configs × subsets in parallel across available GPUs.
        
        Args:
            model_names: List of model names to benchmark
            sparse_attention_configs: List of (name, config) tuples for sparse attention
            benchmark_configs: List of benchmark configurations
            adapter_config: Configuration for model adapter
            generation_kwargs: Optional generation parameters
            request_kwargs: Optional request processing parameters
            
        Returns:
            ExecutionResults containing comprehensive execution information
            
        Example:
            >>> results = executor.run_benchmark_matrix(
            ...     model_names=["meta-llama/Llama-3.1-8B-Instruct"],
            ...     sparse_attention_configs=[("dense", None)],
            ...     benchmark_configs=[BenchmarkConfig("longbench", ["narrativeqa"])],
            ...     adapter_config=AdapterConfig()
            ... )
            >>> print(f"Completed: {results.completed_count}/{results.total_count}")
        """
        # Initialize execution
        self.logger.info("Starting benchmark matrix execution")
        
        try:
            # Step 1: Validate input parameters
            self._validate_matrix_parameters(
                model_names, sparse_attention_configs, benchmark_configs, adapter_config
            )
            
            # Step 2: Generate benchmark stubs (matrix expansion)
            self.logger.info("Generating benchmark matrix...")
            all_stubs = generate_benchmark_stubs(
                model_names=model_names,
                sparse_attention_configs=sparse_attention_configs,
                benchmark_configs=benchmark_configs,
                adapter_config=adapter_config,
                generation_kwargs=generation_kwargs,
                request_kwargs=request_kwargs,
                base_result_dir=str(self.base_result_dir)
            )
            
            total_combinations = len(all_stubs)
            self.logger.info(f"Generated {total_combinations} benchmark combinations:")
            self.logger.info(f"  {len(model_names)} models × {len(sparse_attention_configs)} sparse configs × "
                           f"{sum(len(b.subsets) if b.subsets else 1 for b in benchmark_configs)} benchmark-subsets")
            
            # Step 3: Filter for resumability (if enabled)
            if self.enable_resumability:
                self.logger.info("Checking for existing results (resumability)...")
                pending_stubs, completed_stubs = filter_existing_results(
                    all_stubs,
                    check_result_files=self.result_file_validation,
                    required_files=self.required_result_files,
                    verbose=self.verbose
                )
                
                self.logger.info(f"Resumability check: {len(completed_stubs)} completed, {len(pending_stubs)} pending")
            else:
                pending_stubs = all_stubs
                completed_stubs = []
                self.logger.info("Resumability disabled - will execute all combinations")
            
            # Step 4: Early exit if nothing to do
            if not pending_stubs:
                self.logger.info("No pending benchmarks to execute. All experiments already completed!")
                
                execution_summary = {
                    "total_combinations": total_combinations,
                    "models": model_names,
                    "sparse_configs": [name for name, _ in sparse_attention_configs],
                    "benchmarks": [cfg.benchmark_name for cfg in benchmark_configs]
                }
                
                progress = ExecutionProgress(
                    total_stubs=total_combinations,
                    skipped_stubs=len(completed_stubs),
                    completed_stubs=0,
                    failed_stubs=0
                )
                
                return ExecutionResults(
                    execution_summary=execution_summary,
                    individual_results=[],
                    failed_executions=[],
                    execution_time=0.0,
                    progress=progress
                )
            

            
            # Step 6: Execute pending benchmarks with worker processes
            self.logger.info(f"Executing {len(pending_stubs)} pending benchmarks with {self.max_concurrent_runs} workers...")
            
            start_time = time.time()
            results = self._execute_with_workers(pending_stubs)
            execution_time = time.time() - start_time
            
            # Step 7: Aggregate results
            execution_summary = {
                "total_combinations": total_combinations,
                "models": model_names,
                "sparse_configs": [name for name, _ in sparse_attention_configs],
                "benchmarks": [cfg.benchmark_name for cfg in benchmark_configs],
                "execution_time_seconds": execution_time,
                "gpu_ids_used": self.gpu_ids,
                "max_concurrent_runs": self.max_concurrent_runs
            }
            
            progress = ExecutionProgress(
                total_stubs=total_combinations,
                skipped_stubs=len(completed_stubs),
                completed_stubs=len(results.individual_results),
                failed_stubs=len(results.failed_executions)
            )
            
            final_results = ExecutionResults(
                execution_summary=execution_summary,
                individual_results=results.individual_results,
                failed_executions=results.failed_executions,
                execution_time=execution_time,
                progress=progress
            )
            
            self.logger.info(f"Benchmark matrix execution completed:")
            self.logger.info(f"  Total time: {execution_time:.2f}s")
            self.logger.info(f"  Successful: {len(results.individual_results)}")
            self.logger.info(f"  Failed: {len(results.failed_executions)}")
            self.logger.info(f"  Skipped: {len(completed_stubs)}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Benchmark execution failed: {e}")
            raise
    
    def _execute_with_workers(self, pending_stubs: List[BenchmarkStub]) -> ExecutionResults:
        """Execute benchmark stubs using worker processes.
        
        This method implements comprehensive process pool management with:
        - Dynamic work distribution through shared queues
        - Graceful process startup and shutdown
        - Real-time result collection and error handling
        - Process health monitoring and recovery
        
        Args:
            pending_stubs: List of benchmark stubs to execute
            
        Returns:
            ExecutionResults containing all results and failures
        """
        self.logger.info(f"Starting execution with {len(pending_stubs)} stubs using {self.max_concurrent_runs} workers")
        
        # Step 1: Create and configure shared queues
        stub_queue = multiprocessing.Queue(maxsize=len(pending_stubs) + 100)  # Buffer for queue operations
        gpu_pool = create_gpu_pool(self.gpu_ids)
        result_queue = multiprocessing.Queue()
        error_queue = multiprocessing.Queue()
        
        # Step 2: Populate work queue with all stubs
        self.logger.info("Populating work queue...")
        for stub in pending_stubs:
            stub_queue.put(stub)
        
        # Add sentinel values to signal workers to stop
        for _ in range(self.max_concurrent_runs):
            stub_queue.put(None)  # Sentinel value
        
        # Step 3: Create and start worker processes
        workers = []
        worker_pids = []
        
        self.logger.info(f"Starting {self.max_concurrent_runs} worker processes...")
        for i in range(self.max_concurrent_runs):
            worker = multiprocessing.Process(
                target=_benchmark_worker,
                args=(stub_queue,
                        gpu_pool, 
                        result_queue, 
                        error_queue, 
                        self.timeout_per_benchmark, 
                        self.micro_metric_logger_max_records, 
                        self.micro_metric_logger_sampling_factor
                        ),
                name=f"benchmark_worker_{i}"
            )
            worker.daemon = True  # Ensure workers are terminated when main process exits
            worker.start()
            workers.append(worker)
            worker_pids.append(worker.pid)
            self.logger.debug(f"Started worker {i+1}/{self.max_concurrent_runs} (PID: {worker.pid})")
        
        self.logger.info(f"All {self.max_concurrent_runs} workers started successfully")
        
        # Step 4: Monitor and collect results in real-time
        individual_results: List[BenchmarkResult] = []
        failed_executions: List[BenchmarkFailure] = []
        active_workers = set(worker_pids)
        
        self.logger.info("Monitoring worker processes and collecting results...")
        
        try:
            # Monitor workers and collect results until all work is complete
            while active_workers:
                # Check for completed workers
                completed_workers = []
                for worker in workers:
                    if worker.is_alive():
                        # Worker is still running, check for results
                        pass
                    else:
                        # Worker has completed
                        if worker.pid in active_workers:
                            active_workers.remove(worker.pid)
                            completed_workers.append(worker)
                            
                            if worker.exitcode != 0:
                                self.logger.warning(f"Worker {worker.pid} exited with code {worker.exitcode}")
                            else:
                                self.logger.debug(f"Worker {worker.pid} completed successfully")
                
                # Collect any available results (non-blocking)
                self._collect_available_results(result_queue, error_queue, individual_results, failed_executions)
                
                # Log progress periodically
                if len(individual_results) + len(failed_executions) > 0 and (len(individual_results) + len(failed_executions)) % 5 == 0:
                    self.logger.info(f"Progress: {len(individual_results)} completed, {len(failed_executions)} failed, {len(active_workers)} workers active")
                
                # Small delay to prevent busy waiting
                time.sleep(0.1)
            
            # Step 5: Final result collection
            self.logger.info("All workers completed, collecting final results...")
            self._collect_available_results(result_queue, error_queue, individual_results, failed_executions)
            # If a worker reported a critical filesystem error, raise it here
            exc_map = {
                "OSError": OSError,
                "PermissionError": PermissionError,
                "FileNotFoundError": FileNotFoundError
            }

            for fe in failed_executions:
                if isinstance(fe, dict) and fe.get("error_type") == "WORKER_CRITICAL_ERROR":
                    exc_name = fe.get("exception_class")
                    if exc_name in exc_map:
                        raise exc_map[exc_name](fe.get("exception_message", fe.get("error_message", "")))
        except KeyboardInterrupt:
            self.logger.warning("Received interrupt signal, initiating graceful shutdown...")
            self._graceful_shutdown(workers, stub_queue, gpu_pool, result_queue, error_queue)
            raise
        
        except Exception as e:
            error_category = _categorize_error(e, "main execution")
            self.logger.error(f"Error during execution ({error_category}): {e}")
            self.logger.debug(f"Execution error details: {traceback.format_exc()}")
            self._graceful_shutdown(workers, stub_queue, gpu_pool, result_queue, error_queue)
            raise
        
        finally:
            # Step 6: Cleanup resources
            self._cleanup_resources(workers, stub_queue, gpu_pool, result_queue, error_queue)
        
        # Step 7: Final summary
        total_processed = len(individual_results) + len(failed_executions)
        self.logger.info(f"Execution completed: {len(individual_results)} successful, {len(failed_executions)} failed")
        self.logger.info(f"Success rate: {len(individual_results)/total_processed*100:.1f}%" if total_processed > 0 else "No work processed")
        
        return ExecutionResults(
            execution_summary={},  # Will be filled by caller
            individual_results=individual_results,
            failed_executions=failed_executions,
            execution_time=0.0,  # Will be filled by caller
            progress=ExecutionProgress(
                total_stubs=len(pending_stubs),
                completed_stubs=len(individual_results),
                failed_stubs=len(failed_executions)
            )
        )
    
    def _collect_available_results(
        self,
        result_queue: multiprocessing.Queue,
        error_queue: multiprocessing.Queue,
        individual_results: List[BenchmarkResult],
        failed_executions: List[BenchmarkFailure]
    ) -> None:
        """Collect all available results from queues without blocking.
        
        Args:
            result_queue: Queue containing successful benchmark results
            error_queue: Queue containing benchmark failures
            individual_results: List to append successful results to
            failed_executions: List to append failures to
        """
        # Collect all available successful results
        while True:
            try:
                result = result_queue.get_nowait()
                individual_results.append(result)
            except Empty:
                break
        
        # Collect all available errors
        while True:
            try:
                error = error_queue.get_nowait()
                if isinstance(error, BenchmarkFailure):
                    failed_executions.append(error)
                else:
                    # Worker-level dicts (critical errors) will be appended so caller can decide to raise
                    # e.g. {'worker_id': ..., 'error_type': 'WORKER_CRITICAL_ERROR', 'exception_class': 'OSError', ...}
                    failed_executions.append(error)
                    self.logger.error(f"Worker error (queued): {error}")
            except Empty:
                break
    
    def _graceful_shutdown(
        self,
        workers: List[multiprocessing.Process],
        stub_queue: multiprocessing.Queue,
        gpu_pool: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
        error_queue: multiprocessing.Queue
    ) -> None:
        """Perform graceful shutdown of worker processes and queues.
        
        Args:
            workers: List of worker processes to terminate
            stub_queue: Work queue to close
            gpu_pool: GPU pool queue to close
            result_queue: Result queue to close
            error_queue: Error queue to close
        """
        self.logger.info("Initiating graceful shutdown...")
        
        try:
            # Signal workers to stop by adding sentinel values
            sentinel_count = 0
            for _ in range(len(workers)):
                try:
                    stub_queue.put_nowait(None)
                    sentinel_count += 1
                except Exception as e:
                    self.logger.debug(f"Could not add sentinel to queue: {e}")
                    break  # Queue is full, stop adding sentinels
            
            self.logger.info(f"Added {sentinel_count} shutdown signals to work queue")
            
            # Wait for workers to finish with timeout
            self.logger.info("Waiting for workers to finish...")
            for i, worker in enumerate(workers):
                try:
                    worker.join(timeout=30.0)  # 30 second timeout per worker
                    if worker.is_alive():
                        self.logger.warning(f"Force terminating worker {worker.pid}")
                        worker.terminate()
                        worker.join(timeout=5.0)
                        if worker.is_alive():
                            self.logger.error(f"Failed to terminate worker {worker.pid}, killing process")
                            worker.kill()
                            worker.join(timeout=2.0)
                    else:
                        self.logger.debug(f"Worker {worker.pid} terminated successfully")
                except Exception as e:
                    self.logger.error(f"Error during worker {worker.pid} shutdown: {e}")
            
            self.logger.info("Graceful shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {e}")
            # Force cleanup of remaining workers
            for worker in workers:
                if worker.is_alive():
                    try:
                        worker.kill()
                    except Exception:
                        pass
    
    def _cleanup_resources(
        self,
        workers: List[multiprocessing.Process],
        stub_queue: multiprocessing.Queue,
        gpu_pool: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
        error_queue: multiprocessing.Queue
    ) -> None:
        """Clean up all multiprocessing resources.
        
        Args:
            workers: List of worker processes
            stub_queue: Work queue to close
            gpu_pool: GPU pool queue to close
            result_queue: Result queue to close
            error_queue: Error queue to close
        """
        self.logger.debug("Cleaning up multiprocessing resources...")
        
        # Close all queues
        try:
            stub_queue.close()
            gpu_pool.close()
            result_queue.close()
            error_queue.close()
        except Exception as e:
            self.logger.warning(f"Error closing queues: {e}")
        
        # Join any remaining workers
        for worker in workers:
            if worker.is_alive():
                self.logger.warning(f"Force terminating remaining worker {worker.pid}")
                worker.terminate()
                worker.join(timeout=5.0)
                if worker.is_alive():
                    worker.kill()
        
        self.logger.debug("Resource cleanup completed")
    
    def _validate_gpu_ids(self, gpu_ids: List[int]) -> List[int]:
        """Validate and return GPU IDs."""
        if not gpu_ids:
            raise ValueError("gpu_ids cannot be empty")
        
        if not all(isinstance(gpu_id, int) and gpu_id >= 0 for gpu_id in gpu_ids):
            raise ValueError("All GPU IDs must be non-negative integers")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_gpu_ids = []
        for gpu_id in gpu_ids:
            if gpu_id not in seen:
                seen.add(gpu_id)
                unique_gpu_ids.append(gpu_id)
        
        return unique_gpu_ids
    
    def _validate_max_concurrent_runs(self, max_concurrent_runs: int) -> int:
        """Validate and return max_concurrent_runs."""
        if not isinstance(max_concurrent_runs, int) or max_concurrent_runs <= 0:
            raise ValueError("max_concurrent_runs must be a positive integer")
        
        if max_concurrent_runs > len(self.gpu_ids) if hasattr(self, 'gpu_ids') else True:
            # This is just a warning - user might want more concurrent runs than GPUs
            # for benchmarks that don't require much GPU memory
            pass
        
        return max_concurrent_runs
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger(f"{__name__}.BenchmarkExecutor")
        
        if self.verbose and not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _validate_gpu_setup(self) -> None:
        """Validate GPU availability."""
        try:
            validate_gpu_availability(self.gpu_ids)
            self.logger.info(f"GPU validation passed for devices: {self.gpu_ids}")
        except (RuntimeError, ValueError) as e:
            self.logger.error(f"GPU validation failed: {e}")
            raise
    
    def _validate_matrix_parameters(
        self,
        model_names: List[str],
        sparse_attention_configs: List[Tuple[str, Optional[SparseAttentionConfig]]],
        benchmark_configs: List[BenchmarkConfig],
        adapter_config: AdapterConfig
    ) -> None:
        """Validate benchmark matrix parameters."""
        if not model_names:
            raise ValueError("model_names cannot be empty")
        
        if not sparse_attention_configs:
            raise ValueError("sparse_attention_configs cannot be empty")
        
        if not benchmark_configs:
            raise ValueError("benchmark_configs cannot be empty")
        
        # Validate model names
        for model_name in model_names:
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError(f"Invalid model name: {model_name}")
        
        # Validate sparse attention configs
        for sparse_config in sparse_attention_configs:
            if not isinstance(sparse_config, tuple) or len(sparse_config) != 2:
                raise ValueError(f"sparse_attention_configs must be list of (name, config) tuples")
            
            config_name, config_obj = sparse_config
            if not isinstance(config_name, str) or not config_name.strip():
                raise ValueError(f"Sparse config name must be non-empty string: {config_name}")
        
        # Validate benchmark configs
        for benchmark_config in benchmark_configs:
            if not isinstance(benchmark_config, BenchmarkConfig):
                raise ValueError(f"benchmark_configs must contain BenchmarkConfig instances")
            # Additional validation could be added here if needed
        
        # Validate adapter config
        if not isinstance(adapter_config, AdapterConfig):
            raise ValueError("adapter_config must be an AdapterConfig instance")
            # Additional validation could be added here if needed
        
        self.logger.info("Matrix parameter validation passed")
