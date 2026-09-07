# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark package for evaluating various language model capabilities.
"""

# Import all benchmark submodules to make them available for import
from . import (
    AIME2024,
    AIME2025,
    infinite_bench,
    loft,
    longbench,
    longbenchv2,
    loogle,
    mock_benchmark,
    ruler,
    ruler16k,
    ruler32k,
    ruler64k,
    ruler128k,
    zero_scrolls,
)

# Import the base benchmark class
from .base import Benchmark

# Import registry functions
from .benchmark_registry import (
    create_benchmark_instance,
    ensure_benchmarks_loaded,
    get_available_benchmark_names,
    get_benchmark_subsets,
    get_registered_benchmarks,
    register_benchmark,
    validate_benchmark_config,
)

# Import concrete benchmark implementations
from .loft import LoftRag
from .longbench import LongBench
from .loogle import Loogle
from .mock_benchmark import MockBenchmark

__all__ = [
    "Benchmark",
    "register_benchmark",
    "get_registered_benchmarks",
    "get_available_benchmark_names",
    "create_benchmark_instance",
    "ensure_benchmarks_loaded",
    "validate_benchmark_config",
    "get_benchmark_subsets",
    "LoftRag",
    "LongBench",
    "MockBenchmark",
    "Loogle",
    "AIME2024",
    "AIME2025",
    "infinite_bench",
    "loft",
    "longbench",
    "longbenchv2",
    "loogle",
    "mock_benchmark",
    "ruler",
    "zero_scrolls",
    "ruler32k",
    "ruler16k",
    "ruler64k",
    "ruler128k",
]
