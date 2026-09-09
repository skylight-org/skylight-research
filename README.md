# Sparse Attention Hub

A comprehensive framework for implementing, experimenting with, and benchmarking sparse attention mechanisms in transformer models. This repository provides a unified interface for various sparse attention algorithms, seamless integration with HuggingFace Transformers, and extensive benchmarking capabilities across multiple long-context evaluation datasets.

## 🏗️ Repository Structure

```
sparse-attention-hub/
├── sparse_attention_hub/           # Core package
│   ├── adapters/                   # Model integration adapters
│   │   ├── huggingface.py         # HuggingFace Transformers integration
│   │   └── README.md              # Adapter documentation
│   ├── sparse_attention/          # Sparse attention implementations
│   │   ├── research_attention/    # Research-focused attention mechanisms
│   │   │   ├── maskers/          # Masker implementations
│   │   │   │   ├── fixed/        # Fixed pattern maskers
│   │   │   │   └── sampling/     # Sampling-based maskers
│   │   │   └── README.md         # Research attention documentation
│   │   └── efficient_attention/   # Production-optimized attention
│   └── metric_logging/           # Micro metric logging
├── benchmark/                     # Benchmarking suite
│   ├── raytune/                  # Ray Tune optimization framework
│   │   └── README.md             # Optimization documentation
│   ├── longbench/                # LongBench evaluation
│   ├── infinite_bench/           # InfiniteBench evaluation
│   ├── ruler/                    # RULER evaluation
│   ├── zero_scrolls/             # Zero Scrolls evaluation
│   ├── loogle/                   # Loogle evaluation
│   ├── AIME2025/                 # AIME 2025 mathematical reasoning
│   └── executor.py               # Main benchmark executor
├── tests/                        # Comprehensive test suite
├── tutorials/                    # Usage tutorials and examples
└── scripts/                      # Utility scripts
```

## 🎭 What are Masks and Maskers?

### Mask Objects

A `Mask` object represents attention patterns that control which tokens can attend to each other. The framework supports two main representations:

1. **Dense Representation**: Full tensor of shape `(batch_size, num_heads, seq_len_queries, seq_len_keys)`
2. **Sparse Representation**: Compressed format using indices and pointer arrays for memory efficiency

**Special masks:**
- **Empty Mask**: All elements are 0.0 (no attention connections)
- **Full Mask**: All elements are 1.0 (dense attention, memory-optimized)

### Maskers

A `Masker` is a component that applies specific masking logic to attention computation. Each masker implements the `add_mask()` method which:

1. Takes attention tensors (queries, keys, values) and a previous mask
2. Applies its specific masking logic, adding more active elements to the mask
3. Returns a new mask that can be further processed by subsequent maskers

**Key Concept**: Maskers are **additive** - they add attention connections to the existing mask rather than replacing it entirely. This allows for composition of different attention patterns.

For detailed information about masks and maskers, see the [Research Attention README](sparse_attention_hub/sparse_attention/research_attention/README.md).

## ⚙️ Creating Attention Configs

The framework provides a flexible configuration system for creating sparse attention mechanisms. You can combine multiple maskers to create complex attention patterns:

### Basic Configuration

```python
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig
)

# Create a basic sparse attention configuration
config = ResearchAttentionConfig(
    masker_configs=[
        SinkMaskerConfig(sink_size=128),      # Keep first 128 tokens
        LocalMaskerConfig(window_size=256)    # Local attention window
    ]
)
```

### Advanced Configurations

The framework supports various state-of-the-art sparse attention mechanisms:

- **HashAttention** (Desai et al. 2024): Hash-based attention selection
- **vAttention** (Desai et al. 2025): Adaptive sampling mechanisms
- **MagicPig** (Chen et al. 2024): LSH-based similarity sampling
- **Oracle-based methods**: Research-only mechanisms using ground truth attention

For comprehensive examples and detailed masker implementations, see the [Research Attention README](sparse_attention_hub/sparse_attention/research_attention/README.md).

## 🔧 Optimizing Configurations

The framework includes an optimization system using Ray Tune for hyperparameter search. **Note**: Ray Tune is required for optimization features but optional for basic sparse attention usage.

### Phase 1: Configuration Optimization

```bash
python3 benchmark/raytune/run_optimize_configs.py \
  --objective sparsity_10 \
  --optimal-configs-dir <base_dir> \
  --num-samples 1 \
  --search-max-new-tokens 5 \
  --search-max-context-length 32768 \
  --search-max-requests 2 \
  --actors-per-gpu 1
```

### Phase 2: Benchmark Execution

```bash
python3 benchmark/raytune/run_config_dir.py \
  --configs-dir <base_dir/config_dir> \
  --max-new-tokens 100 \
  --max-context-length 32768 \
  --max-requests 2 \
  --actors-per-gpu 1 \
  --benchmark-results-dir ./benchmark_results/
```

The optimization system supports:
- **Distributed Execution**: Ray-based parallel processing across multiple GPUs
- **Automatic Resource Management**: Efficient GPU utilization and task scheduling
- **Comprehensive Metrics**: Detailed performance and accuracy measurements
- **Search Space Definition**: Customizable hyperparameter search spaces

For detailed optimization documentation, see the [Ray Tune README](benchmark/raytune/README.md).

## 🏃‍♂️ Running Benchmarks

The framework provides a comprehensive benchmarking system that can evaluate sparse attention configurations across multiple datasets:

### Quick Start

```python
from benchmark.executor import BenchmarkExecutor
from benchmark.executor_config import BenchmarkConfig, AdapterConfig

# Define your models and configurations
models = ["meta-llama/Llama-3.2-1B-Instruct"]
sparse_configs = [
    ("dense", None),  # Dense baseline
    ("sparse", your_sparse_config)  # Your sparse configuration
]

# Define benchmarks
benchmarks = [
    BenchmarkConfig(benchmark_name="longbench", subsets=["narrativeqa"]),
    BenchmarkConfig(benchmark_name="ruler", subsets=["4096"]),
    BenchmarkConfig(benchmark_name="infinite_bench", subsets=["passkey"])
]

# Run benchmarks
executor = BenchmarkExecutor(
    gpu_ids=[0, 1, 2],
    max_concurrent_runs=3,
    base_result_dir="./results"
)

results = executor.run_benchmark_matrix(
    model_names=models,
    sparse_attention_configs=sparse_configs,
    benchmark_configs=benchmarks,
    adapter_config=AdapterConfig()
)
```

### Using Pre-configured Scripts

```bash
# Run a minimal benchmark
python benchmark/scripts/benchmark.py

# Run full benchmarking suite
python benchmark/scripts/full_benchmarking/full_benchmark.py
```

## 📊 Supported Benchmarks

The framework supports a comprehensive suite of long-context evaluation benchmarks:

| Benchmark | Description | Context Length | Tasks |
|-----------|-------------|----------------|-------|
| **LongBench** | Long-context understanding | Up to 100K tokens | 6 tasks (narrative QA, summarization, etc.) |
| **LongBench-v2** | Extended long-context evaluation | Up to 100K tokens | Enhanced version of LongBench |
| **InfiniteBench** | Infinite context evaluation | Up to 1M+ tokens | 12 major tasks including passkey retrieval |
| **RULER** | Synthetic long-context evaluation | 4K-128K tokens | 13 tasks in 4 categories (needle-in-haystack, QA, etc.) |
| **Zero Scrolls** | Multi-domain evaluation | Variable | 10 tasks across summarization, QA, sentiment |
| **Loogle** | Short and long dependency understanding | Variable | 7 major tasks |
| **AIME 2025** | Mathematical reasoning | Variable | 30 competition problems |

### Benchmark Features

- **HuggingFace Integration**: All benchmarks use processed HuggingFace datasets
- **Automatic Evaluation**: Built-in metrics calculation and result aggregation
- **Resumability**: Skip completed experiments and resume from interruptions
- **Parallel Execution**: Multi-GPU support with dynamic resource allocation
- **Comprehensive Logging**: Detailed performance and accuracy metrics

## 🚀 Quick Start with HuggingFace Integration

```python
import torch
from sparse_attention_hub.adapters import ModelAdapterHF, Request
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig
)

# 1. Create sparse attention configuration
sparse_config = ResearchAttentionConfig(
    masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=256)
    ]
)

# 2. Initialize adapter
adapter = ModelAdapterHF(
    model_name="meta-llama/Llama-3.2-1B",
    sparse_attention_config=sparse_config,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda"
)

# 3. Process requests
request = Request(
    context="The capital of France is Paris. It is known for the Eiffel Tower.",
    questions="What is the capital of France?",
    answer_prefix="Answer: "
)

response = adapter.process_request(
    request=request,
    generation_kwargs={"max_new_tokens": 50},
    request_kwargs={"max_context_length": 1024}
)

print(response.responses)  # "Answer: The capital of France is Paris."
```

## 📚 Installation

```bash
# Clone the repository
git clone https://github.com/xAlg-ai/sparse-attention-hub.git
cd sparse-attention-hub

# Install the package
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests
pytest -m integration   # Integration tests

```


## 📖 Documentation

- [Adapters Module](sparse_attention_hub/adapters/README.md) - Model integration and HuggingFace support
- [Research Attention](sparse_attention_hub/sparse_attention/research_attention/README.md) - Sparse attention mechanisms and maskers
- [Ray Tune Optimization](benchmark/raytune/README.md) - Hyperparameter optimization and search


## Research 

If you find this repository useful in your research, please consider citing our work supported by this repository
```
@inproceedings{
desai2025hashattention,
title={HashAttention: Semantic Sparsity for Faster Inference},
author={Aditya Desai and Shuo Yang and Alejandro Cuadron and Matei Zaharia and Joseph E. Gonzalez and Ion Stoica},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=Em2oaXd8Dc}
}

@inproceedings{desai2026vattention,
  title={vAttention: Verified Sparse Attention via Sampling},
  author={Desai, Aditya and Agrawal, Kumar Krishna and Yang, Shuo and Cuadron, Alejandro and Schroeder, Luis Gaspar and Zaharia, Matei and Gonzalez, Joseph E and Stoica, Ion},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
