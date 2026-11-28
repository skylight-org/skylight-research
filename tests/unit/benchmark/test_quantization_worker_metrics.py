"""Exercise worker diagnostics with real codecs and CPU-only adapter/GPU boundaries."""

import json
import logging
from contextlib import nullcontext
from queue import Queue

import pytest
import torch

from benchmark import executor
from benchmark.executor_config import AdapterConfig, BenchmarkStub
from sparse_attention_hub.adapters import huggingface
from sparse_attention_hub.kv_quantization import (
    FullQuantizationCachePolicyConfig,
    KVQuantization,
    KVQuantizationConfig,
)
from sparse_attention_hub.kv_quantization.metrics.quant_error import (
    METRIC_IDS,
    QuantErrorMetric,
)
from sparse_attention_hub.metric_logging import MicroMetricLogger
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def metric_logger(monkeypatch):
    monkeypatch.setattr(MicroMetricLogger, "_instance", None)
    monkeypatch.setattr(MicroMetricLogger, "_initialized", False)
    logger = MicroMetricLogger()
    yield logger
    logger.log_queue.clear()
    logger.log_path = None


def config(enabled=True):
    return ResearchAttentionConfig(
        masker_configs=[],
        kv_quantization_config=KVQuantizationConfig(
            error_metric=enabled,
            cache_policy=FullQuantizationCachePolicyConfig(),
        ),
    )


def stub(path, sparse_config):
    return BenchmarkStub(
        model_name="local/cpu-fixture",
        sparse_config_name="fixture",
        sparse_attention_config=sparse_config,
        benchmark_name="cpu-fixture",
        subset=None,
        adapter_config=AdapterConfig(),
        generation_kwargs={},
        request_kwargs={},
        result_dir=str(path),
    )


def run_worker(monkeypatch, jobs, *, requested=None, fail_first=False, **limits):
    work, results, errors = Queue(), Queue(), Queue()
    for job in jobs:
        work.put(job)
    work.put(None)
    cleanup = []
    calls = []

    class CPUAdapter:
        def __init__(self, *, sparse_attention_config, **kwargs):
            kv_config = getattr(sparse_attention_config, "kv_quantization_config", None)
            self.quantization = (
                KVQuantization(kv_config) if kv_config is not None else None
            )

    class CPUBenchmark:
        def run_benchmark(self, *, adapter, result_dir, **kwargs):
            calls.append(result_dir)
            quant = adapter.quantization
            if quant is not None:
                metadata = {}
                x = torch.randn(
                    1, 1, 8, 32, generator=torch.Generator().manual_seed(11)
                )
                try:
                    quant.process(0, x, x, metadata, num_new_tokens=0)
                finally:
                    quant.clear_request(metadata)
            logger = MicroMetricLogger()
            if logger.is_metric_enabled("research_attention_density"):
                logger.log(
                    "research_attention_density", 1.0, metadata={"job": result_dir}
                )
            if fail_first and len(calls) == 1:
                raise RuntimeError(
                    "simulated inference failure after recording diagnostics"
                )
            return {"fixture_score": 1.0}

    monkeypatch.setattr(executor, "_shutdown_requested", False)
    monkeypatch.setattr(executor.signal, "signal", lambda *args: None)
    monkeypatch.setattr(executor, "set_gpu", lambda gpu_id: nullcontext())
    monkeypatch.setattr(executor, "acquire_gpu_from_pool", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        executor, "_cleanup_worker_resources", lambda *args: cleanup.append(args[1])
    )
    monkeypatch.setattr(
        executor, "create_benchmark_instance", lambda **kwargs: CPUBenchmark()
    )
    monkeypatch.setattr(huggingface, "ModelAdapterHF", CPUAdapter)
    executor._benchmark_worker(
        work,
        Queue(),
        results,
        errors,
        micro_metric_logger_enabled_metrics=requested,
        **limits,
    )
    return results, errors, cleanup


def read_events(path):
    file = path / "micro_metrics.jsonl"
    return (
        [json.loads(line) for line in file.read_text().splitlines()]
        if file.exists()
        else []
    )


def test_default_worker_persists_requested_quantization_diagnostics(
    monkeypatch, tmp_path, metric_logger
):
    results, errors, cleanup = run_worker(monkeypatch, [stub(tmp_path, config())])
    assert results.qsize() == 1 and errors.empty() and cleanup == [0]
    events = read_events(tmp_path)
    setup = json.loads((tmp_path / "micro_metrics_config.json").read_text())
    assert set(METRIC_IDS).issubset(setup["enabled_metrics"])
    assert setup["sampling_factor"] == 1.0 and setup["max_records"] is None
    quant_events = {
        event["metric"]: event for event in events if event["metric"] in METRIC_IDS
    }
    assert set(quant_events) == set(METRIC_IDS)
    assert quant_events["kv_quantization_store_bytes"]["value"] == 2048
    for kind in ("key", "value"):
        event = quant_events[f"kv_quantization_{kind}_error"]
        assert event["value"] > 0 and event["metadata"] == {"layer_idx": 0, "tokens": 8}


@pytest.mark.parametrize(
    "requested,expected",
    [([], set()), (["kv_quantization_store_bytes"], {"kv_quantization_store_bytes"})],
)
def test_disabled_or_bytes_only_logging_does_not_compute_reconstruction_error(
    requested,
    expected,
    monkeypatch,
    tmp_path,
    metric_logger,
    capsys,
):
    def unexpected(*args, **kwargs):
        raise AssertionError("Disabled reconstruction diagnostics were computed")

    monkeypatch.setattr(QuantErrorMetric, "compute", unexpected)
    results, errors, _ = run_worker(
        monkeypatch, [stub(tmp_path, config())], requested=requested
    )
    assert results.qsize() == 1 and errors.empty()
    assert {event["metric"] for event in read_events(tmp_path)} == expected
    setup = json.loads((tmp_path / "micro_metrics_config.json").read_text())
    assert set(setup["enabled_metrics"]) == expected
    assert "which is not enabled" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "sparse_config", [None, ResearchAttentionConfig(masker_configs=[]), config(False)]
)
def test_quantization_off_does_not_enable_quantization_events(
    sparse_config, monkeypatch, tmp_path, metric_logger
):
    results, errors, _ = run_worker(monkeypatch, [stub(tmp_path, sparse_config)])
    assert results.qsize() == 1 and errors.empty()
    assert not set(metric_logger.get_enabled_metrics()).intersection(METRIC_IDS)
    assert not {event["metric"] for event in read_events(tmp_path)}.intersection(
        METRIC_IDS
    )


def test_failed_job_diagnostics_are_flushed_to_its_directory_before_next_job(
    monkeypatch, tmp_path, metric_logger
):
    failed, next_job = tmp_path / "failed", tmp_path / "next"
    results, errors, cleanup = run_worker(
        monkeypatch,
        [stub(failed, config()), stub(next_job, config(False))],
        fail_first=True,
    )
    assert results.qsize() == 1 and errors.qsize() == 1 and cleanup == [0, 0]
    assert errors.get().stub.result_dir == str(failed)
    assert set(METRIC_IDS).issubset({event["metric"] for event in read_events(failed)})
    assert not set(METRIC_IDS).intersection(
        event["metric"] for event in read_events(next_job)
    )
    assert not metric_logger.log_queue


@pytest.mark.parametrize(
    "limits,count",
    [
        ({}, 3),
        ({"micro_metric_logger_max_records": 1}, 1),
        ({"micro_metric_logger_sampling_factor": 0}, 0),
    ],
)
def test_worker_respects_record_and_sampling_limits(
    limits, count, monkeypatch, tmp_path, metric_logger
):
    results, errors, _ = run_worker(
        monkeypatch, [stub(tmp_path, config())], requested=list(METRIC_IDS), **limits
    )
    assert results.qsize() == 1 and errors.empty()
    assert len(read_events(tmp_path)) == count


@pytest.mark.parametrize(
    "requested,error",
    [
        (["unknown_metric"], "Unknown micro metrics"),
        (["kv_quantization_key_error"], "error_metric=True"),
    ],
)
def test_invalid_metric_selection_fails_clearly(requested, error):
    with pytest.raises(ValueError, match=error):
        executor._get_enabled_micro_metrics(config(False), requested)


def test_executor_copies_explicit_selection_for_worker_configuration(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        executor.BenchmarkExecutor,
        "_setup_logging",
        lambda self: setattr(self, "logger", logging.getLogger("fixture")),
    )
    monkeypatch.setattr(
        executor.BenchmarkExecutor, "_setup_signal_handlers", lambda self: None
    )
    monkeypatch.setattr(
        executor.BenchmarkExecutor, "_validate_gpu_setup", lambda self: None
    )
    monkeypatch.setattr(executor, "ensure_benchmarks_loaded", lambda: None)
    requested = []
    instance = executor.BenchmarkExecutor(
        [0],
        1,
        base_result_dir=str(tmp_path),
        micro_metric_logger_enabled_metrics=requested,
    )
    requested.append("kv_quantization_key_error")
    assert instance.micro_metric_logger_enabled_metrics == []
