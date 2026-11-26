"""Real worker -> benchmark -> adapter -> local HF model -> quantization -> exports."""

import json
from contextlib import nullcontext
from queue import Queue

import pandas as pd
import pytest
import torch

from benchmark import executor
from benchmark.base import Benchmark
from benchmark.executor_config import AdapterConfig, BenchmarkStub
from sparse_attention_hub.adapters import Request, huggingface
from sparse_attention_hub.kv_quantization import (
    CACHE_KEY,
    DefaultCachePolicyConfig,
    HadamardTransformConfig,
    KIVIConfig,
    KVQuantizationConfig,
    QuaRotKVConfig,
    quarot_int_quantizer,
    RegionConfig,
    TurboQuantKVConfig,
    TurboQuantQuantizerConfig,
)
from sparse_attention_hub.kv_quantization.metrics.quant_error import METRIC_IDS
from sparse_attention_hub.metric_logging import MicroMetricLogger
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)
from tests.integration.test_kv_quantization_flow import (
    tiny_model_dir as shared_tiny_model_dir,
)
from tests.integration.test_kv_quantization_workflow_matrix import (
    CODECS,
    prepare_calibration,
)

pytestmark = pytest.mark.integration
tiny_model_dir = shared_tiny_model_dir


@pytest.mark.parametrize("method_name", ["quarot", "turboquant"])
def test_invalid_decode_releases_prefill_store(
    tiny_model_dir, monkeypatch, method_name
):
    config = (
        QuaRotKVConfig(
            key_quantizer=quarot_int_quantizer("key_head", group_size=32),
            value_quantizer=quarot_int_quantizer("value_group", group_size=32),
        )
        if method_name == "quarot"
        else TurboQuantKVConfig()
    )
    adapter = huggingface.ModelAdapterHF(
        model_name=tiny_model_dir,
        sparse_attention_config=ResearchAttentionConfig(
            masker_configs=[], kv_quantization_config=config
        ),
        device="cpu",
    )
    method = adapter.sparse_attention.kv_quantization
    original_process = method.process
    original_clear = adapter.sparse_attention.clear_request_state
    released = []

    def invalid_decode(layer, keys, values, metadata, **kwargs):
        if kwargs["phase"] == "decode":
            keys = keys.clone()
            keys[..., -1, 0] = float("nan")
        return original_process(layer, keys, values, metadata, **kwargs)

    def clear(metadata):
        store = metadata[CACHE_KEY]
        assert store.layers
        original_clear(metadata)
        released.append((metadata, store))

    monkeypatch.setattr(method, "process", invalid_decode)
    monkeypatch.setattr(adapter.sparse_attention, "clear_request_state", clear)
    try:
        with pytest.raises(ValueError, match="NaN or infinity"):
            adapter.process_request(
                Request(context="w1 w2 w3 w4", questions="w5 w6", answer_prefix=""),
                generation_kwargs={"max_new_tokens": 2},
                request_kwargs={},
            )
        assert len(released) == 1
        metadata, store = released[0]
        assert CACHE_KEY not in metadata and not store.layers
    finally:
        adapter._cleanup_attention_registration()


class OfflineWorkflowBenchmark(Benchmark):
    benchmark_name = "preflight-offline-fixture"
    all_datasets = ["fixture"]
    huggingface_dataset_id = "local/preflight"

    def _load_datasets(self):
        return pd.DataFrame(
            {
                "context": ["w1 " * 80, "w1 " * 80, "w2 " * 83],
                "question": ["w3 w4", "w5 w6", "w7 w8"],
                "answer_prefix": ["", "", ""],
                "max_new_tokens": [4, 4, 4],
            }
        )

    def post_run_evaluate(self, results_df):
        assert len(results_df) == 3
        assert results_df["predicted_answer"].map(lambda x: isinstance(x, str)).all()
        return {"completed_fixture_questions": len(results_df)}


@pytest.mark.parametrize(
    "codec",
    [
        *CODECS,
        pytest.param(None, id="kivi"),
        pytest.param("quarot", id="quarot"),
        pytest.param("turboquant_mse", id="turboquant_mse"),
        pytest.param("turboquant_prod", id="turboquant_prod"),
    ],
)
@pytest.mark.parametrize("packed", [False, True])
def test_real_benchmark_worker_with_local_model_and_saved_outputs(
    codec, packed, tiny_model_dir, monkeypatch, tmp_path
):
    monkeypatch.setattr(MicroMetricLogger, "_instance", None)
    monkeypatch.setattr(MicroMetricLogger, "_initialized", False)
    logger = MicroMetricLogger()
    if codec is None:
        quant = KIVIConfig.with_settings(
            group_size=4,
            residual_length=8,
            store_codes=packed,
            error_metric=True,
        )
    elif codec in ("turboquant_mse", "turboquant_prod"):
        quant = TurboQuantKVConfig(
            key_quantizer=TurboQuantQuantizerConfig(variant=codec.rsplit("_", 1)[1]),
            store_codes=packed,
            error_metric=True,
        )
    elif isinstance(codec, str):
        quant = QuaRotKVConfig(
            key_quantizer=quarot_int_quantizer("key_head", group_size=32),
            value_quantizer=quarot_int_quantizer("value_group", group_size=32),
            store_codes=packed,
            error_metric=True,
        )
    else:
        quant = KVQuantizationConfig(
            key_quantizer=codec,
            value_quantizer=codec,
            key_transform=HadamardTransformConfig(),
            value_transform=HadamardTransformConfig(),
            cache_policy=DefaultCachePolicyConfig(
                key_local=RegionConfig(2),
                value_local=RegionConfig(2),
                key_block_tokens=8,
                value_block_tokens=8,
            ),
            store_codes=packed,
            error_metric=True,
        )
    config = ResearchAttentionConfig(masker_configs=[], kv_quantization_config=quant)
    queue, results, errors = Queue(), Queue(), Queue()
    queue.put(
        BenchmarkStub(
            model_name=tiny_model_dir,
            sparse_config_name="preflight",
            sparse_attention_config=config,
            benchmark_name="preflight-offline-fixture",
            subset="fixture",
            adapter_config=AdapterConfig(),
            generation_kwargs={"max_new_tokens": 4},
            request_kwargs={},
            result_dir=str(tmp_path),
        )
    )
    queue.put(None)
    captured, adapters, releases = [], [], []
    original_adapter = huggingface.ModelAdapterHF

    def cpu_adapter(**kwargs):
        # Load and execute the real model; only the GPU placement is substituted.
        adapter = original_adapter(**kwargs, device="cpu")
        adapters.append(adapter)
        prepare_calibration(adapter.sparse_attention.kv_quantization)
        original_clear = adapter.sparse_attention.clear_request_state

        def clear(metadata):
            store = metadata[CACHE_KEY]
            stats = store.stats()
            assert set(stats["layers"]) == {0, 1}
            assert all(i["key_compressed_tokens"] > 0 for i in stats["layers"].values())
            captured.append((metadata, store, stats))
            original_clear(metadata)

        monkeypatch.setattr(adapter.sparse_attention, "clear_request_state", clear)
        return adapter

    monkeypatch.setattr(huggingface, "ModelAdapterHF", cpu_adapter)
    monkeypatch.setattr(executor, "_shutdown_requested", False)
    monkeypatch.setattr(executor.signal, "signal", lambda *args: None)
    monkeypatch.setattr(executor, "set_gpu", lambda *args: nullcontext())
    monkeypatch.setattr(executor, "acquire_gpu_from_pool", lambda *args, **kwargs: 0)
    monkeypatch.setattr(executor, "cleanup_gpu_memory", lambda *args: None)
    monkeypatch.setattr(
        executor, "release_gpu_to_pool", lambda *args: releases.append(0)
    )
    monkeypatch.setattr(
        executor,
        "create_benchmark_instance",
        lambda **kwargs: OfflineWorkflowBenchmark(),
    )
    try:
        executor._benchmark_worker(queue, Queue(), results, errors)
        assert errors.empty(), errors.get().error_message
        assert results.qsize() == 1 and releases == [0]
        assert results.get().metrics == {"completed_fixture_questions": 3}
        assert len(captured) == 3
        assert len({id(store) for _, store, _ in captured}) == 3
        assert all(
            CACHE_KEY not in meta and not store.layers for meta, store, _ in captured
        )
        assert len(pd.read_csv(tmp_path / "raw_results.csv")) == 3
        assert json.loads((tmp_path / "metrics.json").read_text()) == {
            "completed_fixture_questions": 3
        }
        exported = json.loads((tmp_path / "config.json").read_text())
        saved_quant = exported["sparse_attention_config"]["kv_quantization_config"]
        assert saved_quant["store_codes"] == packed
        expected_class = (
            "KIVIConfig"
            if codec is None
            else "TurboQuantKVConfig"
            if codec in ("turboquant_mse", "turboquant_prod")
            else "QuaRotKVConfig"
            if isinstance(codec, str)
            else "KVQuantizationConfig"
        )
        assert saved_quant["__type__"].endswith("." + expected_class)
        events = [
            json.loads(line)
            for line in (tmp_path / "micro_metrics.jsonl").read_text().splitlines()
        ]
        assert set(METRIC_IDS).issubset(event["metric"] for event in events)
        assert all(torch.isfinite(torch.tensor(event["value"])) for event in events)
    finally:
        logger.log_queue.clear()
        logger.log_path = None
        for adapter in adapters:
            adapter._cleanup_attention_registration()
