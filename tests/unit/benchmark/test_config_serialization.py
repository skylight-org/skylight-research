"""Stable, readable experiment configuration and real benchmark export."""

import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from benchmark.base import Benchmark
from benchmark.utils.data import make_serializable
from sparse_attention_hub.adapters import RequestResponse
from sparse_attention_hub.kv_quantization import (
    KIVIConfig,
    KVQuantizationConfig,
    ParamDType,
    QuantGranularity,
    QuaRotKVConfig,
    TurboQuantKVConfig,
    TurboQuantQuantizerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)

pytestmark = pytest.mark.unit


class Label(Enum):
    TOKEN = "per_token"


class Number(IntEnum):
    TWO = 2


@pytest.mark.parametrize(
    "value,expected",
    [
        (Label.TOKEN, "per_token"),
        (Number.TWO, 2),
        (QuantGranularity.PER_CHANNEL, "per_channel"),
        (ParamDType.FP16, "fp16"),
    ],
)
def test_enum_values_are_exported_as_primitives(value, expected):
    result = make_serializable(value)
    assert result == expected and type(result) is type(expected)


@pytest.mark.parametrize(
    "config",
    [KVQuantizationConfig(), KIVIConfig(), QuaRotKVConfig(), TurboQuantKVConfig()],
)
def test_nested_configs_keep_class_identity_and_are_stable_across_instances(config):
    result = make_serializable(config)
    assert (
        result["__type__"] == f"{type(config).__module__}.{type(config).__qualname__}"
    )
    assert result["key_quantizer"]["__type__"].endswith(
        "." + type(config.key_quantizer).__name__
    )
    if isinstance(config, KIVIConfig):
        # kivi_int_quantizer(): the preset's numerical settings are recorded.
        assert result["key_quantizer"]["compute_dtype"] == "input"
        assert result["key_quantizer"]["parameter_storage"] == "compute"
        assert result["key_quantizer"]["degenerate_range"] == "zero_scale"
        assert result["key_quantizer"]["decode_dtype"] == "fp16"
    if isinstance(config, QuaRotKVConfig):
        # quarot_int_quantizer("key_head") / ("value_group").
        assert result["key_quantizer"]["compute_dtype"] == "fp32"
        assert result["key_quantizer"]["range_includes_zero"] is True
        assert result["key_quantizer"]["zero_point"] == "integer"
        assert result["key_quantizer"]["clip_ratio"] == 0.95
        assert result["value_quantizer"]["compute_dtype"] == "input"
        assert result["value_quantizer"]["range_includes_zero"] is False
        assert result["key_transform"]["preserve_input_dtype"] is True
        assert result["quantize_prefill"] is True
    if isinstance(config, TurboQuantKVConfig):
        assert result["key_quantizer"]["parameter_dtype"] == "fp32"
        assert result["key_quantizer"]["variant"] == "mse"
        assert result["key_quantizer"]["codebook_distribution"] == "sphere_beta"
        assert (
            result["key_quantizer"]["reference"] == "arxiv:2504.19874v1:algorithms-1-2"
        )
        assert result["key_transform"]["__type__"].endswith(".IdentityTransformConfig")
    assert result["key_quantizer"]["scale_dtype"] == "fp16"
    assert isinstance(result["key_quantizer"]["granularity"], str)
    text = json.dumps(result, sort_keys=True, allow_nan=False)
    assert "_value_" not in text and "_name_" not in text and "0x" not in text
    assert text == json.dumps(
        make_serializable(type(config)()), sort_keys=True, allow_nan=False
    )


def test_dataclass_subclass_identity_is_not_lost_in_parent_configs():
    @dataclass
    class Experiment:
        quantization: KVQuantizationConfig

    result = make_serializable(Experiment(KIVIConfig()))
    assert result["quantization"]["__type__"].endswith(".KIVIConfig")


def test_turboquant_variant_and_seed_are_visible_in_exported_configuration():
    config = TurboQuantKVConfig(
        key_quantizer=TurboQuantQuantizerConfig(variant="prod", bits=4, qjl_seed=19)
    )
    result = json.loads(json.dumps(make_serializable(config), allow_nan=False))
    assert result["key_quantizer"]["variant"] == "prod"
    assert result["key_quantizer"]["bits"] == 4
    assert result["key_quantizer"]["qjl_seed"] == 19
    assert result["key_quantizer"]["numerics_version"].startswith("fp32-beta-")


def test_types_functions_devices_and_paths_have_stable_representations():
    value = {
        "type": KIVIConfig,
        "function": make_serializable,
        "dtype": torch.bfloat16,
        "device": torch.device("cpu"),
        "path": Path("results/run"),
        "opaque": object(),
    }
    result = make_serializable(value)
    assert result["type"].endswith(".KIVIConfig")
    assert result["function"] == "benchmark.utils.data.make_serializable"
    assert result["dtype"] == "torch.bfloat16" and result["device"] == "cpu"
    assert result["path"] == "results/run" and result["opaque"] == {
        "__type__": "builtins.object"
    }
    assert "0x" not in json.dumps(result)


def test_nested_collections_and_enum_dictionary_keys_are_serializable():
    value = {Label.TOKEN: [ParamDType.FP32, (Number.TWO, None)], "set": {3, 1, 2}}
    assert json.loads(json.dumps(make_serializable(value))) == {
        "per_token": ["fp32", [2, None]],
        "set": [1, 2, 3],
    }


class FixtureBenchmark(Benchmark):
    benchmark_name = "offline-config-fixture"
    all_datasets = ["fixture"]
    huggingface_dataset_id = "local/fixture"

    def _load_datasets(self):
        return pd.DataFrame(
            {
                "context": ["context"],
                "question": ["question"],
                "answer_prefix": [""],
                "max_new_tokens": [1],
            }
        )

    def post_run_evaluate(self, results_df):
        assert results_df["predicted_answer"].tolist() == ["answer"]
        return {"fixture_score": 1.0}


def test_real_benchmark_writes_readable_method_configuration(tmp_path):
    sparse = ResearchAttentionConfig(
        masker_configs=[], kv_quantization_config=KIVIConfig()
    )
    adapter = SimpleNamespace(
        sparse_attention_config=sparse,
        model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer_kwargs={},
        process_request=lambda *args: RequestResponse(responses=["answer"]),
    )
    result = FixtureBenchmark().run_benchmark(
        adapter,
        str(tmp_path),
        generation_kwargs={"max_new_tokens": 1},
        request_kwargs={},
    )
    assert result == {"fixture_score": 1.0}
    text = (tmp_path / "config.json").read_text()
    data = json.loads(text)
    kv = data["sparse_attention_config"]["kv_quantization_config"]
    assert kv["__type__"].endswith(".KIVIConfig")
    assert kv["key_quantizer"]["granularity"] == "per_channel"
    assert kv["key_quantizer"]["scale_dtype"] == "fp16"
    assert data["model_kwargs"]["torch_dtype"] == "torch.bfloat16"
    assert "0x" not in text and "_value_" not in text
    assert (tmp_path / "raw_results.csv").exists() and (
        tmp_path / "metrics.json"
    ).exists()


def test_benchmark_config_exports_stable_primitive_enum_values():
    from benchmark.utils.data import make_serializable

    result = make_serializable(KVQuantizationConfig())
    assert result["key_quantizer"]["granularity"] == "per_token"
    assert result["key_quantizer"]["scale_dtype"] == "fp16"
