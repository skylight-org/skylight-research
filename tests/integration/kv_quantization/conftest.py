"""Shared fixtures for the heavy GPU suites.

Everything here needs CUDA; tests skip cleanly without it. Measurements are
collected into one JSON report (not just pass/fail), written at the end of the
session to ``$KVQ_REPORT_PATH`` (default ``./gpu_validation_report.json``), so
results can be uploaded and copied into the docs.

Environment variables:
    KVQ_MODEL: model for the real-model tests (default ``Qwen/Qwen3-4B``).
    KVQ_PPL_WINDOWS: WikiText-2 windows of 2048 tokens for perplexity (default 20).
    KVQ_LONG_CONTEXT: prompt length for the long-context test (default 32768).
    KVQ_HEAVY_NUMEL: elements per synthetic distribution (default 2**22).
    KVQ_REPORT_PATH: where to write the JSON report.
"""

import json
import os
import platform
from typing import Any, Dict

import pytest
import torch

REPORT: Dict[str, Any] = {}

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    if not REPORT:
        return
    REPORT.setdefault("environment", _environment())
    path: str = os.environ.get("KVQ_REPORT_PATH", "gpu_validation_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(REPORT, handle, indent=2, default=str)


def _environment() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["compute_capability"] = ".".join(
            map(str, torch.cuda.get_device_capability(0))
        )
        info["cuda"] = torch.version.cuda
    for name in ("transformers", "modelopt", "datasets"):
        try:
            info[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - optional packages
            info[name] = None
    return info


@pytest.fixture(scope="session")
def report() -> Dict[str, Any]:
    """The session-wide measurement report; tests add their numbers to it."""
    return REPORT


@pytest.fixture(scope="session")
def model_and_tokenizer() -> Any:
    """The real model (bf16, on GPU) and its tokenizer, loaded once."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name: str = os.environ.get("KVQ_MODEL", "Qwen/Qwen3-4B")
    tokenizer: Any = AutoTokenizer.from_pretrained(name)
    model: Any = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    REPORT.setdefault("model", name)
    return model, tokenizer


@pytest.fixture(scope="session")
def wikitext_test_ids(model_and_tokenizer: Any) -> torch.Tensor:
    from sparse_attention_hub.kv_quantization.evaluation import wikitext2_test_ids

    return wikitext2_test_ids(model_and_tokenizer[1])
