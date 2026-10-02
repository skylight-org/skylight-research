"""Tests that the quantize_kv_cache flag is wired through ModelAdapterHF.

These run without a GPU and without nvidia-modelopt: the quantized-cache module
is replaced by a stub, so what is under test is the adapter's plumbing (does the
flag reach the forward pass) rather than the quantization numerics, which
``test_fake_nvfp4.py`` and ``test_nvfp4_cache.py`` cover.
"""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch

from sparse_attention_hub.adapters.huggingface import ModelAdapterHF

STUB_MODULE_NAME: str = "sparse_attention_hub.kv_quantization"


class _SentinelCache:
    """Stands in for a quantized cache so call sites can be identified by type."""

    def __init__(self, config: Any, backend: str, **kwargs: Dict[str, Any]) -> None:
        self.config = config
        self.backend = backend
        self.kwargs = kwargs


@pytest.fixture
def stub_kv_quantization(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Install a stub kv_quantization module so modelopt is not needed."""
    module: ModuleType = ModuleType(STUB_MODULE_NAME)
    module.create_quantized_kv_cache = (  # type: ignore[attr-defined]
        lambda config, backend, **kwargs: _SentinelCache(config, backend, **kwargs)
    )
    module.validate_backend = lambda backend: backend  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, STUB_MODULE_NAME, module)
    return module


def _make_adapter(**attrs: Any) -> ModelAdapterHF:
    """Build an adapter without loading a model.

    ``ModelAdapterHF.__init__`` downloads and loads a model, which these tests
    do not need, so the instance is created bare and only the attributes under
    test are set.
    """
    adapter: ModelAdapterHF = object.__new__(ModelAdapterHF)
    adapter.model = SimpleNamespace(config=SimpleNamespace(name="stub-config"))
    adapter.quantize_kv_cache = False
    adapter.kv_quantization_kwargs = {}
    # __del__ unregisters attention functions and reads these
    adapter._registered_attention_name = None
    adapter._custom_attention_fn = None
    for name, value in attrs.items():
        setattr(adapter, name, value)
    return adapter


@pytest.mark.unit
class TestCreateKVCache:
    """Behaviour of ModelAdapterHF._create_kv_cache."""

    def test_false_uses_huggingface_default(self) -> None:
        """quantize_kv_cache=False means the adapter builds no cache of its own."""
        adapter: ModelAdapterHF = _make_adapter(quantize_kv_cache=False)
        assert adapter._create_kv_cache() is None

    @pytest.mark.parametrize("backend", ["nvfp4", "fake_nvfp4"])
    def test_backend_name_selects_that_backend(
        self, backend: str, stub_kv_quantization: ModuleType
    ) -> None:
        """The flag's value is passed through as the backend to build."""
        adapter: ModelAdapterHF = _make_adapter(quantize_kv_cache=backend)
        cache: Any = adapter._create_kv_cache()
        assert isinstance(cache, _SentinelCache)
        assert cache.backend == backend
        assert cache.config is adapter.model.config

    def test_kwargs_are_forwarded(self, stub_kv_quantization: ModuleType) -> None:
        """Cache options reach the cache constructor."""
        adapter: ModelAdapterHF = _make_adapter(
            quantize_kv_cache="fake_nvfp4",
            kv_quantization_kwargs={"axis_key": 0, "residual_length": 64},
        )
        cache: Any = adapter._create_kv_cache()
        assert cache.kwargs == {"axis_key": 0, "residual_length": 64}

    def test_each_call_returns_a_fresh_cache(
        self, stub_kv_quantization: ModuleType
    ) -> None:
        """Caches hold per-request state, so they must not be shared."""
        adapter: ModelAdapterHF = _make_adapter(quantize_kv_cache="fake_nvfp4")
        assert adapter._create_kv_cache() is not adapter._create_kv_cache()


class _RecordingModel:
    """Minimal stand-in for an HF causal LM that records what it is called with."""

    def __init__(self, vocab_size: int = 16) -> None:
        self.config = SimpleNamespace(name="stub-config")
        self.device = torch.device("cpu")
        self.generation_config = SimpleNamespace(eos_token_id=[vocab_size - 1])
        self.vocab_size = vocab_size
        self.calls: List[Dict[str, Any]] = []

    def eval(self) -> "_RecordingModel":
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            past_key_values=kwargs.get("past_key_values"),
            logits=torch.zeros(1, 1, self.vocab_size),
        )


class _StubTokenizer:
    """Tokenizer stub: encodes to a fixed short sequence, decodes to a marker."""

    chat_template = None
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(
        self, text: str, return_tensors: str = "pt", add_special_tokens: bool = True
    ) -> torch.Tensor:
        return torch.zeros(1, 4, dtype=torch.long)

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "stub-answer"


@pytest.mark.unit
class TestProcessRequestUsesCache:
    """The cache built from the flag must reach the prefill forward pass."""

    def _run(self, quantize: Any) -> _RecordingModel:
        from sparse_attention_hub.adapters.base import Request

        model: _RecordingModel = _RecordingModel()
        adapter: ModelAdapterHF = _make_adapter(
            quantize_kv_cache=quantize,
            model=model,
            tokenizer=_StubTokenizer(),
            device="cpu",
            hybrid=False,
            random_separator="SEP",
            _sparse_attention_available=False,
        )
        adapter.process_request(
            request=Request(context="ctx", questions="q", answer_prefix=""),
            generation_kwargs={"max_new_tokens": 1},
            request_kwargs={"max_context_length": 128},
        )
        return model

    def test_prefill_gets_quantized_cache_when_enabled(
        self, stub_kv_quantization: ModuleType
    ) -> None:
        """The context prefill receives the quantized cache."""
        model: _RecordingModel = self._run(quantize="fake_nvfp4")
        assert isinstance(model.calls[0]["past_key_values"], _SentinelCache)

    def test_prefill_gets_none_when_disabled(self) -> None:
        """Without the flag the prefill passes None, HF's default behaviour."""
        model: _RecordingModel = self._run(quantize=False)
        assert model.calls[0]["past_key_values"] is None
