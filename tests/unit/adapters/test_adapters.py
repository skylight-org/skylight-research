"""Unit tests for the adapter implementation."""

from pathlib import Path
from typing import Any, List, Optional, Tuple
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import (
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
)

from sparse_attention_hub.adapters import (
    ModelAdapter,
    ModelAdapterHF,
    ModelHubAdapterInterface,
    Request,
    RequestResponse,
    SparseAttentionAdapterInterface,
)
from sparse_attention_hub.adapters.model_servers.base import ModelServer
from sparse_attention_hub.sparse_attention import (
    LocalMaskerConfig,
    ResearchAttentionConfig,
    SparseAttentionConfig,
)


@pytest.mark.unit
class TestRequest:
    """Test the Request class."""

    def test_request_single_question(self) -> None:
        """Test Request with single question."""
        request = Request(
            context="This is a test context.",
            questions="What is this test about?",
            answer_prefix="Answer: ",
        )

        assert request.context == "This is a test context."
        assert request.questions == "What is this test about?"
        assert isinstance(request.questions, str)

    def test_request_multiple_questions(self) -> None:
        """Test Request with multiple questions."""
        questions = ["Question 1?", "Question 2?", "Question 3?"]
        request = Request(
            context="Test context", questions=questions, answer_prefix="Answer: "
        )

        assert request.context == "Test context"
        assert request.questions == questions
        assert isinstance(request.questions, list)
        assert len(request.questions) == 3

    def test_request_empty_context(self) -> None:
        """Test Request with empty context."""
        request = Request(
            context="", questions="What is this test about?", answer_prefix="Answer: "
        )

        assert request.context == ""
        assert request.questions == "What is this test about?"

    def test_request_empty_questions(self) -> None:
        """Test Request with empty questions list."""
        request = Request(
            context="Test context", questions=[], answer_prefix="Answer: "
        )

        assert request.context == "Test context"
        assert request.questions == []
        assert isinstance(request.questions, list)
        assert len(request.questions) == 0


@pytest.mark.unit
class TestRequestResponse:
    """Test the RequestResponse class."""

    def test_response_single_answer(self) -> None:
        """Test RequestResponse with single answer."""
        response = RequestResponse(responses="This is a test response.")

        assert response.responses == "This is a test response."
        assert isinstance(response.responses, str)

    def test_response_multiple_answers(self) -> None:
        """Test RequestResponse with multiple answers."""
        answers = ["Answer 1", "Answer 2", "Answer 3"]
        response = RequestResponse(responses=answers)

        assert response.responses == answers
        assert isinstance(response.responses, list)
        assert len(response.responses) == 3

    def test_response_empty_answer(self) -> None:
        """Test RequestResponse with empty answer."""
        response = RequestResponse(responses="")

        assert response.responses == ""
        assert isinstance(response.responses, str)

    def test_response_empty_answers_list(self) -> None:
        """Test RequestResponse with empty answers list."""
        response = RequestResponse(responses=[])

        assert response.responses == []
        assert isinstance(response.responses, list)
        assert len(response.responses) == 0


@pytest.mark.unit
class TestInterfaces:
    """Test the interface definitions."""

    def test_model_hub_adapter_interface(self) -> None:
        """Test ModelHubAdapterInterface is abstract."""
        with pytest.raises(TypeError):
            ModelHubAdapterInterface()

    def test_sparse_attention_adapter_interface(self) -> None:
        """Test SparseAttentionAdapterInterface is abstract."""
        with pytest.raises(TypeError):
            SparseAttentionAdapterInterface()

    def test_model_adapter_is_abstract(self) -> None:
        """Test ModelAdapter is abstract."""
        config = SparseAttentionConfig()
        with pytest.raises(TypeError):
            ModelAdapter(config, "test-model")

    def test_model_hub_adapter_interface_methods(self) -> None:
        """Test ModelHubAdapterInterface has required abstract methods."""
        interface_methods = dir(ModelHubAdapterInterface)
        assert "process_request" in interface_methods

    def test_sparse_attention_adapter_interface_methods(self) -> None:
        """Test SparseAttentionAdapterInterface has required abstract methods."""
        interface_methods = dir(SparseAttentionAdapterInterface)
        assert "get_custom_attention_function" in interface_methods


@pytest.mark.unit
class TestModelAdapterHF:
    """Test the ModelAdapterHF class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.masker_config = LocalMaskerConfig(window_size=10)
        self.sparse_attention_config = ResearchAttentionConfig(
            masker_configs=[self.masker_config]
        )
        ModelServer._instance = None

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelServer._instance = None

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_create_model(self, mock_tokenizer, mock_model) -> None:
        """Test model creation."""
        # Mock the tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<EOS>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Create adapter
        adapter = ModelAdapterHF(
            sparse_attention_config=self.sparse_attention_config,
            model_name="test-model",
            model_kwargs={"torch_dtype": torch.float16},
            device=0,
        )

        # Check that model and tokenizer were created
        assert adapter.model is not None
        assert adapter.tokenizer is not None

        # Check that pad_token was set
        assert adapter.tokenizer.pad_token == "<EOS>"

        # Check that the correct methods were called
        mock_tokenizer.from_pretrained.assert_called_once_with("test-model")
        mock_model.from_pretrained.assert_called_once_with(
            "test-model", torch_dtype=torch.float16
        )

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_create_model_with_torch_dtype(self, mock_tokenizer, mock_model) -> None:
        """Test model creation with torch_dtype parameter."""

        # Mock the tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<EOS>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Create adapter
        adapter = ModelAdapterHF(
            sparse_attention_config=self.sparse_attention_config,
            model_name="test-model",
            model_kwargs={"torch_dtype": torch.float16},
            device="cpu",
        )

        # Check that model was created with correct parameters
        mock_model.from_pretrained.assert_called_once_with(
            "test-model", torch_dtype=torch.float16
        )
        assert adapter.device == "cpu"

    @patch("sparse_attention_hub.adapters.huggingface.ModelServerHF")
    def test_allow_unregistered_models_plumbed(self, mock_model_server_hf) -> None:
        """ModelAdapterHF should forward allow_unregistered_models into ModelServerConfig."""
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "<PAD>"
        mock_tokenizer_instance.eos_token = "<EOS>"
        mock_model_server = Mock()
        mock_model_server.get_tokenizer.return_value = mock_tokenizer_instance
        mock_model_server.get_model.return_value = Mock()
        mock_model_server_hf.return_value = mock_model_server

        adapter = ModelAdapterHF(
            sparse_attention_config=self.sparse_attention_config,
            model_name="test-model",
            model_registry_path="/tmp/registry.yaml",
            allow_unregistered_models=False,
            model_kwargs={"torch_dtype": torch.float16},
        )

        mock_model_server_hf.assert_called_once()
        config_arg = mock_model_server_hf.call_args[0][0]
        assert config_arg.model_registry_path == "/tmp/registry.yaml"
        assert config_arg.allow_unregistered_models is False
        assert adapter.tokenizer.pad_token == "<PAD>"
        assert adapter.torch_dtype == torch.float16
        # Check that adapter was created successfully
        assert adapter is not None

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_create_model_with_existing_pad_token(
        self, mock_tokenizer, mock_model
    ) -> None:
        """Test model creation when tokenizer already has pad_token."""
        # Mock the tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "<PAD>"
        mock_tokenizer_instance.eos_token = "<EOS>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Create adapter
        adapter = ModelAdapterHF(
            sparse_attention_config=self.sparse_attention_config,
            model_name="test-model",
        )
        assert adapter.torch_dtype is not None
        assert adapter.model is not None
        assert adapter.tokenizer is not None
        # Check that pad_token was not changed
        assert adapter.tokenizer.pad_token == "<PAD>"

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_get_custom_attention_function(self, mock_tokenizer, mock_model) -> None:
        """Test get_custom_attention_function returns a callable."""
        # Mock the tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        adapter = ModelAdapterHF(
            sparse_attention_config=self.sparse_attention_config,
            model_name="test-model",
            model_kwargs={"torch_dtype": torch.float16},
        )
        assert adapter.torch_dtype == torch.float16
        assert adapter.model is not None
        assert adapter.tokenizer is not None
        custom_fn = adapter.get_custom_attention_function(adapter.sparse_attention)
        assert callable(custom_fn)

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_custom_attention_forwards_explicit_softcap(
        self, mock_tokenizer, mock_model
    ) -> None:
        """Softcap passed by caller should be forwarded unchanged to sparse attention."""
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        adapter = ModelAdapterHF(
            sparse_attention_config=self.sparse_attention_config,
            model_name="test-model",
            model_kwargs={"torch_dtype": torch.float16},
        )

        custom_fn = adapter.get_custom_attention_function(adapter.sparse_attention)

        queries = torch.randn(1, 1, 2, 4)
        keys = torch.randn(1, 1, 2, 4)
        values = torch.randn(1, 1, 2, 4)

        adapter.sparse_attention.custom_attention = Mock(
            return_value=(torch.zeros_like(queries), None)
        )

        module = torch.nn.Module()

        custom_fn(
            module=module,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
            sparse_meta_data={},
            softcap=30.0,
        )

        assert adapter.sparse_attention.custom_attention.call_count == 1
        assert (
            adapter.sparse_attention.custom_attention.call_args.kwargs["softcap"]
            == 30.0
        )

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_generate_unique_attention_name(self, mock_tokenizer, mock_model) -> None:
        """Test unique attention name generation."""
        # Mock the tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        adapter = ModelAdapterHF(
            sparse_attention_config=self.sparse_attention_config,
            model_name="test-model",
        )

        name1 = adapter._generate_unique_attention_name()
        name2 = adapter._generate_unique_attention_name()

        assert isinstance(name1, str)
        assert isinstance(name2, str)
        assert name1.startswith("sparse_attention_")
        assert name2.startswith("sparse_attention_")
        assert name1 != name2  # Should be unique

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_empty_masker_config_still_uses_backend(
        self, mock_tokenizer, mock_model
    ) -> None:
        """Empty masker configs should still initialize sparse backend (for softcap, etc)."""
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        adapter = ModelAdapterHF(
            sparse_attention_config=ResearchAttentionConfig(masker_configs=[]),
            model_name="test-model",
        )

        # Sparse backend should be created even with empty masker_configs
        assert adapter.sparse_attention is not None
        assert adapter._sparse_attention_available is True

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_enable_sparse_mode_when_not_available(
        self, mock_tokenizer, mock_model
    ) -> None:
        """Test enable_sparse_mode raises error when sparse attention is not available."""
        # Mock the tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Create adapter without sparse attention (None config)
        adapter = ModelAdapterHF(
            sparse_attention_config=None,
            model_name="test-model",
        )

        with pytest.raises(RuntimeError) as exc_info:
            with adapter.enable_sparse_mode():
                pass

        assert "Cannot enable sparse mode: sparse attention is not available" in str(
            exc_info.value
        )

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_enable_sparse_and_dense_modes(self, mock_tokenizer, mock_model) -> None:
        """Test enable_sparse_mode and enable_dense_mode context managers."""
        # Mock the tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        # Ensure named_modules is a proper mock method that returns an empty iterator
        mock_model_instance.named_modules = Mock(return_value=iter([]))
        mock_model.from_pretrained.return_value = mock_model_instance

        adapter = ModelAdapterHF(
            sparse_attention_config=self.sparse_attention_config,
            model_name="test-model",
        )

        # Override the model's named_modules method directly
        adapter.model.named_modules = Mock(return_value=[])

        # Test sparse mode context manager works
        with adapter.enable_sparse_mode():
            # Should not raise any errors
            pass

        # Test that custom attention name is reused
        with adapter.enable_sparse_mode():
            first_name = adapter._registered_attention_name

        with adapter.enable_sparse_mode():
            second_name = adapter._registered_attention_name

        assert first_name == second_name

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_inheritance(self, mock_tokenizer, mock_model) -> None:
        """Test that ModelAdapterHF properly inherits from ModelAdapter."""
        # Mock the tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        adapter = ModelAdapterHF(
            sparse_attention_config=self.sparse_attention_config,
            model_name="test-model",
        )

        assert isinstance(adapter, ModelAdapter)
        assert isinstance(adapter, ModelHubAdapterInterface)
        assert isinstance(adapter, SparseAttentionAdapterInterface)

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_adapter_properties(self, mock_tokenizer, mock_model) -> None:
        """Test adapter properties are set correctly."""
        # Mock the tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        adapter = ModelAdapterHF(
            sparse_attention_config=self.sparse_attention_config,
            model_name="test-model",
        )
        # Check properties
        assert adapter.model_name == "test-model"
        assert adapter.sparse_attention_config == self.sparse_attention_config
        assert adapter.sparse_attention is not None
        assert adapter._registered_attention_name is None


@pytest.mark.unit
class TestSoftcapEndToEndTinyGemma:
    """End-to-end dense-vs-sparse-backend logit equivalence on tiny Gemma models.

    With an empty masker list, the research-attention backend must reproduce the
    dense eager logits exactly (up to fp32 numerical noise). This covers the
    softcap plumbing end-to-end: Gemma-2 applies attention-logit softcapping
    (which the sparse backend must honor), while Gemma-4 does not (softcap must
    stay None all the way down).
    """

    def setup_method(self) -> None:
        """Reset the ModelServer singleton so each test loads its own model."""
        ModelServer._instance = None

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelServer._instance = None

    @staticmethod
    def _build_adapter(model_dir: Path) -> ModelAdapterHF:
        """Build a ModelAdapterHF (empty masker list) around a saved tiny model.

        Only AutoTokenizer is mocked (we never tokenize -- input_ids are fed
        directly); AutoModelForCausalLM loads the real model from model_dir.
        """
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = "<PAD>"
        with patch(
            "sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer"
        ) as mock_tokenizer:
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            adapter = ModelAdapterHF(
                model_name=str(model_dir),
                sparse_attention_config=ResearchAttentionConfig(masker_configs=[]),
                model_kwargs={
                    "torch_dtype": torch.float32,
                    "attn_implementation": "eager",
                },
                device="cpu",
            )
        return adapter

    @staticmethod
    def _dense_and_sparse_logits(
        adapter: ModelAdapterHF, input_ids: torch.Tensor, **forward_kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[float]]]:
        """Run a dense forward and a sparse-mode forward on the same inputs.

        The sparse run wraps sparse_attention.custom_attention with a spy that
        records the softcap kwarg seen on every attention call.

        Returns:
            Tuple of (dense_logits, sparse_logits, recorded_softcaps).
        """
        recorded_softcaps: List[Optional[float]] = []
        original_custom_attention = adapter.sparse_attention.custom_attention

        def spying_custom_attention(*args: Any, **kwargs: Any) -> Any:
            recorded_softcaps.append(kwargs.get("softcap"))
            return original_custom_attention(*args, **kwargs)

        adapter.model.eval()
        with torch.no_grad():
            dense_logits = adapter.model(input_ids, **forward_kwargs).logits
            adapter.sparse_attention.custom_attention = spying_custom_attention
            try:
                with adapter.enable_sparse_mode():
                    sparse_logits = adapter.model(
                        input_ids, sparse_meta_data={}, **forward_kwargs
                    ).logits
            finally:
                adapter.sparse_attention.custom_attention = original_custom_attention
        return dense_logits, sparse_logits, recorded_softcaps

    def test_tiny_gemma2_softcap_dense_vs_sparse_empty(self, tmp_path: Path) -> None:
        """Tiny Gemma-2 with an aggressive softcap: sparse backend matches dense.

        attn_logit_softcapping=0.05 forces tanh saturation, so the comparison is
        genuinely sensitive to softcap handling in the sparse backend.
        """
        config = Gemma2Config(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            sliding_window=8,
            attn_logit_softcapping=0.05,
            query_pre_attn_scalar=1,
            final_logit_softcapping=None,
        )
        model = Gemma2ForCausalLM(config)
        model_dir = tmp_path / "m"
        model.save_pretrained(model_dir)

        adapter = self._build_adapter(model_dir)
        generator = torch.Generator().manual_seed(0)
        input_ids = torch.randint(0, 256, (1, 24), generator=generator)

        dense_logits, sparse_logits, recorded_softcaps = self._dense_and_sparse_logits(
            adapter, input_ids
        )

        # Sensitivity guard: with softcapping disabled the dense logits must
        # change materially, proving the comparison actually exercises softcap.
        attention_modules = [layer.self_attn for layer in adapter.model.model.layers]
        saved_softcaps = [attn.attn_logit_softcapping for attn in attention_modules]
        for attn in attention_modules:
            attn.attn_logit_softcapping = None
        try:
            with torch.no_grad():
                dense_logits_nocap = adapter.model(input_ids).logits
        finally:
            for attn, softcap in zip(attention_modules, saved_softcaps):
                attn.attn_logit_softcapping = softcap
        assert (dense_logits - dense_logits_nocap).abs().max().item() > 1e-3

        # Spy: the sparse backend must have seen softcap=0.05 on every call.
        assert len(recorded_softcaps) > 0
        assert all(softcap == 0.05 for softcap in recorded_softcaps)

        assert torch.allclose(dense_logits, sparse_logits, atol=1e-5, rtol=1e-5)

    def test_tiny_gemma4_dense_vs_sparse_empty(self, tmp_path: Path) -> None:
        """Tiny Gemma-4 (mixed sliding/full layers, no softcap): sparse matches dense."""
        config = Gemma4TextConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            sliding_window=8,
            num_kv_shared_layers=0,
            # Keep the per-layer input embedding tiny (defaults are huge).
            vocab_size_per_layer_input=256,
            hidden_size_per_layer_input=16,
        )
        assert "sliding_attention" in config.layer_types
        assert "full_attention" in config.layer_types

        model = Gemma4ForCausalLM(config)
        model_dir = tmp_path / "m"
        model.save_pretrained(model_dir)

        adapter = self._build_adapter(model_dir)
        generator = torch.Generator().manual_seed(1)
        # seq len 24 > sliding_window 8, so sliding layers genuinely mask.
        input_ids = torch.randint(0, 256, (1, 24), generator=generator)

        dense_logits, sparse_logits, recorded_softcaps = self._dense_and_sparse_logits(
            adapter, input_ids
        )

        # Gemma-4 has no attention-logit softcapping: every call must see None.
        assert len(recorded_softcaps) > 0
        assert all(softcap is None for softcap in recorded_softcaps)

        assert torch.allclose(dense_logits, sparse_logits, atol=1e-5, rtol=1e-5)

    def test_tiny_gemma4_kv_shared_dense_vs_sparse_empty(self, tmp_path: Path) -> None:
        """Tiny Gemma-4 with KV-shared layers: sparse matches dense with use_cache."""
        config = Gemma4TextConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            sliding_window=8,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            num_kv_shared_layers=2,
            # Keep the per-layer input embedding tiny (defaults are huge).
            vocab_size_per_layer_input=256,
            hidden_size_per_layer_input=16,
        )
        model = Gemma4ForCausalLM(config)
        model_dir = tmp_path / "m"
        model.save_pretrained(model_dir)

        adapter = self._build_adapter(model_dir)
        generator = torch.Generator().manual_seed(2)
        input_ids = torch.randint(0, 256, (1, 24), generator=generator)

        # KV-shared layers reuse cached keys/values, so run with use_cache=True
        # in BOTH paths.
        dense_logits, sparse_logits, recorded_softcaps = self._dense_and_sparse_logits(
            adapter, input_ids, use_cache=True
        )

        assert len(recorded_softcaps) > 0
        assert all(softcap is None for softcap in recorded_softcaps)

        assert torch.allclose(dense_logits, sparse_logits, atol=1e-5, rtol=1e-5)
