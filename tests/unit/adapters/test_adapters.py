"""Unit tests for the adapter implementation."""

from unittest.mock import Mock, patch

import pytest
import torch

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
