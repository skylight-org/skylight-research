"""Integration tests for the adapter implementation."""

from unittest.mock import Mock, patch

import pytest
import psutil
import torch

from sparse_attention_hub.adapters import ModelAdapterHF, Request, RequestResponse
from sparse_attention_hub.sparse_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
    LocalMaskerConfig,
    SinkMaskerConfig,
)


@pytest.fixture
def masker_config():
    """Create a masker configuration for testing."""
    return LocalMaskerConfig(window_size=5)


@pytest.fixture
def sparse_attention_config(masker_config):
    """Create a sparse attention configuration for testing."""
    return ResearchAttentionConfig(masker_configs=[masker_config])


@pytest.fixture
def sparse_attention_config_multi():
    """Create a sparse attention configuration with multiple maskers for testing."""
    return ResearchAttentionConfig(
        masker_configs=[
            LocalMaskerConfig(window_size=128),
            SinkMaskerConfig(sink_size=128),
        ]
    )


@pytest.fixture
def model_name():
    """Model name for testing."""
    return "Qwen/Qwen3-1.7B"


class TestAdapterIntegration:
    """Integration tests for the adapter system with mocked components."""

    def setup_method(self) -> None:
        """Set up test fixtures - reset ModelServer singleton."""
        from sparse_attention_hub.adapters.model_servers.base import ModelServer

        ModelServer._instance = None

    def teardown_method(self) -> None:
        """Clean up after each test - reset ModelServer singleton."""
        from sparse_attention_hub.adapters.model_servers.base import ModelServer

        ModelServer._instance = None

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_full_request_processing_flow(
        self, mock_tokenizer, mock_model, sparse_attention_config
    ):
        """Test complete request processing flow from context to response."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<EOS>"
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer_instance.eos_token_id = 1
        mock_tokenizer_instance.chat_template = None
        # Updated to handle the corrected calling pattern: context, then question
        mock_tokenizer_instance.encode.side_effect = [
            torch.tensor([[1, 2, 3, 4]]),  # context tokens
            torch.tensor([[7, 8]]),  # question tokens
        ]
        mock_tokenizer_instance.decode.return_value = "Generated response"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.named_modules.return_value = []
        mock_model_instance.device = "cpu"
        mock_model_instance.to.return_value = mock_model_instance

        # Mock KV cache
        mock_kv_cache = Mock()
        mock_kv_cache.get_seq_length.return_value = 4  # context length

        # Mock context output
        mock_context_output = Mock()
        mock_context_output.past_key_values = mock_kv_cache

        # Mock question output (for processing question tokens)
        mock_question_output = Mock()
        # Make logits return EOS token (index 1) as the argmax
        mock_question_output.logits = torch.tensor(
            [[[0.0, 5.0, 1.0, 2.0]]]
        )  # argmax is 1 (EOS)

        # Mock generation outputs (for generating new tokens)
        mock_gen_output1 = Mock()
        # Make this return EOS token (index 1) to stop generation
        mock_gen_output1.logits = torch.tensor(
            [[[0.0, 5.0, 1.0, 2.0]]]
        )  # argmax is 1 (EOS)

        # Configure model to return different outputs based on call sequence
        mock_model_instance.side_effect = [
            mock_context_output,  # context processing
            mock_question_output,  # question processing
            mock_gen_output1,  # first generation step (will be EOS)
        ]

        # Mock generation config
        mock_model_instance.generation_config.eos_token_id = 1

        mock_model.from_pretrained.return_value = mock_model_instance

        # Create adapter with model_kwargs
        adapter = ModelAdapterHF(
            model_name="test-model",
            sparse_attention_config=sparse_attention_config,
            model_kwargs={"torch_dtype": torch.float16},
        )

        # Create request
        request = Request(
            context="What is the capital of France?",
            questions="Tell me about Paris.",
            answer_prefix="Answer: ",
        )

        # Process request
        response = adapter.process_request(request, {}, {})

        # Verify response
        assert isinstance(response, RequestResponse)
        assert response.responses == "Generated response"

        # Verify tokenizer was called correctly
        assert mock_tokenizer_instance.encode.call_count == 2
        mock_tokenizer_instance.decode.assert_called_once()

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_multiple_questions_processing(
        self, mock_tokenizer, mock_model, sparse_attention_config
    ):
        """Test processing multiple questions in a single request."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<EOS>"
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer_instance.eos_token_id = 1
        mock_tokenizer_instance.chat_template = None
        # Updated to handle the corrected calling pattern: context once, then questions
        mock_tokenizer_instance.encode.side_effect = [
            torch.tensor([[1, 2, 3, 4]]),  # context tokens (once)
            torch.tensor([[7, 8]]),  # question 1 tokens
            torch.tensor([[9, 10]]),  # question 2 tokens
        ]
        mock_tokenizer_instance.decode.side_effect = ["Response 1", "Response 2"]
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.named_modules.return_value = []
        mock_model_instance.device = "cpu"
        mock_model_instance.to.return_value = mock_model_instance

        # Mock KV caches (one for each question)
        mock_kv_cache1 = Mock()
        mock_kv_cache1.get_seq_length.return_value = 4  # context length
        mock_kv_cache2 = Mock()
        mock_kv_cache2.get_seq_length.return_value = 4  # context length

        # Mock context outputs (processed for each question)
        mock_context_output1 = Mock()
        mock_context_output1.past_key_values = mock_kv_cache1

        mock_context_output2 = Mock()
        mock_context_output2.past_key_values = mock_kv_cache2

        # Mock question outputs (for processing question tokens)
        mock_question_output1 = Mock()
        # Make logits return EOS token (index 1) as the argmax
        mock_question_output1.logits = torch.tensor(
            [[[0.0, 5.0, 1.0, 2.0]]]
        )  # argmax is 1 (EOS)

        mock_question_output2 = Mock()
        # Make logits return EOS token (index 1) as the argmax
        mock_question_output2.logits = torch.tensor(
            [[[0.0, 5.0, 1.0, 2.0]]]
        )  # argmax is 1 (EOS)

        # Mock generation outputs - make them return EOS to stop generation quickly
        mock_gen_output1_1 = Mock()
        mock_gen_output1_1.logits = torch.tensor(
            [[[0.0, 5.0, 1.0, 2.0]]]
        )  # argmax is 1 (EOS)

        mock_gen_output2_1 = Mock()
        mock_gen_output2_1.logits = torch.tensor(
            [[[0.0, 5.0, 1.0, 2.0]]]
        )  # argmax is 1 (EOS)

        # Configure model to return different outputs based on call sequence
        mock_model_instance.side_effect = [
            mock_context_output1,  # context processing for question 1
            mock_question_output1,  # question 1 processing
            mock_gen_output1_1,  # question 1 generation step 1 (will be EOS)
            mock_context_output2,  # context processing for question 2
            mock_question_output2,  # question 2 processing
            mock_gen_output2_1,  # question 2 generation step 1 (will be EOS)
        ]

        # Mock generation config
        mock_model_instance.generation_config.eos_token_id = 1

        mock_model.from_pretrained.return_value = mock_model_instance

        # Create adapter with model_kwargs
        adapter = ModelAdapterHF(
            model_name="test-model",
            sparse_attention_config=sparse_attention_config,
            model_kwargs={"torch_dtype": torch.float16},
        )

        # Create request with multiple questions
        request = Request(
            context="Context about France",
            questions=["What is the capital?", "What is the population?"],
            answer_prefix="Answer: ",
        )

        # Process request
        response = adapter.process_request(request, {}, {})

        # Verify response
        assert isinstance(response, RequestResponse)
        assert response.responses == ["Response 1", "Response 2"]

        # Verify tokenizer was called correctly (context once + each question)
        assert mock_tokenizer_instance.encode.call_count == 3
        assert mock_tokenizer_instance.decode.call_count == 2

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_mode_switching_integration(
        self, mock_tokenizer, mock_model, sparse_attention_config
    ):
        """Test mode switching between sparse and dense modes."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<EOS>"
        mock_tokenizer_instance.chat_template = None
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.named_modules.return_value = []
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Create adapter with model_kwargs
        adapter = ModelAdapterHF(
            model_name="test-model",
            sparse_attention_config=sparse_attention_config,
            model_kwargs={"torch_dtype": torch.float16},
        )

        # Test switching to sparse mode
        with adapter.enable_sparse_mode():
            # Should not raise any errors
            pass

            # Test nested sparse mode switching (sparse mode within sparse mode)
            with adapter.enable_sparse_mode():
                # Should not raise any errors
                pass

        # Test that we can enable sparse mode again after exiting
        with adapter.enable_sparse_mode():
            # Should not raise any errors
            pass

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_custom_attention_function_integration(
        self, mock_tokenizer, mock_model, sparse_attention_config
    ):
        """Test custom attention function integration."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<EOS>"
        mock_tokenizer_instance.chat_template = None
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.named_modules.return_value = []
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Create adapter with model_kwargs
        adapter = ModelAdapterHF(
            model_name="test-model",
            sparse_attention_config=sparse_attention_config,
            model_kwargs={"torch_dtype": torch.float16},
        )

        # Get custom attention function
        assert adapter.sparse_attention is not None
        custom_fn = adapter.get_custom_attention_function(adapter.sparse_attention)

        # Test custom function signature
        assert callable(custom_fn)

        # Test custom function with mock parameters
        mock_module = Mock()
        mock_module.layer_idx = 0

        mock_queries = torch.randn(1, 8, 10, 64)
        mock_keys = torch.randn(1, 8, 10, 64)
        mock_values = torch.randn(1, 8, 10, 64)

        # Mock sparse attention custom_attention method
        with patch.object(
            adapter.sparse_attention, "custom_attention"
        ) as mock_custom_attention:
            mock_custom_attention.return_value = (torch.randn(1, 8, 10, 64), None)

            # Call custom function
            result = custom_fn(
                module=mock_module,
                queries=mock_queries,
                keys=mock_keys,
                values=mock_values,
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data={},
            )

            # Verify custom attention was called
            mock_custom_attention.assert_called_once()
            assert result is not None

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_error_handling_integration(
        self, mock_tokenizer, mock_model, sparse_attention_config
    ):
        """Test error handling in integration scenarios."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<EOS>"
        mock_tokenizer_instance.chat_template = None
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.named_modules.return_value = []
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Create adapter with model_kwargs
        adapter = ModelAdapterHF(
            model_name="test-model",
            sparse_attention_config=sparse_attention_config,
            model_kwargs={"torch_dtype": torch.float16},
        )

        # Test error when custom attention function is called without sparse_meta_data
        assert adapter.sparse_attention is not None
        custom_fn = adapter.get_custom_attention_function(adapter.sparse_attention)

        mock_module = Mock()
        mock_module.layer_idx = 0

        mock_queries = torch.randn(1, 8, 10, 64)
        mock_keys = torch.randn(1, 8, 10, 64)
        mock_values = torch.randn(1, 8, 10, 64)

        with pytest.raises(ValueError, match="sparse_meta_data must be provided"):
            custom_fn(
                module=mock_module,
                queries=mock_queries,
                keys=mock_keys,
                values=mock_values,
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                # missing sparse_meta_data
            )

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_adapter_with_device_configuration(
        self, mock_tokenizer, mock_model, sparse_attention_config
    ):
        if torch.cuda.device_count() == 0:
            # cannot execute this test
            return
        """Test adapter with device configuration."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<EOS>"
        mock_tokenizer_instance.chat_template = None
        mock_tokenizer_instance.encode.return_value = torch.tensor([[1, 2, 3, 4]])
        mock_tokenizer_instance.decode.return_value = "Generated response"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.named_modules.return_value = []
        mock_model_instance.to.return_value = mock_model_instance

        # Mock tensor operations
        mock_tensor = Mock()
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.shape = [1, 4]

        # Mock context output
        mock_context_output = Mock()
        mock_context_output.past_key_values = "mock_kv_cache"

        # Mock question output
        mock_question_output = Mock()
        mock_question_output.logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

        mock_model_instance.side_effect = [mock_context_output, mock_question_output]
        mock_model.from_pretrained.return_value = mock_model_instance

        # Create adapter with device configuration in model_kwargs
        adapter = ModelAdapterHF(
            model_name="test-model",
            sparse_attention_config=sparse_attention_config,
            model_kwargs={"device_map": "cuda", "torch_dtype": torch.float16},
        )

        # Verify device configuration
        assert adapter.device == "cuda"

        # Verify model was created with device configuration
        mock_model.from_pretrained.assert_called_once_with(
            "test-model", device_map="cuda", torch_dtype=torch.float16
        )

    @patch(
        "sparse_attention_hub.adapters.model_servers.huggingface.AutoModelForCausalLM"
    )
    @patch("sparse_attention_hub.adapters.model_servers.huggingface.AutoTokenizer")
    def test_adapter_cleanup_on_exception(
        self, mock_tokenizer, mock_model, sparse_attention_config
    ):
        """Test adapter cleanup when exceptions occur during mode switching."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<EOS>"
        mock_tokenizer_instance.chat_template = None
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.named_modules.return_value = []
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Create adapter with model_kwargs
        adapter = ModelAdapterHF(
            model_name="test-model",
            sparse_attention_config=sparse_attention_config,
            model_kwargs={"torch_dtype": torch.float16},
        )

        # Test cleanup on exception
        with pytest.raises(ValueError):
            with adapter.enable_sparse_mode():
                # Should not raise any errors from sparse mode itself
                raise ValueError("Test exception")

        # Test that sparse mode can still be enabled after exception
        with adapter.enable_sparse_mode():
            # Should work fine - cleanup was successful
            pass


class TestAdapterManual:
    """Manual tests for adapter functionality with real models."""

    def test_adapter_basic_process_request(
        self, sparse_attention_config_multi, model_name
    ):
        """Test basic process request."""
        if torch.cuda.device_count() == 0:
            # cannot test this
            return
        adapter = ModelAdapterHF(
            model_name=model_name,
            sparse_attention_config=sparse_attention_config_multi,
            model_kwargs={"device_map": "cuda", "torch_dtype": torch.bfloat16},
            device="cuda",
        )

        request = Request(
            context="""
         From July 4 through July 7, 2025, a destructive and deadly flood took place in
         the Texas Hill Country, particularly in Kerr County, in the U.S. state of Texas.
         During the flooding, water levels along the Guadalupe River rose rapidly in a short time.
         As a result, at least 129 fatalities have been confirmed, of which at least 103 are in
         Kerr County, with about 170 reported missing. The flooding was caused by a mesoscale
         convective vortex with enhanced tropical moisture from the remnants of Tropical Storm Barry.
         Answer the following question based on the paragraph above:
            """,
            questions="What county was the most affected by the flood?",
            answer_prefix="Answer: ",
        )
        response = adapter.process_request(request, {}, {})
        assert isinstance(response.responses, str)

    def test_adapter_basic_process_request_with_multiple_questions(
        self, sparse_attention_config_multi, model_name
    ):
        """Test basic process request with multiple questions."""
        if torch.cuda.device_count() == 0:
            # cannot test this
            return
        adapter = ModelAdapterHF(
            model_name=model_name,
            sparse_attention_config=sparse_attention_config_multi,
            model_kwargs={"device_map": "cuda", "torch_dtype": torch.bfloat16},
        )

        request = Request(
            context="""
         From July 4 through July 7, 2025, a destructive and deadly flood took place in
         the Texas Hill Country, particularly in Kerr County, in the U.S. state of Texas.
         During the flooding, water levels along the Guadalupe River rose rapidly in a short time.
         As a result, at least 129 fatalities have been confirmed, of which at least 103 are in
         Kerr County, with about 170 reported missing. The flooding was caused by a mesoscale
         convective vortex with enhanced tropical moisture from the remnants of Tropical Storm Barry.
         Answer the following question based on the paragraph above:
            """,
            questions=[
                "What county was the most affected by the flood?",
                "What was the cause of the flood?",
            ],
            answer_prefix="Answer: ",
        )
        response = adapter.process_request(request, {}, {})
        assert len(response.responses) == 2
    @pytest.mark.skipif(
    psutil.virtual_memory().total < 8 * 1024**3,
    reason="Insufficient memory for large model (Qwen 1.7B)"
    )
    def test_enable_sparse_mode(self, sparse_attention_config_multi, model_name):
        """Test enable_sparse_mode with real model."""
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

        adapter = ModelAdapterHF(
            model_name=model_name,
            sparse_attention_config=sparse_attention_config_multi,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

        for name, module in adapter.model.named_modules():
            if isinstance(module, Qwen3Attention):
                assert not module.config._attn_implementation.startswith(
                    "sparse_attention"
                )

        # Test sparse mode
        with adapter.enable_sparse_mode():
            # Should not raise any errors - sparse mode is working
            for name, module in adapter.model.named_modules():
                if isinstance(module, Qwen3Attention):
                    assert (
                        module.config._attn_implementation
                        == adapter._registered_attention_name
                    )
                    assert module.config._attn_implementation.startswith(
                        "sparse_attention"
                    )

        assert (
            adapter._registered_attention_name in ALL_ATTENTION_FUNCTIONS.valid_keys()
        )
        assert (
            adapter._registered_attention_name
            in ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        )

        for name, module in adapter.model.named_modules():
            if isinstance(module, Qwen3Attention):
                assert not module.config._attn_implementation.startswith(
                    "sparse_attention"
                )

        # Test dense mode (default mode after exiting sparse mode)
        for name, module in adapter.model.named_modules():
            if isinstance(module, Qwen3Attention):
                assert not module.config._attn_implementation.startswith(
                    "sparse_attention"
                )

        assert (
            adapter._registered_attention_name in ALL_ATTENTION_FUNCTIONS.valid_keys()
        )
        assert (
            adapter._registered_attention_name
            in ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        )

        registered_attention_name = adapter._registered_attention_name
        del adapter
        # delete function is called by garbage collector . so force it
        import gc

        gc.collect()
        assert registered_attention_name not in ALL_ATTENTION_FUNCTIONS.valid_keys()
        assert (
            registered_attention_name not in ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        )
