"""HuggingFace adapter implementation for sparse attention."""

import random
import string
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..sparse_attention.base import SparseAttention, SparseAttentionConfig
from ..sparse_attention.research_attention.base import ResearchAttention
from .base import ModelAdapter, Request, RequestResponse
from .model_servers.huggingface import ModelServerHF
from .utils.config import ModelServerConfig

INT_MAX = 2**31 - 1


class ModelAdapterHF(ModelAdapter):
    """ModelAdapter for HuggingFace integration. Provides concrete implementations for huggingface's
    transformer library.
    """

    def __init__(
        self,
        model_name: str,
        sparse_attention_config: Optional[SparseAttentionConfig],
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        hybrid: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize HuggingFace adapter.

        Args:
            model_name: Name of the HuggingFace model to use
            sparse_attention_config: Configuration for sparse attention. If None, adapter runs in dense-only mode.
            model_kwargs: Additional keyword arguments for model creation
            device: Device to run the model on TODO: support dynamic and multipledevice placement
            tokenizer_kwargs: Additional keyword arguments for tokenizer creation
        """
        super().__init__(model_name, sparse_attention_config, **kwargs)
        self._registered_attention_name: Optional[str] = None
        self._custom_attention_fn: Optional[Callable] = None
        self.model_kwargs: Dict[str, Any] = model_kwargs or {}
        self.tokenizer_kwargs: Dict[str, Any] = tokenizer_kwargs or {}

        raw_registry_path: Any = kwargs.get("model_registry_path", "")
        self.model_registry_path: str = (
            raw_registry_path if isinstance(raw_registry_path, str) else ""
        )

        raw_allow_unregistered: Any = kwargs.get("allow_unregistered_models", True)
        self.allow_unregistered_models: bool = (
            raw_allow_unregistered if isinstance(raw_allow_unregistered, bool) else True
        )

        # more useful parameters to store
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.torch_dtype = self.model_kwargs.get("torch_dtype", torch.float32)

        # Handle dense-only mode when sparse_attention_config is None
        self._sparse_attention_available: bool = sparse_attention_config is not None
        # Control token-by-token question processing (for hybrid models)
        self.hybrid = hybrid if hybrid is not None else False
        # Convert device string to GPU ID for ModelServer
        gpu_id: Optional[int] = None
        if self.device.startswith("cuda"):
            if self.device == "cuda":
                gpu_id = 0 if torch.cuda.is_available() else None
            else:
                # Extract GPU ID from "cuda:0", "cuda:1", etc.
                try:
                    gpu_id = int(self.device.split(":")[1])
                except (IndexError, ValueError):
                    gpu_id = None

        model_server = ModelServerHF(
            ModelServerConfig(
                model_registry_path=self.model_registry_path,
                allow_unregistered_models=self.allow_unregistered_models,
            )
        )
        self.model = model_server.get_model(self.model_name, gpu_id, self.model_kwargs)
        self.tokenizer = model_server.get_tokenizer(
            self.model_name, self.tokenizer_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.random_separator = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=100)
        )

    def __del__(self) -> None:
        """Clean up registered attention functions when the adapter is destroyed."""
        self._cleanup_attention_registration()

    def process_request(
        self,
        request: Request,
        generation_kwargs: Dict[str, Any],
        request_kwargs: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> RequestResponse:
        """Processes request with optimized tokenization but independent question processing.
        Context is tokenized once but each question is processed independently to avoid KV cache contamination.

        Args:
            request: The request to process

        Returns:
            response: The response to the request
        """
        max_context_length: int = request_kwargs.get("max_context_length", INT_MAX)
        max_new_tokens: int = generation_kwargs.get("max_new_tokens", INT_MAX)
        print(
            " Processing request with max_context_length: ",
            max_context_length,
            " and max_new_tokens: ",
            max_new_tokens,
            flush=True,
        )

        questions: List[str] = (
            request.questions
            if isinstance(request.questions, list)
            else [request.questions]
        )
        context: str = request.context
        answer_prefix: str = request.answer_prefix

        context, questions = self._preprocess_context_and_questions(
            context, questions, answer_prefix
        )

        context_tokens = self.tokenizer.encode(context, return_tensors="pt")
        context_tokens = context_tokens[
            :, :max_context_length
        ]  # truncate context to max_context_length

        input_device = (
            self.model.device if hasattr(self.model, "device") else self.device
        )
        if input_device is not None:
            context_tokens = context_tokens.to(input_device)
        print(f"Context tokens: {context_tokens.shape}")
        responses: List[str] = []

        self.model.eval()
        with torch.no_grad():
            for question in questions:
                sparse_meta_data: Dict[str, Any] = {}

                question_tokens = self.tokenizer.encode(question, return_tensors="pt")
                if input_device is not None:
                    question_tokens = question_tokens.to(input_device)

                if self._sparse_attention_available:
                    context_outputs = self.model(
                        context_tokens,
                        past_key_values=None,
                        use_cache=True,
                        sparse_meta_data=sparse_meta_data,
                    )

                    with self.enable_sparse_mode():
                        response_text = self._generate_response(
                            question_tokens,
                            context_outputs,
                            sparse_meta_data,
                            generation_kwargs,
                            **kwargs,
                        )
                else:
                    context_outputs = self.model(
                        context_tokens,
                        past_key_values=None,
                        use_cache=True,
                    )

                    response_text = self._generate_response(
                        question_tokens,
                        context_outputs,
                        sparse_meta_data,
                        generation_kwargs,
                        **kwargs,
                    )

                responses.append(response_text)

        if isinstance(request.questions, str):
            return RequestResponse(responses=responses[0])
        else:
            return RequestResponse(responses=responses)

    def _preprocess_context_and_questions(
        self, context: str, questions: List[str], answer_prefix: str
    ) -> Tuple[str, List[str]]:
        """Preprocess the context and questions -- apply chat template if needed

        Args:
            context: The context to preprocess
            questions: The questions to preprocess
        """
        context = context + self.random_separator
        if self.tokenizer.chat_template is not None:
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context}],
                tokenize=False,
                add_generation_prompt=True,
            )
            new_context = context.split(self.random_separator)[0]
            new_questions = [
                question + context.split(self.random_separator)[1] + answer_prefix
                for question in questions
            ]
        else: 
            new_context = context.split(self.random_separator)[0]
            new_questions = [
                f"{context.split(self.random_separator)[0]}\n\nQuestion: {question}\n{answer_prefix}"
                for question in questions
            ]
        return new_context, new_questions

    def get_custom_attention_function(
        self, sparse_attention: SparseAttention
    ) -> Callable:
        """Returns custom_attention_fn callable with the correct signature required for HuggingFace.

        Args:
            sparse_attention: The sparse attention instance

        Returns:
            custom_attention_fn: Callable with correct signature for HuggingFace
        """

        def custom_attention_callable(
            module: torch.nn.Module,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            scaling: float = 1.0,
            dropout: float = 0.0,
            **kwargs: Dict[str, Any],
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """Custom attention callable for HuggingFace integration."""
            if hasattr(module, "layer_idx"):
                layer_idx = getattr(module, "layer_idx", None)
                if layer_idx is not None:
                    kwargs["layer_idx"] = layer_idx

            if "sparse_meta_data" in kwargs:
                sparse_meta_data: Dict[Any, Any] = kwargs["sparse_meta_data"]
                kwargs.pop("sparse_meta_data", None)
            else:
                raise ValueError(
                    "sparse_meta_data must be provided while calling model.forward()"
                )

            return sparse_attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                **kwargs,
            )

        return custom_attention_callable

    def _generate_unique_attention_name(self) -> str:
        """Generate a unique name not present in ALL_ATTENTION_FUNCTIONS."""
        base_name: str = "sparse_attention"
        existing_keys: List[str] = (
            ALL_ATTENTION_FUNCTIONS.valid_keys()
            + ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        )

        while True:
            suffix: str = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=8)
            )
            name: str = f"{base_name}_{suffix}"

            if name not in existing_keys:
                return name

    def _ensure_attention_registered(self) -> str:
        """Ensure custom attention function is registered and return the name.
        Caches the registration to avoid repeated registration overhead.

        Returns:
            The name of the registered attention function
        """
        if self._registered_attention_name is None:
            if not self._sparse_attention_available or self.sparse_attention is None:
                raise RuntimeError(
                    "Cannot register attention function: sparse attention is not available"
                )
            self._custom_attention_fn = self.get_custom_attention_function(
                self.sparse_attention
            )
            self._registered_attention_name = self._generate_unique_attention_name()

            from transformers.masking_utils import eager_mask

            ALL_ATTENTION_FUNCTIONS.register(
                self._registered_attention_name, self._custom_attention_fn
            )
            if isinstance(self.sparse_attention, ResearchAttention):
                ALL_MASK_ATTENTION_FUNCTIONS.register(
                    self._registered_attention_name, eager_mask
                )
            else:
                raise NotImplementedError(
                    "Sparse attention is not supported for this model yet"
                )

        return self._registered_attention_name

    def _cleanup_attention_registration(self) -> None:
        """Clean up registered attention functions."""
        if (
            self._registered_attention_name is not None
            and self._registered_attention_name in ALL_ATTENTION_FUNCTIONS.valid_keys()
        ):
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(self._registered_attention_name)
        if (
            self._registered_attention_name is not None
            and self._registered_attention_name
            in ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        ):
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(
                self._registered_attention_name
            )
        self._registered_attention_name = None
        self._custom_attention_fn = None

    @contextmanager
    def enable_sparse_mode(self) -> Generator[None, None, None]:
        """Context manager to temporarily enable sparse attention mode.

        Yields:
            None
        """
        # If sparse attention is not available, raise an error
        if not self._sparse_attention_available:
            raise RuntimeError(
                "Cannot enable sparse mode: sparse attention is not available"
            )

        # Store original implementations to restore later
        original_implementations: Dict[str, str] = {}

        # First, store the original implementations before registering custom attention
        for name, module in self.model.named_modules():
            if hasattr(module, "config") and hasattr(
                module.config, "_attn_implementation"
            ):
                original_implementations[name] = module.config._attn_implementation

        # Ensure custom attention function is registered (reuse if already registered)
        custom_attention_name: str = self._ensure_attention_registered()

        try:
            # Switch to sparse attention
            for name, module in self.model.named_modules():
                if hasattr(module, "config") and hasattr(
                    module.config, "_attn_implementation"
                ):
                    module.config._attn_implementation = custom_attention_name

            yield

        finally:
            # Restore original implementations
            for name, module in self.model.named_modules():
                if name in original_implementations:
                    module.config._attn_implementation = original_implementations[name]

    def _generate_response(
        self,
        question_tokens: torch.Tensor,
        context_outputs: Any,
        sparse_meta_data: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generate text response.

        For hybrid models, after context prefill, feed the question tokens one-by-one so the
        linear-attention cached path uses seq_len == 1. For dense models, process all question tokens at once.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")

        max_new_tokens: int = generation_kwargs.get("max_new_tokens", 50)

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        # Start from cached context state
        past_key_values = context_outputs.past_key_values

        last_outputs = None

        if self.hybrid:
            # TODO(sahil, P1): add regression coverage for hybrid models that
            # verifies question/prefix tokens are consumed one-by-one after
            # context prefill, while non-hybrid models keep the batched path.
            # ------------------------------------------------------------------
            # Step 1: consume the question/prefix ONE TOKEN AT A TIME
            # ------------------------------------------------------------------
            for idx in range(question_tokens.shape[1]):
                step_input = question_tokens[:, idx : idx + 1]

                model_kwargs = {
                    "input_ids": step_input,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "logits_to_keep": 1,
                }

                # Only pass sparse_meta_data when sparse mode is actually enabled
                if self._sparse_attention_available:
                    model_kwargs["sparse_meta_data"] = sparse_meta_data

                with torch.no_grad():
                    last_outputs = self.model(**model_kwargs)

                past_key_values = last_outputs.past_key_values
        else:
            # Dense models: process all question tokens at once
            model_kwargs = {
                "input_ids": question_tokens,
                "past_key_values": past_key_values,
                "use_cache": True,
            }
            if self._sparse_attention_available:
                model_kwargs["sparse_meta_data"] = sparse_meta_data
            with torch.no_grad():
                last_outputs = self.model(**model_kwargs)
            past_key_values = last_outputs.past_key_values

        if last_outputs is None:
            raise ValueError("Question tokens are empty; cannot generate response")

        generated_ids = [last_outputs.logits[0, -1].argmax()]

        # ------------------------------------------------------------------
        # Step 2: continue normal autoregressive generation, one token at a time
        # ------------------------------------------------------------------
        for _ in tqdm(range(max_new_tokens - 1), disable=(max_new_tokens < 1000)):
            model_kwargs = {
                "input_ids": generated_ids[-1].unsqueeze(0).unsqueeze(0),
                "past_key_values": past_key_values,
                "use_cache": True,
            }

            if self._sparse_attention_available:
                model_kwargs["sparse_meta_data"] = sparse_meta_data

            with torch.no_grad():
                outputs = self.model(**model_kwargs)

            past_key_values = outputs.past_key_values
            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)

            if new_id.item() in should_stop_token_ids:
                break

        answer: str = self.tokenizer.decode(
            torch.stack(generated_ids), skip_special_tokens=True
        )

        return answer
