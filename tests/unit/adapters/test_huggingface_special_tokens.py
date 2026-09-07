"""Unit tests for special-token handling in :class:`ModelAdapterHF.process_request`.

``ModelAdapterHF._preprocess_context_and_questions`` renders the chat template
with ``tokenize=False``, so the returned string already carries the model's BOS
marker. Fast tokenizers for Llama-3.x and Gemma additionally carry a
``TemplateProcessing`` post-processor that prepends BOS on *every* ``encode()``
call, so leaving ``add_special_tokens`` at its default of ``True`` duplicates
BOS at the head of the context and splices a third one in mid-sequence at the
question boundary. These tests pin the corrected call sites.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest
import torch

from sparse_attention_hub.adapters import ModelAdapterHF, Request
from sparse_attention_hub.adapters.model_servers.base import ModelServer

BOS_ID: int = 128000
BOS_MARKER: str = "<|begin_of_text|>"


class EncodeCall:
    """One recorded invocation of :meth:`FakeTokenizer.encode`."""

    def __init__(self, text: str, add_special_tokens: bool) -> None:
        """Store the text and the ``add_special_tokens`` value that was used."""
        self.text: str = text
        self.add_special_tokens: bool = add_special_tokens

    def __repr__(self) -> str:
        """Return a compact representation useful in assertion output."""
        return (
            f"EncodeCall(add_special_tokens={self.add_special_tokens!r}, "
            f"text={self.text!r})"
        )


class FakeTokenizer:
    """A ``TemplateProcessing``-style tokenizer stand-in.

    Two independent sources can emit a BOS id, exactly as in the real Llama-3.x
    and Gemma fast tokenizers:

    1. the literal :data:`BOS_MARKER` that ``apply_chat_template`` writes into
       the rendered string, and
    2. the post-processor, simulated here by prepending :data:`BOS_ID` whenever
       ``add_special_tokens`` is true.

    Every ``encode()`` call is recorded in :attr:`encode_calls` so tests can
    assert on the keyword the adapter actually passed.
    """

    def __init__(self, chat_template: Optional[str]) -> None:
        """Create a tokenizer whose ``chat_template`` may be ``None``."""
        self.chat_template: Optional[str] = chat_template
        self.pad_token: str = "<pad>"
        self.eos_token: str = "<eos>"
        self.bos_token_id: int = BOS_ID
        self.encode_calls: List[EncodeCall] = []
        self._vocab: Dict[str, int] = {}

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        """Render a Llama-3-shaped prompt, BOS marker included."""
        assert tokenize is False, "adapter must render the template as text"
        content: str = conversation[0]["content"]
        return (
            f"{BOS_MARKER}<|start_header_id|>user<|end_header_id|>\n{content}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    def _word_id(self, word: str) -> int:
        """Return a stable, BOS-free id for ``word``."""
        if word not in self._vocab:
            self._vocab[word] = 10 + len(self._vocab)
        return self._vocab[word]

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
    ) -> Any:
        """Encode ``text``, recording the ``add_special_tokens`` kwarg received."""
        self.encode_calls.append(
            EncodeCall(text=text, add_special_tokens=add_special_tokens)
        )

        ids: List[int] = [BOS_ID] if add_special_tokens else []
        for index, segment in enumerate(text.split(BOS_MARKER)):
            if index > 0:
                ids.append(BOS_ID)
            ids.extend(self._word_id(word) for word in segment.split())

        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids


def build_adapter(
    mock_model_server_hf: Mock, chat_template: Optional[str]
) -> ModelAdapterHF:
    """Build a dense-only adapter wired to :class:`FakeTokenizer` and a mock model."""
    tokenizer: FakeTokenizer = FakeTokenizer(chat_template=chat_template)

    mock_model: Mock = Mock()
    mock_model.device = torch.device("cpu")

    mock_model_server: Mock = Mock()
    mock_model_server.get_tokenizer.return_value = tokenizer
    mock_model_server.get_model.return_value = mock_model
    mock_model_server_hf.return_value = mock_model_server

    return ModelAdapterHF(
        model_name="test-model",
        sparse_attention_config=None,
        device="cpu",
    )


def run_request(adapter: ModelAdapterHF) -> torch.Tensor:
    """Drive ``process_request`` once and return the assembled context+question ids."""
    captured: Dict[str, torch.Tensor] = {}

    def fake_generate_response(
        question_tokens: torch.Tensor,
        context_outputs: Any,
        sparse_meta_data: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
        **kwargs: Any,
    ) -> str:
        captured["question_tokens"] = question_tokens
        return "answer"

    adapter._generate_response = fake_generate_response  # type: ignore[assignment]

    request: Request = Request(
        context="the quick brown fox",
        questions="who jumped ?",
        answer_prefix="Answer:",
    )
    adapter.process_request(request, generation_kwargs={}, request_kwargs={})

    context_tokens: torch.Tensor = adapter.model.call_args_list[0][0][0]
    return torch.cat([context_tokens, captured["question_tokens"]], dim=1)


@pytest.mark.unit
class TestProcessRequestSpecialTokens:
    """Pin the ``add_special_tokens`` contract of the two ``encode()`` call sites."""

    def setup_method(self) -> None:
        """Reset the ModelServer singleton before each test."""
        ModelServer._instance = None

    def teardown_method(self) -> None:
        """Reset the ModelServer singleton after each test."""
        ModelServer._instance = None

    @patch("sparse_attention_hub.adapters.huggingface.ModelServerHF")
    def test_chat_template_yields_exactly_one_bos(
        self, mock_model_server_hf: Mock
    ) -> None:
        """With a chat template the template's own BOS must be the only one."""
        adapter: ModelAdapterHF = build_adapter(
            mock_model_server_hf, chat_template="{{ messages }}"
        )
        assembled: torch.Tensor = run_request(adapter)

        ids: List[int] = assembled[0].tolist()
        assert ids.count(BOS_ID) == 1, (
            "expected exactly one BOS in the assembled sequence, got "
            f"{ids.count(BOS_ID)} at positions "
            f"{[i for i, t in enumerate(ids) if t == BOS_ID]}"
        )
        assert ids[0] == BOS_ID, "the single BOS must sit at position 0"

        calls: List[EncodeCall] = adapter.tokenizer.encode_calls
        assert len(calls) == 2, f"expected one context and one question encode: {calls}"
        assert calls[0].add_special_tokens is False, (
            "context encode must suppress special tokens when the chat template "
            f"already emitted BOS; got {calls[0]!r}"
        )
        assert calls[1].add_special_tokens is False, (
            "question encode must never add special tokens mid-sequence; got "
            f"{calls[1]!r}"
        )

    @patch("sparse_attention_hub.adapters.huggingface.ModelServerHF")
    def test_without_chat_template_context_keeps_bos(
        self, mock_model_server_hf: Mock
    ) -> None:
        """Without a chat template the tokenizer must still supply the BOS."""
        adapter: ModelAdapterHF = build_adapter(
            mock_model_server_hf, chat_template=None
        )
        assembled: torch.Tensor = run_request(adapter)

        ids: List[int] = assembled[0].tolist()
        assert ids.count(BOS_ID) == 1, (
            "expected exactly one BOS in the assembled sequence, got "
            f"{ids.count(BOS_ID)} at positions "
            f"{[i for i, t in enumerate(ids) if t == BOS_ID]}"
        )
        assert ids[0] == BOS_ID, "the single BOS must sit at position 0"

        calls: List[EncodeCall] = adapter.tokenizer.encode_calls
        assert len(calls) == 2, f"expected one context and one question encode: {calls}"
        assert calls[0].add_special_tokens is True, (
            "context encode must request special tokens when no chat template "
            f"rendered a BOS; got {calls[0]!r}"
        )
        assert calls[1].add_special_tokens is False, (
            "question encode must never add special tokens mid-sequence; got "
            f"{calls[1]!r}"
        )
