"""Stop tokens, sampling, thinking budget and chat-template options in ModelAdapterHF.

A real in-memory fast tokenizer and scripted LM stand-ins replace the model, so the
adapter's own decode loop and ``apply_chat_template`` call run unmodified.
"""

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sparse_attention_hub.adapters import ModelAdapterHF, Request

VOCAB = {"[UNK]": 0, "<|endoftext|>": 1, "<|im_end|>": 2, "x": 3, "y": 4, "z": 5}
X, Y, Z = VOCAB["x"], VOCAB["y"], VOCAB["z"]
END_OF_TEXT, END_OF_TURN = VOCAB["<|endoftext|>"], VOCAB["<|im_end|>"]

# Qwen3.5-like: the generation prompt opens <think> unless thinking is disabled.
QWEN35_LIKE_TEMPLATE = (
    "{{ messages[0]['content'] }}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}"
    "<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}"
)
# Qwen3-like: when thinking, the model opens <think> itself.
QWEN3_LIKE_TEMPLATE = (
    "{{ messages[0]['content'] }}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}"
    "<think>\n\n</think>\n\n{% endif %}"
)


def make_tokenizer(template: str = QWEN35_LIKE_TEMPLATE) -> PreTrainedTokenizerFast:
    """A word-level tokenizer whose EOS is the chat end-of-turn token, as in Qwen."""
    core = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="[UNK]",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|endoftext|>"],
    )
    tokenizer.add_tokens(["<think>", "</think>"])  # plain added tokens, as in Qwen
    tokenizer.chat_template = template
    return tokenizer


TOKENIZER = make_tokenizer()
THINK_START = TOKENIZER.convert_tokens_to_ids("<think>")
THINK_END = TOKENIZER.convert_tokens_to_ids("</think>")


class ScriptedLM:
    """Greedy LM stand-in: the argmax of forward call ``i`` is ``script[i]``.

    Call 0 is the context prefill, call 1 the question; the last id repeats forever.
    """

    def __init__(self, script: List[int], eos_token_id: Any) -> None:
        self.script = script
        self.calls = 0
        self.device = torch.device("cpu")
        self.generation_config = SimpleNamespace(eos_token_id=eos_token_id)
        self.call_kwargs: List[Dict[str, Any]] = []

    def eval(self) -> "ScriptedLM":
        return self

    def __call__(self, input_ids: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        self.call_kwargs.append(kwargs)
        token = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        logits = torch.zeros(1, input_ids.shape[1], len(TOKENIZER))
        logits[0, -1, token] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=None)


class PolicyLM:
    """LM stand-in whose next-token logits are ``policy(fed)``: a {token: logit} map
    computed from every token id fed so far (prompt, generated and injected)."""

    def __init__(self, policy: Callable[[List[int]], Dict[int, float]]) -> None:
        self.policy, self.fed = policy, []
        self.device = torch.device("cpu")
        self.generation_config = SimpleNamespace(eos_token_id=[END_OF_TURN])

    def eval(self) -> "PolicyLM":
        return self

    def __call__(self, input_ids: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        self.fed.extend(input_ids[0].tolist())
        logits = torch.full((1, input_ids.shape[1], len(TOKENIZER)), -1e4)
        for token, value in self.policy(self.fed).items():
            logits[0, -1, token] = value
        return SimpleNamespace(logits=logits, past_key_values=None)


def build_adapter(model: Any, template: str = QWEN35_LIKE_TEMPLATE) -> ModelAdapterHF:
    server = Mock()
    server.get_model.return_value = model
    server.get_tokenizer.return_value = make_tokenizer(template)
    target = "sparse_attention_hub.adapters.huggingface.ModelServerHF"
    with patch(target, return_value=server):
        return ModelAdapterHF(
            model_name="test-model", sparse_attention_config=None, device="cpu"
        )


def generate(
    adapter: ModelAdapterHF,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    request_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    request = Request(context="x", questions=["x"], answer_prefix="")
    kwargs = {"max_new_tokens": 20, **(generation_kwargs or {})}
    response = adapter.process_request(request, kwargs, request_kwargs or {})
    return response.responses[0]


@pytest.mark.unit
class TestStopTokens:
    def test_first_generated_stop_token_ends_generation(self) -> None:
        # The model ends its turn at once; nothing after that may be decoded.
        model = ScriptedLM([X, END_OF_TURN, Y], eos_token_id=[END_OF_TURN, END_OF_TEXT])
        assert generate(build_adapter(model)) == ""

    def test_stops_on_tokenizer_eos_missing_from_generation_config(self) -> None:
        # Qwen3.5 ships no generation_config.json, so the model config knows only
        # <|endoftext|>; the chat turn still ends on the tokenizer's <|im_end|>.
        model = ScriptedLM([X, X, Y, END_OF_TURN, Y], eos_token_id=END_OF_TEXT)
        assert generate(build_adapter(model)) == "x y"


@pytest.mark.unit
class TestChatTemplateKwargs:
    def test_chat_template_kwargs_reach_the_template(self) -> None:
        adapter = build_adapter(ScriptedLM([X], eos_token_id=END_OF_TURN))
        _, questions = adapter._preprocess_context_and_questions(
            "CTX", ["Q"], "", chat_template_kwargs={"enable_thinking": False}
        )
        assert questions == ["Q<|im_start|>assistant\n<think>\n\n</think>\n\n"]


@pytest.mark.unit
class TestSampling:
    def test_seeded_top_k_sampling_draws_only_from_the_top_k(self) -> None:
        # x and y tie and z is far below; greedy decoding would emit only x.
        kwargs = {"max_new_tokens": 30, "temperature": 1.0, "top_k": 2, "seed": 0}
        policy = lambda fed: {X: 1.0, Y: 1.0, Z: -2.0}  # noqa: E731
        first = generate(build_adapter(PolicyLM(policy)), kwargs)
        again = generate(build_adapter(PolicyLM(policy)), kwargs)
        assert set(first.split()) == {"x", "y"} and first == again

    def test_top_p_keeps_only_the_nucleus(self) -> None:
        # p(x) ~ 0.58 and p(y) = p(z) ~ 0.21, so a 0.5 nucleus holds x alone.
        kwargs = {"max_new_tokens": 30, "temperature": 1.0, "top_p": 0.5, "seed": 0}
        policy = lambda fed: {X: 2.0, Y: 1.0, Z: 1.0}  # noqa: E731
        assert set(generate(build_adapter(PolicyLM(policy)), kwargs).split()) == {"x"}

    def test_presence_penalty_counts_generated_tokens_only(self) -> None:
        # The prompt already holds x; only generated tokens are penalised.
        kwargs = {"max_new_tokens": 4, "presence_penalty": 1.0}  # greedy
        policy = lambda fed: {X: 1.0, Y: 0.5}  # noqa: E731
        assert generate(build_adapter(PolicyLM(policy)), kwargs) == "x y x x"

    def test_unsupported_min_p_is_rejected(self) -> None:
        policy = lambda fed: {X: 1.0}  # noqa: E731
        with pytest.raises(ValueError, match="min_p"):
            generate(build_adapter(PolicyLM(policy)), {"min_p": 0.05})


def thinks_with_y_until_closed(fed: List[int]) -> Dict[int, float]:
    """Opens a thinking block if needed, thinks 'y' until </think>, answers 'x'."""
    if THINK_START not in fed:
        return {THINK_START: 1.0}
    return {Y: 1.0} if THINK_END not in fed else {X: 1.0}


@pytest.mark.unit
class TestThinkingBudget:
    THINKING_ON = {"chat_template_kwargs": {"enable_thinking": True}}

    @pytest.mark.parametrize(
        "template", [QWEN35_LIKE_TEMPLATE, QWEN3_LIKE_TEMPLATE], ids=["qwen35", "qwen3"]
    )
    def test_thinking_budget_forces_stop_then_answers(self, template) -> None:
        adapter = build_adapter(PolicyLM(thinks_with_y_until_closed), template)
        out = generate(
            adapter, {"max_new_tokens": 2, "thinking_budget": 3}, self.THINKING_ON
        )
        thinking, _, answer = out.rpartition("</think>")
        assert thinking.split().count("y") == 3  # budget, not max_new_tokens
        assert answer.split() == ["x", "x"]  # the answer gets max_new_tokens

    def test_thinking_that_closes_naturally_is_untouched(self) -> None:
        def closes_after_one_y(fed: List[int]) -> Dict[int, float]:
            if THINK_END in fed:
                return {X: 1.0}
            return {THINK_END: 1.0} if Y in fed else {Y: 1.0}

        adapter = build_adapter(PolicyLM(closes_after_one_y))
        out = generate(
            adapter, {"max_new_tokens": 2, "thinking_budget": 3}, self.THINKING_ON
        )
        assert out == "y </think> x x"


@pytest.mark.unit
class TestContextPrefill:
    def test_context_prefill_requests_only_the_last_logit(self) -> None:
        # The prefill only fills the KV cache. Full logits for a 128K context and
        # Qwen3.5's 248K vocabulary are a 55 GiB tensor, which OOMs an H200.
        model = ScriptedLM([X, END_OF_TURN], eos_token_id=[END_OF_TURN])
        generate(build_adapter(model))
        assert model.call_kwargs[0].get("logits_to_keep") == 1
