"""Per-task checks on real rows: what the model is sent, and how answers score.

Setup tests run the real datasets (one row per task) through ``_process_all_requests``
and the real Qwen3-4B / Qwen3.5-4B tokenizers; they skip when data or tokenizers cannot
be loaded. Scoring tests replay real greedy outputs of both models, thinking on and off
(``tests/fixtures/real_sample_outputs.json``): the raw output goes through the same
reasoning strip as the pipeline, then the repo's scorer. Every expected score was
computed with upstream code only: google-deepmind/loft @219f68e and NVIDIA/RULER.
"""

import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer

from benchmark.base import strip_reasoning
from benchmark.loft import LoftRag
from benchmark.loft.calculate_metrics import calculate_metrics as loft_metrics
from benchmark.ruler16k import Ruler16K
from sparse_attention_hub.adapters import ModelAdapterHF, Request
from sparse_attention_hub.adapters.base import RequestResponse

pytestmark = pytest.mark.integration

# NVIDIA RULER's tokens_to_generate per task family.
RULER_BUDGET = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}
LOFT_TASKS = ["nq_128k", "hotpotqa_128k", "musique_128k", "qampari_128k", "quest_128k"]
FIXTURE = Path(__file__).parents[1] / "fixtures" / "real_sample_outputs.json"
REAL_OUTPUTS: List[Dict[str, Any]] = json.loads(FIXTURE.read_text())

QWEN_TOKENIZERS = {
    "qwen3_4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    "qwen35_4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
}
# What each chat template emits after the user turn. Qwen3 opens <think> itself when
# thinking; Qwen3.5's prompt already opens it.
GENERATION_PROMPT = {
    ("qwen3_4b", "on"): "<|im_start|>assistant\n",
    ("qwen3_4b", "off"): "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ("qwen35_4b", "on"): "<|im_start|>assistant\n<think>\n",
    ("qwen35_4b", "off"): "<|im_start|>assistant\n<think>\n\n</think>\n\n",
}
# Stop ids from each model's config: Qwen3.5-4B ships no generation_config.json, so
# its config only knows <|endoftext|>.
CONFIG_EOS = {"qwen3_4b": [151645, 151643], "qwen35_4b": 248044}


class RecordingAdapter:
    """Records each request and its generation budget; the model is not needed."""

    def __init__(self) -> None:
        self.calls: List[Tuple[Any, int]] = []

    def process_request(self, request, generation_kwargs, request_kwargs):
        self.calls.append((request, generation_kwargs["max_new_tokens"]))
        return RequestResponse(responses=[""] * len(request.questions))


class ScriptedLM:
    """Greedy LM stand-in: forward call ``i`` puts the argmax on ``script[i]``."""

    def __init__(self, script: List[int], eos_token_id: Any, vocab_size: int) -> None:
        self.script, self.calls, self.vocab_size = script, 0, vocab_size
        self.device = torch.device("cpu")
        self.generation_config = SimpleNamespace(eos_token_id=eos_token_id)

    def eval(self) -> "ScriptedLM":
        return self

    def __call__(self, input_ids: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        token = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        logits = torch.zeros(1, input_ids.shape[1], self.vocab_size)
        logits[0, -1, token] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=None)


@lru_cache(maxsize=None)
def _qwen_tokenizer(model: str):
    repo, revision = QWEN_TOKENIZERS[model]
    try:
        return AutoTokenizer.from_pretrained(repo, revision=revision)
    except Exception as exc:
        pytest.skip(f"{repo} tokenizer unavailable: {exc}")


def _qwen_adapter(tokenizer, lm: ScriptedLM) -> ModelAdapterHF:
    server = Mock()
    server.get_model.return_value = lm
    server.get_tokenizer.return_value = tokenizer
    target = "sparse_attention_hub.adapters.huggingface.ModelServerHF"
    with patch(target, return_value=server):
        return ModelAdapterHF(
            model_name="test-model", sparse_attention_config=None, device="cpu"
        )


def _send(bench, rows: pd.DataFrame, request_kwargs=None) -> Dict[str, Any]:
    """Run rows through the benchmark pipeline; return what each context was sent."""
    adapter = RecordingAdapter()
    bench._process_all_requests(
        adapter, rows, {"thinking_budget": 8192}, request_kwargs or {}
    )
    return {request.context: (request, budget) for request, budget in adapter.calls}


def _first_row_per_task(bench, rows_filter=None) -> Dict[str, Any]:
    try:
        df = bench._load_datasets()
    except Exception as exc:  # offline, or the HF dataset is gone
        pytest.skip(f"{bench.benchmark_name} data unavailable: {exc}")
    if rows_filter is not None:
        df = rows_filter(df)
    rows = df.groupby("task", sort=False).head(1)
    sent = _send(bench, rows)  # all tasks in ONE run
    return {
        row.task: (row, *sent[row.context], rows[rows["task"] == row.task])
        for row in rows.itertuples()
    }


@pytest.fixture(scope="module")
def ruler_sent() -> Dict[str, Any]:
    return _first_row_per_task(Ruler16K())


@pytest.fixture(scope="module")
def loft_sent() -> Dict[str, Any]:
    return _first_row_per_task(
        LoftRag(LOFT_TASKS), rows_filter=lambda df: df[df["split"] == "test"]
    )


@pytest.mark.slow
@pytest.mark.parametrize("task", Ruler16K.all_datasets)
def test_ruler16k_task_keeps_its_budget_and_answer_prefix(ruler_sent, task):
    row, request, budget, _ = ruler_sent[task]
    assert budget == RULER_BUDGET[task.split("_")[0]]
    assert request.questions == [row.question]
    assert request.answer_prefix == row.answer_prefix and row.answer_prefix.strip()


@pytest.mark.slow
@pytest.mark.parametrize("task", LOFT_TASKS)
def test_loft_task_prompt_ends_at_query_with_gold_docs(loft_sent, task):
    row, request, budget, _ = loft_sent[task]
    assert budget == 8192
    assert request.answer_prefix == ""
    assert request.questions[0].rstrip("\n").split("\n")[-1].startswith("query: ")
    assert all(str(g).lower() in request.context.lower() for g in row.answers)


@pytest.mark.slow
@pytest.mark.parametrize(
    "bench, task", [("loft", "nq_128k"), ("ruler", "niah_single_1")]
)
@pytest.mark.parametrize("thinking", ["on", "off"])
@pytest.mark.parametrize("model", ["qwen3_4b", "qwen35_4b"])
def test_qwen_prompt_for_each_thinking_mode(request, model, thinking, bench, task):
    row, _, _, rows = request.getfixturevalue(f"{bench}_sent")[task]
    chat_template_kwargs = {"enable_thinking": thinking == "on"}
    benchmark = Ruler16K([task]) if bench == "ruler" else LoftRag([task])
    sent, budget = next(
        iter(
            _send(
                benchmark, rows, {"chat_template_kwargs": chat_template_kwargs}
            ).values()
        )
    )
    tokenizer = _qwen_tokenizer(model)
    adapter = _qwen_adapter(
        tokenizer, ScriptedLM([0], CONFIG_EOS[model], len(tokenizer))
    )
    _, questions = adapter._preprocess_context_and_questions(
        sent.context, sent.questions, sent.answer_prefix, chat_template_kwargs
    )
    # RULER keeps its completion prefix and budget only without thinking.
    completion = bench == "ruler" and thinking == "off"
    prefix = row.answer_prefix if completion else ""
    assert questions == [
        row.question + "<|im_end|>\n" + GENERATION_PROMPT[model, thinking] + prefix
    ]
    assert budget == (row.max_new_tokens if completion or bench == "loft" else 8192)


@pytest.mark.parametrize("model", ["qwen3_4b", "qwen35_4b"])
def test_qwen_generation_ends_at_end_of_turn(model):
    tokenizer = _qwen_tokenizer(model)
    yes = tokenizer.encode(" yes", add_special_tokens=False)[0]
    no = tokenizer.encode(" no", add_special_tokens=False)[0]
    end_of_turn = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # Calls: context prefill, question -> " yes", then <|im_end|>, then " no" forever.
    lm = ScriptedLM([yes, yes, end_of_turn, no], CONFIG_EOS[model], len(tokenizer))
    request = Request(context="x", questions=["x"], answer_prefix="")
    response = _qwen_adapter(tokenizer, lm).process_request(
        request, {"max_new_tokens": 20}, {}
    )
    assert response.responses == [" yes"]


def _case_id(rec: Dict[str, Any]) -> str:
    mode = "think" if rec["thinking"] == "on" else "nothink"
    return f"{rec['model']}-{mode}-{rec['task']}"


@pytest.mark.parametrize(
    "rec", [r for r in REAL_OUTPUTS if r["bench"] == "ruler16k"], ids=_case_id
)
def test_ruler16k_real_output_scores_like_nvidia_ruler(rec):
    df = pd.DataFrame(
        [
            {
                "task": rec["task"],
                "predicted_answer": strip_reasoning(rec["raw_output"]),
                "answer": rec["gold"],
                "context_length": 16384,
            }
        ]
    )
    scores = Ruler16K().post_run_evaluate(df)["task_scores"][rec["task"]]
    assert scores["string_match"] == rec["expected"]["string_match"]


@pytest.mark.parametrize(
    "rec", [r for r in REAL_OUTPUTS if r["bench"] == "loft"], ids=_case_id
)
def test_loft_real_output_scores_like_upstream_loft(rec):
    df = pd.DataFrame(
        [
            {
                "task": rec["task"],
                "predicted_answer": strip_reasoning(rec["raw_output"]),
                "answers": rec["gold"],
                "answer_prefix": "Final Answer: ",
            }
        ]
    )
    metrics = loft_metrics(df)
    assert {k: metrics.get(k) for k in rec["expected"]} == rec["expected"]
