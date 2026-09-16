"""LOFT RAG evaluation metrics - exact implementation from LOFT.

This module contains the exact evaluation functions from LOFT's evaluation codebase,
ensuring 100% fidelity with LOFT's evaluation methodology.
"""

import ast
import collections
import re
import string
import unicodedata
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy.optimize


def normalize_answer(s: str) -> str:
    """Normalize answer string. Taken from SQuAD evaluation.

    This is LOFT's exact normalization function:
    - Unicode NFD normalization
    - Remove articles (a, an, the)
    - Fix whitespace
    - Remove punctuation
    - Lowercase
    """
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text: str) -> str:
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_answers(answers: List[str]) -> List[str]:
    """Normalize a list of answers."""
    return [normalize_answer(answer) for answer in answers]


def get_tokens(s: str) -> List[str]:
    """Get tokens from normalized string. Taken from SQuAD evaluation."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_em(gold_answers: List[str], pred_answer: str) -> float:
    """Calculates exact match score. Taken from SQuAD evaluation."""
    return max([float(ga == pred_answer) for ga in gold_answers])


def compute_subspan_em(gold_answers: List[str], pred_answer: str) -> float:
    """Calculates subspan match score."""
    return max([1.0 if ga in pred_answer else 0.0 for ga in gold_answers])


def compute_f1(gold_answers: List[str], pred_answer: str) -> float:
    """Calculates F1 score. Taken from SQuAD evaluation."""
    pred_toks: List[str] = get_tokens(pred_answer)

    f1_scores: List[float] = []
    for ga in gold_answers:
        gold_toks: List[str] = get_tokens(ga)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same: int = sum(common.values())

        if num_same == 0:
            f1_scores.append(0.0)
            continue

        if not gold_toks or not pred_toks:
            f1: float = float(gold_toks == pred_toks)
        else:
            precision: float = 1.0 * num_same / len(pred_toks)
            recall: float = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)

    return max(f1_scores)


def compute_em_multi_value(
    gold_answers: List[str], pred_answers: List[str]
) -> float:
    """Calculates exact match score for multi-value RAG."""
    return float(set(gold_answers) == set(pred_answers))


def compute_coverage(gold_answers: List[str], pred_answers: List[str]) -> float:
    """Calculates coverage of gold_answers in pred_answers."""
    return len(set(pred_answers).intersection(set(gold_answers))) / float(
        len(gold_answers)
    )


def compute_multi_value_subspan_em(
    gold_answers: List[str], pred_answers: List[str]
) -> float:
    """Calculates subspan match score. Adopted from DROP evaluation."""
    scores: np.ndarray = np.zeros([len(gold_answers), len(pred_answers)])
    for gold_index, gold_item in enumerate(gold_answers):
        for pred_index, pred_item in enumerate(pred_answers):
            if gold_item in pred_item or pred_item in gold_item:
                scores[gold_index, pred_index] = 1
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-scores)
    aligned_scores: np.ndarray = np.zeros(len(gold_answers))
    for r, c in zip(row_ind, col_ind):
        aligned_scores[r] = scores[r, c]
    return float(all(aligned_scores))


def extract_prediction(
    model_output: str, answer_prefix: str = "final answer"
) -> List[str]:
    """Extracts the prediction from the model output.

    Looks for format: "Final Answer: [answer1, answer2]" and extracts the list.

    Args:
        model_output: Raw model output string
        answer_prefix: The prefix to look for (default: "final answer")

    Returns:
        List of extracted answers (strings)
    """

    def _escape_single_quotes(s: str) -> str:
        pattern = r"([a-zA-Z0-9])'([a-zA-Z0-9])"
        replacement = r"\1\'\2"
        return re.sub(pattern, replacement, s)

    model_output = model_output.replace("*", "").replace("`", "")
    model_output_lines: List[str] = model_output.strip().split("\n")
    preds: List[str] = []

    # Upstream (utils.py:465-478) stops at the FIRST bracketed line whether or not it
    # parses, with no fallback.  Two earlier deviations here (break inside the try, and an
    # after-the-prefix fallback) were strictly more permissive; on 470 real generations
    # they changed 5 predictions, all upward, and +0.155 subspan_em on oracle-top-k nq.
    for line in model_output_lines:
        if "[" in line and "]" in line:
            pred_start_index: int = line.find("[")
            pred_end_index: int = line.rfind("]") + 1
            pred_as_str: str = line[pred_start_index:pred_end_index].strip()
            try:
                pred_as_str = _escape_single_quotes(pred_as_str)
                parsed = ast.literal_eval(pred_as_str)
                # A slice that starts "[" and ends "]" parses to a list, or to a TUPLE
                # when the line holds several ("[a], [b]").  Upstream returns the raw
                # value and convert_to_str iterates it, so the tuple becomes N
                # predictions; wrapping it whole scored differently in both directions.
                preds = [str(p) for p in parsed]
            except Exception:
                pass
            break

    return preds


def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate LOFT RAG metrics for a DataFrame of results.

    Args:
        df: DataFrame with columns:
            - predicted_answer: Raw model output (string)
            - answers: Ground truth answers (list of strings)
            - task: Task name (e.g., "nq_32k", "qampari_128k")
            - answer_prefix: Answer prefix used

    Returns:
        Dictionary with metrics (em, subspan_em, f1/coverage)
    """
    if len(df) == 0:
        return {"error": "No results to evaluate"}

    task_name: str = df["task"].iloc[0] if "task" in df.columns else ""
    is_multi_value: bool = task_name.startswith("qampari") or task_name.startswith(
        "quest"
    )

    answer_prefix: str = (
        df["answer_prefix"].iloc[0]
        if "answer_prefix" in df.columns
        else "Final Answer: "
    )

    all_em_scores: List[float] = []
    all_subspan_em_scores: List[float] = []
    all_f1_scores: List[float] = []
    all_coverage_scores: List[float] = []

    for _, row in df.iterrows():
        gold_answers = row["answers"]
        if isinstance(gold_answers, np.ndarray):
            gold_answers = gold_answers.tolist()
        elif isinstance(gold_answers, tuple):
            gold_answers = list(gold_answers)
        elif not isinstance(gold_answers, list):
            gold_answers = [] if pd.isna(gold_answers) else [gold_answers]

        # Drop null entries and keep a clean list of answer strings.
        gold_answers = [ga for ga in gold_answers if pd.notna(ga)]

        gold_answers_normalized: List[str] = normalize_answers(
            [str(ga) for ga in gold_answers]
        )

        if not gold_answers_normalized:
            all_em_scores.append(0.0)
            all_subspan_em_scores.append(0.0)
            if is_multi_value:
                all_coverage_scores.append(0.0)
            else:
                all_f1_scores.append(0.0)
            continue

        predicted_output: str = (
            str(row["predicted_answer"]) if pd.notna(row["predicted_answer"]) else ""
        )
        pred_answers_raw: List[str] = extract_prediction(
            predicted_output, answer_prefix.lower()
        )

        if not pred_answers_raw:
            # Upstream (rag.py:80-88) records em/subspan_em/f1 = 0.0 here but NOT
            # coverage, so it averages coverage over PARSED rows only.  Denominator on
            # real data: 70 vs 57 (qampari), 70 vs 60 (quest).
            all_em_scores.append(0.0)
            all_subspan_em_scores.append(0.0)
            # Upstream (rag.py:80-88) appends f1=0.0 here for BOTH task types, so a
            # multi-value aggregate carries f1 whenever any row failed to parse.
            all_f1_scores.append(0.0)
            continue

        pred_answers_normalized: List[str] = normalize_answers(pred_answers_raw)

        if is_multi_value:
            em: float = compute_em_multi_value(
                gold_answers_normalized, pred_answers_normalized
            )
            coverage: float = compute_coverage(
                gold_answers_normalized, pred_answers_normalized
            )
            subspan_em: float = compute_multi_value_subspan_em(
                gold_answers_normalized, pred_answers_normalized
            )
            all_em_scores.append(em)
            all_coverage_scores.append(coverage)
            all_subspan_em_scores.append(subspan_em)
        else:
            pred_answer: str = pred_answers_normalized[0]
            em = compute_em(gold_answers_normalized, pred_answer)
            subspan_em = compute_subspan_em(gold_answers_normalized, pred_answer)
            f1: float = compute_f1(gold_answers_normalized, pred_answer)
            all_em_scores.append(em)
            all_subspan_em_scores.append(subspan_em)
            all_f1_scores.append(f1)

    metrics: Dict[str, Any] = {
        "em": float(np.mean(all_em_scores)),
        "subspan_em": float(np.mean(all_subspan_em_scores)),
    }

    # Upstream's aggregate_metrics averages each metric over the list it actually
    # appended to, and omits a key whose list stayed empty -- it never substitutes 0.0.
    if is_multi_value:
        if all_coverage_scores:
            metrics["coverage"] = float(np.mean(all_coverage_scores))
        metrics["num_scored_for_coverage"] = len(all_coverage_scores)
    if all_f1_scores:
        metrics["f1"] = float(np.mean(all_f1_scores))

    metrics["num_samples"] = len(df)
    return metrics

