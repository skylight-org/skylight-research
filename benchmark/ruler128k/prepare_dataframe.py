# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split raw RULER samples into the columns the benchmark harness expects.

The upstream dataset for this context length stores each sample as a single raw
``input`` string -- context, question and answer prefix already concatenated,
with no chat template applied.  ``ruler16k`` / ``ruler32k`` consume a pre-split
dataset because ``ruler32k/create_huggingface_dataset.py`` performed this same
split offline before pushing to the Hub; no such pre-split dataset exists for
64k / 128k, so the split happens here at load time using the same anchors.

Keeping the split identical to the offline one is what makes the 16k / 32k /
64k / 128k numbers comparable: the question is held out of ``context``, so a KV
compressor that only sees ``context`` remains query-agnostic.
"""

import re
from typing import List, Tuple

import pandas as pd

# Source: https://github.com/hsiehjackson/RULER/blob/main/scripts/data/synthetic/constants.py

QUESTION_PATTERNS = {
    "niah": re.compile(r"What (?:is|are all) the special magic"),
    "vt": re.compile(r"Question: Find all variables that are assigned the value"),
    "cwe": re.compile(
        r"Question: What are the 10 most common words in the above list\?"
    ),
    "fwe": re.compile(r"Question: Do not provide any explanation\."),
    "qa": re.compile(r"Answer the question based on the given documents\."),
}

ANSWER_PATTERNS = {
    "niah": re.compile(r"The special magic"),
    "vt": re.compile(r"Answer:"),
    "cwe": re.compile(r"Answer:"),
    "fwe": re.compile(r"Answer:"),
    "qa": re.compile(r"Answer:"),
}

# Source: https://github.com/hsiehjackson/RULER/blob/main/scripts/data/synthetic/constants.py
MAX_NEW_TOKENS = {
    "niah": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa": 32,
}

# The exact schema (and column order) that ruler16k / ruler32k expose.
OUTPUT_COLUMNS: List[str] = [
    "context",
    "question",
    "answer_prefix",
    "answer",
    "task",
    "max_new_tokens",
    "context_length",
]


class RulerSplitError(ValueError):
    """A raw RULER sample did not contain the expected question/answer anchors."""


def split_context_question(text: str, task: str) -> Tuple[str, str, str]:
    """Split a raw RULER ``input`` into ``(context, question, answer_prefix)``.

    ``context + question + answer_prefix == text`` holds exactly.

    The ``cwe`` and ``qa_*`` templates state their instruction twice -- once as a
    preamble before the haystack and once again as the real question -- so the
    *last* match is the boundary.  Using the first match would move the entire
    document set into ``question`` and leave a two-sentence ``context``, which
    produces plausible-looking but meaningless scores.

    Args:
        text: The raw RULER sample text.
        task: Task name, e.g. ``"niah_multikey_2"``.  Only its family prefix is
            used to pick the anchors.

    Returns:
        Tuple of ``(context, question, answer_prefix)``.

    Raises:
        RulerSplitError: If either anchor is missing.  Never returns a
            partially-split sample, and never drops one.
    """
    family: str = task.split("_")[0]
    question_pattern = QUESTION_PATTERNS[family]
    answer_pattern = ANSWER_PATTERNS[family]

    question_matches = list(question_pattern.finditer(text))
    if not question_matches:
        raise RulerSplitError(
            f"task={task}: question anchor {question_pattern.pattern!r} not found "
            f"in a sample of {len(text)} chars; tail={text[-200:]!r}"
        )
    index: int = question_matches[-1].start()
    context, question_and_answer = text[:index], text[index:]

    answer_match = answer_pattern.search(question_and_answer)
    if answer_match is None:
        raise RulerSplitError(
            f"task={task}: answer anchor {answer_pattern.pattern!r} not found after "
            f"the question anchor; question_and_answer={question_and_answer!r}"
        )
    index = answer_match.start()
    question = question_and_answer[:index]
    answer_prefix = question_and_answer[index:]
    return context, question, answer_prefix


def prepare_dataframe(df: pd.DataFrame, task: str, context_length: int) -> pd.DataFrame:
    """Convert a raw RULER dataframe into the schema ``Benchmark`` expects.

    Args:
        df: ``to_pandas()`` of one upstream split, with columns
            ``index, input, outputs, length``.  Mutated in place rather than
            copied: a single 128k split holds roughly 660 MB of strings.
        task: Split name, e.g. ``"niah_multikey_2"``.
        context_length: Nominal context length, recorded on every row.

    Returns:
        DataFrame containing exactly ``OUTPUT_COLUMNS``, in that order.

    Raises:
        RulerSplitError: If any row fails to split.  Rows are never dropped --
            silently shrinking the sample would shrink the scoring denominator
            too, and produce a normal-looking score over fewer rows.
    """
    if len(df) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df["context"], df["question"], df["answer_prefix"] = zip(
        *(split_context_question(text, task) for text in df["input"])
    )
    df["task"] = task
    df["max_new_tokens"] = MAX_NEW_TOKENS[task.split("_")[0]]
    df["context_length"] = context_length
    df = df.rename(columns={"outputs": "answer"})
    return df[OUTPUT_COLUMNS]
