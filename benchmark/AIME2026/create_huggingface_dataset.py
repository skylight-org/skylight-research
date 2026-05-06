# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper to build/push an AIME2026 benchmark dataset."""

from datasets import Dataset, load_dataset
import pandas as pd

from ..aime_common import MAX_NEW_TOKENS_CAP


def create_aime2026_dataset(source_dataset_id: str = "yentinglin/aime_2026") -> Dataset:
    """Process raw AIME2026 data into benchmark format."""
    dataset = load_dataset(source_dataset_id)
    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    df = dataset[split_name].to_pandas()

    processed_data = []
    for _, row in df.iterrows():
        problem = row.get("problem", row.get("Problem"))
        answer = row.get("answer", row.get("Answer"))
        problem_id = row.get("id", row.get("ID"))

        context = (
            "Solve the following AIME (American Invitational Mathematics Examination) problem.\n\n"
            f"Problem: {problem}\n\n"
            "Instructions:\n"
            "- The answer should be an integer between 0 and 999\n"
            "- You must wrap your final answer in \\boxed{...} format"
        )
        processed_data.append(
            {
                "context": context,
                "question": "What is the answer to this problem?",
                "answer_prefix": "",
                "answer": [str(answer)],
                "task": "aime2026",
                "max_new_tokens": MAX_NEW_TOKENS_CAP,
                "problem_id": problem_id,
            }
        )

    processed_df = pd.DataFrame(processed_data)
    final_df = processed_df[
        ["context", "question", "answer_prefix", "answer", "task", "max_new_tokens"]
    ]
    return Dataset.from_pandas(final_df)


if __name__ == "__main__":
    processed_dataset = create_aime2026_dataset()
    print(f"Processed {len(processed_dataset)} AIME2026 problems")
    print("Sample processed example:")
    print(processed_dataset[0])
    # Update this target if you want a different hub location.
    processed_dataset.push_to_hub("xAlg-AI/att-hub-aime2026", config_name="aime2026", split="test")

