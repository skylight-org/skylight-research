# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from datasets import Dataset, load_dataset
import pandas as pd

"""
AIME2024 Dataset Processing

The AIME (American Invitational Mathematics Examination) 2024 dataset contains mathematical competition problems.
Each problem requires a numerical answer between 0 and 999.

Dataset structure:
- ID: Problem identifier (e.g., "2024-I-1", "2024-II-4")
- Problem: The mathematical problem statement
- Solution: The solution explanation (not used for evaluation)
- Answer: The correct numerical answer (integer between 0-999)

For evaluation, we format the problems to instruct the model to wrap its answer in \boxed{...} format,
which is standard in mathematical competition contexts.
"""

def create_aime2024_dataset():
    """
    Process the AIME2024 dataset and convert it to the standardized benchmark format.
    """
    # Load the original dataset
    dataset = load_dataset("Maxwell-Jia/AIME_2024")
    df = dataset["train"].to_pandas()
    
    # Create the standardized format
    processed_data = []
    
    for _, row in df.iterrows():
        # Format the problem with clear instructions about the boxed answer format
        context = f"""Solve the following AIME (American Invitational Mathematics Examination) problem.

Problem: {row['Problem']}
f
Instructions:
- The answer should be an integer between 0 and 999
- Please reason step by step, and put your final answer within \\boxed{{...}} format"""
        
        question = "What is the answer to this problem?"
        
        # The answer prefix encourages the model to show work before the final answer
        answer_prefix = ""
        
        processed_data.append({
            'context': context,
            'question': question,
            'answer_prefix': answer_prefix,
            'answer': str(row['Answer']),  # Convert to string for consistency
            'id': row['ID'],
            'max_new_tokens': 65536,  # Allow comprehensive step-by-step solutions
        })
    
    # Convert to Dataset
    processed_dataset = Dataset.from_pandas(pd.DataFrame(processed_data))
    
    return processed_dataset

if __name__ == "__main__":
    # Test the dataset creation
    processed_dataset = create_aime2024_dataset()
    print(f"Created dataset with {len(processed_dataset)} examples")
    print("\nFirst example:")
    print(processed_dataset[0])

    processed_dataset.push_to_hub("xAlg-AI/att-hub-aime2024", config_name=f"aime2024", split="test")
