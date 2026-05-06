# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from datasets import Dataset, load_dataset
import pandas as pd

"""
AIME2025 Dataset Processing

The AIME (American Invitational Mathematics Examination) 2025 dataset contains mathematical competition problems.
Each problem requires a numerical answer between 0 and 999.

Dataset structure:
- problem: The mathematical problem statement
- answer: The correct numerical answer (integer between 0-999)
- solution: The solution explanation (not used for evaluation)
- id: Problem identifier
- url: Source URL
- year: Competition year

For evaluation, we format the problems to instruct the model to wrap its answer in \boxed{...} format,
which is standard in mathematical competition contexts.
"""

def create_aime2025_dataset():
    """
    Process the AIME2025 dataset and convert it to the standardized benchmark format.
    """
    # Load the original dataset
    dataset = load_dataset("yentinglin/aime_2025")
    df = dataset["train"].to_pandas()
    
    # Create the standardized format
    processed_data = []
    
    for _, row in df.iterrows():
        # Format the problem with clear instructions about the boxed answer format
        context = f"""Solve the following AIME (American Invitational Mathematics Examination) problem.

Problem: {row['problem']}

Instructions:
- The answer should be an integer between 0 and 999
- You must wrap your final answer in \\boxed{{...}} format"""
        
        question = "What is the answer to this problem?"
        
        # The answer prefix encourages the model to show work before the final answer
        answer_prefix = ""
        
        # Convert answer to list format (some benchmarks expect this)
        answer = [str(row['answer'])]
        
        processed_data.append({
            'context': context,
            'question': question,
            'answer_prefix': answer_prefix,
            'answer': answer,
            'task': 'aime2025',
            'max_new_tokens': 65536,  # Allow comprehensive step-by-step solutions
            'problem_id': row['id'],
            'year': row['year']
        })
    
    # Convert to DataFrame and then to Dataset
    processed_df = pd.DataFrame(processed_data)
    
    # Select only the required columns for the benchmark
    final_df = processed_df[['context', 'question', 'answer_prefix', 'answer', 'task', 'max_new_tokens']]
    
    return Dataset.from_pandas(final_df)

if __name__ == "__main__":
    # Create the processed dataset
    processed_dataset = create_aime2025_dataset()
    
    # Push to hub (you would need to set up your own repo)
    # For now, we'll just save locally or use the existing dataset
    print(f"Processed {len(processed_dataset)} AIME2025 problems")
    print("Sample processed example:")
    print(processed_dataset[0])
    
    # Optionally save locally for testing
    processed_dataset.push_to_hub("xAlg-AI/att-hub-aime2025", config_name=f"aime2025", split="test")
