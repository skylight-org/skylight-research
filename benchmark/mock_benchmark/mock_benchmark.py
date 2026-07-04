# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock benchmark implementation for testing and demonstration purposes."""

from typing import Dict, Any, List
import pandas as pd

from ..base import Benchmark
from ..benchmark_registry import register_benchmark


@register_benchmark("mock_benchmark")
class MockBenchmark(Benchmark):
    """Mock benchmark for testing and demonstration purposes.

    This benchmark contains 5 simple samples with short contexts (<250 words)
    and basic comprehension questions. Two samples share the same context to
    test context grouping functionality.

    Example:
        >>> mock_benchmark = MockBenchmark()
        >>> results = mock_benchmark.run_benchmark(adapter, result_dir="/path/to/results")
        >>> print(f"Accuracy: {results['accuracy']}")
    """

    # Class attributes required by base Benchmark class
    all_datasets: List[str] = ["reading_comprehension"]
    benchmark_name: str = "mock_benchmark"
    huggingface_dataset_id: str = "mock/dataset"  # Not actually used since we override _load_datasets

    def _load_datasets(self) -> pd.DataFrame:
        """Load mock dataset with hardcoded samples.
        
        Returns:
            DataFrame containing 5 mock samples with context, question, and answers.
        """
        # Sample contexts - all under 250 words
        contexts: Dict[str, str] = {
            "science_context": """
                Photosynthesis is the process by which green plants and some other organisms 
                use sunlight to synthesize foods from carbon dioxide and water. Photosynthesis 
                in plants generally involves the green pigment chlorophyll and generates oxygen 
                as a byproduct. The process occurs in two main stages: the light-dependent 
                reactions and the light-independent reactions (Calvin cycle). During the 
                light-dependent reactions, chlorophyll absorbs light energy and converts it 
                into chemical energy in the form of ATP and NADPH. These energy carriers are 
                then used in the Calvin cycle to convert carbon dioxide into glucose. This 
                process is essential for life on Earth as it produces the oxygen we breathe 
                and forms the base of most food chains. Plants typically perform photosynthesis 
                in their leaves, where specialized organelles called chloroplasts contain the 
                chlorophyll necessary for the process.
            """.strip(),
            
            "history_context": """
                The Renaissance was a period of cultural, artistic, and intellectual revival 
                that began in Italy during the 14th century and spread throughout Europe. 
                It marked the transition from medieval to modern times and was characterized 
                by renewed interest in classical Greek and Roman culture. Key figures of the 
                Renaissance included Leonardo da Vinci, who was both an artist and inventor; 
                Michelangelo, famous for his sculptures and paintings; and Galileo Galilei, 
                who made significant contributions to astronomy and physics. The period saw 
                major developments in art, literature, science, and philosophy. The invention 
                of the printing press by Johannes Gutenberg around 1440 revolutionized the 
                spread of knowledge and ideas. Renaissance art was characterized by realistic 
                portrayals, perspective, and detailed studies of human anatomy. This period 
                laid the foundation for the Scientific Revolution and the Enlightenment that 
                would follow.
            """.strip(),
            
            "geography_context": """
                The Amazon rainforest, often called the "lungs of the Earth," is the world's 
                largest tropical rainforest. Located primarily in Brazil, it also extends 
                into Peru, Colombia, Venezuela, Ecuador, Bolivia, Guyana, Suriname, and 
                French Guiana. The Amazon covers approximately 5.5 million square kilometers 
                and contains about 10% of the world's known biodiversity. The forest plays 
                a crucial role in regulating the global climate by absorbing large amounts 
                of carbon dioxide and producing oxygen. It is home to thousands of species 
                of plants, animals, and indigenous communities. The Amazon River, which flows 
                through the rainforest, is the longest river in the world and has the largest 
                drainage basin. Unfortunately, the rainforest faces threats from deforestation, 
                mining, and climate change, making its conservation critical for the health 
                of our planet.
            """.strip()
        }

        # Create sample data - 5 samples total, 2 sharing the same context
        sample_data: List[Dict[str, Any]] = [
            {
                "context": contexts["science_context"],
                "question": "What are the two main stages of photosynthesis?",
                "answers": ["light-dependent reactions and light-independent reactions", 
                          "light-dependent reactions and Calvin cycle"],
                "task": "reading_comprehension"
            },
            {
                "context": contexts["science_context"],  # Same context as sample 1
                "question": "What gas is produced as a byproduct of photosynthesis?",
                "answers": ["oxygen"],
                "task": "reading_comprehension"
            },
            {
                "context": contexts["history_context"],
                "question": "In which century did the Renaissance begin?",
                "answers": ["14th century", "14th"],
                "task": "reading_comprehension"
            },
            {
                "context": contexts["geography_context"],
                "question": "Why is the Amazon rainforest called the 'lungs of the Earth'?",
                "answers": ["because it absorbs carbon dioxide and produces oxygen", 
                          "it regulates global climate by absorbing CO2 and producing oxygen"],
                "task": "reading_comprehension"
            },
            {
                "context": contexts["geography_context"],
                "question": "Which river flows through the Amazon rainforest?",
                "answers": ["Amazon River", "the Amazon River"],
                "task": "reading_comprehension"
            }
        ]

        # Convert to DataFrame
        df: pd.DataFrame = pd.DataFrame(sample_data)
        
        # Add sample IDs for tracking
        df["sample_id"] = range(1, len(df) + 1)
        
        print(f"Loaded {len(df)} mock samples")
        print(f"Unique contexts: {df['context'].nunique()}")
        
        return df

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics for mock benchmark results.

        Uses simple exact match and substring matching for evaluation.

        Args:
            results_df: DataFrame containing benchmark results with columns:
                - context: The input context
                - question: The input question
                - predicted_answer: Model's predicted answer
                - answers: Ground truth answers (list)
                - task: Task name
                - sample_id: Sample identifier

        Returns:
            Dictionary containing computed metrics including accuracy and per-sample scores.
        """
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        total_samples: int = len(results_df)
        correct_predictions: int = 0
        sample_scores: List[Dict[str, Any]] = []

        for _, row in results_df.iterrows():
            predicted_answer: str = str(row["predicted_answer"]).strip().lower()
            ground_truth_answers: List[str] = row["answers"]
            sample_id: int = row["sample_id"]
            
            # Check if prediction matches any ground truth answer
            is_correct: bool = False
            for gt_answer in ground_truth_answers:
                gt_answer_normalized: str = str(gt_answer).strip().lower()
                
                # Check exact match or substring match
                if (predicted_answer == gt_answer_normalized or 
                    gt_answer_normalized in predicted_answer):
                    is_correct = True
                    break
            
            if is_correct:
                correct_predictions += 1
            
            sample_scores.append({
                "sample_id": sample_id,
                "question": row["question"],
                "predicted_answer": row["predicted_answer"],
                "ground_truth": ground_truth_answers,
                "correct": is_correct
            })

        # Calculate metrics
        accuracy: float = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        metrics: Dict[str, Any] = {
            "accuracy": round(accuracy, 3),
            "correct_predictions": correct_predictions,
            "total_samples": total_samples,
            "sample_scores": sample_scores,
            "summary": {
                "benchmark": self.benchmark_name,
                "task": "reading_comprehension",
                "unique_contexts": results_df["context"].nunique(),
                "evaluation_method": "exact_match_and_substring"
            }
        }

        return metrics 