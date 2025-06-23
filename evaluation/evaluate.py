from evaluation.deepeval_components.synthesizer import synthesize
import os
import json
from deepeval.test_case import LLMTestCase
from deepeval.dataset import Golden
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

from dotenv import load_dotenv
load_dotenv()


class DatasetGenerator:
    """Generates synthetic queries and golden answers using DeepEval Synthesizer."""
    
    def __init__(self):
        self.synthesizer,self.goldens = synthesize()
        
    def save_dataset(self):
        self.synthesizer.save_as(
            file_type='json',
            directory="data/goldens",
            file_name="synthetic_dataset"
        )
        print(f"✅ Synthetic dataset saved to data/goldens/synthetic_dataset.json")


class EvaluationPipeline:
    """Runs evaluation for each golden dataset entry using DeepEval."""
    def generate_testcases(self):
        """Generates test cases for each golden output."""
        gen_obj = DatasetGenerator()
        self.test_cases = []
        for golden in gen_obj.goldens:
            res,text_chunks = rag(golden.input)
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=res,
                retrieval_context=text_chunks,
                expected_output=golden.expected_output,
            )
            self.test_cases.append(test_case)

        return self.test_cases

    def run_evaluation(self, test_cases):
        """Evaluates the test cases using the custom LLM and embedding model."""
        
        result = evaluate(
            test_cases=test_cases,
            metrics=[
                AnswerRelevancyMetric(),
                FaithfulnessMetric(),
                ContextualPrecisionMetric(),
                ContextualRecallMetric(),
                ContextualRelevancyMetric()
            ]
        )
        return result

    def save_results(self, result, file_path="evaluation/data/results/evaluation_results.txt"):
        """Saves the evaluation results to a text file."""
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        
    
        with open(file_path, "w") as file:
            file.write(str(result))
        print(f"✅ Evaluation results saved to {file_path}")



