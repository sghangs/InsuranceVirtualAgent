from evaluation.evaluate import EvaluationPipeline
from src.exception.exception import InsuranceAgentException
import sys


def evaluate_test():
    """
    Run the entire evaluation pipeline and save the results 
    """
    try:
        eval_obj = EvaluationPipeline()
        goldens = eval_obj.read_goldens()
        test_cases = eval_obj.generate_testcases(goldens)
        result = eval_obj.run_evaluation(test_cases)
        eval_obj.save_results(result)
    except Exception as e:
        raise InsuranceAgentException(e,sys)


