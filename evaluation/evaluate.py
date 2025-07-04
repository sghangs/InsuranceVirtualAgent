from evaluation.deepeval_components.synthesizer import synthesize
from src.constant import LLM_MODEL_ID
import os
import json
from datetime import datetime
import pandas as pd
from evaluation.rag import RagPipeline
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


class EvaluationPipeline:
    """Runs evaluation for each golden dataset entry using DeepEval."""

    def read_goldens(self):
        """
        Read the goldens from json file
        """
        with open("data/goldens/synthetic_dataset.json") as file:
            goldens = json.load(file)

        return goldens

    def generate_testcases(self,goldens):
        """Generates test cases for each golden output."""
        rag_obj = RagPipeline()
        test_cases = []
        for golden in goldens:
            res,text_chunks = rag_obj.execute_rag(golden["input"])
            test_case = LLMTestCase(
                input=golden["input"],
                actual_output=res,
                retrieval_context=text_chunks,
                expected_output=golden["expected_output"],
            )
            test_cases.append(test_case)

        return test_cases

    def run_evaluation(self, test_cases):
        """Evaluates the test cases using the custom LLM and embedding model."""
        
        results = evaluate(
            test_cases=test_cases,
            metrics=[
                AnswerRelevancyMetric(),
                FaithfulnessMetric(),
                ContextualPrecisionMetric(),
                ContextualRecallMetric(),
                ContextualRelevancyMetric()
            ]
        )
        return results

    def save_results(self, results):
        """
        Saves the evaluation results to a csv, json and md files with versioning.
        """

        #Creating file with current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        version_dir = f"data/evaluation_results/{timestamp}"
        os.makedirs(version_dir, exist_ok=True)

        res1,res2=results
        _,res3=res1
        #Extracting test results in list
        rows = []
        for case in res3:
            for met in case.metrics_data:
                rows.append({
                    "TestCase":case.name,
                    "Input": case.input,
                    "Expected Output": case.expected_output,
                    "Actual Output": case.actual_output,
                    "Success":case.success,
                    "Matrics": met.name,
                    "Score": met.score,
                    "Reason": met.reason,
                    "Cost":met.evaluation_cost
                })

        #Creating csv file if not exists
        if not os.path.exists(os.path.dirname(f"{version_dir}/results.csv")):
            os.makedirs(os.path.dirname(f"{version_dir}/results.csv"))

        #saving data to csv file
        print("💾 Saving results...")
        pd.DataFrame(rows).to_csv(f"{version_dir}/results.csv", index=False)

        #Creating file if not exists
        if not os.path.exists(os.path.dirname(f"{version_dir}/results.json")):
            os.makedirs(os.path.dirname(f"{version_dir}/results.json"))
        #saving data to json file
        with open(f"{version_dir}/results.json", "w") as f:
            json.dump(rows, f, indent=2)

        #Creating file if not exists
        if not os.path.exists(os.path.dirname(f"{version_dir}/results.md")):
            os.makedirs(os.path.dirname(f"{version_dir}/results.md"))
        #saving data to md file
        with open(f"{version_dir}/results.md", "w") as f:
            f.write("# RAG Evaluation Report\n")
            f.write(f"**Run Time**: {timestamp}\n")
            f.write(f"**Dataset**: {version_dir}/results.md\n")
            f.write(f"**LLM Model**: {LLM_MODEL_ID} \n\n")

            for i, case in enumerate(res3, 1):
                f.write(f"### Test Case {i}\n")
                f.write(f"- **Input**: {case.input}\n")
                f.write(f"- **Expected**: {case.expected_output}\n")
                f.write(f"- **Actual**: {case.actual_output}\n")
                f.write(f"- **Success**: {case.success}\n")
                for met in case.metrics_data:
                    f.write(f"#### Metrics Data \n")
                    f.write(f"- **Metrics**: {met.name}\n")
                    f.write(f"- **Score**: {met.score}\n")
                    f.write(f"- *Cost**: {met.evaluation_cost:.4f}$\n\n")
                    f.write(f"- **Reason**: {met.reason}\n\n")

        print("✅ Evaluation completed.")
        print(f"🟢 Passed {sum(c.success for c in res3)}/{len(res3)} test cases")





