import argparse
import streamlit.web.bootstrap as st_wb
from evaluation.run_evaluation import evaluate_test
from scripts.generate_goldens import DatasetGenerator
from src.exception.exception import InsuranceAgentException
import sys



def main():
    """
    Run the main.py in three modes -- Interactive, Evaluate and Generate
    """
    try:
        parser = argparse.ArgumentParser(description="Run RAG System")

        parser.add_argument(
            "--mode",choices=["app","evaluate","generate"],default="app",
            help="Choose which component to launch"
        )
        args = parser.parse_args()

        if args.mode == "app":
            print("🚀 Launching Streamlit app...")
            st_wb.run("app/Insurance_Virtual_Agent.py","",[], flag_options={})
        
        elif args.mode == "evaluate":
            print("🧪 Running DeepEval evaluation on test cases...")
            evaluate_test()
        
        elif args.mode == "generate":
            print("📄 Generating synthetic test dataset...")
            gen_obj = DatasetGenerator()
            gen_obj.save_dataset()
    
    except Exception as e:
        raise InsuranceAgentException(e,sys)

if __name__ == "__main__":
    main()


