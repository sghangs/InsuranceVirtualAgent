from evaluation.deepeval_components.synthesizer import synthesize
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
import json
from evaluation.deepeval_components.models import ChatGroqLLM, HuggingFaceModel
from src.testrag import rag
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

class ModelFactory:
    """Creates instances of both ChatGroq and Hugging Face models."""
    def __init__(self, chatgroq_api_key):
        self.custom_model = ChatGroq(groq_api_key=chatgroq_api_key,model_name="gemma2-9b-it")
        self.chat_groq_llm = ChatGroqLLM(model=self.custom_model)
        self.embeddings_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.huggingface_embedding_model = HuggingFaceModel(model=self.embeddings_model)

    def get_models(self):
        return self.chat_groq_llm, self.huggingface_embedding_model


class DatasetGenerator:
    """Generates synthetic queries and golden answers using DeepEval Synthesizer."""
    goldens = [Golden(input="Is Plumbing or drainage problems related to septic tanks covered under my policy?",
                               expected_output="Yes, plumbing or drainage problems related to septic tanks are covered under your policy as per the terms and conditions outlined in the document." 
                               )]
    def __init__(self):
        #self.synthesizer,self.goldens = synthesize()
        pass
        

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
        self.test_cases = []
        for golden in DatasetGenerator.goldens:
            res,text_chunks = rag(golden.input)
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=res,
                retrieval_context=text_chunks,
                expected_output=golden.expected_output,
            )
            self.test_cases.append(test_case)

        return self.test_cases

    def run_evaluation(self, test_cases,chat_groq_llm,huggingface_embedding_model):
        """Evaluates the test cases using the custom LLM and embedding model."""
        
        result = evaluate(
            test_cases=test_cases,
            metrics=[
                AnswerRelevancyMetric(model=chat_groq_llm),
                FaithfulnessMetric(model=chat_groq_llm),
                ContextualPrecisionMetric(model=chat_groq_llm),
                ContextualRecallMetric(model=chat_groq_llm),
                ContextualRelevancyMetric(model=chat_groq_llm)
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



