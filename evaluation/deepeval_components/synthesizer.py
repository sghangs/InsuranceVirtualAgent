from deepeval.synthesizer import Synthesizer 
from evaluation.deepeval_components.models import ChatGroqLLM, HuggingFaceModel
from langchain_huggingface import HuggingFaceEmbeddings
from deepeval.synthesizer.config import ContextConstructionConfig
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the synthesizer with custom LLM and embedding models
def synthesize():
    """
    This function initializes the synthesizer with custom LLM and embedding models,
    and generates golden outputs from documents.
    """
    # Initialize the synthesizer with the custom LLM and embedding model
    groq_api_key = os.getenv("GROQ_API_KEY")
    custom_model = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    synthesizer = Synthesizer(model=ChatGroqLLM(model=custom_model))

    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=["evaluation/data/docs/sample_policy_doc_AU1234.pdf"],
        include_expected_output=True,
        max_goldens_per_context=2,  # Limit to one golden per context
        context_construction_config=ContextConstructionConfig(
            embedder=HuggingFaceModel(model=embeddings_model),
            chunk_size=500,
            chunk_overlap=50,
            critic_model=ChatGroqLLM(model=custom_model),
            max_contexts_per_document=1,  # Limit to one context per document
            max_context_length=1000  # Limit context length to 1000 tokens  
        )
        
    )

    return synthesizer, goldens
