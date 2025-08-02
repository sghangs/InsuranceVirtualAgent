import os
import sys

"""
Defining common constant for synthesizer
"""
POLICY_DOCUMENTS_PATH:str = "data/policy_file"
INPUT_FORMAT:str = "Questions about homewoner insurance policy. What cover/not cover depending on home damage situation"
EXPECTED_OUTPUT_FORMAT:str = "Detail answer on what covered/not covered for that accident based on policy"
TASK:str = "RAG chatbot for homewner insurance policy documents"
SCENARIO:str = "Customer asking queries about their policy information"
MAX_GOLDEN_PER_CONTEXT:int = 2
"""
Defining common constant for retriever
"""
RERANK_MODEL:str = "bge-reranker-v2-m3"
PINECONE_INDEX_NAMESPACE:str = "policy-documents"
RETRIEVE_TOP_K_DOCUMENTS:int = 3

"""
Defining common constant for graph
"""
LLM_MODEL_ID:str = "gpt-4o"
PINECONE_DENSE_INDEX_NAME:str = "insurance-virtual-agent-dense"
PINECONE_SPARSE_INDEX_NAME:str = "insurance-virtual-agent-sparse"
MESSAGES_COUNT:int = 3

"""
Defining common constant for front end
"""
RAG_ENDPOINTS:str = "http://localhost:8080/rag"
SIGNUP_ENDPOINT:str = "http://localhost:8080/signup"
TOKEN_ENDPOINT:str = "http://localhost:8080/token"
ME_ENDPOINT:str = "http://localhost:8080/me"

"""
Defining common constant for long term memory
"""
MIN_CONN:int = 1
MAX_CONN:int = 5

TRIVIAL_MESSAGES = {"hello", "hi", "yes", "no", "ok", "thanks", "thank you", "okay", "sure", "goodbye", "bye", "see you", "later", "welcome", "help", "please", "sorry"}

ALGORITHM:str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES:int = 60 * 1  # 1 hour

TOP_K_CONVERSATIONS:int = 2
SIMILARITY_THRESHOLD:float = 0.75