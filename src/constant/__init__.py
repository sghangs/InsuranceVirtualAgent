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
MESSAGES_COUNT:int = 6

"""
Defining common constant for front end
"""
RAG_ENDPOINTS:str = "http://34.229.162.28:8080/rag"