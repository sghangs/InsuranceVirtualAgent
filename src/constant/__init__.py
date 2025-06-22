import os
import sys

"""
Defining common constant for retriever
"""
SERVICE_NAME:str = "bedrock-runtime"
DENSE_EMBEDDING_MODEL_ID:str = "amazon.titan-embed-text-v2:0"
DENSE_EMBEDDING_DIMENSIONS:int = 256
NORMALIZE_FLAG:bool = True
ACCEPT:str = "application/json"
CONTENT_TYPE:str = "application/json"
SPARSE_EMBEDDING_MODEL_ID:str = "pinecone-sparse-english-v0"
PINECONE_INDEX_NAMESPACE:str = "policy-documents"
RETRIEVE_TOP_K_DOCUMENTS:int = 3
RETRIEVE_ALPHA:float = 0.5

"""
Defining common constant for graph
"""
LLM_MODEL_ID:str = "gpt-4o"
PINECONE_INDEX_NAME:str = "insurance-virtual-agent-hybrid"
MESSAGES_COUNT:int = 6