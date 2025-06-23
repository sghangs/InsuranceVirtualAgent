import os
import sys

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