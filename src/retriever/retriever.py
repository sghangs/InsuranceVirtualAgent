import boto3
import json
import sys
from pinecone import Pinecone
from langchain_core.messages.utils import get_buffer_string
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from src.exception.exception import InsuranceAgentException
from src.loggers.logger import logging
from src.entity.schema_entity import RetrieverInput
from src.constant import (
    RETRIEVE_TOP_K_DOCUMENTS,
    PINECONE_INDEX_NAMESPACE,
    RERANK_MODEL 
)

    
class HybridRetriever:
    """ 
    Hybrid Retriever using Bedrock's hosted dense embedding model and pinecone's self 
    hosted embedding model for sparse embeddings.
    """
    def __init__(self,pinecone_api_key,dense_index_name,sparse_index_name) -> None:
        """ 
        Initialize pinecone client and aws services
        """
        try:
            self.pc = Pinecone(api_key=pinecone_api_key)
            dense_index_response = self.pc.describe_index(name=dense_index_name)
            dense_dns_host = dense_index_response["host"]
            self.dense_index = self.pc.Index(host=dense_dns_host)

            sparse_index_response = self.pc.describe_index(name=sparse_index_name)
            sparse_dns_host = sparse_index_response["host"]
            self.sparse_index = self.pc.Index(host=sparse_dns_host)
           
            self.s3_client = boto3.client("s3")
        except Exception as e:
            raise InsuranceAgentException(e,sys)

    def search_dense_index(self,query:str,policy_number:str):
        """
        Search the dense index for the given query
        """
        try:
            dense_results = self.dense_index.search(
                namespace = PINECONE_INDEX_NAMESPACE,
                query = {
                    "top_k":RETRIEVE_TOP_K_DOCUMENTS,
                    "inputs":{"text":query},
                    "filter":{"policy_number":policy_number}
                }
            )
            return dense_results
        except Exception as e:
            raise InsuranceAgentException(e,sys)
    
    def search_sparse_index(self,query:str,policy_number:str):
        """
        Search the sparse index for the given query
        """
        try:
            sparse_results = self.sparse_index.search(
                namespace = PINECONE_INDEX_NAMESPACE,
                query = {
                    "top_k":RETRIEVE_TOP_K_DOCUMENTS,
                    "inputs":{"text":query},
                    "filter":{"policy_number":policy_number}
                }
            )
            return sparse_results
        except Exception as e:
            raise InsuranceAgentException(e,sys)
    
    def merge_documents(self,dense_results,sparse_results):
        """
        Get the unique hits from two search results and return them as single array 
        of {'id', 'chunk_text'} dicts, printing each dict on a new line.
        """
        try:
            #Deduplicate by id
            deduplicate_hits = {hit["_id"]: hit for hit in dense_results["result"]["hits"] + sparse_results["result"]["hits"]}.values()
            # Sort by _score descending
            sorted_hits = sorted(deduplicate_hits, key=lambda x: x['_score'], reverse=True)
            # Transform to format for reranking
            result = [{'_id': hit['_id'], 'chunk_text': hit['fields']['chunk_text']} for hit in sorted_hits]

            return result
        
        except Exception as e:
            raise InsuranceAgentException(e,sys)
    
    def rerank_documents(self,merge_results,query):
        """
        Rerank the documents based on the semantic relevance to the query
        """
        try:
            result = self.pc.inference.rerank(
                model = RERANK_MODEL,
                query = query,
                documents=merge_results,
                rank_fields=["chunk_text"],
                top_n=RETRIEVE_TOP_K_DOCUMENTS,
                return_documents=True,
                parameters={
                    "truncate":"END"
                }
            )

            return result
        except Exception as e:
            raise InsuranceAgentException(e,sys)

    def retrieve_documents(self,query:str,policy_number: str):
        """ 
        Perform hybrid retrieval with metadata filtering
        """
        try:
            dense_results = self.search_dense_index(query,policy_number)
            logging.info("Dense search is completed")
            sparse_results = self.search_sparse_index(query,policy_number)
            logging.info("Sparse search is completed")
            merge_results = self.merge_documents(dense_results,sparse_results)
            logging.info("Merging of docuemnts is completed")
            rerank_results = self.rerank_documents(merge_results,query)

            document_list = []
            for row in rerank_results.data:
                document_list.append(row['document']['chunk_text'])

            logging.info("Search results for documents are completed")
            return document_list
        
        except Exception as e:
            raise InsuranceAgentException(e,sys)
    
    def get_tools(self):
        """ 
        Get the tools for hybrid retriever
        """
        try:
            return [StructuredTool.from_function(
                name="hybrid_retriever",
                func=self.retrieve_documents,
                description="""Use this tool only when query is related to policy documents.
                    Do not use for general queries even though if policy number is provided.""",
                args_schema=RetrieverInput
            )]
        
        except Exception as e:
            raise InsuranceAgentException(e,sys)
    
    
    
    










