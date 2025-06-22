import boto3
import json
import sys
from pinecone import Pinecone
from langchain_core.messages.utils import get_buffer_string
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from src.exception.exception import InsuranceAgentException
from src.logging.logger import logging
from src.entity.schema_entity import RetrieverInput
from src.constant import (
    SERVICE_NAME,
    DENSE_EMBEDDING_DIMENSIONS,
    DENSE_EMBEDDING_MODEL_ID,
    NORMALIZE_FLAG,
    ACCEPT,
    RETRIEVE_ALPHA,
    CONTENT_TYPE,
    SPARSE_EMBEDDING_MODEL_ID,
    RETRIEVE_TOP_K_DOCUMENTS,
    PINECONE_INDEX_NAMESPACE   
)

    
class HybridRetriever:
    """ 
    Hybrid Retriever using Bedrock's hosted dense embedding model and pinecone's self 
    hosted embedding model for sparse embeddings.
    """
    def __init__(self,pinecone_api_key,index_name) -> None:
        """ 
        Initialize pinecone client and aws services
        """
        try:
            self.bedrock = boto3.client(service_name=SERVICE_NAME)
            self.pc = Pinecone(api_key=pinecone_api_key)
            index_response = self.pc.describe_index(name=index_name)
            dns_host = index_response["host"]
            self.index = self.pc.Index(host=dns_host)
            self.s3_client = boto3.client("s3")
        except Exception as e:
            raise InsuranceAgentException(e,sys)
        

    def generate_dense_embeddings(self,text,model_id=DENSE_EMBEDDING_MODEL_ID):
        """ 
        Generate dense vectors of the document text using aws bedrock 

        Args:
            text (str): text which need to be transformed into dense vectors
            bedrock (aws object): bedrock client for model access
            model_id (str): name of bedrock model
        Returns:
            embedding (List[int]): Dense vector for text data
        """
        try:
            body = json.dumps({
                    "inputText": text,
                    "dimensions": DENSE_EMBEDDING_DIMENSIONS,
                    "normalize": NORMALIZE_FLAG
                })
            response = self.bedrock.invoke_model(
                body=body,
                modelId=model_id,
                accept=ACCEPT,
                contentType=CONTENT_TYPE
            )
            response_body=json.loads(response.get("body").read())
            response=response_body["embedding"]
            return response
        except Exception as e:
            raise InsuranceAgentException(e,sys)

    def generate_sparse_embeddings(self,text_chunk: str):
        """ 
        Generate sparse embeddings using pinecone hosted embedding models 
        Args:
            pc (Pinecone object): Pinecone client object
            text_chunks (str): Text chunk to be embedded.

        Returns:
            EmbeddingsList: A list of sparse embeddings data 

        """
        try:
            sparse_embeddings = self.pc.inference.embed(
                model=SPARSE_EMBEDDING_MODEL_ID,
                inputs=[text_chunk],
                parameters={"input_type": "passage", "truncate": "END"}

            )
            return sparse_embeddings
        except Exception as e:
            raise InsuranceAgentException(e,sys)

    def retrieve_documents(self,query:str,policy_number: str):
        """ 
        Perform hybrid retrieval with metadata filtering
        """
        try:
            dense_vector = self.generate_dense_embeddings(query)
            logging.info("Dense embeddings are generated")
            sparse_vector = self.generate_sparse_embeddings(query)
            logging.info("Sparse Embeddings are generated")

            #Define metadata filter
            filter_query = {"policy_number":policy_number} if policy_number else {}
            search_results = self.index.query(
                namespace=PINECONE_INDEX_NAMESPACE,
                vector = dense_vector,
                sparse = sparse_vector,
                top_k = RETRIEVE_TOP_K_DOCUMENTS,
                alpha = RETRIEVE_ALPHA,
                filter = filter_query,
                include_metadata=True,
                include_values=False,
            )

            logging.info("Search results for documents are completed")
            return search_results["matches"]
        
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
    
    
    
    










