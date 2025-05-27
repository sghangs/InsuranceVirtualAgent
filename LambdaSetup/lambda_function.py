import boto3
import os
import uuid
from typing import List
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.llms.bedrock import Bedrock
from urllib.parse import unquote_plus
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from pinecone_text.sparse import BM25Encoder


#Environment variables
os.environ["PINECONE_API_KEY"]=os.getenv("PINECONE_API_KEY")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

#Bedrock client initialization
bedrock = boto3.client(service_name="bedrock-runtime")

#pinecone client
pc = Pinecone()

#create hybrid(dense + sparse) index with external embeddings 
index_name = "insurance-virtual-agent-hybrid"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws",region="us-east-1"),
        deletion_protection="disabled",
        tags={
            "environment":"dev"
        }

    )

#Get the index host
index_response = pc.describe_index(name=index_name)
dns_host = index_response["host"]

index = pc.Index(host=dns_host)


#LLM initialization
llm=ChatGroq(model_name="Llama3-8b-8192")

#S3 client initialization
s3_client = boto3.client("s3")

def document_processing(file_path,policy_number):
    #load the pdf file
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    #split the docs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    docs = splitter.split_documents(docs)
    
    #adding policy number information in the document metadata
    for doc in docs:
        doc.metadata["policy_number"] = policy_number

    return docs

def generate_dense_embeddings(text,model_id="amazon.titan-embed-text-v2:0"):
    response = bedrock.invoke_model(
        body={"inputText":text,"dimensions": 256},
        modelId=model_id
    )
    return response["embedding"]

def generate_sparse_embeddings(corpus: List[str],text_chunk: str):
    """ 
    Generate BM25 sparse embeddings for a list of text chunks after fitting the BM25 
    encoder to the corpus.

    Args:
        corpus (List[str]): A list of text chunks to fit the BM25 encoder.
        text_chunks (str): Text chunk to be embedded.

    Returns:
        List[dict]: A list of BM25 sparse embeddings for each text chunk in the format 
        {indices, values}.
    """
    #Initialize BM25 encoder
    encoder = BM25Encoder()
    #Fit the encoder to the corpus
    encoder.fit(corpus)
    #Generate sparse embeddings for each text chunk
    sparse_embeddings = encoder.encode_documents(text_chunk)

    return sparse_embeddings




#Define handler function
def handler_function(event,context):
    # Iterate over the S3 event object and get the key for all uploaded files
    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"]) # Decode the S3 object key to remove any URL-encoded characters
        policy_number = key.split("_")[-1].split(".")[0]  # Get the policy number from the name of file
        download_path = f'/temp/{uuid.uuid4()}.pdf'  # Create a path in the lambda temp directory to save the file

        #check if the file is pdf file
        if key.lower().endswith('.pdf'):
            s3_client.download_file(bucket,key,download_path)
            #call document processing function to process the documents
            documents = document_processing(download_path,policy_number)
            #create corpus of all text inside the policy document
            corpus = [doc.page_content for doc in documents]
            #store embeddings in pinecone
            vectors = []
            for doc in documents:
                dense_embedding = generate_dense_embeddings(doc.page_content)
                sparse_embedding = generate_sparse_embeddings(corpus,doc.page_content)
                vectors.append({
                    "id":str(uuid.uuid4()),
                    "dense_vector":dense_embedding,
                    "sparse_vector":sparse_embedding,
                    "metadata":{"text":doc.page_content,"policy_number":policy_number}
                })
            
            #upsert the records into hybrid index
            index.upsert(
                vectors=vectors,
                namespace="policy-documents"
            )








