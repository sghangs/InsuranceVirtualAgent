import boto3
import os
import uuid
from typing import List
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.llms.bedrock import Bedrock
from urllib.parse import unquote_plus
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from pinecone_text.sparse import BM25Encoder


#Environment variables
os.environ["PINECONE_API_KEY"]=os.getenv("PINECONE_API_KEY")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


def initialization():
    """ 
    Initialize the aws services and pinecone vector db
    Args:
        None
    Returns:
        bedrock: bedrock client accessed through boto3
        pc: pinecone client
        index: pinecone index
        s3_client: s3 client access  through boto3
    """
    try:
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
            print("Pinecone index created successfully")
        
        #Get the index host
        index_response = pc.describe_index(name=index_name)
        dns_host = index_response["host"]

        index = pc.Index(host=dns_host)

        #S3 client initialization
        s3_client = boto3.client("s3")
        print("Initialization completed successfully")
        return bedrock,pc,index,s3_client
    except Exception as e:
        print(f"Error occured in Initialization : {e}")

def document_processing(file_path,policy_number):
    """ 
    Create the documents from the pdf file with added policy number
    Args:
        file_path (Any): Pdf file path
        policy_number (str): policy number
    
    Returns:
        doc (List[Document]): List of Documents
    
    """
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

def generate_dense_embeddings(text,bedrock,model_id="amazon.titan-embed-text-v2:0"):
    """ 
    Generate dense vectors of the document text using bedrock 

    Args:
        text (str): text which need to be transformed into dense vectors
        bedrock (aws object): bedrock client for model access
        model_id (str): name of bedrock model
    Returns:
        embedding (List[int]): Dense vector for text data
    """
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
    #call initialization function to create aws client and pinecone index
    bedrock,pc,index,llm,s3_client = initialization()
    try:
        # Iterate over the S3 event object and get the key for all uploaded files
        for record in event["Records"]:
            bucket = record["s3"]["bucket"]["name"]
            key = unquote_plus(record["s3"]["object"]["key"],encoding='utf-8') # Decode the S3 object key to remove any URL-encoded characters
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
                    dense_embedding = generate_dense_embeddings(doc.page_content,bedrock)
                    sparse_embedding = generate_sparse_embeddings(corpus,doc.page_content)
                    vectors.append({
                        "id":str(uuid.uuid4()),
                        "dense_vector":dense_embedding,
                        "sparse_vector":sparse_embedding,
                        "metadata":{"text":doc.page_content,"policy_number":policy_number}
                    })
                
                #upsert the records into hybrid index
                upsert_response = index.upsert(
                    vectors=vectors,
                    namespace="policy-documents"
                )
                if upsert_response["status"] == "success":
                    print(f"Total number of vectors are loaded in database : {upsert_response.upserted_count}")
                else:
                    print(f"Error occured in upserting {upsert_response.errors}")
            else:
                print(f"File uploaded is not pdf file : {key}")
    except Exception as e:
        print(f"Error occured : {e}")








