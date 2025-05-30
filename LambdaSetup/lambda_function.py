import boto3
import os
import uuid
from typing import List
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Bedrock
from urllib.parse import unquote_plus
from pinecone import Pinecone
from pinecone import ServerlessSpec
from botocore.exceptions import ClientError


def get_secret():
    """ 
    Get pinecone api key using aws secretmanager services.
    Args:
        None
    Returns:
        api_key:str
    """
    secret_name = "pinecone_key"
    region_name = "us-east-1"
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    try:
        response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    secret = json.loads(response["SecretString"])
    return secret["pinecone_key"]

def initialization(pinecone_api_key:str):
    """ 
    Initialize the aws services and pinecone vector db
    Args:
        pinecone_api_key:str -> Pinecone api key
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
        pc = Pinecone(api_key=pinecone_api_key)

        #create hybrid(dense + sparse) index with external embeddings 
        index_name = "insurance-virtual-agent-hybrid"
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                vector_type="dense",
                dimension=256,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws",region="us-east-1"),
                deletion_protection="disabled",
                tags={
                    "environment":"dev"
                }

            )
            print("Pinecone index created successfully")
        else:
            print(f"Pinecone index {index_name} has already created")
        
        #Get the index host
        index_response = pc.describe_index(name=index_name)
        dns_host = index_response["host"]

        index = pc.Index(host=dns_host)

        #S3 client initialization
        s3_client = boto3.client("s3")
        
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
        chunk_size = 500,
        chunk_overlap = 50,
        length_function = len
    )
    docs = splitter.split_documents(docs)
    
    #adding policy number information in the document metadata
    for doc in docs:
        doc.metadata["policy_number"] = policy_number

    return docs

def generate_dense_embeddings(text,bedrock,model_id="amazon.titan-embed-text-v2:0"):
    """ 
    Generate dense vectors of the document text using aws bedrock 

    Args:
        text (str): text which need to be transformed into dense vectors
        bedrock (aws object): bedrock client for model access
        model_id (str): name of bedrock model
    Returns:
        embedding (List[int]): Dense vector for text data
    """
    body = json.dumps({
            "inputText": text,
            "dimensions": 256,
            "normalize": True
        })
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    response_body=json.loads(response.get("body").read())
    response=response_body["embedding"]
    return response

def generate_sparse_embeddings(pc,text_chunk: str):
    """ 
    Generate sparse embeddings using pinecone hosted embedding models 
    Args:
        pc (Pinecone object): Pinecone client object
        text_chunks (str): Text chunk to be embedded.

    Returns:
        EmbeddingsList: A list of sparse embeddings data 

    """
    sparse_embeddings = pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=[text_chunk],
        parameters={"input_type": "passage", "truncate": "END"}

    )
    return sparse_embeddings
    


#Define handler function
def lambda_handler(event,context):
    print("Entering lambda handler function")
    #Get pinecone api key
    pinecone_api_key = get_secret()
    print("Pinecone key fetched successfully")

    #call initialization function to create aws client and pinecone index
    bedrock,pc,index,s3_client = initialization(pinecone_api_key)
    print("Initialization completed successfully")
    try:
        # Iterate over the S3 event object and get the key for all uploaded files
        for record in event["Records"]:
            bucket = record["s3"]["bucket"]["name"]
            key = unquote_plus(record["s3"]["object"]["key"],encoding='utf-8') # Decode the S3 object key to remove any URL-encoded characters
            print("Key/File name is : ",key)
            policy_number = key.split("_")[-1].split(".")[0]  # Get the policy number from the name of file
            print("Policy number is : ",policy_number)
            download_path = f'/tmp/{key}'  # Create a path in the lambda temp directory to save the file

            #check if the file is pdf file
            if key.lower().endswith('.pdf'):
                s3_client.download_file(bucket,key,download_path)
            
                #call document processing function to process the documents
                documents = document_processing(download_path,policy_number)
                print("Document processing completed successfully")

                #store embeddings in pinecone
                vectors = []
                print("Starting embedding generation")
                for doc in documents:
                    sparse_embedding = generate_sparse_embeddings(pc,doc.page_content)
                    sparse_indices = sparse_embedding.data[0]["sparse_indices"]
                    sparse_values = sparse_embedding.data[0]["sparse_values"]

                    dense_embedding = generate_dense_embeddings(doc.page_content,bedrock)
                    
                    vectors.append({
                        "id":str(uuid.uuid4()),
                        "values":dense_embedding,
                        "sparse_values":{'indices': sparse_indices, 'values': sparse_values},
                        "metadata":{"text":doc.page_content,"policy_number":policy_number}
                    })

                #upsert the records into hybrid index
                upsert_response = index.upsert(
                    vectors=vectors,
                    namespace="policy-documents"
                )
                print("Upsert response is : ", upsert_response.status)
                print(f"Total number of vectors are loaded in database : {upsert_response.upserted_count}")
                
            else:
                print(f"File uploaded is not pdf file : {key}")
    except Exception as e:
        print(f"Error occured : {e}")

