import boto3
import os
import uuid
from typing import List
import json
from urllib.parse import unquote_plus
from pinecone import Pinecone
from botocore.exceptions import ClientError
from llama_cloud_services import LlamaParse
from langchain_core.documents import Document


def get_secret():
    """ 
    Get pinecone and llama parse api key using aws secretmanager services.
    Args:
        None
    Returns:
        secret:dict
    """
    secret_list = ["pinecone_key","llama_api_key"]
    region_name = "us-east-1"
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    try:
        response = client.batch_get_secret_value(
            SecretIdList=secret_list
        )
    except ClientError as e:
        raise e

    secrets = {item['Name']: item['SecretString'] for item in response['SecretValues']}
    
    return secrets

def initialization(pinecone_api_key:str):
    """ 
    Initialize the aws services and pinecone vector db
    Args:
        pinecone_api_key:str -> Pinecone api key
    Returns:
        pc: pinecone client
        index: pinecone index
        s3_client: s3 client access  through boto3
    """
    try:
        #pinecone client
        pc = Pinecone(api_key=pinecone_api_key)

        #create hybrid(dense + sparse) index with external embeddings 
        dense_index_name = "insurance-virtual-agent-dense"
        if not pc.has_index(dense_index_name):
            pc.create_index_for_model(
                name=dense_index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model":"llama-text-embed-v2",
                    "field_map":{"text": "chunk_text"}
                }

            )
            print("Pinecone dense index created successfully")
        else:
            print(f"Pinecone index {dense_index_name} has already created")

        sparse_index_name = "insurance-virtual-agent-sparse"
        if not pc.has_index(sparse_index_name):
            pc.create_index_for_model(
                name=sparse_index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model":"pinecone-sparse-english-v0",
                    "field_map":{"text": "chunk_text"}
                }

            )
            print("Pinecone sparse index created successfully")
        else:
            print(f"Pinecone index {sparse_index_name} has already created")
        
        #Get the index host
        dense_index_response = pc.describe_index(name=dense_index_name)
        dense_dns_host = dense_index_response["host"]
        dense_index = pc.Index(host=dense_dns_host)

        sparse_index_response = pc.describe_index(name=sparse_index_name)
        sparse_dns_host = sparse_index_response["host"]
        sparse_index = pc.Index(host=sparse_dns_host)

        #S3 client initialization
        s3_client = boto3.client("s3")
        
        return pc,dense_index,sparse_index,s3_client
    except Exception as e:
        print(f"Error occured in Initialization : {e}")

def document_processing(file_path,policy_number,llama_api_key):
    """ 
    Create the documents from the pdf file with added policy number
    Args:
        file_path (Any): Pdf file path
        policy_number (str): policy number
        llama_api_key (str): Llama API key

    Returns:
        doc (List[Document]): List of Documents
    
    """
    try:
        #load the pdf file
        parser = LlamaParse(
            api_key=llama_api_key,
            result_type="markdown",
            system_prompt_append=(
                """
                This is an Homeowner insurance policy document. 
                1. Each document should clearly state what should be covered and what should 
                    not be covered in the respective categories. 
                2. Categories can be found in the headings of the pages with largest font size of the page.
                3. If there is no heading found on that page, look for previous pages for headings. 
                4. Each page has written "Home Insurance" on the top of the page. Do not consider that for headings.
                5. Real headings are like "Buildings Insurance", "Contents Insurance" etc.
                6. Extract the tables and each documents extracted from tables should have headings to
                    differentiate the columns name.

                """
            ),
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name="openai-gpt4o",
            show_progress=True,
        )
        md_json_objs = parser.get_json_result(file_path)  # extract markdown data for insurance claim document
        print("Markdown data extracted successfully")

        md_json_list = []
        for obj in md_json_objs:
            md_json_list.extend(obj["pages"])

        # Convert markdown data to Document objects
        document_list = [Document(page_content=doc["md"],metadata={"page_number": i+1}) for i, doc in enumerate(md_json_list)]
        print("Document objects created successfully")
        
        #adding policy number information in the document metadata
        for doc in document_list:
            doc.metadata["policy_number"] = policy_number

        return document_list
    except Exception as e:
        print(f"Error occured in document processing : {e}")
    


#Define handler function
def lambda_handler(event,context):
    print("Entering lambda handler function")
    #Get pinecone and llama api key
    secrets = get_secret()
    pinecone_key_data = json.loads(secrets["pinecone_key"])
    llama_key_data = json.loads(secrets["llama_api_key"])
    
    pinecone_api_key = pinecone_key_data["pinecone_key"]
    llama_api_key = llama_key_data["llama_api_key"]
    print("Pinecone and llama key fetched successfully")

    #call initialization function to create aws client and pinecone index
    pc,dense_index,sparse_index,s3_client = initialization(pinecone_api_key)
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
                documents = document_processing(download_path,policy_number,llama_api_key)
                print("Document processing completed successfully")

                #store embeddings in pinecone
                vectors = []
                print("Starting embedding generation")
                for doc in documents:
                    vectors.append({
                        "id":str(uuid.uuid4()),
                        "chunk_text":doc.page_content,
                        "policy_number":policy_number
                    })

                #upsert the records into hybrid index
                dense_upsert_response = dense_index.upsert_records(
                    "policy-documents",
                    vectors
                )
                print("Dense Upsert response is : ", dense_upsert_response.get("status_code"))
                print(f"Total number of dense vectors are loaded in database : {dense_upsert_response.get("upsertedCount")}")
                sparse_upsert_response = sparse_index.upsert_records(
                    "policy-documents",
                    vectors
                )
                print("Upsert response is : ", sparse_upsert_response.get("status_code"))
                print(f"Total number of sparse vectors are loaded in database : {sparse_upsert_response.get("upsertedCount")}")
                
            else:
                print(f"File uploaded is not pdf file : {key}")
    except Exception as e:
        print(f"Error occured : {e}")

