## Insurance Virtual Agent 🚀
### Overview
The Insurance Virtual Agent is an AI-powered chatbot designed to provide seamless assistance for insurance-related queries. Leveraging retrieval-augmented generation (RAG) and metadata filtering, this agent improves document retrieval accuracy and delivers precise responses to customer inquiries.

### Key Features
✅ Conversational AI – Engages users in natural conversations.
✅ RAG System for Policies – Retrieves accurate policy information using Pinecone and LangChain.
✅ Metadata Filtering – Optimizes search accuracy for insurance documents.
✅ Hybrid Search Support – Combines dense and sparse embeddings for better relevance.
✅ AWS-Powered Deployment – Runs efficiently on AWS Lambda and Bedrock.

### Tech Stack
🔹 Programming Language: Python
🔹 Embedding Models: amazon.titan-embed-text-v2:0 (Amazon), pinecone-sparse-english-v0 (Pinecone)
🔹 Retrieval System: Pinecone, LangChain
🔹 Deployment: AWS Lambda, AWS Bedrock
🔹 Data Processing: UUID-based indexing, Metadata filtering

### Setup Instructions

#### Setup AWS resources
1. Create S3 bucket (rag-source-bucket13).
2. Create policy to enable the lambda to get s3 object(policy documents) and do the 
    processing. policy name -> (s3-trigger-lambda)
    Add below services to this policy
    Bedrock
    CloudWatch Logs
    S3
3. Create execution role that grants a lambda function permission to access aws resources  and services.In this step, create an execution role using the permissions policy that you created in the previous step. role -> (s3-trigger-lambda-role)
Add the below permissions.
    AWSLambdaVPCAccessExecutionRole
    SecretsManagerReadWrite
    AWSLambdaBasicExecutionRole
    s3-trigger-lambda
4.  Create lambda function to have the logic of indexing the policy document in RAG process. This function automatically triggered by S3 to create the embeddings of policy document and store it in the pineconne vectore db.
    lambda function -> (Insurance-Virtual-Assistant)
    logic has been written in lambda_function.py file
5. Create secrets in secretmanagers to store the pinecone api key. -> (pinecone_api_key) 

6. Create layer

You need to package the distributions compatible with Amazon Linux and the architecture of your Lambda function (x86_64 or arm).

The easiest way to do this is create a layer.

    1. Put the list of your dependencies (Pillow, opencv-python-headless, plus whatever else you need) into requirements.txt

    Run these commands:

    pip install \
    --platform manylinux2014_x86_64 \
    --target=python \
    --implementation cp \
    --python-version 3.12 \
    --only-binary=:all: \
    --upgrade \
    -r requirements.txt
    2. zip -r imaging.zip python/
    3. If your Lambda is running on ARM, replace manylinux2014_x86_64 with manylinux2014_aarch64

    Create the layer from the resulting zip file and use it in your lambda.

7. Add triggers : Add trigger to trigger the lambda function whenever policy documents gets
    created in S3 bucket.

    Use below link to follow the steps
    https://docs.aws.amazon.com/lambda/latest/dg/with-s3-example.html




### Usage Guide

### Query Insurance Policies


Contributing
🤝 We welcome contributions! Please submit pull requests or report issues via GitHub Issues.
