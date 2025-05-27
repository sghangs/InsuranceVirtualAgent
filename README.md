### Overview:

    This project is about Insurance Virtual Agent which can help insured to find the details regardings their policy.

    Insurance is a document-heavy industry with numerous terms and conditions, making it challenging for policyholders to find accurate answers to their queries regarding policy details or the claims process. This often leads to higher customer churn due to frustration and misinformation. This article explores how to address this issue using Generative AI by building an end-to-end Retrieval-Augmented Generation (RAG) chatbot for insurance. We call it IVA(Insurance Virtual Agent), which is built over the robust AWS stack.

### Solution:

    1. When a policy is issued, the policy document is stored in an S3 bucket.
    2. An S3 notification triggers a Lambda function upon document upload. This function   tokenizes the document, generates vector embeddings via AWS Bedrock, and store it in pineconne vector db.
    3. When a user queries the chatbot, it retrieves the relevant vector index based on the policy number. The chatbot then uses this index and the user’s query, processed through a Large Language Model (LLM) with AWS Bedrock and LangChain, to generate an accurate response.

    image.png

### Process:

1. Setup AWS resources
    a. create S3 bucket (rag-source-bucket13).
    b. create lambda function to have the logic of indexing the policy document in RAG process. This function automatically triggered by S3 to create the embeddings of policy document and store it in the pineconne vectore db.

