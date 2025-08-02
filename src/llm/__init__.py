import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

#load openai api key in environment
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
#initialize gpt-4o llm from openai
llm = ChatOpenAI(model="gpt-4o")

#initialize openai embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

