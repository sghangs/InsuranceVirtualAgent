import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

#load openai api key in environment
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
#initialize gpt-4o llm from openai
llm = ChatOpenAI(model="gpt-4o")