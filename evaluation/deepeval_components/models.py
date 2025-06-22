from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models import DeepEvalBaseEmbeddingModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from typing import List
import os


## This file contains the implementation of custom LLM and embedding models for DeepEval using Groq and HuggingFace.
class ChatGroqLLM(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model
    
    def load_model(self):
        return self.model
    
    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content
    
    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content
    
    def get_model_name(self):
        return "ChatGroq LLM"


class HuggingFaceModel(DeepEvalBaseEmbeddingModel):
    def __init__(self, model):
        self.model = model
    
    def load_model(self):
        return self.model
    
    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return embedding_model.embed_query(text)

    def embed_texts(self, text: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return embedding_model.embed_documents(text)

    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_query(text)
    
    async def a_embed_texts(self, text: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_documents(text)

    def get_model_name(self):
        return "HuggingFace Embedding Model"



