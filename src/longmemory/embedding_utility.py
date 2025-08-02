import sys
import os
from src.llm import embedding_model
from src.constant import TRIVIAL_MESSAGES
from src.exception.exception import InsuranceAgentException

def get_embedding(text: str) -> list:
    """Generate an embedding for the given text using the OpenAI embedding model.
    Args:
        text (str): The input text to generate an embedding for.
    Returns:
        list: The generated embedding as a list of floats.
    """
    try:
        response = embedding_model.embed_query(text)
        return response
    except Exception as e:
        raise InsuranceAgentException(sys, e)
        return []

# Define trivial messages that do not require embedding
def should_embed(message: str) -> bool:
    return (
        len(message.strip().split()) > 2 and
        message.lower().strip() not in TRIVIAL_MESSAGES
    )