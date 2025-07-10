from langchain_core.prompts import ChatPromptTemplate
from src.llm import llm


def generate_response_chain():
    """
    Build generation chain to generate the llm response
    """
    template = """"
        You are a helpful assistant that answers questions based on the following context
        Context: {context}
        Question: {question}
        Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    generation_chain = prompt | llm

    return generation_chain

def generate_input_prompt():
    """
    Generate prompt for input node which respond directly or make a tool call
    """
    # prompt for the input node
    input_system_message = """You are an P&C insurance assistant for question-answering tasks. \n
        You will be provided with a user question and a policy number.\n
        If the question is related to the insurance policy, Make a tool call to the 
        retriever tool to retrieve relevant documents.\n If the question is general 
        and not related to any insurance policy documents, you will respond directly to 
        the user irrespective of the policy number provided."""
    
    input_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", input_system_message),
            ("human", "User question: \n\n {question} \n\n Policy Number: {policy_number}"),
        ]
    )

    return input_prompt

def generate_rewrite_chain():
    """
    Generate the chain which rewrite the original query
    """
    # prompt for the input node
    rewrite_system_message = """You are an AI assistant tasked with reformulating user queries to improve 
    retrieval in a RAG system. Given the original query, rewrite it to be more specific, 
    detailed, and likely to retrieve relevant information.Look at the input and try to 
    reason about the underlying semantic intent / meaning.
    """
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rewrite_system_message),
            ("human", "User question: \n\n {question}"),
        ]
    )
    rewrite_chain = rewrite_prompt | llm

    return rewrite_chain
