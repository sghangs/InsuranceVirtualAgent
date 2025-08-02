from langchain_core.prompts import ChatPromptTemplate
from src.llm import llm


def generate_response_chain():
    """
    Build generation chain to generate the llm response
    """
    template = """"
        You are an P&C insurance assistant for question-answering tasks. \n
        You will be provided with a user question, context, user profile and long-term context.\n
        Provide the precise and correct answer in detail based on the context.\n
        Do not make up any information or hallucinate.\n
        Use user profile and long-term context to provide more personalized and relevant answers if not empty\n
        Context: {context}
        Question: {question}
        User Profile: {user_profile_info}
        Long-term Context: {long_term_context}
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
        You will be provided with a user question, policy number, user profile and long-term context.\n
        If the question is related to the insurance policy, Make a tool call to the
        retriever tool to retrieve relevant documents.\n If the question is general
        and not related to any insurance policy documents, you will respond directly to
        the user irrespective of the policy number provided and use user profile and 
        long-term context to provide more personalized and relevant answers if not empty."""

    input_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", input_system_message),
            ("human", "User question: \n\n {question} \n\n "
            "Policy Number: {policy_number} \n\n "
            "User Profile: {user_profile_info} \n\n "
            "Long-term Context: {long_term_context}"),
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

def user_profile_prompt(user_profile: dict):
    """
    Generate prompt for user profile information retrieval
    :param user_profile: Dictionary containing user profile information
    """
    user_profile_info = (
        f"User profile:\n"
        f"Name: {user_profile['name']}\n"
        f"Policy Number: {user_profile['policy_number']}\n"
        f"Policy Type: {user_profile['policy_type']}\n"
        f"DOB: {user_profile['dob']}\n"
        f"Phone: {user_profile['phone']}"
    )

    return user_profile_info




def profile_extraction_prompt():
    """
    Build generation chain to generate the llm response
    """
    template = """"
        Extract any explicit user profile facts (name, phone, policy_number, policy_type, dob) from the following message. 
        If a field is not explicitly mentioned, leave it blank. Do not assume any information.\n\n
        Message: {message}
    """
    prompt = ChatPromptTemplate.from_template(template)

    return prompt
