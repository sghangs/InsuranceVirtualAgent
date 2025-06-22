from langchain_core.prompts import ChatPromptTemplate
from src.entity.schema_entity import GradeDocuments,GradeHallucinations,GradeAnswer
from src.llm import llm


def grade_documents():
    """
    Grade each retrieved document for the user question
    """
    documents_system_message = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", documents_system_message),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    structured_llm_retrieval_grader = llm.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_retrieval_grader

    return retrieval_grader


def grade_hallucinations():
    """
    Grade LLM response generation for set of retrieved facts.
    """
    hallucination_system_message = """You are a grader assessing whether an LLM generation 
    is grounded in / supported by a set of retrieved facts. \n Give a binary score 'yes' 
    or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", hallucination_system_message),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)
    hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader

    return hallucination_grader

def grade_answer():
    """
    Grade LLM response generation for set of retrieved facts.
    """
    answer_system_message = """You are a grader assessing whether an answer 
    addresses / resolves a question \n Give a binary score 'yes' or 'no'. Yes' means that 
    the answer resolves the question."""

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", answer_system_message),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)
    answer_grader = answer_prompt | structured_llm_answer_grader

    return answer_grader