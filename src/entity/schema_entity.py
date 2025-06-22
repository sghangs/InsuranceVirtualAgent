from pydantic import BaseModel, Field


# Schema for retriever tool
class RetrieverInput(BaseModel):
    """ 
    Input schema for the retriever tool.
    """
    query: str = Field(description="The query text to retrieve documents for.")
    policy_number: str = Field(description="The policy number to filter documents by.")

# Schema for llm output to grade the retrieved documents
class GradeDocuments(BaseModel):
    """ 
    Grade the retrieved documents based on their relevance to the query.
    """
    binary_score: str = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )

# Schema for llm output to check hallucination in the answer
class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination present in generation answer.
    """
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# Schema for llm output to grade the answer
class GradeAnswer(BaseModel):
    """
    Binary score to assess answer addresses question.
    """
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )