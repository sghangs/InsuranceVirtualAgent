from pydantic import BaseModel, Field
from src.llm import llm
from src.prompts.prompts import profile_extraction_prompt


# Pydantic schema for user profile updates
# This schema defines the fields we want to extract from the message.
class UserProfileUpdate(BaseModel):
    name: str | None = Field(None, description="User's full name")
    phone: str | None = Field(None, description="User's phone number")
    policy_number: str | None = Field(None, description="User's policy number")
    policy_type: str | None = Field(None, description="Type of insurance policy")
    dob: str | None = Field(None, description="User's date of birth in YYYY-MM-DD format")

async def extract_profile_updates(message: str) -> dict:
    """
    Extract user profile updates from a message using an LLM and Pydantic schema.
    Returns a dict of updated fields, e.g., {"phone": "...", "name": "..."}
    """
    try:
        llm_with_structured_output= llm.with_structured_output(UserProfileUpdate)
        profile_chain = profile_extraction_prompt() | llm_with_structured_output 
        profile_update = await profile_chain.ainvoke({"message": message})
      
        return profile_update.dict(exclude_none=True)
    except Exception as e:
        print(f"Fact extraction error: {e}")
        return {}