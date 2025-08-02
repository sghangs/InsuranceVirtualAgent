from pydantic import BaseModel, Field
from typing import Annotated

# Schema for Request 
class RagInput(BaseModel):
    session_id: Annotated[str, Field(..., description="Session ID for tracking user session")]
    query: Annotated[str, Field(..., description="Input query for the rag")]
    policy_number: Annotated[str, Field(..., description="Policy number")]
 

# Schema for Response
class MessageResponse(BaseModel):
    message: str

# Schema for signup form
class SignupForm(BaseModel):
    name: str
    email: str
    password: str

# Schema for login form
class LoginForm(BaseModel):
    email: str
    password: str

# Schema for token response
class Token(BaseModel):
    access_token: str
    token_type: str

# Schema for user output
class UserOut(BaseModel):
    user_id: str
    name: str
    email: str