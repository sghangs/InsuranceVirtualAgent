import sys
import os
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from passlib.context import CryptContext
from sqlalchemy.orm import Session
from session import get_db
from src.pipeline.rag import RagPipeline
from evaluation.run_evaluation import evaluate_test
from scripts.generate_goldens import DatasetGenerator
from src.exception.exception import InsuranceAgentException
from src.loggers.logger import logging
from fastapi.security import OAuth2PasswordRequestForm
from jwt_utils import create_access_token, verify_password, get_password_hash, get_user_by_email, get_user_by_id
from jwt_utils import get_current_user

from schemas import RagInput, MessageResponse, SignupForm, LoginForm , UserOut, Token
from models import User
from contextlib import asynccontextmanager
from uvicorn import run as app_run

# Lifespan context replacing deprecated on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = await RagPipeline.create()
    logging.info("Starting Insurance Virtual Agent API...")
    yield
    logging.info("Shutting down Insurance Virtual Agent API...")

app = FastAPI(
    title="Insurance Virtual Agent API",
    version="1.0.0",
    description="Backend API for Insurance Virtual Agent",
    lifespan=lifespan
)

# Restrict CORS in production
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)



# Health check endpoint
@app.get("/health", response_model=MessageResponse)
async def check_health():
    return {"message": "OK"}

# signup endpoint
@app.post("/signup", response_model=UserOut)
async def signup(form: SignupForm, db=Depends(get_db)):
    if await get_user_by_email(db, form.email):
        raise HTTPException(status_code=400, detail="Email already exists")
    user = User(
        name=form.name,
        email=form.email,
        password_hash=get_password_hash(form.password)
    )
    await db.add(user)
    await db.commit()
    await db.refresh(user)
    return UserOut(user_id=str(user.user_id), name=user.name, email=user.email)

# Get access token 
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    user = await get_user_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": str(user.user_id)})
    return {"access_token": access_token, "token_type": "bearer"}

# Get current user endpoint
@app.get("/me", response_model=UserOut)
def read_users_me(current_user: User = Depends(get_current_user)):
    return UserOut(user_id=str(current_user.user_id), name=current_user.name, email=current_user.email)

# RAG endpoint
@app.post("/rag", response_model=MessageResponse)
async def run_rag(rag_input: RagInput,current_user: User = Depends(get_current_user)):
    try:
        pipeline= app.state.pipeline
        user_id = str(current_user.user_id)
        response, context = await pipeline.execute_rag(
            user_input=rag_input.query,
            policy_number=rag_input.policy_number,
            session_id=rag_input.session_id,
            user_id=user_id
        )
        return {"message": response}
    except InsuranceAgentException as e:
        logging.error(f"InsuranceAgentException: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.exception("Unhandled exception in /rag endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")

# Evaluation endpoint
@app.get("/evaluate", response_model=MessageResponse)
async def run_evaluation():
    try:
        result = await evaluate_test()
        return {"message": result}
    except Exception as e:
        logging.exception("Unhandled exception in /evaluate endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")

# Dataset generation endpoint
@app.get("/generate", response_model=MessageResponse)
async def generate_dataset():
    try:
        data = DatasetGenerator()
        data.save_dataset()
        return {"message": "Synthetic Dataset generated successfully"}
    except Exception as e:
        logging.exception("Unhandled exception in /generate endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "Internal server error"}
    )

# InsuranceAgentException handler
@app.exception_handler(InsuranceAgentException)
async def insurance_agent_exception_handler(request: Request, exc: InsuranceAgentException):
    logging.error(f"InsuranceAgentException: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": str(exc)}
    )

# For local development only. Use gunicorn/uvicorn in production.
if __name__ == "__main__":
    app_run("backend.main:app", host="0.0.0.0", port=8080, reload=False)
