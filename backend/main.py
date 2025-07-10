import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Annotated
from src.pipeline.rag import RagPipeline
from evaluation.run_evaluation import evaluate_test
from scripts.generate_goldens import DatasetGenerator
from src.exception.exception import InsuranceAgentException
from src.logging.logger import logging
from contextlib import asynccontextmanager
from uvicorn import run as app_run



# Lifespan context replacing deprecated on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
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

# Request and Response Schemas
class RagInput(BaseModel):
    session_id: Annotated[str, Field(..., description="Session ID for tracking user session")]
    query: Annotated[str, Field(..., description="Input query for the rag")]
    policy_number: Annotated[str, Field(..., description="Policy number")]

class MessageResponse(BaseModel):
    message: str

# Initialize the RAG pipeline object
rag_obj = RagPipeline()

@app.get("/health", response_model=MessageResponse)
async def check_health():
    return {"message": "OK"}

@app.post("/rag", response_model=MessageResponse)
async def run_rag(rag_input: RagInput):
    try:
        response, context = await rag_obj.execute_rag(
            user_input=rag_input.query,
            policy_number=rag_input.policy_number,
            session_id=rag_input.session_id
        )
        return {"message": response}
    except InsuranceAgentException as e:
        logging.error(f"InsuranceAgentException: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.exception("Unhandled exception in /rag endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/evaluate", response_model=MessageResponse)
async def run_evaluation():
    try:
        result = await evaluate_test()
        return {"message": result}
    except Exception as e:
        logging.exception("Unhandled exception in /evaluate endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")

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
