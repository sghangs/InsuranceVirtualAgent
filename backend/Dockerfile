# Backend Dockerfile for FastAPI app (main.py) with shared src/ import support

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y build-essential python3-dev libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend app code
COPY backend/ .

# Copy shared src directory from project root
COPY src/ ./src/

# Set PYTHONPATH so FastAPI app can import from src
ENV PYTHONPATH=/app:/app/src

EXPOSE 8080

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]