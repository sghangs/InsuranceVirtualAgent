# Frontend Dockerfile for Streamlit app (app.py) with shared src/ import support

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y build-essential python3-dev libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy frontend requirements for better caching
COPY frontend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend app code
COPY frontend/ .

# Copy shared src directory from project root
COPY src/ ./src/

# Set PYTHONPATH so Streamlit app can import from src
ENV PYTHONPATH=/app:/app/src

EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]