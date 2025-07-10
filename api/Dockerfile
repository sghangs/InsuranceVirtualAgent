# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app
COPY . .

# Run with Uvicorn CLI for clean production entry
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

