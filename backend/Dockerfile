FROM python:3.10-slim

WORKDIR /app

# Copy main.py and requirements.txt from backend directory
COPY main.py .
COPY requirements.txt .

# Copy src directory from parent directory
COPY ../src ./src
COPY ../scripts ./scripts

# Add environment path for imports
ENV PYTHONPATH=/app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
