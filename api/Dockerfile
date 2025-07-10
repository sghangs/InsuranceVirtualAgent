FROM python:3.10-slim

WORKDIR /app

# Copy only needed directories
COPY main.py .
COPY ../src ./src

# Add environment path for imports
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
