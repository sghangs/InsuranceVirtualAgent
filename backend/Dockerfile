FROM python:3.12-slim

WORKDIR /app

# Copy main.py and requirements.txt from backend directory
COPY requirements.txt .

# Copy src directory from parent directory
COPY . .


# Expose FastAPI port
EXPOSE 8080

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
