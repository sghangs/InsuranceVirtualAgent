# Use the official Python 3.10 slim image as the base image
FROM python:3.10-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies without using cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from the current directory to the container's /app directory
COPY ../src ./src
COPY app.py .
COPY ../scripts ./scripts

# Set the default command to run the Streamlit app on port 8501 and listen on all interfaces
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]