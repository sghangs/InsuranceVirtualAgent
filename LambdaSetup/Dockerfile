FROM public.ecr.aws/lambda/python:3.12

# Install dependencies in /opt/python (AWS Lambda Layer structure)
RUN pip install --upgrade pip

# Copy requirements and install into /opt/python
COPY requirements.txt .
RUN pip install -r requirements.txt -t /opt/python

# Copy function code (not needed in layer, but for local testing)
COPY lambda_function.py .

# (Optional) Set the default command for local testing
CMD ["lambda_function.lambda_handler"]