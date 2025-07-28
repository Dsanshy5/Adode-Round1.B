# Use a standard Python 3.9 image for AMD64 architecture
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your pre-downloaded AI models into the container
COPY ./models /app/models

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your Python source code into the container
COPY . .

# This command will run your analysis automatically when the container starts
CMD ["python", "main.py", "--input-dir", "/app/input", "--output-dir", "/app/output"]
