# Use NVIDIA PyTorch base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /w4-caption

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your files
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the server
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]