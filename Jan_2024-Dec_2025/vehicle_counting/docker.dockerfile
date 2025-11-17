# Dockerfile for YOLO detection project
# Use an official Python runtime as a base image
FROM python:3.9-slim

# Install system dependencies required for video processing
RUN apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY . .

# Run the main application script
CMD ["python", "run.py"]
