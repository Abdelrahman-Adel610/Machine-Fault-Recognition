# Use official lightweight Python image
FROM python:3.10-slim

# Install system dependencies required for audio processing (librosa/soundfile)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /code

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /code/models/exports /code/data/raw

# Copy the rest of the codebase
COPY . .

# Expose the port Gradio runs on
EXPOSE 7860

# Command to run the app
CMD["python", "infer.py"]
