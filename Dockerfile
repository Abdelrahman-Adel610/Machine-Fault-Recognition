# ============================================================
# Machine Fault Recognition – Hugging Face Spaces Deployment
# ============================================================

# Use official lightweight Python image
FROM python:3.10-slim

# Install system dependencies required for audio processing
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# Set the working directory (HF Spaces convention)
WORKDIR /code

# ── Install Python dependencies ──────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install CPU-only PyTorch first (avoids pulling the ~2 GB CUDA wheel)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────
COPY . .

# ── Hugging Face Spaces permission fix ───────────────────────
# HF runs containers as a non-root user (uid=1000); pre-create
# writable directories so the app can write cache/temp files.
RUN mkdir -p /code/models/exports /code/data/raw \
    && chmod -R 777 /code/models /code/data

# ── Runtime configuration ─────────────────────────────────────
# HF Spaces routes external traffic to port 7860
EXPOSE 7860

# Launch the Gradio app (app/app.py), NOT the batch infer.py
CMD ["python", "app/app.py"]
