FROM python:3.10-slim
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p /code/models/exports

COPY . .

# In your Hugging Face Space, the model files MUST be in models/exports/
# best_machine_model.pth, best_fault_model_m1.pth, etc.

EXPOSE 7860
CMD ["python", "app/app.py"]