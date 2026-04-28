import os
import torch
import gradio as gr
import numpy as np
import collections
import warnings

# Resolve repo root from this file's location so all paths work in Docker/HF
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from src.utils.config import load_config
from src.utils.labels import LABEL_NAMES
from src.preprocessing.preprocess import preprocess_audio
from src.feature_extraction.extract import extract_mel_spectrograms
from src.models.resnet import get_resnet18_model

warnings.filterwarnings('ignore')

# 1. Setup Device and Config
CONFIG = load_config(os.path.join(REPO_ROOT, "config", "default.yaml"))
device = torch.device("cpu") # Inference is usually fine on CPU for single files

# 2. Load the trained model
MODEL_PATH = os.path.join(REPO_ROOT, "models", "exports", "best_resnet18.pth")
model = get_resnet18_model(num_classes=6)

# Load weights (map_location='cpu' ensures it works even if trained on Kaggle GPU)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")
else:
    print(f"⚠️ Warning: Model not found at {MODEL_PATH}. App will crash on prediction.")

def predict_audio(audio_filepath):
    """End-to-end inference pipeline for a single audio file."""
    if not audio_filepath:
        return "Please upload an audio file."

    try:
        # Step 1: Preprocess (Wav -> Chunks)
        chunks = preprocess_audio(
            file_path=audio_filepath,
            target_sr=CONFIG['audio']['target_sr'],
            chunk_duration=CONFIG['audio']['chunk_duration'],
            step_duration=CONFIG['audio']['step_duration'],
            trim_top_db=CONFIG['audio']['trim_top_db']
        )

        if len(chunks) == 0:
            return "Audio too short or silent."

        # Step 2: Feature Extraction (Chunks -> Mel Spectrograms)
        features = extract_mel_spectrograms(
            chunks=chunks,
            sr=CONFIG['audio']['target_sr'],
            n_fft=CONFIG['features']['n_fft'],
            hop_length=CONFIG['features']['hop_length'],
            n_mels=CONFIG['features']['n_mels']
        ) # Shape: (Num_Chunks, 128, 94)

        # Step 3: Prepare Tensor for ResNet
        tensor = torch.from_numpy(features).float()
        tensor = tensor.unsqueeze(1).repeat(1, 3, 1, 1) # Shape: (N, 3, 128, 94)
        tensor = tensor.to(device)

        # Step 4: Model Prediction
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted_chunks = torch.max(outputs, 1)

        # Step 5: Majority Voting
        votes = predicted_chunks.cpu().tolist()
        most_common_class = collections.Counter(votes).most_common(1)[0][0]
        
        # Format output
        result_label = LABEL_NAMES[most_common_class]
        
        return f"Prediction: {result_label}\n(Based on {len(chunks)} audio chunks)"
        
    except Exception as e:
        return f"Error processing audio: {str(e)}"

# 3. Build the Gradio UI
interface = gr.Interface(
    fn=predict_audio,
    inputs=gr.Audio(type="filepath", label="Upload Machine Audio (.wav)"),
    outputs=gr.Textbox(label="Diagnosis"),
    title="⚙️ Machine Fault Recognition AI",
    description="Upload an audio recording of a machine to detect if it is operating Normally or Abnormally.",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # On Hugging Face Spaces the SPACE_ID env var is always set.
    # Locally we bind to 127.0.0.1 so the browser opens automatically.
    in_hf = os.environ.get("SPACE_ID") is not None
    interface.launch(
        server_name="0.0.0.0" if in_hf else "127.0.0.1",
        server_port=7860,
        share=False,
    )