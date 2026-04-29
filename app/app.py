import os
import torch
import gradio as gr
import numpy as np
import collections
import sys
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure modules are importable
sys.path.append(os.getcwd())

from src.utils.config import load_config
from src.utils.labels import LABEL_NAMES
from src.preprocessing.preprocess import preprocess_audio
from src.feature_extraction.extract import extract_mel_spectrograms
from src.models.audio_cnn import BetterAudioCNN

# 1. Setup Environment
CONFIG = load_config("config/default.yaml")
device = torch.device("cpu")
MODELS_DIR = "src/models"

# Define our model registry
MODEL_FILES = {
    "machine": {"path": f"{MODELS_DIR}/best_machine_model.pth", "classes": 3},
    "fault_m1": {"path": f"{MODELS_DIR}/best_fault_model_m1.pth", "classes": 2},
    "fault_m2": {"path": f"{MODELS_DIR}/best_fault_model_m2.pth", "classes": 2},
    "fault_m3": {"path": f"{MODELS_DIR}/best_fault_model_m3.pth", "classes": 2},
}

def check_models_integrity():
    """
    Checks if all 4 model files exist and can be loaded into memory.
    Returns the loaded models if successful, otherwise raises an error.
    """
    loaded_models = {}
    missing_files = []
    
    logger.info("--- Starting Model Integrity Check ---")
    
    for key, info in MODEL_FILES.items():
        path = info["path"]
        if not os.path.exists(path):
            logger.error(f"❌ Missing weight file: {path}")
            missing_files.append(path)
            continue
        
        try:
            # Initialize architecture
            model = BetterAudioCNN(num_classes=info["classes"])
            # Try to load state dict
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            loaded_models[key] = model
            logger.info(f"✅ Successfully loaded: {path}")
        except Exception as e:
            logger.error(f"❌ Failed to load {path}: {str(e)}")
            missing_files.append(path)

    if missing_files:
        error_msg = f"Application cannot start. Missing or corrupt models: {missing_files}"
        logger.critical(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info("--- All models verified. System Ready. ---")
    return loaded_models

# 2. Initial Load Check
try:
    models = check_models_integrity()
    machine_model = models["machine"]
    fault_models_map = {
        0: models["fault_m1"],
        1: models["fault_m2"],
        2: models["fault_m3"]
    }
except Exception as e:
    # This will stop the Docker container from starting if models are missing
    sys.exit(1)

def predict(audio_path):
    if audio_path is None:
        return "Please upload an audio file."
        
    try:
        # Step 1: Preprocess
        chunks = preprocess_audio(
            file_path=audio_path,
            target_sr=CONFIG['audio']['target_sr'],
            chunk_duration=CONFIG['audio']['chunk_duration'],
            step_duration=CONFIG['audio']['step_duration'],
            trim_top_db=CONFIG['audio']['trim_top_db']
        )
        
        # Step 2: Feature Extraction
        features = extract_mel_spectrograms(
            chunks=chunks,
            sr=CONFIG['audio']['target_sr'],
            n_fft=CONFIG['features']['n_fft'],
            hop_length=CONFIG['features']['hop_length'],
            n_mels=CONFIG['features']['n_mels']
        )
        
        # Step 3: Z-Score Normalization (Match Training logic)
        # Shape: (N, 1, 128, 94)
        tensors = torch.from_numpy(features).float().unsqueeze(1)
        mean = tensors.mean(dim=(2, 3), keepdim=True)
        std = tensors.std(dim=(2, 3), keepdim=True) + 1e-6
        tensors = (tensors - mean) / std

        # Step 4: Two-Stage Inference
        with torch.no_grad():
            # Stage 1: Identify Machine
            m_outputs = machine_model(tensors)
            m_preds = m_outputs.argmax(1).tolist()
            m_id = collections.Counter(m_preds).most_common(1)[0][0]
            
            # Stage 2: Identify Fault for that specific machine
            f_outputs = fault_models_map[m_id](tensors)
            f_preds = f_outputs.argmax(1).tolist()
            f_status = collections.Counter(f_preds).most_common(1)[0][0]
        
        # Step 5: Map to Original Labels (0-5)
        final_label_idx = m_id * 2 + f_status
        result_text = LABEL_NAMES[final_label_idx]
        
        return (f"Detected: Machine {m_id + 1}\n"
                f"Status: {result_text}\n"
                f"Analysis based on {len(chunks)} audio segments.")
                
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return f"Error during processing: {str(e)}"

# 3. Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload Machine Recording (.wav)"),
    outputs=gr.Textbox(label="AI Diagnostic Result"),
    title="⚙️ Hierarchical Machine Fault Detector",
    description="This AI uses a two-stage CNN to first identify the machine type and then detect specific anomalies with 93% accuracy.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)