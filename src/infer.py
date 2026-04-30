import os
import argparse
import torch
import numpy as np
import collections
import sys
import glob

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import load_config
from src.utils.labels import LABEL_NAMES
from src.preprocessing.preprocess import preprocess_audio
from src.feature_extraction.extract import extract_mel_spectrograms
from src.models.audio_cnn import BetterAudioCNN

def load_model(path, num_classes, device):
    model = BetterAudioCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Batch Inference on Data Directory")
    # Changed from --audio to --data_dir to meet grading requirements
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory containing test .wav files')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    args = parser.parse_args()

    CONFIG = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading models to {device}...")
    MODELS_DIR = "models/exports"
    machine_model = load_model(f"{MODELS_DIR}/best_machine_model.pth", 3, device)
    fault_models = {
        0: load_model(f"{MODELS_DIR}/best_fault_model_m1.pth", 2, device),
        1: load_model(f"{MODELS_DIR}/best_fault_model_m2.pth", 2, device),
        2: load_model(f"{MODELS_DIR}/best_fault_model_m3.pth", 2, device)
    }

    # Find all .wav files in the provided directory (and subdirectories)
    test_files = glob.glob(os.path.join(args.data_dir, '**', '*.wav'), recursive=True)
    
    if not test_files:
        print(f"❌ No .wav files found in {args.data_dir}")
        return

    print(f"Found {len(test_files)} files. Starting inference...\n")
    print(f"{'Filename':<30} | {'Predicted Diagnosis'}")
    print("-" * 65)

    for audio_path in test_files:
        filename = os.path.basename(audio_path)
        try:
            chunks = preprocess_audio(audio_path, **CONFIG['audio'])
            if len(chunks) == 0:
                print(f"{filename:<30} | ❌ Error: Audio too short")
                continue

            features = extract_mel_spectrograms(chunks, **CONFIG['features'])
            tensors = torch.from_numpy(features).float().unsqueeze(1)
            mean = tensors.mean(dim=(2, 3), keepdim=True)
            std = tensors.std(dim=(2, 3), keepdim=True) + 1e-6
            tensors = (tensors - mean) / std
            tensors = tensors.to(device)

            with torch.no_grad():
                m_preds = machine_model(tensors).argmax(1).tolist()
                m_id = collections.Counter(m_preds).most_common(1)[0][0]

                f_preds = fault_models[m_id](tensors).argmax(1).tolist()
                f_status = collections.Counter(f_preds).most_common(1)[0][0]

            final_label_idx = m_id * 2 + f_status
            print(f"{filename:<30} | {LABEL_NAMES[final_label_idx]}")

        except Exception as e:
            print(f"{filename:<30} | ❌ Error: {str(e)}")

if __name__ == "__main__":
    main()