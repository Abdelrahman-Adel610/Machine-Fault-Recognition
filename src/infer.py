import os
import sys
import time
import re
import warnings
import collections
import torch
import torch.nn as nn
import numpy as np
import librosa

# Suppress warnings for a clean competition output
warnings.filterwarnings('ignore')

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class CustomAudioCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomAudioCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Layer 4 added in new version
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==========================================
# 2. INFERENCE ENGINE
# ==========================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python infer.py <path_to_data_directory>")
        sys.exit(1)
        
    data_dir = sys.argv[1]
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = CustomAudioCNN(num_classes=6).to(device)
    
    # Resolve model path (looks in the same directory as this script)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "machine_fault_resnet18.pth")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"FATAL ERROR: Weight file '{model_path}' not found.")
        sys.exit(1)
        
    model.eval()

    # ==========================================
    # 3. WARM-UP (Optimizing for Running Time Grade)
    # ==========================================
    with torch.no_grad():
        # Initialize CUDA/GPU and JIT compilers
        fake_wave = np.random.randn(16000 * 3).astype(np.float32)
        _ = librosa.effects.trim(fake_wave, top_db=20)
        _S = librosa.feature.melspectrogram(y=fake_wave, sr=16000, n_fft=1024, hop_length=512, n_mels=128)
        _ = librosa.power_to_db(_S, ref=np.max)
        
        fake_input = torch.zeros((1, 3, 128, 94)).to(device)
        _ = model(fake_input)

    # ==========================================
    # 4. PROCESSING LOOP
    # ==========================================
    # Get and Sort files NUMERICALLY (1.wav, 2.wav, 10.wav)
    files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    files.sort(key=lambda f: int(re.sub(r'\D', '', f)))

    all_labels = []
    all_times = []

    with torch.no_grad():
        for file_name in files:
            file_path = os.path.join(data_dir, file_name)
            
            # PDF Rule: Timer starts AFTER reading the file
            y, sr = librosa.load(file_path, sr=16000)
            
            start_time = time.perf_counter()
            
            # A. Preprocessing
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            chunk_len = int(3.0 * 16000)
            step_len = int(1.5 * 16000)
            
            chunks = []
            for start in range(0, len(y_trimmed) - chunk_len + 1, step_len):
                chunks.append(y_trimmed[start:start + chunk_len])
                
            if not chunks: # Fallback for very short files
                if len(y_trimmed) < chunk_len:
                    chunks.append(np.pad(y_trimmed, (0, chunk_len - len(y_trimmed))))
                else:
                    chunks.append(y_trimmed[:chunk_len])
            
            # B. Feature Extraction
            mel_specs = []
            for chunk in chunks:
                S = librosa.feature.melspectrogram(y=chunk, sr=16000, n_fft=1024, hop_length=512, n_mels=128)
                S_db = librosa.power_to_db(S, ref=np.max)
                # Force exact shape (128, 94)
                if S_db.shape[1] < 94:
                    S_db = np.pad(S_db, ((0, 0), (0, 94 - S_db.shape[1])), mode='constant')
                else:
                    S_db = S_db[:, :94]
                mel_specs.append(S_db)
                
            features = np.array(mel_specs)
            
            # C. Model Prediction
            tensor = torch.from_numpy(features).float()
            tensor = tensor.unsqueeze(1).repeat(1, 3, 1, 1).to(device)
            
            outputs = model(tensor)
            _, predicted_chunks = torch.max(outputs, 1)
            
            # Majority Voting for the whole file
            predictions_list = predicted_chunks.cpu().tolist()
            most_common_prediction = collections.Counter(predictions_list).most_common(1)[0][0]
            
            # PDF Rule: Timer ends AFTER generating prediction
            end_time = time.perf_counter()
            
            # Store results
            all_labels.append(str(most_common_prediction))
            all_times.append(f"{end_time - start_time:.3f}")

    # ==========================================
    # 5. FINAL OUTPUT (PDF Compliance)
    # ==========================================
    with open("results.txt", "w") as f:
        f.write("\n".join(all_labels) + "\n")
        
    with open("time.txt", "w") as f:
        f.write("\n".join(all_times) + "\n")

if __name__ == "__main__":
    main()