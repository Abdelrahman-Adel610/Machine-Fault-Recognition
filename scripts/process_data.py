import os
import numpy as np
from src.preprocessing.preprocess import preprocess_audio

for filename in os.listdir("data/raw/"):
    if filename.endswith(".wav"):
        filepath = os.path.join("data/raw/", filename)
        chunks = preprocess_audio(filepath)
        
        save_path = os.path.join("data/processed/", filename.replace(".wav", ".npy"))
        np.save(save_path, chunks)