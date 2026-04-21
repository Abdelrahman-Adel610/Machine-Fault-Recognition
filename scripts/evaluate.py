import os
import argparse
import torch
import pandas as pd
import numpy as np
import collections
from tqdm import tqdm

from src.utils.config import load_config
from src.models.resnet import get_resnet18_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate Audio Model")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config')
    parser.add_argument('--model_path', type=str, default='models/checkpoints/best_resnet18.pth', help='Path to weights')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Test Metadata
    data_dir = config['paths']['output_dir']
    test_df = pd.read_csv(os.path.join(data_dir, "test_metadata.csv"))
    
    # 2. Load Model
    model = get_resnet18_model(num_classes=6)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    correct_files = 0
    total_files = len(test_df)

    print(f"Evaluating {total_files} files using Majority Voting...")

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=total_files):
            # Load .npz file (all chunks)
            # Re-pathing for local/cloud compatibility
            file_name = os.path.basename(row['file_path'])
            machine_folder = row['machine']
            full_path = os.path.join(data_dir, machine_folder, file_name)
            
            true_label = int(row['label'])
            
            # Load chunks
            chunks = np.load(full_path)['features'] # (N, 128, 94)
            
            # Prepare tensor for ResNet (N, 3, 128, 94)
            tensor = torch.from_numpy(chunks).float()
            tensor = tensor.unsqueeze(1).repeat(1, 3, 1, 1)
            tensor = tensor.to(device)
            
            # Predict
            outputs = model(tensor)
            _, predicted_chunks = torch.max(outputs, 1)
            
            # Majority Vote
            votes = predicted_chunks.cpu().tolist()
            most_common = collections.Counter(votes).most_common(1)[0][0]
            
            if most_common == true_label:
                correct_files += 1

    accuracy = 100 * correct_files / total_files
    print(f"\n🎉 FINAL TEST ACCURACY (File-level): {accuracy:.2f}%")

if __name__ == "__main__":
    main()