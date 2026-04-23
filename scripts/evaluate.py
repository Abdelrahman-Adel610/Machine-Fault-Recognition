import os
import yaml
import torch
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from src.models.cnn import CustomAudioCNN

def main():
    # Load config
    with open("config/kaggle.yaml", "r") as f:
        config = yaml.safe_load(f)

    DATA_DIR = config['paths']['input_dir']
    OUTPUT_DIR = config['paths']['output_dir']
    TEST_CSV = os.path.join(OUTPUT_DIR, "test_metadata.csv")
    MODEL_PATH = "models/machine_fault_custom_cnn.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data (Saved by train.py)
    test_df = pd.read_csv(TEST_CSV)
    
    # Load Model
    model = CustomAudioCNN(num_classes=6).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    correct_test_predictions = 0
    total_test_files = len(test_df)

    print("Starting Final Test Set Evaluation using Majority Voting...\n")

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=total_test_files, desc="Testing"):
            file_name = os.path.basename(row['file_path']) 
            machine_folder = row['machine']
            full_path = os.path.join(DATA_DIR, machine_folder, file_name)
            true_label = int(row['label'])
            
            chunks = np.load(full_path)['features']
            tensor = torch.from_numpy(chunks).float()
            tensor = tensor.unsqueeze(1).repeat(1, 3, 1, 1).to(device)
            
            outputs = model(tensor)
            _, predicted_chunks = torch.max(outputs, 1)
            
            predictions_list = predicted_chunks.cpu().tolist()
            most_common_prediction = collections.Counter(predictions_list).most_common(1)[0][0]
            
            if most_common_prediction == true_label:
                correct_test_predictions += 1

    final_accuracy = 100 * correct_test_predictions / total_test_files
    print(f"\n🎉 FINAL MODULAR PIPELINE TEST ACCURACY: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()