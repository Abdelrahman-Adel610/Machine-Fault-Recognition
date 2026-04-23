import os
import yaml
import torch
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.dataset import AudioChunkDataset
from src.models.cnn import CustomAudioCNN

def main():
    # Load config
    with open("config/kaggle.yaml", "r") as f:
        config = yaml.safe_load(f)

    DATA_DIR = config['paths']['input_dir']
    OUTPUT_DIR = config['paths']['output_dir']
    CSV_PATH = os.path.join(OUTPUT_DIR, config['paths']['csv_name'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data & Split
    df = pd.read_csv(CSV_PATH)
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.176, random_state=42, stratify=train_val_df['label'])

    # IMPORTANT FIX: Save test_df so evaluate.py can find it!
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_metadata.csv"), index=False)

    # 2. Datasets & Class Imbalance handling
    train_dataset = AudioChunkDataset(train_df, DATA_DIR)
    val_dataset   = AudioChunkDataset(val_df, DATA_DIR)

    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for _, _, label in train_dataset.samples]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # 3. DataLoaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 4. Initialize Model
    model = CustomAudioCNN(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # 5. Training Loop
    epochs = config['training']['epochs']
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/machine_fault_custom_cnn.pth"

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        print(f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {100*correct_train/total_train:.2f}% | Val Acc: {100*correct_val/total_val:.2f}%")

    torch.save(model.state_dict(), best_model_path)
    print(f"✅ Model Saved Successfully to {best_model_path}")

if __name__ == "__main__":
    main()