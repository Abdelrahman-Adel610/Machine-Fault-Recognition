import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.utils.config import load_config
from src.data.dataset import AudioChunkDataset
from src.models.resnet import get_resnet18_model

def main():
    parser = argparse.ArgumentParser(description="Train Audio Model")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config')
    args = parser.parse_args()
    config = load_config(args.config)

    # Configs
    data_dir = config['paths']['output_dir']
    csv_path = os.path.join(data_dir, config['paths']['csv_name'])
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # # 1. Load Data
    # print("Loading metadata...")
    # df = pd.read_csv(csv_path)
    
    # # 2. Split Data (File-level split to prevent data leakage)
    # train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
    # train_df, val_df = train_test_split(train_val_df, test_size=0.176, random_state=42, stratify=train_val_df['label'])
    
    # print(f"Train files: {len(train_df)} | Val files: {len(val_df)} | Test files: {len(test_df)}")
    # 1. Load Data
    print("Loading metadata...")
    df = pd.read_csv(csv_path)
    
    # 2. Split Data (File-level split to prevent data leakage)
    if len(df) < 100:
        print("⚠️ Tiny dataset detected. Disabling stratification for local testing.")
        train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
        # Avoid zero-sized validation sets for tiny data
        train_df, val_df = train_test_split(train_val_df, test_size=0.20, random_state=42) 
    else:
        print("🌍 Large dataset detected. Using stratified split.")
        train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
        train_df, val_df = train_test_split(train_val_df, test_size=0.176, random_state=42, stratify=train_val_df['label'])
    
    print(f"Train files: {len(train_df)} | Val files: {len(val_df)} | Test files: {len(test_df)}")

    # 3. Create Datasets
    train_dataset = AudioChunkDataset(train_df, data_dir)
    val_dataset   = AudioChunkDataset(val_df, data_dir)

    # 4. Handle Class Imbalance
    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for _, _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # 5. Create DataLoaders (num_workers=0 for Windows local testing stability)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 6. Initialize Model, Loss, Optimizer
    model = get_resnet18_model(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 7. Training Loop
    best_val_acc = 0.0
    os.makedirs("models/checkpoints", exist_ok=True)
    
    print("\nStarting Training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # --- TRAIN ---
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
            
        train_acc = 100 * correct_train / total_train
        
        # --- VALIDATE ---
        model.eval()
        correct_val, total_val = 0, 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = 100 * correct_val / total_val
        
        print(f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = "models/checkpoints/best_resnet18.pth"
            torch.save(model.state_dict(), save_path)
            print(f"🌟 New best model saved to {save_path}")

    # Save the test split so evaluate.py can use it
    test_csv_path = os.path.join(data_dir, "test_metadata.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"✅ Test split saved to {test_csv_path}")

if __name__ == "__main__":
    main()
