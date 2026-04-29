import os
import yaml
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.dataset import AudioChunkDataset
from src.models.cnn import CustomAudioCNN

def sequential_split(group, train_frac):
    split_idx = int(len(group) * train_frac)
    return group.iloc[:split_idx], group.iloc[split_idx:]

def main():
    with open("config/kaggle.yaml", "r") as f:
        config = yaml.safe_load(f)

    DATA_DIR = config['paths']['input_dir']
    OUTPUT_DIR = config['paths']['output_dir']
    CSV_PATH = os.path.join(OUTPUT_DIR, config['paths']['csv_name'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load and Sort Data Chronologically
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(by=['machine', 'label', 'file_path']).reset_index(drop=True)

    # 2. Sequential Splitting
    train_val_list, test_list = [], []
    for _, group in df.groupby(['machine', 'label']):
        tv_group, test_group = sequential_split(group, train_frac=0.85)
        train_val_list.append(tv_group)
        test_list.append(test_group)

    train_val_df = pd.concat(train_val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    # Save test metadata for evaluation script
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_metadata.csv"), index=False)

    train_list, val_list =[], []
    for _, group in train_val_df.groupby(['machine', 'label']):
        t_group, v_group = sequential_split(group, train_frac=0.823)
        train_list.append(t_group)
        val_list.append(v_group)

    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)

    # 3. Create Datasets (Enable Augmentation for Train)
    train_dataset = AudioChunkDataset(train_df, DATA_DIR, is_train=True)
    val_dataset   = AudioChunkDataset(val_df, DATA_DIR, is_train=False)

    # 4. Handle Class Imbalance
    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for _, _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # 5. DataLoaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 6. Initialize Model
    model = CustomAudioCNN(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # ADDED: weight_decay (L2 Regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)

    # 7. Training Loop
    epochs = config['training']['epochs']
    os.makedirs("models", exist_ok=True)
    best_model_path = "models/machine_fault_resnet18.pth"

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