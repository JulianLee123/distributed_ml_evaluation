"""
Train a 2-layer Deep Neural Network using PyTorch for multi-class classification.
Saves model as TorchScript for generic loading.

Example usage:
    python train_classification_model.py ../models/classification1 ../datasets/classification1_train.csv ../datasets/classification1_val.csv --e 20

Parameters:
    opt_path: Path to save the trained model (without extension)
    train_csv: Path to training dataset CSV file
    val_csv: Path to validation dataset CSV file
    --lr: Learning rate (default: 0.001)
    --e: Number of epochs (default: 100)
    --hs: Hidden layer size (default: 64)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse

class CustomDataset(Dataset):
    """Custom PyTorch Dataset for CSV data."""
    
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        feature_cols = [col for col in self.data.columns if col.startswith('feature_')]
        self.features = self.data[feature_cols].values.astype(np.float32)
        self.labels = self.data['class'].values.astype(np.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

class TwoLayerDNN(nn.Module):
    """2-layer Deep Neural Network for classification."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, num_classes: int = 10, dropout_rate: float = 0.3):
        super(TwoLayerDNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """Train the neural network model."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
    
    print(f"Starting training for {num_epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        scheduler.step()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            print("-" * 60)

def save_torchscript_model(model, opt_path, input_size):
    """Save model as TorchScript with fallbacks."""
    
    model.eval()
    example_input = torch.randn(1, input_size, dtype=torch.float32)
    
    try:
        print("Saving model with TorchScript (scripting)...")
        scripted_model = torch.jit.script(model)
        scripted_model.save(opt_path + '.pt')
        
        # Verify
        with torch.no_grad():
            original_output = model(example_input)
            scripted_output = scripted_model(example_input)
            
            if torch.allclose(original_output, scripted_output, atol=1e-5):
                print(f"✓ TorchScript model saved to {opt_path}.pt")
                return
            else:
                print("⚠ Scripted model verification failed, trying tracing...")
                
    except Exception as e:
        print(f"TorchScript scripting failed: {e}")
        print("Trying tracing method...")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train a 2-layer DNN for classification')
    parser.add_argument('opt_path', help='Path to output file (without extension)')
    parser.add_argument('train_csv', help='Path to training dataset CSV file')
    parser.add_argument('val_csv', help='Path to validation dataset CSV file')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--e', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--hs', type=int, default=64, help='Hidden layer size (default: 64)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Loading dataset...")
    train_dataset = CustomDataset(args.train_csv)
    val_dataset = CustomDataset(args.val_csv)
    
    # Determine model architecture from data
    input_size = train_dataset.features.shape[1]
    num_classes = len(np.unique(train_dataset.labels))
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TwoLayerDNN(input_size=input_size, hidden_size=args.hs, num_classes=num_classes)
    
    # Print model info
    print(f"\nModel: {input_size} → {args.hs} → {num_classes}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs=args.e, learning_rate=args.lr)
    
    # Save as TorchScript
    save_torchscript_model(model, args.opt_path, input_size)

if __name__ == "__main__":
    main()