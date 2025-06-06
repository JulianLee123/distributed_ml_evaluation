"""
Train a 2-layer Deep Neural Network using PyTorch for multi-class classification.
Designed for datasets with 10 classes and 10 features.

python train_classification_model.py ../models/classification1 ../datasets/classification1_train.csv ../datasets/classification1_tal.csv --e 20
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

class CustomDataset(Dataset):
    """Custom dataset class for loading CSV data."""
    
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with dataset.
        """
        self.data = pd.read_csv(csv_file)
        
        # Separate features and labels
        feature_cols = [col for col in self.data.columns if col.startswith('feature_')]
        self.features = self.data[feature_cols].values.astype(np.float32)
        self.labels = self.data['class'].values.astype(np.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

class TwoLayerDNN(nn.Module):
    """2-layer Deep Neural Network for classification."""
    
    def __init__(self, input_size=10, hidden_size=64, num_classes=10, dropout_rate=0.3):
        super(TwoLayerDNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """Train the neural network."""
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"Starting training for {num_epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
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
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Store history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            print("-" * 60)
    print("")
    return train_losses, val_losses, train_accuracies, val_accuracies

def main():
    parser = argparse.ArgumentParser(description='Train a 2-layer DNN for classification')
    parser.add_argument('opt_path', help='Path to output file, including output file name without exension')
    parser.add_argument('train_csv', help='Path to training dataset CSV file')
    parser.add_argument('val_csv', help='Path to validation dataset CSV file')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--e', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--hs', type=int, default=64, help='Hidden layer size (default: 64)')
    
    args = parser.parse_args()
    
    opt_path = args.opt_path
    train_csv_file = args.train_csv
    val_csv_file = args.val_csv
    learning_rate = args.lr
    num_epochs = args.e
    hidden_size = args.hs
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Loading dataset...")
    train_dataset = CustomDataset(train_csv_file)
    val_dataset = CustomDataset(val_csv_file)
    
    # Determine input size and number of classes from data
    input_size = train_dataset.features.shape[1]
    num_classes = len(np.unique(train_dataset.labels))
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = TwoLayerDNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    
    print(f"\nModel Architecture:")
    print(f"Input Layer: {input_size} features")
    print(f"Hidden Layer: {hidden_size} neurons")
    print(f"Output Layer: {num_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    # Train model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate
    )
    
    # Save model
    torch.save(model.state_dict(), opt_path + '.pth')
    print(f"Saved model to {opt_path}.pth")

if __name__ == "__main__":
    main()