"""
Evaluate a model using PyTorch.

This script loads a pre-trained model and evaluates it on a test dataset.
It saves only the raw model predictions as tensors.

Example usage:
    python evaluate_classification_model.py ../outputs/classification1_mini_preds.npy ../datasets/classification1_mini_test.csv ../models/classification1.pt

Parameters:
    output_path: Path to save evaluation results and output tensors
    test_csv: Path to test dataset CSV file
    model_path: Path to trained model file (.pt) - complete model expected
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import sys
import os
import argparse

class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading and preprocessing CSV data.
    
    This dataset class handles:
    - Loading data from CSV files
    - Separating features and labels
    - Converting data to appropriate PyTorch tensors
    """
    
    def __init__(self, csv_file, has_labels=True):
        """
        Initialize the dataset.
        
        Args:
            csv_file (string): Path to the csv file with dataset.
            has_labels (bool): Whether the dataset contains labels (default: True)
        """
        self.data = pd.read_csv(csv_file)
        self.has_labels = has_labels
        
        # Extract feature columns (all columns starting with 'feature_')
        feature_cols = [col for col in self.data.columns if col.startswith('feature_')]
        self.features = self.data[feature_cols].values.astype(np.float32)
        
        if self.has_labels:
            self.labels = self.data['class'].values.astype(np.int64)
        else:
            self.labels = None
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return a single sample (features and label) at the given index."""
        if self.has_labels:
            return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])
        else:
            return torch.tensor(self.features[idx]), torch.tensor(-1)  # Dummy label

def evaluate_model(model, test_loader):
    """
    Evaluate the neural network model on test data.
    
    This function implements the evaluation process with:
    - Model inference in evaluation mode
    - Raw prediction collection
    
    Args:
        model (nn.Module): The trained neural network model
        test_loader (DataLoader): DataLoader for test data
        
    Returns:
        torch.Tensor: Raw model outputs (logits)
    """
    
    model.eval()  # Set model to evaluation mode
    
    # Initialize list to store results
    all_outputs = []
    
    print("Evaluating model...")
    print("-" * 60)
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            # Forward pass
            outputs = model(features)
            
            # Store raw outputs
            all_outputs.append(outputs.cpu())
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * test_loader.batch_size} samples...")
    
    # Concatenate all results
    all_outputs = torch.cat(all_outputs, dim=0)
    
    print(f"Evaluation complete. Generated predictions for {all_outputs.shape[0]} samples.")
    print("-" * 60)
    
    return all_outputs.numpy()

def main():
    """
    Main function to handle argument parsing, model loading, evaluation, and result saving.
    """
    parser = argparse.ArgumentParser(description='Evaluate a trained model for classification')
    parser.add_argument('output_path', help='Path to output folder for results and tensors')
    parser.add_argument('test_csv', help='Path to test dataset CSV file')
    parser.add_argument('model_path', help='Path to trained model file (.pth) - complete model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation (default: 32)')
    
    args = parser.parse_args()
        
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Loading test dataset...")
    
    # Check if dataset has labels
    test_data = pd.read_csv(args.test_csv)
    has_labels = 'class' in test_data.columns
    test_dataset = CustomDataset(args.test_csv, has_labels=has_labels)
    
    # Determine input size from data
    input_size = test_dataset.features.shape[1]
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load complete trained model
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found!")
        sys.exit(1)
    
    print(f"Loading complete model from {args.model_path}...")
    
    try:
        # Load the complete model
        model = torch.jit.load(args.model_path, map_location='cpu')
        model.eval()
        
        print("Successfully loaded complete model")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model file contains a complete model, not just state_dict")
        sys.exit(1)
    
    # Evaluate model and get raw predictions
    raw_predictions = evaluate_model(model, test_loader)
    
    # Save raw predictions as the output tensor
    torch.save(raw_predictions, args.output_path)
    print(f"Saved raw model predictions to {args.output_path}")
    
    print(f"\nEvaluation complete! Raw predictions saved to {args.output_path}")

if __name__ == "__main__":
    main()