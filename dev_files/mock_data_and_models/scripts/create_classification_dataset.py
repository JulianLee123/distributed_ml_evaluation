#!/usr/bin/env python3
"""
Creates a synthetic classification dataset with Gaussian clusters.

This script generates a dataset where each class is represented by a cluster of points
in n-dimensional space. The points are generated using a normal distribution around
randomly placed cluster centers.

Parameters:
    opt_path: Output directory path, including the name of the dataset (without extension)
    num_entries_per_class: Number of data points per cluster/class
    num_classes: Number of clusters/classes
    num_input_dim: Number of features/dimensions
    spread: Standard deviation around cluster centers (controls tightness)
    train_split: Proportion of data to use for training (0.0 to 1.0)
    val_split: Proportion of data to use for validation (0.0 to 1.0)
    seed: Random seed for reproducibility

Example usage:
    python create_classification_dataset.py ../datasets/classification1 10000 10 10 0.1 0.2 0.05 1

Note: Test data size is intentionally augmented to test larger dataset handling in the evaluation pipeline.
"""

import sys
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import argparse

def generate_clustered_dataset(num_entries_per_class, num_classes, num_input_dim, spread, random_seed):
    """
    Generates synthetic data points arranged in clusters.
    
    Args:
        num_entries_per_class (int): Number of data points to generate per class
        num_classes (int): Number of distinct classes/clusters
        num_input_dim (int): Number of features/dimensions for each data point
        spread (float): Standard deviation for the normal distribution around cluster centers
        random_seed (int): Seed for random number generation
    
    Returns:
        tuple: (features, labels) where features is a numpy array of shape (n_samples, n_features)
               and labels is a numpy array of shape (n_samples,)
    """
    np.random.seed(random_seed)

    # Generate random cluster centers in the range [-1, 1]
    cluster_centers = np.random.uniform(-1, 1, (num_classes, num_input_dim))
    
    features = []
    labels = []
    
    # Generate points for each class/cluster
    for class_id in range(num_classes):
        center = cluster_centers[class_id]
        
        # Generate points using normal distribution around the cluster center
        cluster_points = np.random.normal(
            loc=center, 
            scale=spread, 
            size=(num_entries_per_class, num_input_dim)
        )
        
        features.extend(cluster_points)
        labels.extend([class_id] * num_entries_per_class)
    
    return np.array(features), np.array(labels)

def save_dataset(features, labels, output_path, train_pct, val_pct, random_seed):
    """
    Saves the generated dataset to CSV files, split into train/validation/test sets.
    
    Args:
        features (np.ndarray): Feature matrix
        labels (np.ndarray): Class labels
        output_path (str): Directory to save the dataset files
        train_pct (float): Proportion of data to use for training
        val_pct (float): Proportion of data to use for validation
        random_seed (int): Seed for random splitting
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Create a dictionary for the DataFrame
    num_dims = features.shape[1]
    data_dict = {}
    
    # Add each feature dimension to the dictionary
    for i in range(num_dims):
        data_dict[f'feature_{i}'] = features[:, i]
    
    # Add class labels
    data_dict['output'] = labels
    
    # Convert to DataFrame and split into train/val/test sets
    df = pd.DataFrame(data_dict)
    
    # First split: separate training data
    val_test_size = 1 - train_pct
    train_df, temp_df = train_test_split(df,
        train_size=train_pct,
        random_state=random_seed
    )
    
    # Second split: separate validation and test data
    val_relative_size = val_pct / val_test_size    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative_size,
        random_state=random_seed
    )
    
    # Save each split to a separate CSV file
    for data, split_name in [[train_df, '_train'], [val_df, '_val'], [test_df, '_test']]:
        filepath = os.path.join(output_path, f"{split_name}.csv")
        data.to_csv(filepath, index=False)

def main():
    """
    Main function to parse command line arguments and generate the dataset.
    """
    parser = argparse.ArgumentParser(description='Generate clustered dataset for classification')
    parser.add_argument('opt_path', help='Output directory path')
    parser.add_argument('num_entries_per_class', type=int, help='Number of data points per cluster/class')
    parser.add_argument('num_classes', type=int, help='Number of clusters/classes')
    parser.add_argument('num_input_dim', type=int, help='Number of input dimensions/features')
    parser.add_argument('spread', type=float, help='Standard deviation around cluster centers')
    parser.add_argument('train_pct', type=float, help='Training data percentage (0.0 to 1.0)')
    parser.add_argument('val_pct', type=float, help='Validation data percentage (0.0 to 1.0)')
    parser.add_argument('seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Generate the dataset
    features, labels = generate_clustered_dataset(
        args.num_entries_per_class,
        args.num_classes,
        args.num_input_dim,
        args.spread,
        args.seed
    )
    
    # Save the dataset with the specified splits
    save_dataset(features, labels, args.opt_path, args.train_pct, args.val_pct, args.seed)

if __name__ == "__main__":
    main()