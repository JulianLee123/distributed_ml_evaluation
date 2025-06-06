#!/usr/bin/env python3
"""
Creates a dataset with num_classes clusters of points in num_input_dim dimensions.
Spread controls the standard deviation of the normal distribution around cluster centers,
determining how tightly clustered the points are (smaller spread = tighter clusters).

Usage: create_classification_dataset.py <opt_path> <num_entries_per_class> <num_classes> <num_input_dim> <spread> <train_split> <val_split> <seed>
    opt_path: Output directory path, including the name of the dataset (without extension)
    num_entries_per_class: Number of data points per cluster/class
    num_classes: Number of clusters/classes
    num_input_dim: Number of features
    spread: Standard deviation around cluster centers (controls tightness
    train_split: Proportion of data to use for training (0.0 to 1.0)
    val_split: Proportion of data to use for validation (0.0 to 1.0)
    seed: seed number for reproducibility

python create_classification_dataset.py ../datasets/classification1 10000 10 10 0.1 0.2 0.05 1

Note: purposely augmenting size of test data to test larget dataset size on the evaluation pipeline
"""

import sys
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import argparse

def generate_clustered_dataset(num_entries_per_class, num_classes, num_input_dim, spread, random_seed):
    np.random.seed(random_seed)

    cluster_centers = np.random.uniform(-1, 1, (num_classes, num_input_dim))
    
    features = []
    labels = []
    
    for class_id in range(num_classes):
        center = cluster_centers[class_id]
        
        cluster_points = np.random.normal(
            loc=center, 
            scale=spread, 
            size=(num_entries_per_class, num_input_dim)
        )
        
        features.extend(cluster_points)
        labels.extend([class_id] * num_entries_per_class)
    
    return np.array(features), np.array(labels)

def save_dataset(features, labels, output_path, train_pct, val_pct, random_seed):
    os.makedirs(output_path, exist_ok=True)
    
    num_dims = features.shape[1]
    data_dict = {}
    
    for i in range(num_dims):
        data_dict[f'feature_{i}'] = features[:, i]
    
    data_dict['class'] = labels
    
    df = pd.DataFrame(data_dict)
    val_test_size = 1 - train_pct
    train_df, temp_df = train_test_split(df,
        train_size=train_pct,
        random_state=random_seed
    )
    val_relative_size = val_pct / val_test_size    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative_size,
        random_state=random_seed
    )
    for data, split_name in [[train_df, '_train'], [val_df, '_val'], [test_df, '_test']]:
        filepath = os.path.join(output_path, f"{split_name}.csv")
        data.to_csv(filepath, index=False)

def main():
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
    
    opt_path = args.opt_path
    num_entries_per_class = args.num_entries_per_class
    num_classes = args.num_classes
    num_input_dim = args.num_input_dim
    spread = args.spread
    train_pct = args.train_pct
    val_pct = args.val_pct
    seed = args.seed
    
    features, labels = generate_clustered_dataset(num_entries_per_class, num_classes, num_input_dim, spread, seed)
        
    save_dataset(features, labels, opt_path, train_pct, val_pct, seed)

if __name__ == "__main__":
    main()