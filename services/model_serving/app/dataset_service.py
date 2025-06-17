"""
Dataset service for loading and preprocessing datasets.
"""

import pandas as pd
import numpy as np
import torch
from fastapi import HTTPException
from typing import Dict, Any, List
from shared.src.storage.storage_service import StorageService
import os

class DatasetService:
    def __init__(self, storage_service: StorageService):
        """Initialize dataset service with storage service."""
        self.storage_service = storage_service

    def load_dataset(self, dataset_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Load a dataset from storage."""
        try:
            # Fetch dataset from storage
            dataset_query = {
                "dataset_name": dataset_metadata["dataset_name"],
                "version": dataset_metadata["version"]
            }
            dataset_data = self.storage_service.fetch("dataset", dataset_query, metadata_only=False)
            
            if not dataset_data or "download_path" not in dataset_data:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            # Load dataset
            dataset = pd.read_csv(dataset_data["download_path"])
            
            # Clean up temporary file
            os.unlink(dataset_data["download_path"])
            
            return dataset
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess the dataset for model input. """
        data = data.drop(columns=["output"])
        return data.values

    def create_chunks(self, data: np.ndarray) -> List[np.ndarray]:
        """Split data into chunks for batch processing."""
        return [data[i:i+int(os.getenv("CHUNK_SIZE"))] for i in range(0, len(data), int(os.getenv("CHUNK_SIZE")))] 