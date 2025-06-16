"""
Model service for loading and running ML models.
"""

import pickle
import torch
import os
from fastapi import HTTPException
from typing import Dict, Any
from shared.src.storage.storage_service import StorageService

class ModelService:
    def __init__(self, storage_service: StorageService):
        """Initialize model service with storage service."""
        self.storage_service = storage_service

    def load_model(self, model_metadata: Dict[str, Any]) -> torch.nn.Module:
        """Load a model from storage."""
        try:
            # Fetch model from storage
            model_query = {
                "model_name": model_metadata["model_name"],
                "version": model_metadata["version"]
            }
            model_data = self.storage_service.fetch("model", model_query, metadata_only=False)
            
            if not model_data or "download_path" not in model_data:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Load model from local path
            with open(model_data["download_path"], 'rb') as f:
                model = pickle.load(f)
            
            # Clean up temporary file
            os.unlink(model_data["download_path"])
            
            model.eval()  # Set model to evaluation mode
            return model
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    def predict(self, model: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Make predictions using the model."""
        with torch.no_grad():
            return model(data) 