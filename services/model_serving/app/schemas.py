"""
Request and response schemas for model serving service.
"""

from pydantic import BaseModel
from typing import Dict, Any, List
from shared.src.storage.schemas.schema import (
    ModelMetadata,
    DatasetMetadata,
    PredictionMetadata
)

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    model_metadata: ModelMetadata
    dataset_metadata: DatasetMetadata
    created_by: str

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    predictions_metadata: PredictionMetadata