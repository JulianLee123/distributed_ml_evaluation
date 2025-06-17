"""
Shared schema definitions for ML Evaluation platform.
"""

from pydantic import BaseModel, Field, ConfigDict, field_serializer
from typing import Dict, Any, List, Optional
from datetime import datetime

class ModelMetadata(BaseModel):
    """Model metadata schema."""
    model_config = ConfigDict()
    
    model_name: str
    version: str
    output_type: str
    size: int
    created_by: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer('timestamp')
    def serialize_timestamp(self, timestamp: datetime) -> str:
        return timestamp.isoformat()

class DatasetMetadata(BaseModel):
    """Dataset metadata schema."""
    model_config = ConfigDict()
    
    dataset_name: str
    version: str
    num_entries: int
    size: int
    created_by: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer('timestamp')
    def serialize_timestamp(self, timestamp: datetime) -> str:
        return timestamp.isoformat()

class PredictionMetadata(BaseModel):
    """Prediction metadata schema."""
    model_config = ConfigDict()
    
    model_name: str
    model_version: str
    dataset_name: str
    dataset_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    num_predictions: int
    created_by: str

    @field_serializer('timestamp')
    def serialize_timestamp(self, timestamp: datetime) -> str:
        return timestamp.isoformat()

class EvaluationMetadata(BaseModel):
    """Evaluation metadata schema."""
    model_config = ConfigDict()
    
    model_name: str
    model_version: str
    dataset_name: str
    dataset_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    evaluations: List[Dict[str, Any]]
    created_by: str

    @field_serializer('timestamp')
    def serialize_timestamp(self, timestamp: datetime) -> str:
        return timestamp.isoformat()