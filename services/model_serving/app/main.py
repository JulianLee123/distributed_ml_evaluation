"""
Main FastAPI application for model serving service.
"""

from fastapi import FastAPI, HTTPException
import ray
import numpy as np
from datetime import datetime
import os
import tempfile
import gzip

from .schemas import (
    PredictionRequest, 
    PredictionResponse,
    ModelMetadata,
    DatasetMetadata,
    PredictionMetadata
)
from .model_service import ModelService
from .dataset_service import DatasetService
from shared.src.storage.storage_service import StorageService

# Initialize Ray
ray.init()

# Initialize FastAPI app
app = FastAPI(title="ML Evaluation Model Serving Service")

# Initialize services
storage_service = StorageService()
model_service = ModelService(storage_service)
dataset_service = DatasetService(storage_service)

@ray.remote
def predict_batch(model, data_chunk):
    """erform predictions on a batch of data using Ray for parallel processing."""
    return model_service.predict(model, torch.tensor(data_chunk)).numpy()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Handle prediction requests and store results.
    
    Args:
        request (PredictionRequest): Request containing model and dataset metadata
        
    Returns:
        PredictionResponse: Model predictions and metadata
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Load dataset and model
        dataset = dataset_service.load_dataset(request.dataset_metadata.dict())
        model = model_service.load_model(request.model_metadata.dict())

        # Preprocess data and create chunks
        processed_data = dataset_service.preprocess_data(dataset)
        chunks = dataset_service.create_chunks(processed_data)

        # Use Ray to make predictions in parallel on chunks
        prediction_futures = [predict_batch.remote(model, chunk) for chunk in chunks]
        predictions = ray.get(prediction_futures)

        # Flatten the predictions
        all_predictions = np.concatenate(predictions, axis=0)
        
        # Create prediction metadata
        prediction_metadata = PredictionMetadata(
            model_name=request.model_metadata.model_name,
            model_version=request.model_metadata.version,
            dataset_name=request.dataset_metadata.dataset_name,
            dataset_version=request.dataset_metadata.version,
            timestamp=datetime.utcnow(),
            num_predictions=len(all_predictions),
            created_by=os.getenv("SERVICE_USER", "model_serving_service")
        )
        
        # Save predictions to a temporary .npy.gz file
        with tempfile.NamedTemporaryFile(suffix='.npy.gz', delete=False) as temp_file:
            with gzip.open(temp_file.name, 'wb') as f:
                np.save(f, all_predictions)
            temp_path = temp_file.name

        try:
            # Store prediction in storage service
            prediction_data = {
                "model_name": prediction_metadata.model_name,
                "model_version": prediction_metadata.model_version,
                "dataset_name": prediction_metadata.dataset_name,
                "dataset_version": prediction_metadata.dataset_version,
                "timestamp": prediction_metadata.timestamp,
                "num_predictions": prediction_metadata.num_predictions,
                "created_by": prediction_metadata.created_by
            }
            
            # Upload the predictions file to storage
            storage_service.create("prediction", prediction_data, object_path=temp_path)

            return PredictionResponse(
                predictions_metadata=prediction_metadata
            )

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
