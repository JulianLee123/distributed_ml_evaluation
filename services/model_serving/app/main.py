"""
Main FastAPI application for model serving service.
"""

from fastapi import FastAPI, HTTPException
import ray
import numpy as np
import os
import tempfile
import gzip

from .schemas import (
    PredictionRequest, 
    PredictionResponse,
    PredictionMetadata
)
from .model_actor import ModelActor
from .dataset_service import DatasetService
from shared.src.storage.storage_service import StorageService

# Initialize Ray
ray.init()

# Initialize FastAPI app
app = FastAPI(title="ML Evaluation Model Serving Service")

# Initialize services
if os.getenv("DEV") == "TRUE":
    storage_service = StorageService(is_testing=True)
else:
    storage_service = StorageService()
dataset_service = DatasetService(storage_service)

# Initialize Ray actors pool
actor_pool = [ModelActor.remote() for _ in range(int(os.getenv("RAY_NUM_ACTORS")))]

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
        # Load and preprocess dataset
        dataset = dataset_service.load_dataset(request.dataset_metadata.model_dump())
        processed_data = dataset_service.preprocess_data(dataset)
        chunks = dataset_service.create_chunks(processed_data)

        # Load model
        model_metadata = request.model_metadata.model_dump()
        model_query = {
            "model_name": model_metadata["model_name"],
            "version": model_metadata["version"]
        }
        model_data = storage_service.fetch("model", model_query, metadata_only=False)

        if not model_data or "download_path" not in model_data:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_path = model_data["download_path"]

        # Load model bytes and create unique model ID
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
        model_id = f"{request.model_metadata.model_name}_{request.model_metadata.version}"
        model_bytes_ref = ray.put(model_bytes)
        
        # Distribute predictions across actor pool
        prediction_futures = []
        for i, chunk in enumerate(chunks):
            actor = actor_pool[i % len(actor_pool)]
            future = actor.predict.remote(model_bytes_ref, model_id, chunk)
            prediction_futures.append(future)
        
        predictions = ray.get(prediction_futures)
        os.unlink(model_path)
        
        # Flatten the predictions
        all_predictions = np.concatenate(predictions, axis=0)

        # Create prediction metadata
        prediction_metadata = PredictionMetadata(
            model_name=request.model_metadata.model_name,
            model_version=request.model_metadata.version,
            dataset_name=request.dataset_metadata.dataset_name,
            dataset_version=request.dataset_metadata.version,
            num_predictions=len(all_predictions),
            created_by=request.created_by
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