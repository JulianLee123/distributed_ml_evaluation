"""
Test suite for model serving prediction endpoint.
"""

import pytest
import os
import numpy as np
from pathlib import Path
import sys
import tempfile
import gzip
from bson import Int64

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.main import app
from app.schemas import PredictionRequest, ModelMetadata, DatasetMetadata
from shared.src.storage.storage_service import StorageService
from fastapi.testclient import TestClient

# Initialize test client
client = TestClient(app)

@pytest.fixture(scope="session")
def storage_service():
    """Initialize storage service for testing."""
    service = StorageService(is_testing=True)
    yield service
    service.close()

@pytest.fixture(scope="session")
def test_data():
    """Load test data and expected outputs."""
    test_dir = Path(__file__).parent
    
    # Load test dataset
    dataset_path = test_dir / "datasets" / "classification1_test.csv"
    with open(dataset_path, "rb") as f:
        dataset_content = f.read()
    
    # Load test model
    model_path = test_dir / "models" / "classification1.pt"
    with open(model_path, "rb") as f:
        model_content = f.read()
    
    # Load expected predictions
    expected_path = test_dir / "expected_outputs" / "classification1_pred.npy"
    expected_predictions = np.load(expected_path)
    
    return {
        "dataset": dataset_content,
        "model": model_content,
        "expected_predictions": expected_predictions
    }

def test_prediction_endpoint(storage_service, test_data):
    """Test the prediction endpoint with test data."""
    # First, upload test dataset and model to storage
    dataset_metadata = {
        "dataset_name": "test_dataset",
        "version": "1.0",
        "num_entries": 100,
        "size": Int64(len(test_data["dataset"])),
        "created_by": "test_user"
    }
    
    model_metadata = {
        "model_name": "test_model",
        "version": "1.0",
        "output_type": "classification",
        "size": Int64(len(test_data["model"])),
        "created_by": "test_user"
    }
    
    # Save test files temporarily
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as dataset_file, \
         tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as model_file:
        
        dataset_file.write(test_data["dataset"])
        model_file.write(test_data["model"])
        
        dataset_path = dataset_file.name
        model_path = model_file.name
    
    try:
        # Upload to storage
        storage_service.create("dataset", dataset_metadata, object_path=dataset_path)
        storage_service.create("model", model_metadata, object_path=model_path)
        
        # Create prediction request
        request = PredictionRequest(
            model_metadata=ModelMetadata(**model_metadata),
            dataset_metadata=DatasetMetadata(**dataset_metadata),
            created_by="test_user"
        )
        
        # Make prediction request
        response = client.post("/predict", json=request.model_dump())
        assert response.status_code == 200
        
        # Get prediction results from storage
        prediction_query = {
            "model_name": "test_model",
            "dataset_name": "test_dataset"
        }
        prediction_result = storage_service.fetch("prediction", prediction_query, metadata_only=False)
        assert prediction_result is not None
        
        # Load predictions from stored file
        with gzip.open(prediction_result["download_path"], 'rb') as f:
            stored_predictions = np.load(f)
        
        # Compare with expected predictions
        np.testing.assert_array_almost_equal(stored_predictions, test_data["expected_predictions"], decimal=4)
        
    finally:
        # Clean up temporary files
        os.unlink(dataset_path)
        os.unlink(model_path)
        
        # Clean up test data from storage
        storage_service.delete("dataset", {"dataset_name": "test_dataset", "version": "1.0"})
        storage_service.delete("model", {"model_name": "test_model", "version": "1.0"})
        storage_service.delete("prediction", {"model_name": "test_model", "dataset_name": "test_dataset"})

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
