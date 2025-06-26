"""
Test suite for StorageService class.
Run with: pytest test_storage_service.py -v
"""

import pytest
import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

# Add the src directory to Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.storage.storage_service import StorageService

# Import test data from data subfolder
from tests.data.test_metadata import (
    create_test_model,
    create_test_dataset,
    create_test_prediction,
    create_test_evaluation
)


@pytest.fixture(scope="session")
def storage_client():
    """Initialize StorageService client for the entire test session."""
    service = StorageService(is_testing=True)  # Use test prefix
    yield service
    service.close()


def test_create_fetch(storage_client):
    """Test creating artifacts and then fetching them for all 4 collection types."""
    # Create temporary files for model and dataset
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as model_file, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as dataset_file:
        
        # Write some dummy content
        dummy_model_content = "dummy model content"
        model_file.write(dummy_model_content.encode('utf-8'))
        dataset_file.write(b"dummy dataset content")
        model_path = model_file.name
        dataset_path = dataset_file.name

    try:
        # Test data for all 4 collection types
        model_data = create_test_model("fetch_test_model", "1.0", "classification")
        dataset_data = create_test_dataset("fetch_test_dataset", "1.0", 100, "target")
        prediction_data = create_test_prediction("fetch_test_model", "fetch_test_dataset")
        evaluation_data = create_test_evaluation("fetch_test_model", "fetch_test_dataset", {"accuracy": 0.88, "precision": 0.85})
        
        # Test Model collection
        storage_client.create("model", model_data, object_path=model_path)
        model_query = {"model_name": "fetch_test_model", "version": "1.0"}
        fetched_model = storage_client.fetch("model", model_query, metadata_only=False)
        assert fetched_model is not None
        assert fetched_model["model_name"] == "fetch_test_model"
        assert fetched_model["output_type"] == "classification"
        assert "_id" in fetched_model
        assert "download_path" in fetched_model
        with open(fetched_model["download_path"], 'r') as f:
            assert f.read() == dummy_model_content
        # Test Dataset collection
        storage_client.create("dataset", dataset_data, object_path=dataset_path)
        dataset_query = {"dataset_name": "fetch_test_dataset", "version": "1.0"}
        fetched_dataset = storage_client.fetch("dataset", dataset_query, metadata_only=False)
        assert fetched_dataset is not None
        assert fetched_dataset["dataset_name"] == "fetch_test_dataset"
        assert fetched_dataset["num_entries"] == 100
        assert fetched_dataset["output_column"] == "target"
        assert "_id" in fetched_dataset
        assert "download_path" in fetched_dataset

        
        # Test Prediction collection
        storage_client.create("prediction", prediction_data)
        prediction_query = {"model_name": "fetch_test_model", "dataset_name": "fetch_test_dataset"}
        fetched_prediction = storage_client.fetch("prediction", prediction_query)
        assert fetched_prediction is not None
        assert fetched_prediction["model_name"] == "fetch_test_model"
        assert "_id" in fetched_prediction
        
        # Test Evaluation collection
        storage_client.create("evaluation", evaluation_data)
        evaluation_query = {"model_name": "fetch_test_model", "dataset_name": "fetch_test_dataset"}
        fetched_evaluation = storage_client.fetch("evaluation", evaluation_query)
        assert fetched_evaluation is not None
        assert fetched_evaluation["model_name"] == "fetch_test_model"
        assert fetched_evaluation["evaluations"]["accuracy"] == 0.88
        assert fetched_evaluation["evaluations"]["precision"] == 0.85
        assert "_id" in fetched_evaluation
        
    finally:
        # Clean up all test artifacts
        try:
            storage_client.delete("model", {"model_name": "fetch_test_model", "version": "1.0"})
            storage_client.delete("dataset", {"dataset_name": "fetch_test_dataset", "version": "1.0"})
            storage_client.delete("prediction", {"model_name": "fetch_test_model", "dataset_name": "fetch_test_dataset"})
            storage_client.delete("evaluation", {"model_name": "fetch_test_model", "dataset_name": "fetch_test_dataset"})
            os.unlink(model_path)
            os.unlink(dataset_path)
        except:
            pass


def test_create_delete_with_metadata(storage_client):
    """Test creating and deleting dataset entries with metadata."""
    # Create temporary files for datasets
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as dataset1_file, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as dataset2_file:
        
        # Write some dummy content
        dataset1_file.write(b"dummy dataset 1 content")
        dataset2_file.write(b"dummy dataset 2 content")
        dataset1_path = dataset1_file.name
        dataset2_path = dataset2_file.name

    try:
        # Create test data
        dataset1_data = create_test_dataset("delete_test_dataset1", "1.0", 100, "target")
        dataset2_data = create_test_dataset("delete_test_dataset2", "1.0", 200, "label")
        
        # Create both datasets
        storage_client.create("dataset", dataset1_data, object_path=dataset1_path)
        storage_client.create("dataset", dataset2_data, object_path=dataset2_path)
        
        # Verify they exist
        dataset1_query = {"dataset_name": "delete_test_dataset1", "version": "1.0"}
        dataset2_query = {"dataset_name": "delete_test_dataset2", "version": "1.0"}
        
        assert storage_client.fetch("dataset", dataset1_query) is not None
        assert storage_client.fetch("dataset", dataset2_query) is not None
        
        # Delete first dataset
        storage_client.delete("dataset", dataset1_query)
        assert storage_client.fetch("dataset", dataset1_query) is None
        
        # Delete second dataset
        storage_client.delete("dataset", dataset2_query)
        assert storage_client.fetch("dataset", dataset2_query) is None
        
    finally:
        # Clean up
        try:
            storage_client.delete("dataset", {"dataset_name": "delete_test_dataset1", "version": "1.0"})
            storage_client.delete("dataset", {"dataset_name": "delete_test_dataset2", "version": "1.0"})
            os.unlink(dataset1_path)
            os.unlink(dataset2_path)
        except:
            pass


def test_create_delete_without_metadata(storage_client):
    """Test creating and deleting evaluation entries without metadata."""
    try:
        # Create test data
        evaluation1_data = create_test_evaluation("delete_test_model1", "delete_test_dataset1", 
                                                {"accuracy": 0.85, "precision": 0.82})
        evaluation2_data = create_test_evaluation("delete_test_model2", "delete_test_dataset2", 
                                                {"precision": 0.92, "recall": 0.89})
        
        # Create both evaluations
        storage_client.create("evaluation", evaluation1_data)
        storage_client.create("evaluation", evaluation2_data)
        
        # Verify they exist
        eval1_query = {"model_name": "delete_test_model1", "dataset_name": "delete_test_dataset1"}
        eval2_query = {"model_name": "delete_test_model2", "dataset_name": "delete_test_dataset2"}
        
        assert storage_client.fetch("evaluation", eval1_query) is not None
        assert storage_client.fetch("evaluation", eval2_query) is not None
        
        # Delete first evaluation
        storage_client.delete("evaluation", eval1_query)
        assert storage_client.fetch("evaluation", eval1_query) is None
        
        # Delete second evaluation
        storage_client.delete("evaluation", eval2_query)
        assert storage_client.fetch("evaluation", eval2_query) is None
        
    finally:
        # Clean up
        try:
            storage_client.delete("evaluation", {"model_name": "delete_test_model1", "dataset_name": "delete_test_dataset1"})
            storage_client.delete("evaluation", {"model_name": "delete_test_model2", "dataset_name": "delete_test_dataset2"})
        except:
            pass


def test_fetch_multiple_metadata(storage_client):
    """Test creating multiple dataset entries and fetching their metadata."""
    # Create temporary files for datasets
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as dataset1_file, \
         tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as dataset2_file:
        
        # Write some dummy content
        dataset1_file.write(b"dummy dataset 1 content")
        dataset2_file.write(b"dummy dataset 2 content")
        dataset1_path = dataset1_file.name
        dataset2_path = dataset2_file.name

    try:
        # Create test data for multiple datasets
        dataset1_data = create_test_dataset("multi_test_dataset1", "1.0", 100, "target")
        dataset2_data = create_test_dataset("multi_test_dataset2", "1.0", 200, "label")
        
        # Add user field to both datasets
        dataset1_data["user_id"] = "test_user_1"
        dataset2_data["user_id"] = "test_user_1"
        
        # Create both datasets
        storage_client.create("dataset", dataset1_data, object_path=dataset1_path)
        storage_client.create("dataset", dataset2_data, object_path=dataset2_path)
        
        # Fetch all datasets for test_user_1
        user_query = {"user_id": "test_user_1"}
        user_datasets = storage_client.fetch_multiple_metadata("dataset", user_query)
        
        # Verify we got exactly 2 datasets
        assert len(user_datasets) == 2
        
        # Verify both datasets belong to test_user_1
        user_ids = [dataset["user_id"] for dataset in user_datasets]
        assert all(user_id == "test_user_1" for user_id in user_ids)
        
        # Verify we got the correct dataset names
        dataset_names = sorted([dataset["dataset_name"] for dataset in user_datasets])
        expected_names = sorted(["multi_test_dataset1", "multi_test_dataset2"])
        assert dataset_names == expected_names
        
        # Test fetch_multiple with limit parameter
        limited_results = storage_client.fetch_multiple_metadata("dataset", user_query, limit=1)
        assert len(limited_results) == 1
        assert limited_results[0]["user_id"] == "test_user_1"
        
    finally:
        # Clean up
        try:
            storage_client.delete("dataset", {"dataset_name": "multi_test_dataset1", "version": "1.0"})
            storage_client.delete("dataset", {"dataset_name": "multi_test_dataset2", "version": "1.0"})
            os.unlink(dataset1_path)
            os.unlink(dataset2_path)
        except:
            pass


def test_update_metadata(storage_client):
    """Test updating metadata for mutable and immutable artifacts."""
    try:
        # Create test data
        dataset_data = create_test_dataset("update_test_dataset", "1.0", 100, "target")
        evaluation_data = create_test_evaluation("update_test_model", "update_test_dataset", 
                                               {"accuracy": 0.75, "precision": 0.72})
        
        # Create both artifacts
        storage_client.create("dataset", dataset_data)
        storage_client.create("evaluation", evaluation_data)
        
        # Test 1: Try to update dataset entry - should fail/error
        dataset_query = {"dataset_name": "update_test_dataset", "version": "1.0"}
        dataset_updates = {
            "num_entries": 200,
            "size": 2048
        }
        
        # This should raise an error
        try:
            storage_client.update_metadata("dataset", dataset_query, dataset_updates)
            assert False, "Expected schema validation error"
        except Exception as e:
            pass
        
        # Test 2: Try to update evaluation entry - should succeed
        evaluation_query = {"model_name": "update_test_model", "dataset_name": "update_test_dataset"}
        evaluation_updates = {
            "evaluations": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.87
            }
        }
        
        evaluation_update_result = storage_client.update_metadata("evaluation", evaluation_query, evaluation_updates)
        assert evaluation_update_result is True, "Evaluation update should succeed"
        
        # Verify the evaluation was actually updated
        updated_evaluation = storage_client.fetch("evaluation", evaluation_query)
        assert len(updated_evaluation["evaluations"]) == 3
        assert updated_evaluation["evaluations"]["accuracy"] == 0.92
        assert updated_evaluation["evaluations"]["precision"] == 0.89
        assert updated_evaluation["evaluations"]["recall"] == 0.87
        
        # Verify the dataset was NOT updated
        unchanged_dataset = storage_client.fetch("dataset", dataset_query)
        assert unchanged_dataset["num_entries"] == 100  # Should be unchanged
        assert unchanged_dataset["size"] == dataset_data["size"]  # Should be unchanged
        
    finally:
        # Clean up
        try:
            storage_client.delete("dataset", {"dataset_name": "update_test_dataset", "version": "1.0"})
            storage_client.delete("evaluation", {"model_name": "update_test_model", "dataset_name": "update_test_dataset"})
        except:
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
