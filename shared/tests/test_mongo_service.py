"""
Simple test suite for MongoService class.
Run with: pytest test_mongo_service.py -v
"""

import pytest
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Add the src directory to Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
from src.storage.mongo_service import MongoService

# Import test data from data subfolder
from tests.data.test_metadata import (
    create_test_model,
    create_test_dataset,
    create_test_prediction,
    create_test_evaluation
)


@pytest.fixture(scope="session")
def mongo_client():
    """Initialize MongoDB client for the entire test session."""
    service = MongoService(is_testing=True)  # Use test prefix
    yield service
    service.close()


def test_create_fetch(mongo_client):
    """Test creating artifacts and then fetching them for all 4 collection types."""
    # Test data for all 4 collection types
    model_data = create_test_model("fetch_test_model", "1.0", "classification")
    dataset_data = create_test_dataset("fetch_test_dataset", "1.0", 100)
    prediction_data = create_test_prediction("fetch_test_model", "fetch_test_dataset")
    evaluation_data = create_test_evaluation("fetch_test_model", "fetch_test_dataset", [{"metric_name": "accuracy", "value": 0.88}])
    
    try:
        # Test Model collection
        model_id = mongo_client.create("model", model_data)
        assert model_id is not None
        
        model_query = {"model_name": "fetch_test_model", "version": "1.0"}
        fetched_model = mongo_client.fetch("model", model_query)
        assert fetched_model is not None
        assert fetched_model["model_name"] == "fetch_test_model"
        assert fetched_model["output_type"] == "classification"
        assert "_id" in fetched_model
        
        # Test Dataset collection
        dataset_id = mongo_client.create("dataset", dataset_data)
        assert dataset_id is not None
        
        dataset_query = {"dataset_name": "fetch_test_dataset", "version": "1.0"}
        fetched_dataset = mongo_client.fetch("dataset", dataset_query)
        assert fetched_dataset is not None
        assert fetched_dataset["dataset_name"] == "fetch_test_dataset"
        assert fetched_dataset["num_entries"] == 100
        assert "_id" in fetched_dataset
        
        # Test Prediction collection
        prediction_id = mongo_client.create("prediction", prediction_data)
        assert prediction_id is not None
        
        prediction_query = {"model_name": "fetch_test_model", "dataset_name": "fetch_test_dataset"}
        fetched_prediction = mongo_client.fetch("prediction", prediction_query)
        assert fetched_prediction is not None
        assert fetched_prediction["model_name"] == "fetch_test_model"
        assert "_id" in fetched_prediction
        
        # Test Evaluation collection
        evaluation_id = mongo_client.create("evaluation", evaluation_data)
        assert evaluation_id is not None
        
        evaluation_query = {"model_name": "fetch_test_model", "dataset_name": "fetch_test_dataset"}
        fetched_evaluation = mongo_client.fetch("evaluation", evaluation_query)
        assert fetched_evaluation is not None
        assert fetched_evaluation["model_name"] == "fetch_test_model"
        assert fetched_evaluation["evaluations"][0]["metric_name"] == "accuracy"
        assert fetched_evaluation["evaluations"][0]["value"] == 0.88
        assert "_id" in fetched_evaluation
        
    finally:
        # Clean up all test artifacts
        try:
            mongo_client.delete("model", {"model_name": "fetch_test_model", "version": "1.0"})
            mongo_client.delete("dataset", {"dataset_name": "fetch_test_dataset", "version": "1.0"})
            mongo_client.delete("prediction", {"model_name": "fetch_test_model", "dataset_name": "fetch_test_dataset"})
            mongo_client.delete("evaluation", {"model_name": "fetch_test_model", "dataset_name": "fetch_test_dataset"})
        except:
            pass



def test_create_fetch_versioned(mongo_client):
    """Test creating multiple versions (1.0, 1.5, 2.0) and fetching the latest."""
    # Create three versions of the same model: 1.0, 1.5, and 2.0
    version_1_0_data = create_test_model("versioned_model", "1.0", "classification")
    version_1_5_data = create_test_model("versioned_model", "1.5", "regression") 
    version_2_0_data = create_test_model("versioned_model", "2.0", "classification")
    
    try:
        # Create all three versions
        doc_id_1_0 = mongo_client.create("model", version_1_0_data)
        doc_id_1_5 = mongo_client.create("model", version_1_5_data)
        doc_id_2_0 = mongo_client.create("model", version_2_0_data)
        
        assert doc_id_1_0 is not None
        assert doc_id_1_5 is not None
        assert doc_id_2_0 is not None
        assert len({doc_id_1_0, doc_id_1_5, doc_id_2_0}) == 3  # All IDs should be unique
        
        # Fetch specific version 1.0
        v1_0_query = {"model_name": "versioned_model", "version": "1.0"}
        v1_0_result = mongo_client.fetch("model", v1_0_query)
        assert v1_0_result["version"] == "1.0"
        assert v1_0_result["output_type"] == "classification"
        
        # Fetch specific version 1.5
        v1_5_query = {"model_name": "versioned_model", "version": "1.5"}
        v1_5_result = mongo_client.fetch("model", v1_5_query)
        assert v1_5_result["version"] == "1.5"
        assert v1_5_result["output_type"] == "regression"
        
        # Fetch specific version 2.0
        v2_0_query = {"model_name": "versioned_model", "version": "2.0"}
        v2_0_result = mongo_client.fetch("model", v2_0_query)
        assert v2_0_result["version"] == "2.0"
        assert v2_0_result["output_type"] == "classification"
        
        # Fetch latest version (should be 2.0 since it's the highest)
        latest_query = {"model_name": "versioned_model"}
        latest_result = mongo_client.fetch("model", latest_query, get_latest=True)
        assert latest_result["version"] == "2.0"
        assert latest_result["output_type"] == "classification"
        
    finally:
        # Clean up all three versions
        try:
            mongo_client.delete("model", {"model_name": "versioned_model", "version": "1.0"})
            mongo_client.delete("model", {"model_name": "versioned_model", "version": "1.5"})
            mongo_client.delete("model", {"model_name": "versioned_model", "version": "2.0"})
        except:
            pass


def test_fetch_multiple(mongo_client):
    """Test creating multiple models for different users and fetching models for a specific user."""
    # Create test data for multiple models
    # Two models for test_user_1
    user1_model1_data = create_test_model("user1_model_alpha", "1.0", "classification")
    user1_model1_data["user_id"] = "test_user_1"  # Add user field
    
    user1_model2_data = create_test_model("user1_model_beta", "1.2", "regression")
    user1_model2_data["user_id"] = "test_user_1"  # Add user field
    
    # One model for test_user_2
    user2_model_data = create_test_model("user2_model_gamma", "2.0", "classification")
    user2_model_data["user_id"] = "test_user_2"  # Add user field
    
    try:
        # Create all three models
        user1_model1_id = mongo_client.create("model", user1_model1_data)
        user1_model2_id = mongo_client.create("model", user1_model2_data)
        user2_model_id = mongo_client.create("model", user2_model_data)
        
        assert user1_model1_id is not None
        assert user1_model2_id is not None
        assert user2_model_id is not None
        
        # Use fetch_multiple to get all models for test_user_1
        user1_query = {"user_id": "test_user_1"}
        user1_models = mongo_client.fetch_multiple("model", user1_query)
        
        # Verify we got exactly 2 models for test_user_1
        assert len(user1_models) == 2
        
        # Verify both models belong to test_user_1
        user_ids = [model["user_id"] for model in user1_models]
        assert all(user_id == "test_user_1" for user_id in user_ids)
        
        # Verify we got the correct model names
        model_names = sorted([model["model_name"] for model in user1_models])
        expected_names = sorted(["user1_model_alpha", "user1_model_beta"])
        assert model_names == expected_names
        
        # Verify model details
        alpha_model = next(m for m in user1_models if m["model_name"] == "user1_model_alpha")
        beta_model = next(m for m in user1_models if m["model_name"] == "user1_model_beta")
        
        assert alpha_model["version"] == "1.0"
        assert alpha_model["output_type"] == "classification"
        assert "_id" in alpha_model
        
        assert beta_model["version"] == "1.2"
        assert beta_model["output_type"] == "regression"
        assert "_id" in beta_model
        
        # Test fetch_multiple for test_user_2 (should return 1 model)
        user2_query = {"user_id": "test_user_2"}
        user2_models = mongo_client.fetch_multiple("model", user2_query)
        
        assert len(user2_models) == 1
        assert user2_models[0]["user_id"] == "test_user_2"
        assert user2_models[0]["model_name"] == "user2_model_gamma"
        assert user2_models[0]["version"] == "2.0"
        assert user2_models[0]["output_type"] == "classification"
        
        # Test fetch_multiple with non-existent user (should return empty list)
        nonexistent_query = {"user_id": "nonexistent_user"}
        empty_results = mongo_client.fetch_multiple("model", nonexistent_query)
        assert len(empty_results) == 0
        assert empty_results == []
        
        # Test fetch_multiple with limit parameter
        limited_results = mongo_client.fetch_multiple("model", user1_query, limit=1)
        assert len(limited_results) == 1
        assert limited_results[0]["user_id"] == "test_user_1"
        
    finally:
        # Clean up all test artifacts
        try:
            mongo_client.delete("model", {"model_name": "user1_model_alpha", "version": "1.0"})
            mongo_client.delete("model", {"model_name": "user1_model_beta", "version": "1.2"})
            mongo_client.delete("model", {"model_name": "user2_model_gamma", "version": "2.0"})
        except:
            pass


def test_create_delete(mongo_client):
    """Test creating an artifact and then deleting it."""
    # Use helper function to create test data
    delete_test_data = create_test_model("delete_test_model", "1.0", "classification")
    
    try:
        # Create the artifact
        doc_id = mongo_client.create("model", delete_test_data)
        assert doc_id is not None
        
        # Verify it exists
        query = {"model_name": "delete_test_model", "version": "1.0"}
        assert mongo_client.exists("model", query) is True
        
        # Fetch to double-check
        fetched = mongo_client.fetch("model", query)
        assert fetched is not None
        assert fetched["model_name"] == "delete_test_model"
        assert fetched["output_type"] == delete_test_data["output_type"]
        
        # Delete the artifact
        mongo_client.delete("model", query)
        
        # Verify it no longer exists
        assert mongo_client.exists("model", query) is False
        
        # Verify fetch returns None
        deleted_fetch = mongo_client.fetch("model", query)
        assert deleted_fetch is None
        
    finally:
        # Cleanup (in case test fails partway through)
        try:
            mongo_client.delete("model", {"model_name": "delete_test_model", "version": "1.0"})
        except:
            pass


def test_create_update(mongo_client):
    """Test creating artifacts and then updating them - model should fail, evaluation should succeed."""
    # Test data for both model and evaluation
    model_data = create_test_model("update_test_model", "1.0", "classification")
    evaluation_data = create_test_evaluation("update_test_model", "test_dataset", [{"metric_name": "accuracy", "value": 0.75}])
    
    try:
        # Create both artifacts
        model_id = mongo_client.create("model", model_data)
        evaluation_id = mongo_client.create("evaluation", evaluation_data)
        assert model_id is not None
        assert evaluation_id is not None
        
        # Test 1: Try to update model entry - should fail/error
        model_query = {"model_name": "update_test_model", "version": "1.0"}
        model_updates = {
            "output_type": "regression",
            "size": 2048
        }
        
        # This should raise an error
        try:
            mongo_client.update("model", model_query, model_updates)
            assert False, "Expected schema validation error"
        except Exception as e:
            pass

        # Test 2: Try to update evaluation entry - should succeed
        evaluation_query = {"model_name": "update_test_model", "dataset_name": "test_dataset"}
        evaluation_updates = {
            "evaluations": [
                {"metric_name": "accuracy", "value": 0.92},
                {"metric_name": "precision", "value": 0.89}
            ]
        }
        
        evaluation_update_result = mongo_client.update("evaluation", evaluation_query, evaluation_updates)
        assert evaluation_update_result is True, "Evaluation update should succeed"
        
        # Verify the evaluation was actually updated
        updated_evaluation = mongo_client.fetch("evaluation", evaluation_query)
        assert len(updated_evaluation["evaluations"]) == 2
        assert updated_evaluation["evaluations"][0]["metric_name"] == "accuracy"
        assert updated_evaluation["evaluations"][0]["value"] == 0.92
        assert updated_evaluation["evaluations"][1]["metric_name"] == "precision"
        assert updated_evaluation["evaluations"][1]["value"] == 0.89
        assert "updated_at" in updated_evaluation
        
        # Verify the model was NOT updated
        unchanged_model = mongo_client.fetch("model", model_query)
        assert unchanged_model["output_type"] == "classification"  # Should be unchanged
        assert unchanged_model["size"] == model_data["size"]  # Should be unchanged
        
    finally:
        # Clean up
        try:
            mongo_client.delete("model", {"model_name": "update_test_model", "version": "1.0"})
            mongo_client.delete("evaluation", {"model_name": "update_test_model", "dataset_name": "test_dataset"})
        except:
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])