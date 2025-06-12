"""
Test data fixtures for MongoDB service tests.
Contains reusable test data objects for different artifact types.
"""

from datetime import datetime, UTC
from bson import Int64


def get_current_timestamp():
    """Helper function to get current timestamp for test data."""
    return datetime.now(UTC)


# Model test data
TEST_MODEL_TYPE_1 = {
    "model_name": "test_model",
    "version": "1.0",
    "timestamp": get_current_timestamp(),
    "output_type": "classification",
    "size": Int64(1024), 
    "created_by": "test_user"
}

# Dataset test data
TEST_DATASET = {
    "dataset_name": "test_dataset",
    "version": "1.0",
    "timestamp": get_current_timestamp(),
    "num_entries": 5,
    "size": Int64(1024), 
    "created_by": "test_user"
}

# Prediction test data
TEST_PREDICTION = {
    "model_name": "test_model",
    "model_version": "1.0",
    "dataset_name": "test_dataset",
    "dataset_version": "1.0",
    "timestamp": get_current_timestamp(),
    "predictions": [1, 2, 1, 1, 1],  # Fixed syntax error: was "predictions"; should be "predictions":
    "created_by": "test_user"
}

# Evaluation test data
TEST_EVALUATION = {
    "model_name": "test_model",
    "model_version": "1.0",
    "dataset_name": "test_dataset",
    "dataset_version": "1.0",
    "timestamp": get_current_timestamp(),
    "evaluations": [{"metric_name": "accuracy", "value": 0.95}],
    "created_by": "test_user"
}


# Helper functions to create modified copies of test data
def create_test_model(model_name="test_model", version="1.0", output_type="classification"):
    """Create a test model with custom parameters."""
    data = TEST_MODEL_TYPE_1.copy()
    data.update({
        "model_name": model_name,
        "version": version,
        "output_type": output_type,
        "timestamp": get_current_timestamp()
    })
    return data


def create_test_dataset(dataset_name="test_dataset", version="1.0", num_entries=5):
    """Create a test dataset with custom parameters."""
    data = TEST_DATASET.copy()
    data.update({
        "dataset_name": dataset_name,
        "version": version,
        "num_entries": num_entries,
        "timestamp": get_current_timestamp()
    })
    return data


def create_test_prediction(model_name="test_model", dataset_name="test_dataset", predictions=None):
    """Create a test prediction with custom parameters."""
    if predictions is None:
        predictions = [1, 2, 1, 1, 1]
    
    data = TEST_PREDICTION.copy()
    data.update({
        "model_name": model_name,
        "dataset_name": dataset_name,
        "predictions": predictions,
        "timestamp": get_current_timestamp()
    })
    return data


def create_test_evaluation(model_name="test_model", dataset_name="test_dataset", evaluations=None):
    """Create a test evaluation with custom parameters."""
    if evaluations is None:
        evaluations = [{"metric_name": "accuracy", "value": 0.95}]
    
    data = TEST_EVALUATION.copy()
    data.update({
        "model_name": model_name,
        "dataset_name": dataset_name,
        "evaluations": evaluations,
        "timestamp": get_current_timestamp()
    })
    return data