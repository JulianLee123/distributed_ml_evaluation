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
    "output_column": "target",
    "created_by": "test_user"
}

# Prediction test data
TEST_PREDICTION = {
    "model_name": "test_model",
    "model_version": "1.0",
    "dataset_name": "test_dataset",
    "dataset_version": "1.0",
    "timestamp": get_current_timestamp(),
    "created_by": "test_user"
}

# Evaluation test data
TEST_EVALUATION = {
    "model_name": "test_model",
    "model_version": "1.0",
    "dataset_name": "test_dataset",
    "dataset_version": "1.0",
    "timestamp": get_current_timestamp(),
    "evaluations": {"accuracy": 0.95, "precision": 0.92, "recall": 0.88},
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


def create_test_dataset(dataset_name="test_dataset", version="1.0", num_entries=5, output_column="target"):
    """Create a test dataset with custom parameters."""
    data = TEST_DATASET.copy()
    data.update({
        "dataset_name": dataset_name,
        "version": version,
        "num_entries": num_entries,
        "output_column": output_column,
        "timestamp": get_current_timestamp()
    })
    return data


def create_test_prediction(model_name="test_model", dataset_name="test_dataset"):
    """Create a test prediction."""
    
    data = TEST_PREDICTION.copy()
    data.update({
        "model_name": model_name,
        "dataset_name": dataset_name,
        "timestamp": get_current_timestamp()
    })
    return data


def create_test_evaluation(model_name="test_model", dataset_name="test_dataset", evaluations=None):
    """Create a test evaluation with custom parameters."""
    if evaluations is None:
        evaluations = {"accuracy": 0.95, "precision": 0.92, "recall": 0.88}
    
    data = TEST_EVALUATION.copy()
    data.update({
        "model_name": model_name,
        "dataset_name": dataset_name,
        "evaluations": evaluations,
        "timestamp": get_current_timestamp()
    })
    return data