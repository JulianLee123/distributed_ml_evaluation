"""
Model service for loading and running ML models.
"""

import ray
import torch
import io

@ray.remote
class ModelActor:
    """Ray actor for handling model predictions with persistent model loading."""
    
    def __init__(self):
        self.model = None
        self.current_model_id = None
    
    def load_model(self, model_bytes, model_id):
        """Load model if not already loaded or if different model."""
        if self.current_model_id != model_id:
            self.model = torch.jit.load(io.BytesIO(model_bytes), map_location='cpu')
            self.model.eval()
            self.current_model_id = model_id
    
    def predict(self, model_bytes, model_id, data_chunk):
        """Perform predictions on a batch of data."""
        self.load_model(model_bytes, model_id)
        with torch.no_grad():
            return self.model(torch.tensor(data_chunk, dtype=torch.float32)).numpy()