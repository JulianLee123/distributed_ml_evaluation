"""
Shared utilities for ML Evaluation platform.
"""

from .s3.s3_service import S3Service
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

__all__ = ['S3Service']
