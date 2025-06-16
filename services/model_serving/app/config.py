import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS Configuration
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_DEFAULT_REGION = os.getenv("S3_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Validate environment variables
if not all([S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_DEFAULT_REGION, S3_BUCKET_NAME]):
    raise RuntimeError("Missing AWS credentials or S3 configuration in .env file")

# Ray Configuration
RAY_ADDRESS = os.getenv("RAY_ADDRESS", "auto")

# Application Configuration
CHUNK_SIZE = 100  # Number of rows to process in each batch 