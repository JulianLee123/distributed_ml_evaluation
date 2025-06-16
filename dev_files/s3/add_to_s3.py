#!/usr/bin/env python3
"""
Script for uploading files to AWS S3 bucket.

This script handles uploading datasets and models to an S3 bucket for storage.
It uses environment variables for AWS credentials and S3 configuration.

Example usage:
    python add_to_s3.py ../mock_data_and_models/models/classification1.pth model
    python add_to_s3.py ../mock_data_and_models/datasets/classification1Test.csv dataset

Requirements:
    - AWS credentials (access key and secret key)
    - S3 bucket configuration
    - boto3 library
    - python-dotenv for environment variable management
"""

import os
import boto3
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
load_dotenv()

# Get AWS credentials and S3 configuration from environment variables
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_DEFAULT_REGION = os.getenv("S3_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Validate that all required environment variables are set
if not S3_ACCESS_KEY_ID or not S3_SECRET_ACCESS_KEY or not S3_DEFAULT_REGION or not S3_BUCKET_NAME:
    print("Missing AWS credentials or S3 configuration in .env file")
    sys.exit(1)

# Initialize S3 client with credentials
s3 = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY_ID,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
    region_name=S3_DEFAULT_REGION
)

def upload_file_to_s3(file_path, bucket_name, file_type, object_name=None):
    """
    Upload a file to S3 bucket.
    
    Args:
        file_path (str): Local path to the file to upload
        bucket_name (str): Name of the S3 bucket
        file_type (str): Type of file ('dataset' or 'model')
        object_name (str, optional): Custom S3 object name. If None, will use default naming convention
        
    Raises:
        SystemExit: If upload fails
    """
    # If no custom object name provided, use default naming convention
    if object_name is None:
        object_name = file_type + "s/" + os.path.basename(file_path)
    
    try:
        # Upload file to S3
        s3.upload_file(file_path, bucket_name, object_name)
        print(f"Successfully uploaded {file_path} to {bucket_name}/{object_name}")
    except Exception as e:
        print(f"Error uploading file: {e}")
        sys.exit(1)

def main():
    """
    Main function to handle command line arguments and initiate file upload.
    
    Expected arguments:
        file_path: Path to the file to upload
        file_type: Type of file ('dataset' or 'model')
        
    Raises:
        SystemExit: If arguments are invalid or file doesn't exist
    """
    # Validate command line arguments
    if len(sys.argv) != 3:
        print("Usage: python add_to_s3.py <file_path> <file_type = dataset/model>")
        sys.exit(1)

    file_path = sys.argv[1]
    file_type = sys.argv[2]

    # Validate that the file exists
    if not os.path.isfile(file_path):
        print(f"Error: {file_path} is not a valid file.")
        sys.exit(1)

    # Upload the file to S3
    upload_file_to_s3(file_path, S3_BUCKET_NAME, file_type)

if __name__ == "__main__":
    main()

# Example usage:
# python add_to_s3.py ../mock_data_and_models/models/classification1.pth model
# python add_to_s3.py ../mock_data_and_models/datasets/classification1Test.csv dataset