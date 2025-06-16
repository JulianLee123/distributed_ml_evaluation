#!/usr/bin/env python3
"""
Script for downloading files from AWS S3 bucket.

This script handles downloading datasets and models from an S3 bucket to local storage.
It uses environment variables for AWS credentials and S3 configuration.

Example usage:
    python pull_from_s3.py models/classification1.pth models/classification1.pth
    python pull_from_s3.py datasets/classification1Test.csv datasets/classification1Test.csv

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

def download_file_from_s3(bucket_name, object_name, file_path):
    """
    Download a file from S3 bucket to local storage.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        object_name (str): Name of the object in S3 (including path)
        file_path (str): Local path where the file should be saved
        
    Raises:
        SystemExit: If download fails
    """
    try:
        # Print download parameters for debugging
        print(bucket_name, object_name, file_path)
        
        # Download file from S3
        s3.download_file(bucket_name, object_name, file_path)
        print(f"Successfully downloaded {object_name} from {bucket_name} to {file_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)

def main():
    """
    Main function to handle command line arguments and initiate file download.
    
    Expected arguments:
        s3-object-name: Name of the object in S3 (including path)
        local-file-path: Local path where the file should be saved
        
    Raises:
        SystemExit: If arguments are invalid
    """
    # Validate command line arguments
    if len(sys.argv) != 3:
        print("Usage: python pull_from_s3.py <s3-object-name> <local-file-path>")
        sys.exit(1)

    object_name = sys.argv[1]
    file_path = sys.argv[2]

    # Download the file from S3
    download_file_from_s3(S3_BUCKET_NAME, object_name, file_path)

if __name__ == "__main__":
    main()

# Example usage:
# python pull_from_s3.py models/classification1.pth models/classification1.pth
# python pull_from_s3.py datasets/classification1Test.csv datasets/classification1Test.csv