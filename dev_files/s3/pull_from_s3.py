import os
import boto3
from dotenv import load_dotenv
import sys

load_dotenv()

S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_DEFAULT_REGION = os.getenv("S3_DEFAULT_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

if not S3_ACCESS_KEY_ID or not S3_SECRET_ACCESS_KEY or not S3_DEFAULT_REGION or not S3_BUCKET_NAME:
    print("Missing AWS credentials or S3 configuration in .env file")
    sys.exit(1)

s3 = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY_ID,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
    region_name=S3_DEFAULT_REGION
)

def download_file_from_s3(bucket_name, object_name, file_path):
    try:
        print(bucket_name, object_name, file_path)
        s3.download_file(bucket_name, object_name, file_path)
        print(f"Successfully downloaded {object_name} from {bucket_name} to {file_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)

def main():
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