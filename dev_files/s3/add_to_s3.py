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

def upload_file_to_s3(file_path, bucket_name, file_type, object_name=None):
    if object_name is None:
        object_name = file_type + "s/" + os.path.basename(file_path)
    try:
        s3.upload_file(file_path, bucket_name, object_name)
        print(f"Successfully uploaded {file_path} to {bucket_name}/{object_name}")
    except Exception as e:
        print(f"Error uploading file: {e}")
        sys.exit(1)

# Main function to handle command line input and call the upload function
def main():
    if len(sys.argv) != 3:
        print("Usage: python add_to_s3.py <file_path> <file_type = dataset/model>")
        sys.exit(1)

    file_path = sys.argv[1]
    file_type = sys.argv[2]

    if not os.path.isfile(file_path):
        print(f"Error: {file_path} is not a valid file.")
        sys.exit(1)

    # Upload the file to S3
    upload_file_to_s3(file_path, S3_BUCKET_NAME, file_type)

if __name__ == "__main__":
    main()

# python add_to_s3.py ../mock_data_and_models/models/classification1.pth model
# python add_to_s3.py ../mock_data_and_models/datasets/classification1Test.csv dataset