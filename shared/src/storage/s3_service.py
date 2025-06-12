"""
S3 service for ML Evaluation platform.
"""

import os
import boto3
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError

class S3ServiceError(Exception):
    pass

class S3Service:
    def __init__(self, is_testing: bool = False, **kwargs):
        """Initialize S3 service with config from kwargs or environment variables."""
        self.config = {
            'aws_access_key_id': kwargs.get('aws_access_key_id') or os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': kwargs.get('aws_secret_access_key') or os.getenv('AWS_SECRET_ACCESS_KEY'),
            'region_name': kwargs.get('region_name') or os.getenv('AWS_DEFAULT_REGION'),
            'bucket_name': kwargs.get('bucket_name') or os.getenv('S3_BUCKET_NAME')
        }
        
        missing = [k for k, v in self.config.items() if not v]
        if missing:
            raise S3ServiceError(f"Missing config: {', '.join(missing)}")
        
        if is_testing:
            self.path_prefix = "test/"
        else:
            self.path_prefix = ""

        self._connect()

    def _connect(self):
        """Establish S3 connection and test bucket access."""
        try:
            self.client = boto3.client(
                's3',
                aws_access_key_id=self.config['aws_access_key_id'],
                aws_secret_access_key=self.config['aws_secret_access_key'],
                region_name=self.config['region_name']
            )
            self.bucket = self.config['bucket_name']
            self.client.head_bucket(Bucket=self.bucket)  # Test connection
            
        except NoCredentialsError:
            raise S3ServiceError("Invalid AWS credentials")
        except ClientError as e:
            code = e.response['Error']['Code']
            if code == '404':
                raise S3ServiceError(f"Bucket '{self.bucket}' not found")
            elif code == '403':
                raise S3ServiceError(f"Access denied to bucket '{self.bucket}'")
            raise S3ServiceError(f"S3 connection failed: {str(e)}")

    def download_file(self, s3_key: str, local_path: str = None) -> str:
        """Download file from S3 to local filesystem or temp file."""
        try:
            s3_key = self.path_prefix + s3_key
            if not local_path:
                suffix = Path(s3_key).suffix
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                local_path = temp_file.name
                temp_file.close()
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.client.download_file(self.bucket, s3_key, local_path)
            return local_path
            
        except ClientError as e:
            code = e.response['Error']['Code']
            if code == '404':
                raise S3ServiceError(f"Object not found: {s3_key}")
            elif code == '403':
                raise S3ServiceError(f"Access denied: {s3_key}")
            raise S3ServiceError(f"Download failed: {str(e)}")

    def upload_file(self, local_path: str, s3_key: str, metadata: Dict[str, str] = None) -> None:
        """Upload file to S3 with optional metadata."""
        try:
            s3_key = self.path_prefix + s3_key
            if not os.path.exists(local_path):
                raise S3ServiceError(f"File not found: {local_path}")
            
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = {k: str(v) for k, v in metadata.items()}
            
            self.client.upload_file(local_path, self.bucket, s3_key, ExtraArgs=extra_args)
            
        except ClientError as e:
            raise S3ServiceError(f"Upload failed: {str(e)}")

    def delete_file(self, s3_key: str) -> None:
        """Delete file from S3."""
        try:
            s3_key = self.path_prefix + s3_key
            self.client.delete_object(Bucket=self.bucket, Key=s3_key)
        except ClientError as e:
            raise S3ServiceError(f"Delete failed: {str(e)}")

    def file_exists(self, s3_key: str) -> bool:
        """Check if file exists in S3."""
        try:
            s3_key = self.path_prefix + s3_key
            self.client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise S3ServiceError(f"Exists check failed: {str(e)}")

    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """List objects in S3 with prefix filter."""
        try:
            prefix = self.path_prefix + prefix
            objects = []
            paginator = self.client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, MaxKeys=max_keys):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append({
                            'Key': obj['Key'],
                            'Size': obj['Size'],
                            'LastModified': obj['LastModified']
                        })
            return objects
            
        except ClientError as e:
            raise S3ServiceError(f"List failed: {str(e)}")

    def close(self):
        """Close S3 connectoin."""
        if hasattr(self, 'client'):
            self.client.close()