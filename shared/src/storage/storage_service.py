"""
Main storage service for ML Evaluation platform. Coordiantes s3 and mongo services.
"""

from typing import Dict, Any, Optional, List
from .mongo_service import MongoService
from .s3_service import S3Service, S3ServiceError

class StorageService:
    def __init__(self, is_testing: bool = False, **kwargs):
        """Initialize MongoDB and s3 services"""
        self.artifact_types_with_upload = ['model', 'dataset', 'prediction']
        
        # Initialize services
        artifact_type_immutable = ['model', 'dataset', 'prediction']
        self.mongo_service = MongoService(is_testing = is_testing, immutable_artifact_types = artifact_type_immutable, **kwargs)
        self.s3_service = S3Service(is_testing = is_testing, **kwargs)

    def fetch(self, artifact_type: str, query: Dict, get_latest: bool = False, metadata_only: bool = True) -> Optional[Dict]:
        """Fetch single artifact from storage based on specified criteria. Returns error if more than one artifact is found.
       
        Args:
            artifact_type (str): Type of artifact to retrieve
            query (Dict): Query parameters to filter artifacts
            get_latest (bool, optional): Optionally set to true for versioned artifacts (models, datasets) to retreive the latest version that corresponds to a query.
            metadata_only (bool, optional): If True, only fetches metadata from MongoDB and does not download the artifact from S3. If False and the artifact type supports upload, the downloaded artifact path will be listed as 'download_path' in the returned dictionary. 
        
        Returns:
            Optional[Dict]: Dictionary containing artifact metadata if found,
                None if no matching artifacts exist
        
        Raises:
            MongoServiceError: If MongoDB service unsuccessful
            S3ServiceError: If S3 service unsuccessful 
        """
        # Get metadata from MongoDB
        result = self.mongo_service.fetch(artifact_type, query, get_latest)
        if not result:
            return None

        # If metadata_only is False and artifact type supports upload, download from S3
        if not metadata_only and artifact_type in self.artifact_types_with_upload:
            s3_key = f"/{artifact_type}s/{result['_id']}"
            if self.s3_service.file_exists(s3_key):
                download_path = self.s3_service.download_file(s3_key)
                result['download_path'] = download_path

        return result

    def create(self, artifact_type: str, query: Dict, object_path: Optional[str] = None) -> None:
        """Create artifact in storage based on specified criteria.
       
       Args:
           artifact_type (str): Type of artifact to retrieve
           query (Dict): Query parameters to filter artifacts
           object_path (str, optional): should be set if an only if the artifact type supports upload. If set, uploads the artifact to S3 at path '/' + artifact_type + 's/' + artifact_id.  
       
       Raises:
           MongoServiceError: If MongoDB service unsuccessful
           S3ServiceError: If S3 service unsuccessful 
       """
        try:
            # Validate artifact type supports upload if object_path is provided
            if object_path and artifact_type not in self.artifact_types_with_upload:
                raise ValueError(f"Artifact type {artifact_type} does not support file upload")

            # Create metadata in MongoDB
            artifact_id = self.mongo_service.create(artifact_type, query)

            # If object_path is provided, upload to S3
            if object_path:
                s3_key = f"/{artifact_type}s/{artifact_id}"
                self.s3_service.upload_file(object_path, s3_key, metadata=query)

        except S3ServiceError as e:
            # If S3 upload fails, clean up MongoDB entry
            try:
                self.mongo_service.delete(artifact_type, {"_id": artifact_id})
            except:
                pass
            raise S3ServiceError(f"Failed to upload {artifact_type} to S3: {str(e)}")

    def delete(self, artifact_type: str, query: Dict) -> None:
        """Delete artifact from storage (MongoDB and if necessary, S3) based on specified criteria. Returns error if more than one artifact is found.
        
        Args:
            artifact_type (str): Type of artifact to delete
            query (Dict): Query parameters to filter artifacts
        
        Raises:
            MongoServiceError: If MongoDB service unsuccessful
            S3ServiceError: If S3 service unsuccessful 
        """
        # Get artifact metadata first to get ID
        result = self.mongo_service.fetch(artifact_type, query)
        if not result:
            return

        # Delete from MongoDB
        self.mongo_service.delete(artifact_type, query)

        # If artifact type supports upload, delete from S3
        if artifact_type in self.artifact_types_with_upload:
            s3_key = f"/{artifact_type}s/{result['_id']}"
            if self.s3_service.file_exists(s3_key):
                self.s3_service.delete_file(s3_key)

    def fetch_multiple_metadata(self, artifact_type: str, query: Dict, limit: int = 50) -> List[Dict]:
        """Fetches metadata for multiple artifacts. Does not fetch objects from S3. 
       
        Args:
            artifact_type (str): Type of artifact to delete
            query (Dict): Query parameters to filter artifacts
            limit (int); Maximum number of artifacts to be returned
        
        Returns:
            List[Dict]: List of dictionary objcts containing artifact metadata
        
        Raises:
            MongoServiceError: If MongoDB service unsuccessful
        """
        return self.mongo_service.fetch_multiple(artifact_type, query, limit)

    def update_metadata(self, artifact_type: str, query: Dict, updates: Dict) -> None:
        """Updates metadata for a mutable artifact. Returns error if more than one artifact is found, or if the artifact type is immutable.
       
       Args:
           artifact_type (str): Type of artifact to delete
           query (Dict): Query parameters to filter artifacts
       
       Raises:
           MongoServiceError: If MongoDB service unsuccessful
       """
        # Check if artifact type is immutable
        return self.mongo_service.update(artifact_type, query, updates)

    def close(self):
        """Close MongoDB and S3 connections."""
        self.mongo_service.close()
        self.s3_service.close()