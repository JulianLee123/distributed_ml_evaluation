"""
MongoDB service for ML Evaluation platform.
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError, PyMongoError
from urllib.parse import quote_plus
from .schema_manager import SchemaManager

class MongoServiceError(Exception):
    pass

class MongoService:
    def __init__(self, is_testing = False, immutable_artifact_types = [], **kwargs):
        """Initialize MongoDB service with config from kwargs or environment variables."""
        self.config = {
            'uri': kwargs.get('mongodb_uri') or os.getenv('MONGODB_URI'),
            'password': kwargs.get('mongodb_password') or os.getenv('MONGODB_PASSWORD'),
        }
        
        if not self.config['uri']:
            raise MongoServiceError("MONGODB_URI required")
        
        self._connect()

        if is_testing:
            self.is_testing = True
            self.collection_prefix = "test_"
        else:
            self.is_testing = False
            self.collection_prefix = ""

        self.immutable_artifact_types = immutable_artifact_types
        self.schema_manager = SchemaManager(self.db, is_testing = is_testing)
        self.schema_manager.apply_schemas()
        self.supported_collections = self.schema_manager.get_supported_collections()

    def _connect(self):
        """Establish MongoDB connection with credentials and test connectivity."""
        try:
            uri = self.config['uri']
            if self.config.get('password'):
                uri = uri.replace('<db_password>', self.config['password'])
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.db = self.client['ModelEval']
            self.client.admin.command('ping')  # Test connection
            
        except Exception as e:
            raise MongoServiceError(f"Connection failed: {str(e)}")

    def _enforce_zero_or_one_query_results(self, collection: Collection, query: Dict):
        """Enforce that there aren't multiple query results."""
        count = collection.count_documents(query)
        
        if count > 1:
            raise MongoServiceError(f"Query matches {count} documents, expected exactly 1")
        return 

    def _get_latest_version(self, collection: Collection, query: Dict):
        """Get latest version."""
        pipeline = [
            {"$match": query},
            {
                "$addFields": {
                    "version_parts": {"$split": ["$version", "."]},
                }
            },
            {
                "$addFields": {
                    "major": {"$toInt": {"$arrayElemAt": ["$version_parts", 0]}},
                    "minor": {"$toInt": {"$arrayElemAt": ["$version_parts", 1]}}
                }
            },
            {"$sort": {"major": -1, "minor": -1}},
            {"$limit": 1},
            {"$project": {"version_parts": 0, "major": 0, "minor": 0}}
        ]

        ret = list(collection.aggregate(pipeline))
        if len(ret) > 1:
            raise MongoServiceError(f"Query matches {len(ret)} documents, expected exactly 1")

        return ret[0] if ret else None

    def fetch(self, artifact_type: str, query: Dict, get_latest: bool = False) -> Optional[Dict]:
        """Fetch an artifact by the query. Gets latest version of artifact if get_latest set to True, otherwise version should be specified for versioned artifacts."""
        try:
            if self.collection_prefix + artifact_type + 's' not in self.supported_collections:
                raise MongoServiceError(f"Unsupported artifact type: {artifact_type}")
            
            collection = self.db[self.collection_prefix + artifact_type + 's']

            result = None 
            if get_latest:
                result = self._get_latest_version(collection, query)
            else:
                self._enforce_zero_or_one_query_results(collection, query) 
                result = collection.find_one(query)
                
            if result:
                result['_id'] = str(result['_id'])
            return result
        except PyMongoError as e:
            raise MongoServiceError(f"Fetch failed: {str(e)}")

    def create(self, artifact_type: str, query: Dict) -> str:
        """Upload artifact metadata to MongoDB and return document ID."""
        try:
            if self.collection_prefix + artifact_type + 's' not in self.supported_collections:
                raise MongoServiceError(f"Unsupported artifact type: {artifact_type}")
            
            collection = self.db[self.collection_prefix + artifact_type + 's']
            result = collection.insert_one(query)
            return str(result.inserted_id)
            
        except DuplicateKeyError:
            raise MongoServiceError(f"{artifact_type} {name} v{version} already exists")
        except PyMongoError as e:
            raise MongoServiceError(f"Create failed: {str(e)}")

    def delete(self, artifact_type: str, query: Dict) -> None:
        """Delete artifact and return True if deleted, False if not found."""
        try:
            if self.collection_prefix + artifact_type + 's' not in self.supported_collections:
                raise MongoServiceError(f"Unsupported artifact type: {artifact_type}")
            collection = self.db[self.collection_prefix + artifact_type + 's']
            self._enforce_zero_or_one_query_results(collection, query) 
            result = collection.delete_one(query)
            if result.deleted_count == 0: 
                raise MongoServiceError(f"Delete failed because artifact not found: {query}")
            return 
        except PyMongoError as e:
            raise MongoServiceError(f"Delete failed: {str(e)}")

    def fetch_multiple(self, artifact_type: str, query: Dict, limit: int = 50) -> List[Dict]:
        """Fetch multiple artifacts based on the query."""
        try:
            if self.collection_prefix + artifact_type + 's' not in self.supported_collections:
                raise MongoServiceError(f"Unsupported artifact type: {artifact_type}")

            collection = self.db[self.collection_prefix + artifact_type + 's']
            
            results = []
            for doc in collection.find(query).limit(limit):
                doc['_id'] = str(doc['_id'])
                results.append(doc)
            return results
        except PyMongoError as e:
            raise MongoServiceError(f"List failed: {str(e)}")

    def update(self, artifact_type: str, query: Dict, updates: Dict) -> bool:
        """Updates artifact based on the query."""
        try:
            if self.collection_prefix + artifact_type + 's' not in self.supported_collections:
                raise MongoServiceError(f"Unsupported artifact type: {artifact_type}")

            if artifact_type in self.immutable_artifact_types:
                raise MongoServiceError(f"Can't update metadata for immutable artifact type: {artifact_type}")

            collection = self.db[self.collection_prefix + artifact_type + 's']
            self._enforce_zero_or_one_query_results(collection, query) 
            updates["updated_at"] = datetime.utcnow()
            result = collection.update_one(query, {"$set": updates})
            return result.modified_count > 0
           
        except PyMongoError as e:
            raise MongoServiceError(f"Update failed: {str(e)}")

    def exists(self, artifact_type: str, query: Dict) -> bool:
        """Check if artifact exists in MongoDB."""
        try:
            if self.collection_prefix + artifact_type + 's' not in self.supported_collections:
                raise MongoServiceError(f"Unsupported artifact type: {artifact_type}")

            collection = self.db[self.collection_prefix + artifact_type + 's']
            return collection.count_documents(query) > 0
        except PyMongoError as e:
            raise MongoServiceError(f"Exists check failed: {str(e)}")

    def close(self):
        """Close MongoDB connection."""
        if self.is_testing: 
            self.db.drop_collection("test_models")
            self.db.drop_collection("test_datasets")
            self.db.drop_collection("test_predictions")
            self.db.drop_collection("test_evaluations")
        if hasattr(self, 'client'):
            self.client.close()