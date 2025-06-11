# shared_storage/schema_validator.py
import json
import os
from pathlib import Path
from pymongo.database import Database
from pymongo.errors import OperationFailure

class SchemaManager:
    def __init__(self, database: Database, is_testing: bool = False,schema_dir: str = "schemas"):
        self.db = database
        self.schema_dir = Path(schema_dir)
        self.schemas = self._load_schemas()
        self.indexes = self._load_indexes()
        if is_testing:
            self.collection_prefix = "test_"
        else:
            self.collection_prefix = ""
    
    def _load_schemas(self):
        """Load all collection schemas from individual files"""
        schemas = {}
        if not self.schema_dir.exists():
            raise FileNotFoundError(f"Schema directory not found: {self.schema_dir}")
        
        for collection_dir in self.schema_dir.iterdir():
            if collection_dir.is_dir():
                schema_file = collection_dir / "schema.json"
                if schema_file.exists():
                    with open(schema_file, 'r') as f:
                        schemas[collection_dir.name] = json.load(f)
        return schemas
    
    def _load_indexes(self):
        """Load all collection indexes from individual files"""
        indexes = {}
        for collection_dir in self.schema_dir.iterdir():
            if collection_dir.is_dir():
                index_file = collection_dir / "indexes.json"
                if index_file.exists():
                    with open(index_file, 'r') as f:
                        index_data = json.load(f)
                        # Convert to list of tuples for pymongo
                        indexes[collection_dir.name] = [(field, direction) for field, direction in index_data["unique_index"]]
        return indexes
    
    def get_supported_collections(self):
        """Return list of all supported collection names"""
        return list(self.schemas.keys())
    
    def apply_schemas(self):
        """Apply schemas and indexes for all collections"""
        for name, schema in self.schemas.items():
            try:
                name = self.collection_prefix + name
                self.db.run_command({
                    "collMod": name,
                    "validator": {"$jsonSchema": schema},
                    "validationAction": "error"
                })
            except OperationFailure:
                self.db.create_collection(name, validator={"$jsonSchema": schema})
            
            # Apply indexes if they exist
            if name in self.indexes:
                self.db[name].create_index(self.indexes[name], unique=True)