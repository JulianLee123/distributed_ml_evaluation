"""
Simple test suite for S3Service class.
Run with: pytest test_s3_service.py -v
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv()
from storage.s3_service import S3Service


@pytest.fixture(scope="session")
def s3_client():
    """Initialize S3 client for the entire test session."""
    service = S3Service(is_testing=True)  # Use test prefix
    yield service
    service.close()


def test_upload_two_files(s3_client):
    """Test uploading two files is successful."""
    # Create two temporary files with different content
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as file1:
        file1.write("This is test file 1")
        file1_path = file1.name
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as file2:
        file2.write("This is test file 2")
        file2_path = file2.name
    
    try:
        # Upload both files
        s3_client.upload_file(file1_path, "test_file_1.txt")
        s3_client.upload_file(file2_path, "test_file_2.txt")
        
        # Verify files exist in S3
        assert s3_client.file_exists("test_file_1.txt")
        assert s3_client.file_exists("test_file_2.txt")
        
    finally:
        # Clean up local files
        os.unlink(file1_path)
        os.unlink(file2_path)
        # Clean up S3 files (in case other tests fail)
        try:
            s3_client.delete_file("test_file_1.txt")
            s3_client.delete_file("test_file_2.txt")
        except:
            pass


def test_upload_and_download_two_files(s3_client):
    """Test uploading and then downloading two files is successful."""
    # Create two temporary files with different content
    file1_content = "Content for download test file 1"
    file2_content = "Content for download test file 2"
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as file1:
        file1.write(file1_content)
        file1_path = file1.name
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as file2:
        file2.write(file2_content)
        file2_path = file2.name
    
    try:
        # Upload both files
        s3_client.upload_file(file1_path, "download_test_1.txt")
        s3_client.upload_file(file2_path, "download_test_2.txt")
        
        # Download both files
        downloaded_file1 = s3_client.download_file("download_test_1.txt")
        downloaded_file2 = s3_client.download_file("download_test_2.txt")
        
        # Verify downloaded content matches original
        with open(downloaded_file1, 'r') as f:
            assert f.read() == file1_content
        
        with open(downloaded_file2, 'r') as f:
            assert f.read() == file2_content
            
    finally:
        # Clean up local files
        os.unlink(file1_path)
        os.unlink(file2_path)
        if 'downloaded_file1' in locals():
            os.unlink(downloaded_file1)
        if 'downloaded_file2' in locals():
            os.unlink(downloaded_file2)
        # Clean up S3 files
        try:
            s3_client.delete_file("download_test_1.txt")
            s3_client.delete_file("download_test_2.txt")
        except:
            pass


def test_upload_and_list_two_files(s3_client):
    """Test uploading two files and then listing them is successful."""
    # Create two temporary files
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as file1:
        file1.write("List test file 1")
        file1_path = file1.name
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as file2:
        file2.write("List test file 2")
        file2_path = file2.name
    
    try:
        # List objects when no matches for the prefix
        objects = s3_client.list_objects(prefix="list_test/")
        
        assert len(objects) == 0

        # Upload both files with a common prefix for easier listing
        s3_client.upload_file(file1_path, "list_test/file_1.txt")
        s3_client.upload_file(file2_path, "list_test/file_2.txt")
        
        # List objects with the test prefix
        objects = s3_client.list_objects(prefix="list_test/")
        
        # Verify both files are in the list
        file_keys = [obj['Key'] for obj in objects]
        
        # Account for the test prefix that gets added automatically
        expected_keys = []
        for key in file_keys:
            if key.endswith("list_test/file_1.txt") or key.endswith("list_test/file_2.txt"):
                expected_keys.append(key)
        
        assert len(expected_keys) == 2
        assert any("file_1.txt" in key for key in expected_keys)
        assert any("file_2.txt" in key for key in expected_keys)
        
    finally:
        # Clean up local files
        os.unlink(file1_path)
        os.unlink(file2_path)
        # Clean up S3 files
        try:
            s3_client.delete_file("list_test/file_1.txt")
            s3_client.delete_file("list_test/file_2.txt")
        except:
            pass


def test_upload_and_delete_two_files(s3_client):
    """Test uploading two files and deleting them is successful."""
    # Create two temporary files
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as file1:
        file1.write("Delete test file 1")
        file1_path = file1.name
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as file2:
        file2.write("Delete test file 2")
        file2_path = file2.name
    
    try:
        # Upload both files
        s3_client.upload_file(file1_path, "delete_test_1.txt")
        s3_client.upload_file(file2_path, "delete_test_2.txt")
        
        # Verify files exist before deletion
        assert s3_client.file_exists("delete_test_1.txt")
        assert s3_client.file_exists("delete_test_2.txt")
        
        # Delete both files
        s3_client.delete_file("delete_test_1.txt")
        s3_client.delete_file("delete_test_2.txt")
        
        # Verify files no longer exist
        assert not s3_client.file_exists("delete_test_1.txt")
        assert not s3_client.file_exists("delete_test_2.txt")
        
    finally:
        # Clean up local files
        os.unlink(file1_path)
        os.unlink(file2_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])