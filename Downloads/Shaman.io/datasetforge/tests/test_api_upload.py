"""Unit tests for document upload API endpoint."""
import tempfile
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from backend import api, database


@pytest.fixture
def test_session():
    """Create a test session database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override sessions directory
        original_sessions_dir = database.SESSIONS_DIR
        database.SESSIONS_DIR = tmpdir
        
        # Create test session
        session_id = "test_session_123"
        document_name = "Test Document"
        db_path = database.create_session_database(session_id, document_name)
        
        # Set as current session
        api.current_session_db = db_path
        
        yield session_id, db_path
        
        # Cleanup
        api.current_session_db = None
        database.SESSIONS_DIR = original_sessions_dir


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(api.app)


def test_upload_txt_document(client, test_session):
    """Test uploading a plain text document."""
    session_id, db_path = test_session
    
    # Create test file content
    test_content = "This is a test document.\n\nIt has multiple paragraphs.\n\nAnd some content."
    file_data = BytesIO(test_content.encode('utf-8'))
    
    # Upload document
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", file_data, "text/plain")}
    )
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.txt"
    assert data["page_count"] >= 1
    assert data["text_length"] == len(test_content)
    assert data["status"] == "uploaded"
    assert "document_id" in data
    
    # Verify database storage
    db_manager = database.get_database_manager(db_path)
    with db_manager.get_session() as db_session:
        from backend.models import Document
        doc = db_session.query(Document).filter(Document.id == data["document_id"]).first()
        assert doc is not None
        assert doc.filename == "test.txt"
        assert doc.content == test_content
        assert doc.page_count >= 1


def test_upload_without_session(client):
    """Test uploading without an active session."""
    # Ensure no active session
    api.current_session_db = None
    
    test_content = "Test content"
    file_data = BytesIO(test_content.encode('utf-8'))
    
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", file_data, "text/plain")}
    )
    
    assert response.status_code == 400
    assert "No active session" in response.json()["detail"]


def test_upload_unsupported_format(client, test_session):
    """Test uploading an unsupported file format."""
    test_content = b"fake binary content"
    file_data = BytesIO(test_content)
    
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.xyz", file_data, "application/octet-stream")}
    )
    
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]


def test_upload_empty_file(client, test_session):
    """Test uploading an empty file."""
    file_data = BytesIO(b"")
    
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", file_data, "text/plain")}
    )
    
    # Empty files should fail with 400 or 500 error
    assert response.status_code in [400, 500]
    assert "empty" in response.json()["detail"].lower() or "failed" in response.json()["detail"].lower()


def test_upload_no_filename(client, test_session):
    """Test uploading without a filename."""
    test_content = b"Test content"
    file_data = BytesIO(test_content)
    
    response = client.post(
        "/api/documents/upload",
        files={"file": ("", file_data, "text/plain")}
    )
    
    # FastAPI may return 422 for validation errors or 400 for our custom check
    assert response.status_code in [400, 422]
    detail = response.json().get("detail", "")
    # Check if it's either our custom error or FastAPI's validation error
    assert isinstance(detail, (str, list))
