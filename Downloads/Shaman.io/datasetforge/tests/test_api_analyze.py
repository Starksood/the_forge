"""Unit tests for framework analysis API endpoint."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from backend.api import app
from backend.database import DatabaseManager, create_session_database
from backend.models import SessionStatus


@pytest.fixture
def temp_session():
    """Create a temporary session for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override sessions directory
        import backend.database as db_module
        original_sessions_dir = db_module.SESSIONS_DIR
        db_module.SESSIONS_DIR = tmpdir
        
        # Create session
        session_id = "test_session_123"
        document_name = "Test Document"
        db_path = create_session_database(session_id, document_name)
        
        # Create document
        db_manager = DatabaseManager(db_path)
        document_id = "doc_123"
        db_manager.create_document(
            document_id=document_id,
            session_id=session_id,
            filename="test.txt",
            content="This is a test document about psychedelic experiences.",
            page_count=1
        )
        
        yield {
            "session_id": session_id,
            "document_id": document_id,
            "db_path": db_path,
            "db_manager": db_manager
        }
        
        # Restore original sessions directory
        db_module.SESSIONS_DIR = original_sessions_dir


@pytest.mark.asyncio
async def test_analyze_document_success(temp_session):
    """Test successful framework analysis."""
    client = TestClient(app)
    
    # Set current session
    import backend.api as api_module
    api_module.current_session_db = temp_session["db_path"]
    
    # Mock the OllamaClient and FrameworkAnalyzer
    mock_analysis_result = {
        "taxonomy": [
            {"name": "Entity1", "definition": "Definition 1", "source_section": "Section 1"}
        ],
        "framework_summary": "Test framework summary",
        "relationships": [
            {"from": "Section A", "to": "Section B", "connection": "Test connection"}
        ],
        "raw_analysis": "Raw analysis text",
        "thinking_trace": "Thinking trace"
    }
    
    with patch('backend.api.OllamaClient') as mock_ollama_class:
        # Setup mock
        mock_ollama_instance = AsyncMock()
        mock_ollama_class.return_value.__aenter__.return_value = mock_ollama_instance
        
        with patch('backend.phases.phase0_analysis.FrameworkAnalyzer') as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_and_store = AsyncMock(return_value="analysis_123")
            mock_analyzer_class.return_value = mock_analyzer
            
            # Store the analysis in the database manually for the test
            temp_session["db_manager"].create_framework_analysis(
                analysis_id="analysis_123",
                document_id=temp_session["document_id"],
                taxonomy=mock_analysis_result["taxonomy"],
                framework_summary=mock_analysis_result["framework_summary"],
                relationships=mock_analysis_result["relationships"],
                raw_analysis=mock_analysis_result["raw_analysis"]
            )
            
            # Make request
            response = client.post(f"/api/documents/{temp_session['document_id']}/analyze")
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["analysis_id"] == "analysis_123"
            assert data["document_id"] == temp_session["document_id"]
            assert data["status"] == "complete"
            assert len(data["taxonomy"]) == 1
            assert data["framework_summary"] == "Test framework summary"
            assert len(data["relationships"]) == 1


@pytest.mark.asyncio
async def test_analyze_document_no_session():
    """Test analysis fails when no session is active."""
    client = TestClient(app)
    
    # Clear current session
    import backend.api as api_module
    api_module.current_session_db = None
    
    # Make request
    response = client.post("/api/documents/doc_123/analyze")
    
    # Verify error
    assert response.status_code == 400
    assert "No active session" in response.json()["detail"]


@pytest.mark.asyncio
async def test_analyze_document_not_found(temp_session):
    """Test analysis fails when document doesn't exist."""
    client = TestClient(app)
    
    # Set current session
    import backend.api as api_module
    api_module.current_session_db = temp_session["db_path"]
    
    # Make request with non-existent document ID
    response = client.post("/api/documents/nonexistent_doc/analyze")
    
    # Verify error
    assert response.status_code == 404
    assert "Document not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_analyze_document_stores_results(temp_session):
    """Test that analysis results are stored in database."""
    client = TestClient(app)
    
    # Set current session
    import backend.api as api_module
    api_module.current_session_db = temp_session["db_path"]
    
    # Mock the analysis
    mock_analysis_result = {
        "taxonomy": [{"name": "Test", "definition": "Test def", "source_section": "Sec 1"}],
        "framework_summary": "Summary",
        "relationships": [],
        "raw_analysis": "Raw",
        "thinking_trace": "Trace"
    }
    
    with patch('backend.api.OllamaClient') as mock_ollama_class:
        mock_ollama_instance = AsyncMock()
        mock_ollama_class.return_value.__aenter__.return_value = mock_ollama_instance
        
        with patch('backend.phases.phase0_analysis.FrameworkAnalyzer') as mock_analyzer_class:
            mock_analyzer = AsyncMock()
            
            # Mock analyze_and_store to actually store in database
            async def mock_analyze_and_store(document_id, document_text, db_manager, stream_callback=None):
                analysis_id = "analysis_456"
                db_manager.create_framework_analysis(
                    analysis_id=analysis_id,
                    document_id=document_id,
                    taxonomy=mock_analysis_result["taxonomy"],
                    framework_summary=mock_analysis_result["framework_summary"],
                    relationships=mock_analysis_result["relationships"],
                    raw_analysis=mock_analysis_result["raw_analysis"]
                )
                return analysis_id
            
            mock_analyzer.analyze_and_store = mock_analyze_and_store
            mock_analyzer_class.return_value = mock_analyzer
            
            # Make request
            response = client.post(f"/api/documents/{temp_session['document_id']}/analyze")
            
            # Verify response
            assert response.status_code == 200
            
            # Verify data is stored in database
            with temp_session["db_manager"].get_session() as db_session:
                from backend.models import FrameworkAnalysis
                analysis = db_session.query(FrameworkAnalysis).filter(
                    FrameworkAnalysis.document_id == temp_session["document_id"]
                ).first()
                
                assert analysis is not None
                assert analysis.framework_summary == "Summary"
                assert len(analysis.taxonomy_json) == 1
