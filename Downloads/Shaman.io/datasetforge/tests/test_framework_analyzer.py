"""Unit tests for FrameworkAnalyzer class."""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from backend.phases.phase0_analysis import FrameworkAnalyzer
from backend.ollama_client import OllamaClient


@pytest.fixture
def mock_ollama_client():
    """Create a mock OllamaClient for testing."""
    client = MagicMock(spec=OllamaClient)
    return client


@pytest.fixture
def sample_document_text():
    """Sample document text for testing."""
    return """
    Chapter 1: Introduction to Psychedelic States
    
    The dissolution phase is the initial stage where ego boundaries begin to soften.
    This is followed by the integration phase where insights are consolidated.
    """


@pytest.fixture
def sample_analysis_response():
    """Sample analysis response from Ollama."""
    return {
        "taxonomy": [
            {
                "name": "Dissolution Phase",
                "definition": "Initial stage where ego boundaries begin to soften",
                "source_section": "Chapter 1"
            },
            {
                "name": "Integration Phase",
                "definition": "Stage where insights are consolidated",
                "source_section": "Chapter 1"
            }
        ],
        "framework_summary": "The framework describes a journey from dissolution to integration.",
        "relationship_map": [
            {
                "from": "Dissolution Phase",
                "to": "Integration Phase",
                "connection": "Dissolution naturally leads to integration"
            }
        ]
    }


@pytest.mark.asyncio
async def test_analyze_document_success(mock_ollama_client, sample_document_text, sample_analysis_response):
    """Test successful document analysis."""
    # Setup mock to return chunks and parse JSON correctly
    async def mock_stream():
        yield "<think>Analyzing document...</think>"
        yield json.dumps(sample_analysis_response)
    
    mock_ollama_client.generate_stream = MagicMock(return_value=mock_stream())
    mock_ollama_client.parse_json_response = MagicMock(
        return_value=("Analyzing document...", sample_analysis_response)
    )
    
    # Create analyzer and run analysis
    analyzer = FrameworkAnalyzer(mock_ollama_client)
    result = await analyzer.analyze_document(sample_document_text)
    
    # Verify results
    assert "taxonomy" in result
    assert "framework_summary" in result
    assert "relationships" in result
    assert "raw_analysis" in result
    assert "thinking_trace" in result
    
    assert len(result["taxonomy"]) == 2
    assert result["taxonomy"][0]["name"] == "Dissolution Phase"
    assert result["framework_summary"] == "The framework describes a journey from dissolution to integration."
    assert len(result["relationships"]) == 1


@pytest.mark.asyncio
async def test_analyze_document_with_stream_callback(mock_ollama_client, sample_document_text, sample_analysis_response):
    """Test document analysis with streaming callback."""
    # Setup mock
    async def mock_stream():
        yield "<think>"
        yield "Analyzing..."
        yield "</think>"
        yield json.dumps(sample_analysis_response)
    
    mock_ollama_client.generate_stream = MagicMock(return_value=mock_stream())
    mock_ollama_client.parse_json_response = MagicMock(
        return_value=("Analyzing...", sample_analysis_response)
    )
    
    # Track callback invocations
    callback_calls = []
    
    async def stream_callback(msg_type, text):
        callback_calls.append((msg_type, text))
    
    # Create analyzer and run analysis
    analyzer = FrameworkAnalyzer(mock_ollama_client)
    result = await analyzer.analyze_document(sample_document_text, stream_callback)
    
    # Verify callback was called
    assert len(callback_calls) > 0
    assert result is not None


@pytest.mark.asyncio
async def test_analyze_document_json_parse_failure(mock_ollama_client, sample_document_text):
    """Test handling of JSON parsing failure."""
    # Setup mock to return invalid JSON
    async def mock_stream():
        yield "Invalid JSON response"
    
    mock_ollama_client.generate_stream = MagicMock(return_value=mock_stream())
    mock_ollama_client.parse_json_response = MagicMock(return_value=("", None))
    
    # Create analyzer and expect exception
    analyzer = FrameworkAnalyzer(mock_ollama_client)
    
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        await analyzer.analyze_document(sample_document_text)


def test_extract_taxonomy(mock_ollama_client, sample_analysis_response):
    """Test taxonomy extraction from analysis result."""
    analyzer = FrameworkAnalyzer(mock_ollama_client)
    
    taxonomy = analyzer.extract_taxonomy(sample_analysis_response)
    
    assert len(taxonomy) == 2
    assert taxonomy[0]["name"] == "Dissolution Phase"
    assert taxonomy[1]["name"] == "Integration Phase"


def test_extract_relationships(mock_ollama_client):
    """Test relationship extraction from analysis result."""
    analyzer = FrameworkAnalyzer(mock_ollama_client)
    
    # Create a result dict with relationships key (not relationship_map)
    analysis_result = {
        "relationships": [
            {
                "from": "Dissolution Phase",
                "to": "Integration Phase",
                "connection": "Dissolution naturally leads to integration"
            }
        ]
    }
    
    relationships = analyzer.extract_relationships(analysis_result)
    
    assert len(relationships) == 1
    assert relationships[0]["from"] == "Dissolution Phase"
    assert relationships[0]["to"] == "Integration Phase"


@pytest.mark.asyncio
async def test_analyze_and_store(mock_ollama_client, sample_document_text, sample_analysis_response):
    """Test document analysis with database storage."""
    # Setup mocks
    async def mock_stream():
        yield json.dumps(sample_analysis_response)
    
    mock_ollama_client.generate_stream = MagicMock(return_value=mock_stream())
    mock_ollama_client.parse_json_response = MagicMock(
        return_value=("Thinking...", sample_analysis_response)
    )
    
    mock_db_manager = MagicMock()
    mock_db_manager.create_framework_analysis = MagicMock(return_value="analysis_123")
    
    # Create analyzer and run analysis with storage
    analyzer = FrameworkAnalyzer(mock_ollama_client)
    analysis_id = await analyzer.analyze_and_store(
        document_id="doc_123",
        document_text=sample_document_text,
        db_manager=mock_db_manager
    )
    
    # Verify database was called
    assert analysis_id.startswith("analysis_")
    mock_db_manager.create_framework_analysis.assert_called_once()
    
    # Verify the call arguments
    call_args = mock_db_manager.create_framework_analysis.call_args
    assert call_args.kwargs["document_id"] == "doc_123"
    assert call_args.kwargs["taxonomy"] == sample_analysis_response["taxonomy"]
    assert call_args.kwargs["framework_summary"] == sample_analysis_response["framework_summary"]
    assert call_args.kwargs["relationships"] == sample_analysis_response["relationship_map"]
