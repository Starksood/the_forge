"""Tests for OllamaClient class."""
import pytest
import json
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import aiohttp
from backend.ollama_client import (
    OllamaClient, 
    OllamaConnectionError, 
    OllamaGenerationError,
    ping,
    generate,
    parse_think_and_json
)


class TestOllamaClient:
    """Test cases for OllamaClient class."""
    
    @pytest.fixture
    def client(self):
        """Create OllamaClient instance for testing."""
        return OllamaClient(host="http://localhost:11434", model="gemma3:4b")
    
    def test_client_initialization(self, client):
        """Test OllamaClient initialization."""
        assert client.host == "http://localhost:11434"
        assert client.model == "gemma3:4b"
        assert client._session is None
    
    def test_parse_thinking_blocks_with_thinking(self, client):
        """Test parsing response with thinking blocks."""
        response = "<think>This is my reasoning process</think>This is the actual response"
        thinking, content = client.parse_thinking_blocks(response)
        
        assert thinking == "This is my reasoning process"
        assert content == "This is the actual response"
    
    def test_parse_thinking_blocks_without_thinking(self, client):
        """Test parsing response without thinking blocks."""
        response = "This is just a regular response"
        thinking, content = client.parse_thinking_blocks(response)
        
        assert thinking == ""
        assert content == "This is just a regular response"
    
    def test_parse_thinking_blocks_multiple_blocks(self, client):
        """Test parsing response with multiple thinking blocks (all are removed from content)."""
        response = "<think>First thought</think>Content<think>Second thought</think>More content"
        thinking, content = client.parse_thinking_blocks(response)
        
        assert thinking == "First thought"
        assert content == "ContentMore content"  # All think blocks removed
    
    def test_parse_json_response_with_thinking(self, client):
        """Test parsing JSON response with thinking blocks."""
        response = '<think>Let me think about this</think>{"key": "value", "number": 42}'
        thinking, parsed = client.parse_json_response(response)
        
        assert thinking == "Let me think about this"
        assert parsed == {"key": "value", "number": 42}
    
    def test_parse_json_response_with_markdown(self, client):
        """Test parsing JSON response with markdown code fences."""
        response = '<think>Processing</think>```json\n{"result": "success"}\n```'
        thinking, parsed = client.parse_json_response(response)
        
        assert thinking == "Processing"
        assert parsed == {"result": "success"}
    
    def test_parse_json_response_invalid_json(self, client):
        """Test parsing invalid JSON response."""
        response = '<think>Hmm</think>This is not valid JSON'
        thinking, parsed = client.parse_json_response(response)
        
        assert thinking == "Hmm"
        assert parsed is None
    
    def test_parse_json_response_extract_json_block(self, client):
        """Test extracting JSON from mixed content."""
        response = '<think>Working</think>Here is the result: {"status": "ok"} and some more text'
        thinking, parsed = client.parse_json_response(response)
        
        assert thinking == "Working"
        assert parsed == {"status": "ok"}
    
    def test_parse_json_response_nested_json(self, client):
        """Test parsing nested JSON structures."""
        response = '<think>Complex data</think>{"user": {"name": "test", "data": [1, 2, 3]}}'
        thinking, parsed = client.parse_json_response(response)
        
        assert thinking == "Complex data"
        assert parsed == {"user": {"name": "test", "data": [1, 2, 3]}}
    
    def test_parse_json_response_no_thinking_valid_json(self, client):
        """Test parsing JSON without thinking blocks."""
        response = '{"simple": "json"}'
        thinking, parsed = client.parse_json_response(response)
        
        assert thinking == ""
        assert parsed == {"simple": "json"}
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using OllamaClient as async context manager."""
        async with OllamaClient() as client:
            assert client._session is not None
        # Session should be closed after exiting context
    
    @pytest.mark.asyncio
    async def test_close_method(self, client):
        """Test close method."""
        # Create a mock session
        mock_session = AsyncMock()
        client._session = mock_session
        await client.close()
        mock_session.close.assert_called_once()
        assert client._session is None
    
    @pytest.mark.asyncio
    async def test_close_method_no_session(self, client):
        """Test close method when no session exists."""
        assert client._session is None
        await client.close()  # Should not raise an error
        assert client._session is None


class TestBackwardCompatibility:
    """Test backward compatibility functions."""
    
    @pytest.mark.asyncio
    async def test_ping_function(self):
        """Test legacy ping function."""
        with patch('backend.ollama_client.OllamaClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.verify_connection.return_value = True
            mock_client_class.return_value = mock_client
            
            result = await ping("http://localhost:11434")
            assert result is True
            mock_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_function(self):
        """Test legacy generate function."""
        with patch('backend.ollama_client.OllamaClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_complete.return_value = "Generated response"
            mock_client_class.return_value = mock_client
            
            result = await generate("Test prompt", "llama3.2:3b", "http://localhost:11434")
            assert result == "Generated response"
            mock_client.close.assert_called_once()
    
    def test_parse_think_and_json_function(self):
        """Test legacy parse_think_and_json function."""
        response = '<think>Thinking</think>{"data": "test"}'
        thinking, parsed = parse_think_and_json(response)
        
        assert thinking == "Thinking"
        assert parsed == {"data": "test"}
    
    def test_parse_think_and_json_complex(self):
        """Test legacy function with complex JSON."""
        response = '<think>Complex reasoning</think>```json\n{"items": [{"id": 1, "name": "test"}]}\n```'
        thinking, parsed = parse_think_and_json(response)
        
        assert thinking == "Complex reasoning"
        assert parsed == {"items": [{"id": 1, "name": "test"}]}


class TestErrorScenarios:
    """Test error handling scenarios."""
    
    def test_malformed_json_parsing(self):
        """Test handling of malformed JSON (repaired by json-repair)."""
        client = OllamaClient()
        
        # Test with broken JSON
        response = '<think>Trying to parse</think>{"broken": json}'
        thinking, parsed = client.parse_json_response(response)
        
        assert thinking == "Trying to parse"
        assert parsed == {"broken": "json"}  # json-repair fixes it
    
    def test_empty_response_parsing(self):
        """Test handling of empty responses."""
        client = OllamaClient()
        
        # Test with empty response
        response = ""
        thinking, parsed = client.parse_json_response(response)
        
        assert thinking == ""
        assert parsed is None
    
    def test_thinking_block_edge_cases(self):
        """Test edge cases in thinking block parsing."""
        client = OllamaClient()
        
        # Test with nested angle brackets
        response = "<think>I think <this> is confusing</think>Final answer"
        thinking, content = client.parse_thinking_blocks(response)
        
        assert thinking == "I think <this> is confusing"
        assert content == "Final answer"
    
    def test_json_extraction_edge_cases(self):
        """Test edge cases in JSON extraction."""
        client = OllamaClient()
        
        # Multiple objects: result is an array of objects
        response = 'First {"fake": "json"} then {"real": "json"} finally'
        thinking, parsed = client.parse_json_response(response)
        
        assert thinking == ""
        assert parsed == [{"fake": "json"}, {"real": "json"}]


class TestClientConfiguration:
    """Test client configuration options."""
    
    def test_custom_host_and_model(self):
        """Test client with custom host and model."""
        client = OllamaClient(host="http://custom:8080", model="custom-model")
        
        assert client.host == "http://custom:8080"
        assert client.model == "custom-model"
    
    def test_default_configuration(self):
        """Test client with default configuration."""
        client = OllamaClient()
        
        assert client.host == "http://localhost:11434"
        assert client.model == "gemma3:4b"