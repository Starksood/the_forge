"""Integration tests for OllamaClient functionality."""
import pytest
from backend.ollama_client import OllamaClient


class TestOllamaIntegration:
    """Integration tests for OllamaClient."""
    
    def test_client_can_be_instantiated(self):
        """Test that OllamaClient can be instantiated with default settings."""
        client = OllamaClient()
        assert client.host == "http://localhost:11434"
        assert client.model == "llama3.2:3b"
    
    def test_client_can_be_instantiated_with_custom_settings(self):
        """Test that OllamaClient can be instantiated with custom settings."""
        client = OllamaClient(host="http://custom:8080", model="custom-model")
        assert client.host == "http://custom:8080"
        assert client.model == "custom-model"
    
    def test_thinking_block_parsing_comprehensive(self):
        """Test comprehensive thinking block parsing scenarios."""
        client = OllamaClient()
        
        # Test various DeepSeek R1 response formats
        test_cases = [
            # Standard thinking block
            ("<think>I need to analyze this</think>Final answer", "I need to analyze this", "Final answer"),
            
            # No thinking block
            ("Just a direct answer", "", "Just a direct answer"),
            
            # Empty thinking block
            ("<think></think>Answer", "", "Answer"),
            
            # Thinking block with newlines
            ("<think>\nMultiline\nthinking\n</think>Result", "Multiline\nthinking", "Result"),
            
            # Thinking block with special characters
            ("<think>Thinking with $pecial ch@rs!</think>Output", "Thinking with $pecial ch@rs!", "Output"),
        ]
        
        for response, expected_thinking, expected_content in test_cases:
            thinking, content = client.parse_thinking_blocks(response)
            assert thinking == expected_thinking, f"Failed for response: {response}"
            assert content == expected_content, f"Failed for response: {response}"
    
    def test_json_parsing_comprehensive(self):
        """Test comprehensive JSON parsing scenarios."""
        client = OllamaClient()
        
        # Test various JSON response formats
        test_cases = [
            # Standard JSON with thinking
            ('<think>Processing</think>{"result": "success"}', "Processing", {"result": "success"}),
            
            # JSON with markdown fences
            ('<think>Working</think>```json\n{"data": [1, 2, 3]}\n```', "Working", {"data": [1, 2, 3]}),
            
            # Plain JSON without thinking
            ('{"simple": true}', "", {"simple": True}),
            
            # Complex nested JSON
            ('{"user": {"id": 1, "prefs": {"theme": "dark"}}}', "", {"user": {"id": 1, "prefs": {"theme": "dark"}}}),
            
            # Invalid JSON
            ('<think>Hmm</think>Not valid JSON', "Hmm", None),
            
            # Empty response
            ('', "", None),
        ]
        
        for response, expected_thinking, expected_json in test_cases:
            thinking, parsed = client.parse_json_response(response)
            assert thinking == expected_thinking, f"Failed thinking for: {response}"
            assert parsed == expected_json, f"Failed JSON for: {response}"
    
    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self):
        """Test the full lifecycle of OllamaClient as context manager."""
        # Test that context manager properly manages session lifecycle
        async with OllamaClient() as client:
            # Session should be created
            assert client._session is not None
            
            # Client should be usable
            assert client.host == "http://localhost:11434"
            assert client.model == "llama3.2:3b"
            
            # Parsing should work
            thinking, content = client.parse_thinking_blocks("<think>test</think>result")
            assert thinking == "test"
            assert content == "result"
        
        # Session should be closed after context exit
        # Note: We can't easily test this without mocking, but the structure is correct
    
    def test_error_handling_robustness(self):
        """Test that the client handles various error conditions gracefully."""
        client = OllamaClient()
        
        # Test malformed thinking blocks
        malformed_cases = [
            "<think>Unclosed thinking block",
            "think>Missing opening bracket</think>",
            "<think>Nested <think>blocks</think></think>",
            "<THINK>Wrong case</THINK>Answer",  # Should not match
        ]
        
        for case in malformed_cases:
            thinking, content = client.parse_thinking_blocks(case)
            # Should handle gracefully without crashing
            assert isinstance(thinking, str)
            assert isinstance(content, str)
        
        # Test malformed JSON
        json_cases = [
            '{"unclosed": "json"',
            '{"trailing": "comma",}',
            '{invalid: json}',
            'null',  # Valid JSON but not an object
        ]
        
        for case in json_cases:
            thinking, parsed = client.parse_json_response(case)
            # Should handle gracefully without crashing
            assert isinstance(thinking, str)
            # parsed can be None or a valid object, but should not crash
    
    def test_backward_compatibility_maintained(self):
        """Test that backward compatibility functions still work."""
        from backend.ollama_client import parse_think_and_json
        
        # Test the legacy function
        response = '<think>Legacy test</think>{"legacy": true}'
        thinking, parsed = parse_think_and_json(response)
        
        assert thinking == "Legacy test"
        assert parsed == {"legacy": True}
        
        # Test with the new class method for comparison
        client = OllamaClient()
        thinking2, parsed2 = client.parse_json_response(response)
        
        assert thinking == thinking2
        assert parsed == parsed2