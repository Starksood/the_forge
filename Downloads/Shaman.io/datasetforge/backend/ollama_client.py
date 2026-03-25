"""Ollama client — streaming + non-streaming, R1 think-block parsing."""
import json
import re
import logging
from typing import AsyncGenerator, Optional, Tuple, List
from json_repair import repair_json
import aiohttp
import asyncio

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"
GENERATION_TIMEOUT = 600  # 10 minutes — reasoning models need space

logger = logging.getLogger(__name__)

class OllamaConnectionError(Exception):
    """Raised when Ollama connection fails."""
    pass


class OllamaGenerationError(Exception):
    """Raised when generation fails."""
    pass


class OllamaClient:
    """
    Robust Ollama client with connection verification, streaming generation,
    and error handling. Includes DeepSeek R1 thinking block parsing.
    """
    
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        options: Optional[dict] = None,
    ):
        self.host = host
        self.model = model
        """Ollama generate options, e.g. {"num_ctx": 4096} — lowers RAM on small Macs."""
        self.options = options or {}
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None
    
    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create session for requests."""
        if self._session is None:
            # Increase read_bufsize to handle large JSON responses (e.g. from DeepSeek R1 or large datasets)
            self._session = aiohttp.ClientSession(read_bufsize=1024 * 1024)
        return self._session
    
    async def verify_connection(self) -> bool:
        """
        Verify Ollama backend connectivity.
        
        Returns:
            bool: True if Ollama is available, False otherwise
        """
        try:
            session = self._get_session()
            async with session.get(
                f"{self.host}/api/tags", 
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    # Also verify the model is available
                    data = await response.json()
                    models = [model.get("name", "") for model in data.get("models", [])]
                    
                    # Check for model match (handle :latest suffix)
                    model_base = self.model.split(":")[0]
                    for m in models:
                        m_base = m.split(":")[0]
                        if model_base == m_base or self.model == m:
                            logger.info(f"Ollama connection verified with model {m}")
                            # Update model name to the full tag
                            self.model = m
                            return True
                    
                    logger.warning(f"Model {self.model} not found in available models: {models}")
                    return False
                return False
        except Exception as e:
            logger.error(f"Ollama connection verification failed: {e}")
            return False
    
    async def generate_stream(self, prompt: str, model: Optional[str] = None, format: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from Ollama.
        
        Args:
            prompt: The input prompt
            model: Optional model override
            format: Optional format constraint (e.g. "json")
            
        Yields:
            str: Text chunks as they arrive
            
        Raises:
            OllamaConnectionError: If connection fails
            OllamaGenerationError: If generation fails
        """
        model = model or self.model
        payload = {"model": model, "prompt": prompt, "stream": True}
        if format:
            payload["format"] = format
        if self.options:
            payload["options"] = self.options
        
        try:
            session = self._get_session()
            async with session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    connect=10.0,
                    sock_read=600.0,    # wait up to 10 min for any read
                    sock_connect=10.0,
                    total=GENERATION_TIMEOUT
                ),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise OllamaGenerationError(f"HTTP {response.status}: {error_text}")
                
                async for line in response.content:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Check for errors in response
                        if "error" in data:
                            raise OllamaGenerationError(f"Ollama error: {data['error']}")
                        
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                        
                        if data.get("done"):
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON line: {line}, error: {e}")
                        continue
                        
        except aiohttp.ClientError as e:
            raise OllamaConnectionError(f"Connection error: {e}")
        except asyncio.TimeoutError:
            raise OllamaGenerationError("Generation timed out")
    
    async def generate_complete(self, prompt: str, model: Optional[str] = None, format: Optional[str] = None) -> str:
        """
        Generate complete response from Ollama.
        
        Args:
            prompt: The input prompt
            model: Optional model override
            format: Optional format constraint (e.g. "json")
            
        Returns:
            str: Complete generated response
            
        Raises:
            OllamaConnectionError: If connection fails
            OllamaGenerationError: If generation fails
        """
        result = []
        async for chunk in self.generate_stream(prompt, model, format):
            result.append(chunk)
        return "".join(result)
    
    def parse_thinking_blocks(self, response: str) -> Tuple[str, str]:
        """
        Parse DeepSeek R1 thinking blocks from response, handling unclosed tags.
        
        Args:
            response: Raw response from DeepSeek R1
            
        Returns:
            Tuple[str, str]: (thinking_trace, content)
        """
        thinking = ""
        content = response
        
        # 1. Try to find closed think blocks
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            # Remove thinking block(s) from content
            content = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        else:
            # 2. Handle unclosed think blocks (model truncated during thinking)
            if "<think>" in response:
                parts = response.split("<think>", 1)
                pre_think = parts[0].strip()
                post_think = parts[1].strip()
                thinking = post_think
                content = pre_think
        
        return thinking, content
    
    def parse_json_response(self, response: str) -> Tuple[str, Optional[dict]]:
        """
        Parse JSON from response, handling DeepSeek R1 thinking blocks.
        Uses json-repair for robustness.
        
        Args:
            response: Raw response that may contain thinking blocks and JSON
            
        Returns:
            Tuple[str, Optional[dict]]: (thinking_trace, parsed_json)
        """
        # Extract thinking trace using existing helper if needed, 
        # but the user pattern handles cleaning as well.
        thinking, _ = self.parse_thinking_blocks(response)
        
        try:
            # Strip R1 think blocks first
            cleaned = re.sub(
                r'<think>.*?</think>', 
                '', 
                response, 
                flags=re.DOTALL
            ).strip()

            # Find the outermost JSON object
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            if start == -1 or end == 0:
                logger.debug("No JSON object found in model output")
                return thinking, None
            
            json_str = cleaned[start:end]

            # Repair and parse — handles unescaped newlines, 
            # trailing commas, truncated output
            repaired = repair_json(json_str)
            logger.debug("JSON repair applied to model output")
            result = json.loads(repaired)
            return thinking, result
            
        except Exception as e:
            logger.debug(f"JSON parsing failed: {e}")
            return thinking, None

    async def get_embedding(self, text: str, model: str = "nomic-embed-text") -> List[float]:
        """
        Get embedding for text from Ollama.
        """
        payload = {"model": model, "prompt": text}
        try:
            session = self._get_session()
            async with session.post(
                f"{self.host}/api/embeddings",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    raise OllamaGenerationError(f"Embedding failed: {response.status}")
                data = await response.json()
                return data.get("embedding", [])
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            return []
    
    async def close(self):
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None


# Backward compatibility functions
async def ping(host: str = DEFAULT_HOST) -> bool:
    """Legacy function for backward compatibility."""
    client = OllamaClient(host=host)
    try:
        return await client.verify_connection()
    finally:
        await client.close()


async def stream_generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    options: Optional[dict] = None,
    format: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Legacy function for backward compatibility."""
    client = OllamaClient(host=host, model=model, options=options)
    try:
        async for chunk in client.generate_stream(prompt, format=format):
            yield chunk
    finally:
        await client.close()


async def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    options: Optional[dict] = None,
    format: Optional[str] = None,
) -> str:
    """Legacy function for backward compatibility."""
    client = OllamaClient(host=host, model=model, options=options)
    try:
        return await client.generate_complete(prompt, format=format)
    finally:
        await client.close()


def parse_think_and_json(raw: str) -> Tuple[str, Optional[dict]]:
    """Legacy function for backward compatibility."""
    client = OllamaClient()
    return client.parse_json_response(raw)


def generate_sync(prompt: str, model: str = DEFAULT_MODEL,
                  host: str = DEFAULT_HOST, format: str = "json") -> str:
    """Synchronous generate for use in Streamlit (no event loop)."""
    import httpx as _httpx
    with _httpx.Client(timeout=600.0) as client:
        r = client.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "format": format, "stream": False},
        )
        r.raise_for_status()
        return r.json().get("response", "")
