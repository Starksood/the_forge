"""Phase 3 — Cross-reference triples from relationship map."""
import json
from typing import AsyncGenerator, Optional, Any
from ..ollama_client import generate, parse_think_and_json, OllamaClient
from ..settings_store import get_settings

CROSSREF_PROMPT = """KNOWLEDGE CONTEXT:
{KNOWLEDGE_CONTEXT}

PASSAGE A:
{chunk_a_text}

PASSAGE B:
{chunk_b_text}

CONNECTION: {connection}

INSTRUCTIONS:
- These two passages are linked. Generate ONE bridging triple.
- Complete the JSON triple based on these passages.
- Only replace the null values. Follow JSON syntax precisely.

JSON TEMPLATE TO COMPLETE:
{
  "system": null,
  "user": null,
  "assistant": null
}"""


async def run_phase3_streaming(
    relationship_map: list[dict],
    chunks: list[dict],
    knowledge_context: str,
    model: str,
    host: str,
) -> AsyncGenerator[str, None]:
    """
    Yields SSE lines:
      data: {"type": "progress", "index": N, "total": M, "from": "...", "to": "..."}
      data: {"type": "triple", "chunk_index": N, "triple": {...}}
      data: {"type": "done"}
    """
    total = len(relationship_map)
    all_texts = [c.get("text", c.get("content", "")) for c in chunks]

    for i, rel in enumerate(relationship_map):
        yield _sse({
            "type": "progress",
            "index": i + 1,
            "total": total,
            "from": rel.get("from", ""),
            "to": rel.get("to", ""),
        })

        # Find best matching chunks by keyword overlap
        chunk_a_text = _find_best_chunk(rel.get("from", ""), all_texts)
        chunk_b_text = _find_best_chunk(rel.get("to", ""), all_texts)

        prompt = (
            CROSSREF_PROMPT
            .replace("{KNOWLEDGE_CONTEXT}", knowledge_context)
            .replace("{chunk_a_text}", chunk_a_text)
            .replace("{chunk_b_text}", chunk_b_text)
            .replace("{connection}", rel.get("connection", ""))
        )

        ctx = get_settings().get("ollama_num_ctx")
        opts = {"num_ctx": int(ctx)} if ctx else None
        
        # Use constrained JSON format
        raw = await generate(prompt, model, host, opts, format="json")
        think, parsed = parse_think_and_json(raw)

        if parsed and all(k in parsed for k in ("system", "user", "assistant")):
            triple = {
                "system_text": parsed["system"],
                "user_text": parsed["user"],
                "assistant_text": parsed["assistant"],
                "think_block": think,
                "status": "pending",
                "is_cross_reference": 1,
                "tags": "cross-reference",
            }
        else:
            triple = {
                "system_text": "",
                "user_text": "",
                "assistant_text": "",
                "think_block": raw,
                "status": "needs_manual",
                "is_cross_reference": 1,
                "tags": "cross-reference",
            }

        yield _sse({
            "type": "triple",
            "chunk_index": 0,  # cross-ref triples attach to chunk 0 by default
            "triple": triple,
        })

    yield _sse({"type": "done"})


def _find_best_chunk(section_name: str, all_texts: list[str]) -> str:
    """Simple keyword overlap to find the most relevant chunk."""
    if not all_texts:
        return ""
    keywords = set(section_name.lower().split())
    best_score = -1
    best_text = all_texts[0]
    for text in all_texts:
        words = set(text.lower().split())
        score = len(keywords & words)
        if score > best_score:
            best_score = score
            best_text = text
    return best_text


class CrossReferenceGenerator:
    """
    Class-based interface for cross-reference triple generation.

    Requirements: 6.1, 6.2, 6.3, 6.4
    """

    def __init__(self, ollama_client: Optional[OllamaClient] = None,
                 model: str = "gemma3:4b", host: str = "http://localhost:11434"):
        self.ollama_client = ollama_client
        self.model = model
        self.host = host

    async def generate_cross_references(
        self,
        relationship_map: list[dict],
        chunks: list[dict],
        knowledge_context: str,
        db_manager=None,
        default_chunk_id: str = "",
    ) -> list[dict]:
        """Generate cross-reference triples for all relationships and optionally store them."""
        import time as _time

        all_texts = [c.get("content", c.get("text", "")) for c in chunks]
        results = []

        for rel in relationship_map:
            chunk_a_text = _find_best_chunk(rel.get("from", ""), all_texts)
            chunk_b_text = _find_best_chunk(rel.get("to", ""), all_texts)

            prompt = (
                CROSSREF_PROMPT
                .replace("{KNOWLEDGE_CONTEXT}", knowledge_context)
                .replace("{chunk_a_text}", chunk_a_text)
                .replace("{chunk_b_text}", chunk_b_text)
                .replace("{connection}", rel.get("connection", ""))
            )

            ctx = get_settings().get("ollama_num_ctx")
            opts = {"num_ctx": int(ctx)} if ctx else None
            
            # Use constrained JSON format
            raw = await generate(prompt, self.model, self.host, opts, format="json")
            think, parsed = parse_think_and_json(raw)

            if parsed and all(k in parsed for k in ("system", "user", "assistant")):
                triple_data = {
                    "system_text": parsed["system"],
                    "user_text": parsed["user"],
                    "assistant_text": parsed["assistant"],
                    "think_block": think,
                    "status": "pending",
                    "is_cross_reference": True,
                    "tags": ["cross-reference"],
                }
            else:
                triple_data = {
                    "system_text": "",
                    "user_text": "",
                    "assistant_text": "",
                    "think_block": raw,
                    "status": "needs_manual",
                    "is_cross_reference": True,
                    "tags": ["cross-reference"],
                }

            triple_id = f"xref_{int(_time.time() * 1000)}"
            triple_data["id"] = triple_id

            if db_manager and default_chunk_id:
                db_manager.create_triple(
                    triple_id=triple_id,
                    chunk_id=default_chunk_id,
                    angle="cross_reference",
                    intensity="moderate",
                    system_prompt=triple_data.get("system_text", ""),
                    user_message=triple_data.get("user_text", ""),
                    assistant_response=triple_data.get("assistant_text", ""),
                    thinking_trace=triple_data.get("think_block", ""),
                    tags=triple_data.get("tags", []),
                    is_cross_reference=True,
                )

            results.append(triple_data)

        return results


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"
