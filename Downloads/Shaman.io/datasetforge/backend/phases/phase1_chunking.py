"""Phase 1 — Semantic chunking via R1."""
import json
from ..ollama_client import generate, parse_think_and_json

CHUNK_PROMPT = """Read this document and identify every point where the subject matter meaningfully changes — where a new entity, state, phase, concept, or psychological mechanism begins to be described.

Return ONLY a JSON array of character positions marking the start of each new conceptual unit:
{"chunk_boundaries": [0, 1842, 3901, 5240, ...]}

DOCUMENT TEXT:
{document_text}"""


async def run_phase1(
    document_text: str,
    model: str,
    host: str,
    min_chunk_words: int = 150,
) -> list[dict]:
    """
    Returns list of chunk dicts:
    {chunk_index, text, char_start, char_end, page_estimate, status}
    """
    prompt = CHUNK_PROMPT.replace("{document_text}", document_text)
    raw = await generate(prompt, model, host)
    _, parsed = parse_think_and_json(raw)

    boundaries = [0]
    if parsed and "chunk_boundaries" in parsed:
        raw_bounds = parsed["chunk_boundaries"]
        # Validate and filter
        doc_len = len(document_text)
        for b in raw_bounds:
            if isinstance(b, int) and 0 < b < doc_len:
                boundaries.append(b)
    boundaries = sorted(set(boundaries))
    boundaries.append(len(document_text))

    chunks = []
    total_chars = len(document_text)
    total_pages = max(1, total_chars // 3000)

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        text = document_text[start:end].strip()
        if not text:
            continue
        # Enforce minimum chunk size by merging short chunks forward
        if len(text.split()) < min_chunk_words and chunks:
            prev = chunks[-1]
            prev["text"] += "\n\n" + text
            prev["char_end"] = end
            prev["page_estimate"] = round((prev["char_end"] / total_chars) * total_pages, 1)
            continue
        chunks.append({
            "chunk_index": len(chunks),
            "text": text,
            "char_start": start,
            "char_end": end,
            "page_estimate": round((start / total_chars) * total_pages, 1),
            "status": "pending",
        })

    # Re-index after merges
    for i, c in enumerate(chunks):
        c["chunk_index"] = i

    return chunks
