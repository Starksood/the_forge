"""SemanticChunker — deterministic embedding-based chunking. Synchronous version."""
import json
import time
import logging
import httpx
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    logger.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    nlp = None


def get_embedding(text: str, host: str) -> List[float]:
    """Get embedding from Ollama synchronously."""
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                f"{host}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
            )
            r.raise_for_status()
            return r.json().get("embedding", [])
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    va, vb = np.array(a), np.array(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class SemanticChunker:
    """
    Splits documents using sentence embeddings and cosine similarity.
    No LLM calls — purely mathematical.
    """

    def __init__(self, ollama_host: str = "http://localhost:11434",
                 similarity_threshold: float = 0.75,
                 min_chunk_words: int = 150):
        self.host = ollama_host
        self.threshold = similarity_threshold
        self.min_chunk_words = min_chunk_words

    def chunk_document(self, document_text: str, document_id: str,
                       progress_callback=None) -> List[Dict]:
        """Chunk a document and return list of chunk dicts."""
        if not nlp:
            logger.warning("spaCy not loaded — using fallback chunking")
            return self._fallback_chunk(document_text)

        if len(document_text) > 1_000_000:
            nlp.max_length = len(document_text) + 1000

        doc = nlp(document_text)
        sentences = []
        sentence_offsets = []
        for sent in doc.sents:
            txt = sent.text.strip()
            if txt:
                sentences.append(txt)
                sentence_offsets.append(sent.start_char)

        if not sentences:
            return self._fallback_chunk(document_text)

        logger.info(f"Generating embeddings for {len(sentences)} sentences...")
        embeddings = []
        total = len(sentences)
        for i, s in enumerate(sentences):
            emb = get_embedding(s, self.host)
            embeddings.append(emb)
            if progress_callback:
                progress_callback(i + 1, total)
            if i % 50 == 0 and i > 0:
                logger.debug(f"Embedded {i}/{total} sentences")

        boundaries = [0]
        for i in range(1, len(embeddings)):
            sim = cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.threshold:
                boundaries.append(sentence_offsets[i])
        boundaries.append(len(document_text))
        boundaries = sorted(set(boundaries))

        total_chars = len(document_text)
        total_pages = max(1, total_chars // 3000)
        raw_chunks = []

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            text = document_text[start:end].strip()
            if not text:
                continue
            if len(text.split()) < self.min_chunk_words and raw_chunks:
                prev = raw_chunks[-1]
                prev["content"] += "\n\n" + text
                prev["end_char"] = end
                prev["page_estimate"] = round((prev["end_char"] / total_chars) * total_pages, 1)
                continue
            raw_chunks.append({
                "sequence_number": len(raw_chunks),
                "content": text,
                "start_char": start,
                "end_char": end,
                "page_estimate": round((start / total_chars) * total_pages, 1),
            })

        result = []
        for i, c in enumerate(raw_chunks):
            c["sequence_number"] = i
            result.append(c)

        logger.info(f"Created {len(result)} chunks")
        return result

    def _fallback_chunk(self, text: str) -> List[Dict]:
        chunk_size = 5000
        chunks = []
        total_chars = len(text)
        total_pages = max(1, total_chars // 3000)
        for i, start in enumerate(range(0, total_chars, chunk_size)):
            end = min(start + chunk_size, total_chars)
            chunks.append({
                "sequence_number": i,
                "content": text[start:end],
                "start_char": start,
                "end_char": end,
                "page_estimate": round((start / total_chars) * total_pages, 1),
            })
        return chunks
