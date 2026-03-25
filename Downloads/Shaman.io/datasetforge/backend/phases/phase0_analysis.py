"""Phase 0 — Full-book analysis with chunked approach. Synchronous version."""
import json
import logging
import time
import re
import httpx
from typing import Dict, List, Optional, Set
from json_repair import repair_json

logger = logging.getLogger(__name__)

CHUNKED_ANALYSIS_PROMPT = """You are analyzing a passage from a document to extract key concepts for a fine-tuning dataset.
Your goal is to identify concepts, entities, and ideas that are central to the subject matter —
including defined terms, named processes, key distinctions, and important relationships between ideas.

Read this passage and identify:
1. Any named concepts, terms, or entities that are defined or meaningfully described here
2. Any cause-effect, hierarchical, or dependency relationships between concepts

From this candidate term list, confirm which appear and are substantively defined in this passage:
{candidate_terms}

Passage text:
{document_text}

Return JSON with exactly this structure:
{{
  "confirmed_terms": [
    {{"name": "...", "definition": "...", "guidance_relevance": "..."}}
  ],
  "relationships": [
    {{"from": "...", "to": "...", "type": "..."}}
  ]
}}

For each confirmed term, "guidance_relevance" should be a brief phrase describing how an assistant
would apply this concept when answering user questions (e.g., "use as foundation for explaining X",
"distinguish from related term Y", "apply when user asks about Z")."""

SYNTHESIS_PROMPT = """You are synthesizing the results of a chunked analysis of a document.
Below is an aggregated taxonomy of terms and their definitions, along with conceptual relationships.

Aggregated Taxonomy:
{taxonomy}

Relationships:
{relationships}

TASK:
1. Create a definitive, compressed framework summary of the document's subject matter.
2. Ensure it is grounded in the relationships and definitions provided.
3. Keep it maximally dense and informative.
4. Write the "compressed_context" field as exactly 3 dense sentences in the instructional register —
   as if briefing an assistant before answering user questions about this document.
   Each sentence should directly instruct how to recognize, explain, or apply the core concepts.
   Example register: "This document covers X, which is defined as Y." / "When users ask about Z,
   emphasize the distinction between A and B."

Return JSON with exactly this structure:
{{
  "framework_summary": "...",
  "compressed_context": "..."
}}"""


def call_ollama_blocking(prompt: str, host: str, model: str) -> dict:
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"num_ctx": 8192},
    }
    with httpx.Client(timeout=600.0) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()

    raw = data.get("response", "")

    # Strip think blocks
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    start = cleaned.find('{')
    end = cleaned.rfind('}') + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object in response: {raw[:200]}")

    repaired = repair_json(cleaned[start:end])
    result = json.loads(repaired)
    if not isinstance(result, dict):
        raise ValueError(f"Expected dict, got {type(result).__name__}: {str(result)[:100]}")
    return result


def extract_local_taxonomy(text: str) -> List[str]:
    """Step 1: LOCAL TAXONOMY EXTRACTION (no LLM, deterministic)."""
    candidates: Set[str] = set()

    multi_word_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b')
    counts = {}
    for match in multi_word_pattern.finditer(text):
        term = match.group()
        counts[term] = counts.get(term, 0) + 1
    for term, count in counts.items():
        if count >= 3:
            candidates.add(term)

    definition_phrases = [
        "is defined as", "refers to", "means", "is a state of", "is the experience of"
    ]
    definition_pattern = re.compile(
        rf'\b([A-Z][\w\s]+?)\s+(?:{"|".join(definition_phrases)})\b',
        re.IGNORECASE
    )
    for match in definition_pattern.finditer(text):
        candidates.add(match.group(1).strip())

    header_pattern = re.compile(r'^(?:#+\s+|Chapter\s+\d+[:\s]+)(.*)', re.MULTILINE)
    for match in header_pattern.finditer(text):
        candidates.add(match.group(1).strip())

    final_candidates = [c for c in candidates if len(c) > 3 and not c.isupper()]
    return sorted(final_candidates)


class FrameworkAnalyzer:
    """Performs full-document analysis using a chunked approach."""

    def __init__(self, host: str, model: str):
        self.host = host
        self.model = model

    def analyze_document(self, document_text: str) -> Dict:
        logger.info("Starting chunked framework analysis")

        candidate_terms = extract_local_taxonomy(document_text)
        logger.info(f"Extracted {len(candidate_terms)} local candidate terms")

        num_segments = 10
        total_len = len(document_text)
        segment_size = total_len // num_segments

        all_confirmed_terms = {}
        all_relationships = []

        for i in range(num_segments):
            start = i * segment_size
            end = total_len if i == num_segments - 1 else (i + 1) * segment_size
            segment_text = document_text[start:end]

            terms_list = ", ".join(candidate_terms[:100])
            prompt = CHUNKED_ANALYSIS_PROMPT.format(
                document_text=segment_text,
                candidate_terms=terms_list
            )

            try:
                parsed = call_ollama_blocking(prompt, self.host, self.model)
                if parsed:
                    confirmed = parsed.get("confirmed_terms", [])
                    if isinstance(confirmed, list):
                        for item in confirmed:
                            name = item.get("name")
                            if name:
                                if name not in all_confirmed_terms or len(item.get("definition", "")) > len(all_confirmed_terms[name].get("definition", "")):
                                    all_confirmed_terms[name] = item
                    rel = parsed.get("relationships", [])
                    if isinstance(rel, list):
                        all_relationships.extend(rel)
            except Exception as e:
                logger.error(f"Error analyzing segment {i+1}: {e}")
                continue

        taxonomy = list(all_confirmed_terms.values())

        unique_relationships = []
        seen_rels = set()
        for r in all_relationships:
            if not isinstance(r, dict):
                continue
            key = f"{r.get('from')}-{r.get('to')}-{r.get('type')}"
            if key not in seen_rels:
                unique_relationships.append(r)
                seen_rels.add(key)

        synth_prompt = SYNTHESIS_PROMPT.format(
            taxonomy=json.dumps(taxonomy[:30], indent=2),
            relationships=json.dumps(unique_relationships[:20], indent=2)
        )

        try:
            synth_parsed = call_ollama_blocking(synth_prompt, self.host, self.model)
            framework_summary = synth_parsed.get("framework_summary", "")
            compressed_context = synth_parsed.get("compressed_context", "")
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            framework_summary = ""
            compressed_context = ""
            synth_parsed = {}

        return {
            "taxonomy": taxonomy,
            "framework_summary": framework_summary,
            "compressed_context": compressed_context,
            "relationships": unique_relationships,
            "raw_analysis": json.dumps(synth_parsed),
        }
