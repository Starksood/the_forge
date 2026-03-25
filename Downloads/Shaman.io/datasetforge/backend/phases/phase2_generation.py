"""Phase 2 — Triple generation: 5 angles × 2 intensities per chunk. Synchronous version."""
import json
import re
import httpx
from typing import Dict, Optional, Tuple
from json_repair import repair_json

ANGLES = [
    {"key": "first_encounter", "name": "first_encounter",
     "description": "the person doesn't know what they're facing yet, only that something overwhelming is present"},
    {"key": "identification", "name": "identification",
     "description": "they are asking what this thing is, what it wants, whether it is real"},
    {"key": "maximum_resistance", "name": "maximum_resistance",
     "description": "they are fighting it, trying to make it stop, in peak distress"},
    {"key": "yielding", "name": "yielding",
     "description": "something shifts, they stop resisting, moving through rather than away"},
    {"key": "integration", "name": "integration",
     "description": "it is receding, they are trying to name what just happened"},
]

INTENSITIES = [
    {"key": "acute", "name": "acute",
     "description": "barely holding together, fragmented, language breaking down"},
    {"key": "moderate", "name": "moderate",
     "description": "frightened but coherent, can still form full sentences"},
]

ANGLE_INTENSITY_COMBINATIONS = [(a, i) for a in ANGLES for i in INTENSITIES]

GENERATION_PROMPT = """COMPRESSED CONTEXT:
{COMPRESSED_CONTEXT}

SOURCE PASSAGE:
{chunk_text}

ASSIGNED ANGLE: {angle_name} ({angle_description})
INTENSITY LEVEL: {intensity_level} ({intensity_description})

INSTRUCTIONS:
- Complete the JSON triple based on the source passage and knowledge context.
- System field: guide's understanding (max 2 sentences).
- User field: person's authentic voice (1-2 sentences).
- Assistant field: guide's response (3-4 sentences).
- Only replace the null values. Follow JSON syntax precisely.

JSON TEMPLATE TO COMPLETE:
{{
  "system": null,
  "user": null,
  "assistant": null
}}
"""


def _parse_json_response(raw: str) -> Tuple[str, Optional[dict]]:
    """Strip think blocks and parse JSON."""
    think = ""
    think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()

    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    start = cleaned.find('{')
    end = cleaned.rfind('}') + 1
    if start == -1 or end == 0:
        return think, None
    try:
        repaired = repair_json(cleaned[start:end])
        result = json.loads(repaired)
        return think, result if isinstance(result, dict) else None
    except Exception:
        return think, None


def generate_triple(
    chunk_text: str,
    angle: dict,
    intensity: dict,
    knowledge_context: str,
    model: str,
    host: str,
) -> dict:
    """Generate one triple synchronously. Returns {system_text, user_text, assistant_text, think_block, status}."""
    prompt = (
        GENERATION_PROMPT
        .replace("{COMPRESSED_CONTEXT}", knowledge_context)
        .replace("{chunk_text}", chunk_text)
        .replace("{angle_name}", angle["name"])
        .replace("{angle_description}", angle["description"])
        .replace("{intensity_level}", intensity["name"])
        .replace("{intensity_description}", intensity["description"])
    )

    try:
        with httpx.Client(timeout=600.0) as client:
            r = client.post(
                f"{host}/api/generate",
                json={"model": model, "prompt": prompt, "format": "json", "stream": False},
            )
            r.raise_for_status()
            raw = r.json().get("response", "")
    except Exception as e:
        return {"system_text": "", "user_text": "", "assistant_text": "",
                "think_block": str(e), "status": "needs_manual"}

    think, parsed = _parse_json_response(raw)

    if parsed and all(k in parsed for k in ("system", "user", "assistant")):
        return {
            "system_text": parsed.get("system") or "",
            "user_text": parsed.get("user") or "",
            "assistant_text": parsed.get("assistant") or "",
            "think_block": think,
            "status": "pending",
        }

    return {
        "system_text": "",
        "user_text": "",
        "assistant_text": "",
        "think_block": raw,
        "status": "needs_manual",
    }
