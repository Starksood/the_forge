"""The Forge — Document to fine-tuning dataset pipeline."""
# ── SECTION 1: IMPORTS ──────────────────────────────────────────────────────
import streamlit as st
import sqlite3
import json
import os
import uuid
import re
import httpx
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

from backend.phases.phase0_analysis import (
    extract_local_taxonomy, call_ollama_blocking,
    CHUNKED_ANALYSIS_PROMPT, SYNTHESIS_PROMPT,
)
from backend.ollama_client import generate_sync
from json_repair import repair_json
from backend.phases.semantic_chunker import SemanticChunker
from backend.phases.phase2_generation import (
    ANGLE_INTENSITY_COMBINATIONS, generate_triple,
)
from backend.personas import (
    GUIDE_PERSONA,
    GUIDE_INTEGRATION_VOICE,
    TRAVELER_PERSONA,
    ANGLE_DEFINITIONS,
    INTENSITY_DEFINITIONS,
)

st.set_page_config(
    page_title="The Forge",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SECTION 2: SESSION STATE INITIALIZATION ─────────────────────────────────
def _init_session_state():
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "ollama_host": "http://localhost:11434",
            "model": "gemma3:4b",
            "threshold": 0.75,
            "cross_reference": True,
            "triples_per_chunk": 10,
        }
    if "phase" not in st.session_state:
        st.session_state.phase = "upload"
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "session_db_path" not in st.session_state:
        st.session_state.session_db_path = None
    if "document_id" not in st.session_state:
        st.session_state.document_id = None
    if "document_text" not in st.session_state:
        st.session_state.document_text = None
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    if "analysis_id" not in st.session_state:
        st.session_state.analysis_id = None
    if "current_chunk_index" not in st.session_state:
        st.session_state.current_chunk_index = 0
    if "filter_mode" not in st.session_state:
        st.session_state.filter_mode = "all"
    if "tag_filter" not in st.session_state:
        st.session_state.tag_filter = ""

_init_session_state()

# ── SECTION 2.5: VALIDATION ──────────────────────────────────────────────────

ANALYSIS_PHRASES = [
    "this passage", "the text", "the author", "this section",
    "the document", "the excerpt", "in the passage", "as described",
]


def validate_triple(triple: dict) -> list[str]:
    """
    Check a triple dict for structural and tonal quality violations.

    Returns a list of violation message strings. Empty list means the triple passes.
    Never raises — all field accesses use .get() with empty string defaults.
    """
    violations = []

    def _get_field(triple, *keys):
        for k in keys:
            v = triple.get(k)
            if v is not None:
                return str(v)
        return ''

    system = _get_field(triple, 'system_prompt', 'system')
    user = _get_field(triple, 'user_message', 'user')
    assistant = _get_field(triple, 'assistant_response', 'assistant')

    # Check 1: Required fields present and non-empty
    if not system.strip():
        violations.append("system_prompt field is missing or empty")
    if not user.strip():
        violations.append("user_message field is missing or empty")
    if not assistant.strip():
        violations.append("assistant_response field is missing or empty")

    # If core fields missing, skip further checks
    if not user.strip() or not assistant.strip():
        return violations

    # Check 2: user_message must not contain analysis language (case-insensitive)
    user_lower = user.lower()
    for phrase in ANALYSIS_PHRASES:
        if phrase in user_lower:
            violations.append(
                f"user_message contains analysis language: '{phrase}'"
            )
            break  # report first match only

    # Check 3: assistant_response must use first-person present tense
    assistant_stripped = assistant.lstrip()
    has_first_person = (
        assistant_stripped.startswith("I ") or
        bool(re.search(
            r'\bI\s+(am|feel|see|hear|sense|know|want|need|think|notice)\b',
            assistant
        ))
    )
    if not has_first_person:
        violations.append(
            "assistant_response does not use first-person present tense "
            "(must begin with 'I ' or contain a first-person present-tense construction)"
        )

    # Check 4: assistant_response length bounds [40, 600]
    assistant_len = len(assistant)
    if assistant_len < 40:
        violations.append(
            f"assistant_response too short ({assistant_len} chars, minimum 40)"
        )
    elif assistant_len > 600:
        violations.append(
            f"assistant_response too long ({assistant_len} chars, maximum 600)"
        )

    return violations


def _generate_triple_json(system_prompt: str, model: str, host: str) -> dict:
    """
    Call Ollama's chat endpoint with a proper system/user message split,
    then parse the JSON response with repair fallback for truncated outputs.
    """
    with httpx.Client(timeout=600.0) as client:
        r = client.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate the JSON triple now."},
                ],
                "format": "json",
                "stream": False,
                "options": {"num_ctx": 8192},
            },
        )
        r.raise_for_status()
        raw = r.json().get("message", {}).get("content", "")

    # Strip any <think>...</think> blocks (some models emit these)
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Find the start of the JSON object
    start = cleaned.find('{')
    if start == -1:
        raise ValueError(f"No JSON object in response: {raw[:200]}")

    # Use repair_json to handle truncated/incomplete output — it closes open strings/braces
    candidate = cleaned[start:]
    repaired = repair_json(candidate)
    result = json.loads(repaired)
    if not isinstance(result, dict):
        raise ValueError(f"Expected dict, got {type(result).__name__}")
    return result


def build_generation_prompt(
    chunk_content: str,
    angle: str,
    intensity: str,
    compressed_context: str,
) -> dict:
    """
    Build a generation prompt dict for the given angle/intensity combination.

    Returns a dict with exactly four string keys: system, user, assistant, thinking.
    Raises ValueError if angle or intensity are not valid keys.
    """
    if angle not in ANGLE_DEFINITIONS:
        raise ValueError(
            f"Invalid angle '{angle}'. Must be one of: {list(ANGLE_DEFINITIONS.keys())}"
        )
    if intensity not in INTENSITY_DEFINITIONS:
        raise ValueError(
            f"Invalid intensity '{intensity}'. Must be one of: {list(INTENSITY_DEFINITIONS.keys())}"
        )

    angle_def = ANGLE_DEFINITIONS[angle]
    intensity_def = INTENSITY_DEFINITIONS[intensity]

    # Use integration voice for integration angle
    guide_voice = GUIDE_INTEGRATION_VOICE if angle == "integration" else GUIDE_PERSONA

    # Strip register examples from persona strings to keep prompt small
    def _trim_persona(text: str) -> str:
        """Remove 'Register examples:' block and everything after it."""
        marker = "Register examples:"
        idx = text.find(marker)
        return text[:idx].rstrip() if idx != -1 else text

    guide_voice_trimmed = _trim_persona(guide_voice)
    traveler_trimmed = _trim_persona(TRAVELER_PERSONA)

    # Truncate chunk content to 3000 chars.
    # Budget: 8K context, ~4000 chars fixed overhead, 512 tokens reserved for output.
    # Available for passage: ~(8192-512)*4 - 4000 = ~25700 chars. 3000 is safe with margin.
    passage = chunk_content[:3000]

    system = f"""You are generating a training triple from a source passage.

=== THE ASSISTANT ===

{guide_voice_trimmed}

=== THE USER ===

{traveler_trimmed}

=== DOMAIN CONTEXT ===

{compressed_context}

=== SOURCE PASSAGE ===

{passage}

=== YOUR TASK ===

The passage above contains knowledge, concepts, or explanations about a specific topic.
Your task is to translate that knowledge into a realistic conversational exchange between
a User and an Assistant.

DO NOT:
- Quote directly from the passage
- Summarize the passage in the system or user fields
- Reference the source text in any field ("this passage", "the text", "the author", etc.)
- Make the user sound like they are reading from a document

DO:
- Construct a natural exchange that did not exist in the source
- Apply the passage's knowledge through the Assistant's response
- Make the USER field sound like a real person asking a genuine question
- Make the ASSISTANT field apply the passage's knowledge without citing it

=== THE EXCHANGE ===

PERSPECTIVE: {angle_def['name']}
SCENARIO: {angle_def['description']}
ASSISTANT ORIENTATION: {angle_def['guide_orientation']}

STYLE: {intensity_def['name']}
EXCHANGE CHARACTER: {intensity_def['description']}

=== OUTPUT FORMAT ===

Return ONLY valid JSON with exactly these four keys:
{{
  "system": "What the Assistant knows and how it should approach this exchange, drawn from the passage. Written as instruction to the Assistant. 2 sentences maximum.",
  "user": "What the User says. Must sound like a genuine question or statement. 1-2 sentences.",
  "assistant": "What the Assistant says in response. Must respond specifically to what the User just said. 3-4 sentences maximum.",
  "thinking": "Brief note on what passage knowledge informed this exchange."
}}

Do not include any preamble, explanation, or text outside the JSON."""

    return {
        "system": system,
        "user": "",
        "assistant": "",
        "thinking": "",
    }


def repair_triple_with_ollama(triple: dict, host: str, model: str) -> dict:
    """
    Rewrite a flagged triple by calling Ollama with a repair prompt.

    Does NOT write to the database — caller is responsible for persistence.
    Always adds 'repaired' tag to tags_json.
    Sets status to 'pending' if validate_triple() passes, 'needs_manual' if it fails.
    On any error (network, JSON parse), returns original triple with needs_manual + repaired tag.
    """
    def _add_repaired_tag(t: dict) -> dict:
        existing_tags = []
        try:
            existing_tags = json.loads(t.get('tags_json') or '[]')
        except (json.JSONDecodeError, TypeError):
            existing_tags = []
        if 'repaired' not in existing_tags:
            existing_tags.append('repaired')
        t['tags_json'] = json.dumps(existing_tags)
        return t

    violations = validate_triple(triple)

    system_prompt = triple.get('system_prompt', triple.get('system', ''))
    user_message = triple.get('user_message', triple.get('user', ''))
    assistant_response = triple.get('assistant_response', triple.get('assistant', ''))

    violations_text = "\n".join(f"- {v}" for v in violations) if violations else "- No specific violations detected"

    repair_prompt = f"""You are rewriting a training triple that failed quality validation.

VIOLATIONS TO FIX:
{violations_text}

ORIGINAL TRIPLE:
System: {system_prompt}
User: {user_message}
Assistant: {assistant_response}

RULES:
- The assistant response MUST begin with "I" and use first-person present tense.
- The user message MUST NOT reference the source document (no "this passage", "the text", "the author", etc.).
- The assistant response MUST be between 40 and 600 characters.
- Do NOT quote or summarize source material directly.

Return JSON: {{"system": "...", "user": "...", "assistant": "..."}}"""

    try:
        raw = generate_sync(repair_prompt, model=model, host=host, format="json")
        # Use repair_json to handle truncated/malformed responses
        cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        start = cleaned.find('{')
        if start == -1:
            raise ValueError(f"No JSON in repair response: {raw[:100]}")
        parsed = json.loads(repair_json(cleaned[start:]))

        result = dict(triple)
        result['system_prompt'] = parsed.get('system', system_prompt)
        result['user_message'] = parsed.get('user', user_message)
        result['assistant_response'] = parsed.get('assistant', assistant_response)

        result = _add_repaired_tag(result)

        post_violations = validate_triple(result)
        result['status'] = 'pending' if not post_violations else 'needs_manual'
        return result

    except Exception:
        result = dict(triple)
        result['status'] = 'needs_manual'
        result = _add_repaired_tag(result)
        return result


# ── SECTION 3: HELPER FUNCTIONS ─────────────────────────────────────────────

def get_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def get_session_db() -> sqlite3.Connection:
    return get_db(st.session_state.session_db_path)


def create_session_db(path: str):
    conn = get_db(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            filename TEXT,
            content TEXT,
            page_count INTEGER,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS framework_analyses (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            taxonomy_json TEXT,
            framework_summary TEXT,
            compressed_context TEXT,
            relationships_json TEXT,
            raw_analysis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            sequence_number INTEGER,
            start_char INTEGER,
            end_char INTEGER,
            content TEXT,
            page_estimate REAL,
            status TEXT DEFAULT 'pending'
        );
        CREATE TABLE IF NOT EXISTS triples (
            id TEXT PRIMARY KEY,
            chunk_id TEXT,
            angle TEXT,
            intensity TEXT,
            system_prompt TEXT,
            user_message TEXT,
            assistant_response TEXT,
            thinking_trace TEXT,
            status TEXT DEFAULT 'pending',
            tags_json TEXT DEFAULT '[]',
            is_cross_reference INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            framework_id TEXT,
            source_concept TEXT,
            target_concept TEXT,
            relationship_type TEXT,
            description TEXT
        );
    """)
    conn.commit()
    conn.close()


def extract_text(file) -> tuple:
    """Extract text from uploaded file. Returns (text, page_count)."""
    name = file.name.lower()
    if name.endswith(".txt"):
        text = file.read().decode("utf-8", errors="replace")
        return text, max(1, len(text) // 3000)
    elif name.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(file)
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        return text, len(reader.pages)
    elif name.endswith(".docx"):
        from docx import Document
        doc = Document(file)
        text = "\n".join(p.text for p in doc.paragraphs)
        return text, max(1, len(text) // 3000)
    else:
        raise ValueError(f"Unsupported format: {file.name}")


def check_ollama() -> bool:
    try:
        r = httpx.get(
            f"{st.session_state.settings['ollama_host']}/api/tags",
            timeout=5.0,
        )
        return r.status_code == 200
    except Exception:
        return False


def check_model_available(host: str, model: str) -> tuple:
    """Returns (is_available, error_message)."""
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        base_names = [m.split(":")[0] for m in models]
        if model in models or model.split(":")[0] in base_names:
            return True, ""
        return False, (
            f"Model '{model}' not found. Pull it with: `ollama pull {model}`\n\n"
            f"Available: {', '.join(models) or 'none'}"
        )
    except Exception as e:
        return False, f"Cannot reach Ollama at {host}: {e}"


def get_chunks() -> List[dict]:
    conn = get_session_db()
    rows = conn.execute(
        "SELECT * FROM chunks WHERE document_id=? ORDER BY sequence_number",
        (st.session_state.document_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_triples_for_chunk(
    chunk_id: str,
    filter_mode: str = "all",
    tag_filter: Optional[str] = None,
) -> List[dict]:
    conn = get_session_db()
    if filter_mode == "all":
        rows = conn.execute(
            "SELECT * FROM triples WHERE chunk_id=? ORDER BY angle, intensity",
            (chunk_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM triples WHERE chunk_id=? AND status=? ORDER BY angle, intensity",
            (chunk_id, filter_mode),
        ).fetchall()
    conn.close()
    result = [dict(r) for r in rows]

    # Apply tag filter in Python (tags stored as JSON array string)
    if tag_filter:
        filtered = []
        for t in result:
            try:
                tags = json.loads(t.get("tags_json") or "[]")
                if tag_filter in tags:
                    filtered.append(t)
            except (json.JSONDecodeError, TypeError):
                pass  # malformed tags_json — exclude from tag-filtered results
        return filtered

    return result


def get_triple_stats() -> dict:
    conn = get_session_db()
    stats = {}
    for status in ["pending", "approved", "rejected", "needs_manual"]:
        count = conn.execute(
            "SELECT COUNT(*) FROM triples WHERE status=?", (status,)
        ).fetchone()[0]
        stats[status] = count
    stats["total"] = sum(stats.values())
    conn.close()
    return stats


def _collect_all_tags() -> list:
    """Collect all unique tags from all triples in the current session."""
    conn = get_session_db()
    rows = conn.execute(
        "SELECT tags_json FROM triples WHERE tags_json IS NOT NULL"
    ).fetchall()
    conn.close()
    all_tags = []
    for row in rows:
        try:
            tags = json.loads(row[0] or '[]')
            all_tags.extend(tags)
        except (json.JSONDecodeError, TypeError):
            pass
    return all_tags


def update_triple(triple_id: str, system: str, user: str, assistant: str):
    conn = get_session_db()
    conn.execute(
        "UPDATE triples SET system_prompt=?, user_message=?, assistant_response=?, "
        "updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (system, user, assistant, triple_id),
    )
    conn.commit()
    conn.close()


def set_triple_status(triple_id: str, status: str):
    conn = get_session_db()
    conn.execute(
        "UPDATE triples SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (status, triple_id),
    )
    conn.commit()
    conn.close()


def update_triple_tags(triple_id: str, tags: str):
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    conn = get_session_db()
    conn.execute(
        "UPDATE triples SET tags_json=? WHERE id=?",
        (json.dumps(tag_list), triple_id),
    )
    conn.commit()
    conn.close()


def get_analysis() -> Optional[dict]:
    if not st.session_state.analysis_id:
        return None
    conn = get_session_db()
    row = conn.execute(
        "SELECT * FROM framework_analyses WHERE id=?",
        (st.session_state.analysis_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    r = dict(row)
    r["taxonomy"] = json.loads(r.get("taxonomy_json") or "[]")
    r["relationships"] = json.loads(r.get("relationships_json") or "[]")
    return r


def find_taxonomy_matches(text: str, taxonomy: List[dict]) -> List[dict]:
    text_lower = text.lower()
    return [t for t in taxonomy if t.get("name", "").lower() in text_lower]


def auto_backup():
    stats = get_triple_stats()
    if stats["approved"] > 0 and stats["approved"] % 50 == 0:
        src = st.session_state.session_db_path
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = src.replace(".db", f"_backup_{ts}.db")
        if not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)


def list_sessions() -> List[dict]:
    sessions = []
    sessions_dir = Path("sessions")
    if not sessions_dir.exists():
        return sessions
    for db_file in sorted(sessions_dir.glob("session_*.db"), reverse=True):
        try:
            conn = get_db(str(db_file))
            doc = conn.execute("SELECT filename FROM documents LIMIT 1").fetchone()
            triples = conn.execute(
                "SELECT COUNT(*) FROM triples WHERE status='approved'"
            ).fetchone()
            conn.close()
            sessions.append({
                "path": str(db_file),
                "name": db_file.stem,
                "document": doc["filename"] if doc else "unknown",
                "approved": triples[0] if triples else 0,
                "modified": datetime.fromtimestamp(
                    db_file.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M"),
            })
        except Exception:
            continue
    return sessions


def delete_session(path: str):
    """Delete a session DB and all its backup files."""
    import glob
    base = path.replace(".db", "")
    for f in glob.glob(f"{base}*.db"):
        try:
            os.remove(f)
        except Exception:
            pass


def call_ollama_with_retry(prompt: str, host: str, model: str,
                           max_retries: int = 3) -> dict:
    """Call Ollama with exponential backoff. On timeout, halves the prompt text."""
    import time
    last_err = None
    current_prompt = prompt
    for attempt in range(max_retries):
        try:
            return call_ollama_blocking(current_prompt, host, model)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            is_timeout = "timeout" in err_str or "timed out" in err_str
            if is_timeout and attempt < max_retries - 1:
                # Halve the document_text portion of the prompt on timeout
                # Find the passage block and truncate it
                if "Passage text:" in current_prompt:
                    parts = current_prompt.split("Passage text:", 1)
                    truncated = parts[1][:len(parts[1]) // 2]
                    current_prompt = parts[0] + "Passage text:" + truncated
                elif "SOURCE PASSAGE:" in current_prompt:
                    parts = current_prompt.split("SOURCE PASSAGE:", 1)
                    truncated = parts[1][:len(parts[1]) // 2]
                    current_prompt = parts[0] + "SOURCE PASSAGE:" + truncated
            wait = 2 ** attempt
            time.sleep(wait)
    raise last_err


# ── SECTION 4: PHASE RENDER FUNCTIONS ───────────────────────────────────────

def render_repair_pass(scope: str, host: str, model: str):
    """
    Run a batch repair pass over triples matching the given scope.

    scope: one of "needs_manual_only", "all_pending", "all_non_approved"
    """
    st.markdown("---")
    st.markdown("### ⚙ Repair Pass Running")

    # Map scope to SQL WHERE clause
    scope_queries = {
        "needs_manual_only": "SELECT * FROM triples WHERE status='needs_manual'",
        "all_pending": "SELECT * FROM triples WHERE status='pending'",
        "all_non_approved": "SELECT * FROM triples WHERE status != 'approved'",
    }
    query = scope_queries.get(scope, scope_queries["needs_manual_only"])

    conn = get_session_db()
    triples = [dict(r) for r in conn.execute(query).fetchall()]
    conn.close()

    total = len(triples)
    if total == 0:
        st.info("No triples to repair in this scope.")
        return

    st.caption(f"Repairing {total} triples...")

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    counter = st.empty()

    passed = 0
    failed = 0

    for i, triple in enumerate(triples):
        angle = triple.get('angle', 'unknown')
        intensity = triple.get('intensity', 'unknown')
        status_text.markdown(
            f"Repairing **{angle.replace('_', ' ')}** / {intensity} · triple {i+1}/{total}"
        )

        try:
            repaired = repair_triple_with_ollama(triple, host, model)

            # Write updated status and tags back to DB
            conn = get_session_db()
            conn.execute(
                "UPDATE triples SET system_prompt=?, user_message=?, assistant_response=?, "
                "status=?, tags_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (
                    repaired.get('system_prompt', triple.get('system_prompt', '')),
                    repaired.get('user_message', triple.get('user_message', '')),
                    repaired.get('assistant_response', triple.get('assistant_response', '')),
                    repaired.get('status', 'needs_manual'),
                    repaired.get('tags_json', triple.get('tags_json', '[]')),
                    triple['id'],
                )
            )
            conn.commit()
            conn.close()

            if repaired.get('status') == 'pending':
                passed += 1
            else:
                failed += 1

        except Exception:
            failed += 1

        progress_bar.progress((i + 1) / total)
        counter.caption(
            f"{passed} passed validation  ·  {failed} remain flagged  ·  {total - i - 1} remaining"
        )

    st.success(
        f"Repair complete · {passed} triples passed validation · {failed} remain flagged"
    )
    if passed > 0:
        st.info(
            "Repaired triples have been tagged 'repaired' and reset to 'pending'. "
            "Use 'Review Repaired Triples' to filter to them."
        )


def render_upload_phase():
    st.markdown("# ⚗️ The Forge")
    st.markdown("##### document · extract · fine-tune")
    st.divider()

    if check_ollama():
        st.success(f"Ollama connected · {st.session_state.settings['model']}")
    else:
        st.error(
            f"Ollama not detected at {st.session_state.settings['ollama_host']}. "
            f"Install from ollama.ai and run: ollama pull {st.session_state.settings['model']}"
        )
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### New Session")
        uploaded = st.file_uploader(
            "Upload source document",
            type=["txt", "pdf", "docx"],
            help="Your book or document to convert into training data",
        )
        if uploaded:
            if st.button("Begin Processing", type="primary"):
                with st.spinner("Parsing document..."):
                    try:
                        text, pages = extract_text(uploaded)
                        session_id = (
                            datetime.now().strftime("%Y%m%d_%H%M%S")
                            + "_" + str(uuid.uuid4())[:8]
                        )
                        Path("sessions").mkdir(exist_ok=True)
                        db_path = f"sessions/session_{session_id}.db"
                        create_session_db(db_path)

                        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
                        conn = get_db(db_path)
                        conn.execute(
                            "INSERT INTO documents (id, session_id, filename, content, page_count) "
                            "VALUES (?,?,?,?,?)",
                            (doc_id, session_id, uploaded.name, text, pages),
                        )
                        conn.commit()
                        conn.close()

                        st.session_state.session_id = session_id
                        st.session_state.session_db_path = db_path
                        st.session_state.document_id = doc_id
                        st.session_state.document_text = text
                        st.session_state.document_name = uploaded.name
                        st.session_state.phase = "analyze"
                        st.success(
                            f"Loaded {uploaded.name} · {pages} pages · {len(text):,} characters"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to parse document: {e}")

    with col2:
        sessions = list_sessions()
        if sessions:
            st.markdown("### Resume Session")
            for s in sessions:
                with st.container(border=True):
                    st.markdown(f"**{s['document']}**")
                    st.caption(f"{s['approved']} approved · last opened {s['modified']}")
                    btn_col, del_col = st.columns([3, 1])
                    with btn_col:
                        if st.button("Resume", key=f"res_{s['name']}"):
                            conn = get_db(s["path"])
                            doc = conn.execute(
                                "SELECT id, content, filename FROM documents LIMIT 1"
                            ).fetchone()
                            analysis = conn.execute(
                                "SELECT id FROM framework_analyses LIMIT 1"
                            ).fetchone()
                            conn.close()
                            st.session_state.session_db_path = s["path"]
                            st.session_state.document_id = doc["id"]
                            st.session_state.document_text = doc["content"]
                            st.session_state.document_name = doc["filename"]
                            st.session_state.analysis_id = analysis["id"] if analysis else None
                            st.session_state.phase = "review"
                            st.rerun()
                    with del_col:
                        if st.button("🗑", key=f"del_{s['name']}", help="Delete this session"):
                            delete_session(s["path"])
                            st.rerun()


def render_analyze_phase():
    st.markdown("### Phase 0 — Concept Extraction")
    st.info(
        "Analyzes your full document to extract a taxonomy of key concepts, entities, "
        "and relationships. This framework is injected into every generation prompt that follows."
    )

    doc_text = st.session_state.document_text
    st.metric("Document size", f"{len(doc_text):,} characters")

    if st.button("Begin Analysis", type="primary"):
        host = st.session_state.settings["ollama_host"]
        model = st.session_state.settings["model"]

        ok, err = check_model_available(host, model)
        if not ok:
            st.error(err)
            st.stop()

        with st.status("Running concept extraction...", expanded=True) as status:
            # Step 1: local taxonomy
            st.write("Extracting local taxonomy...")
            candidate_terms = extract_local_taxonomy(doc_text)
            st.write(f"Found {len(candidate_terms)} candidate terms using pattern matching")

            # Step 2: segment LLM analysis.
            # Budget: 8K context, ~1500 chars fixed overhead, 1024 tokens reserved for output.
            # Available for document text: ~(8192-1024)*4 - 1500 = ~25000 chars per segment.
            MAX_SEG_CHARS = 24_000
            num_segments = -(-len(doc_text) // MAX_SEG_CHARS)  # ceiling division
            seg_size = len(doc_text) // num_segments
            all_terms = {}
            all_relationships = []

            seg_progress = st.progress(0.0, text="Analyzing segments...")
            for i in range(num_segments):
                start = i * seg_size
                end = len(doc_text) if i == num_segments - 1 else (i + 1) * seg_size
                segment = doc_text[start:end][:MAX_SEG_CHARS]  # hard cap as safety net
                st.write(f"Analyzing segment {i+1} of {num_segments}...")
                terms_list = ", ".join(candidate_terms[:50])  # keep prompt overhead small
                prompt = CHUNKED_ANALYSIS_PROMPT.format(
                    document_text=segment, candidate_terms=terms_list
                )
                try:
                    result = call_ollama_with_retry(prompt, host, model)
                    if isinstance(result, dict):
                        for term in result.get("confirmed_terms", []):
                            name = term.get("name")
                            if name:
                                all_terms[name] = term
                        all_relationships.extend(result.get("relationships", []))
                except Exception as e:
                    st.warning(f"Segment {i+1} failed after retries: {e}")
                seg_progress.progress(
                    (i + 1) / num_segments,
                    text=f"Segment {i+1}/{num_segments} done · {len(all_terms)} terms so far",
                )

            # Step 3: synthesis
            st.write("Synthesizing final framework...")
            taxonomy = list(all_terms.values())
            seen = set()
            unique_rels = []
            for r in all_relationships:
                if not isinstance(r, dict):
                    continue
                key = f"{r.get('from')}-{r.get('to')}-{r.get('type')}"
                if key not in seen:
                    unique_rels.append(r)
                    seen.add(key)

            synth_prompt = SYNTHESIS_PROMPT.format(
                taxonomy=json.dumps(taxonomy[:30], indent=2),
                relationships=json.dumps(unique_rels[:20], indent=2),
            )
            try:
                synth = call_ollama_blocking(synth_prompt, host, model)
                framework_summary = synth.get("framework_summary", "")
                compressed_context = synth.get("compressed_context", "")
            except Exception as e:
                st.warning(f"Synthesis failed: {e}")
                framework_summary = ""
                compressed_context = ""
                synth = {}

            analysis_id = f"analysis_{uuid.uuid4().hex[:12]}"
            conn = get_session_db()
            conn.execute(
                "INSERT INTO framework_analyses "
                "(id, document_id, taxonomy_json, framework_summary, compressed_context, "
                "relationships_json, raw_analysis) VALUES (?,?,?,?,?,?,?)",
                (
                    analysis_id,
                    st.session_state.document_id,
                    json.dumps(taxonomy),
                    framework_summary,
                    compressed_context,
                    json.dumps(unique_rels),
                    json.dumps(synth if isinstance(synth, dict) else {}),
                ),
            )
            conn.commit()
            conn.close()
            st.session_state.analysis_id = analysis_id

            status.update(
                label=f"Analysis complete · {len(taxonomy)} terms · {len(unique_rels)} relationships",
                state="complete",
            )

        analysis = get_analysis()
        if analysis:
            with st.expander(f"Concept preview — {len(analysis['taxonomy'])} terms"):
                for term in analysis["taxonomy"][:20]:
                    st.markdown(f"**{term.get('name')}** — {term.get('definition', '')}")
            with st.expander("Framework summary"):
                st.write(analysis.get("framework_summary", ""))
            with st.expander("Compressed context (used in prompts)"):
                st.write(analysis.get("compressed_context", ""))

    if st.session_state.analysis_id:
        if st.button("Continue to Chunking →", type="primary"):
            st.session_state.phase = "chunk"
            st.rerun()


def render_chunk_phase():
    st.markdown("### Phase 1 — Semantic Chunking")
    st.info(
        "Splits the document into conceptual units using sentence embeddings and cosine "
        "similarity. No LLM is used — this is purely mathematical."
    )

    threshold = st.session_state.settings["threshold"]
    st.metric("Similarity threshold", threshold, help="Lower = more chunks. Adjust in sidebar.")

    if st.button("Create Chunks", type="primary"):
        host = st.session_state.settings["ollama_host"]

        ok, err = check_model_available(host, "nomic-embed-text")
        if not ok:
            st.error("Embedding model missing. Run: `ollama pull nomic-embed-text`")
            st.stop()

        with st.status("Chunking document...", expanded=True) as status:
            doc_text = st.session_state.document_text

            chunker = SemanticChunker(
                ollama_host=host,
                similarity_threshold=threshold,
            )
            st.write("Segmenting into sentences...")
            st.write("Generating embeddings — this takes 2-3 minutes for a full book...")
            embed_progress = st.progress(0.0, text="Starting embeddings...")
            chunks = chunker.chunk_document(
                doc_text,
                st.session_state.document_id,
                progress_callback=lambda done, total: embed_progress.progress(
                    done / max(total, 1),
                    text=f"Embedding sentence {done}/{total}",
                ),
            )
            embed_progress.progress(1.0, text="Embeddings complete")
            st.write(f"Created {len(chunks)} semantic chunks")
            st.write("Storing chunks to database...")

            store_progress = st.progress(0.0, text="Storing chunks...")
            conn = get_session_db()
            for i, chunk in enumerate(chunks):
                chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
                conn.execute(
                    "INSERT OR REPLACE INTO chunks "
                    "(id, document_id, sequence_number, start_char, end_char, content, page_estimate, status) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        chunk_id,
                        st.session_state.document_id,
                        i,
                        chunk.get("start_char", 0),
                        chunk.get("end_char", 0),
                        chunk.get("content", ""),
                        chunk.get("page_estimate", 0),
                        "pending",
                    ),
                )
                store_progress.progress(
                    (i + 1) / len(chunks),
                    text=f"Stored {i+1}/{len(chunks)} chunks",
                )
            conn.commit()
            conn.close()

            status.update(label=f"Chunking complete · {len(chunks)} chunks", state="complete")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total chunks", len(chunks))
        col2.metric("Avg chunk length", f"{len(doc_text) // max(1, len(chunks))} chars")
        col3.metric("Triples to generate", len(chunks) * 10)

    chunks = get_chunks()
    if chunks:
        if st.button("Begin Generation →", type="primary"):
            st.session_state.phase = "generate"
            st.rerun()


def render_generate_phase():
    st.markdown("### Phase 2 — Triple Generation")
    chunks = get_chunks()
    analysis = get_analysis()
    total = len(chunks) * 10

    st.info(
        f"Generating {total} triples from {len(chunks)} chunks "
        f"(5 perspectives × 2 response styles each). "
        f"This will take time — do not close the browser."
    )

    stats = get_triple_stats()
    if stats["total"] > 0:
        st.warning(
            f"{stats['total']} triples already exist ({stats['approved']} approved). "
            "Generation will skip completed chunks."
        )

    if st.button("Start Generation", type="primary"):
        host = st.session_state.settings["ollama_host"]
        model = st.session_state.settings["model"]
        compressed_context = analysis.get("compressed_context", "") if analysis else ""

        angles = [
            ("first_encounter", "The user is encountering this concept for the first time"),
            ("identification", "The user is asking what this concept is and how to name it"),
            ("maximum_resistance", "The user is skeptical or pushing back on this concept"),
            ("yielding", "The user is beginning to accept or engage with this concept"),
            ("integration", "The user is connecting this concept to their broader understanding"),
        ]
        intensities = [
            ("acute", "Brief, direct exchange — terse questions and concise answers"),
            ("moderate", "Detailed exchange — fuller questions and thorough explanations"),
        ]

        progress_bar = st.progress(0.0, text="Starting generation...")
        chunk_progress = st.progress(0.0, text="Chunks...")
        status_text = st.empty()
        counter_text = st.empty()
        generated = 0
        skipped = 0

        conn = get_session_db()
        for ci, chunk in enumerate(chunks):
            chunk_id = chunk["id"]
            chunk_progress.progress(
                ci / len(chunks),
                text=f"Chunk {ci+1}/{len(chunks)} — {chunk['content'][:60].strip()}...",
            )
            for angle_name, angle_desc in angles:
                for intensity_name, intensity_desc in intensities:
                    exists = conn.execute(
                        "SELECT id FROM triples WHERE chunk_id=? AND angle=? AND intensity=?",
                        (chunk_id, angle_name, intensity_name),
                    ).fetchone()
                    if exists:
                        skipped += 1
                        generated += 1
                        progress_bar.progress(
                            min(generated / total, 1.0),
                            text=f"{generated}/{total} triples · {skipped} skipped",
                        )
                        continue

                    status_text.markdown(
                        f"**Chunk {chunk['sequence_number']+1}/{len(chunks)}** · "
                        f"{angle_name.replace('_', ' ')} · {intensity_name}"
                    )

                    try:
                        prompt_dict = build_generation_prompt(
                            chunk_content=chunk["content"],
                            angle=angle_name,
                            intensity=intensity_name,
                            compressed_context=compressed_context,
                        )
                        result = _generate_triple_json(
                            prompt_dict["system"],
                            model=model,
                            host=host,
                        )
                        violations = validate_triple(result)
                        triple_status = "needs_manual" if violations else "pending"
                        if violations:
                            result["thinking"] = (
                                f"VALIDATION FAILED: {'; '.join(violations)}. "
                                f"Original thinking: {result.get('thinking', '')}"
                            )
                        triple_id = f"triple_{uuid.uuid4().hex[:12]}"
                        conn.execute(
                            "INSERT INTO triples "
                            "(id, chunk_id, angle, intensity, system_prompt, user_message, "
                            "assistant_response, thinking_trace, status) "
                            "VALUES (?,?,?,?,?,?,?,?,?)",
                            (
                                triple_id, chunk_id, angle_name, intensity_name,
                                result.get("system", ""),
                                result.get("user", ""),
                                result.get("assistant", ""),
                                result.get("thinking", ""),
                                triple_status,
                            ),
                        )
                        conn.commit()
                    except Exception as e:
                        st.warning(f"Triple failed: {angle_name}/{intensity_name} · {str(e)[:60]}")

                    generated += 1
                    progress_bar.progress(
                        min(generated / total, 1.0),
                        text=f"{generated}/{total} triples · {skipped} skipped",
                    )
                    counter_text.caption(
                        f"{generated}/{total} processed · {skipped} skipped (already done)"
                    )

        chunk_progress.progress(1.0, text="All chunks processed")

        conn.close()
        final_stats = get_triple_stats()
        st.success(f"Generation complete · {final_stats['total']} triples created")

    if get_triple_stats()["total"] > 0:
        if st.button("Go to Review →", type="primary"):
            st.session_state.phase = "review"
            st.rerun()


def render_triple_card(triple: dict, chunk: dict, analysis: Optional[dict]):
    """Render a single triple review card."""
    angle_icons = {
        "first_encounter": "🔵",
        "identification": "🟣",
        "maximum_resistance": "🔴",
        "yielding": "🟢",
        "integration": "🟡",
    }
    status_icons = {
        "approved": "✅",
        "rejected": "❌",
        "pending": "⏳",
        "needs_manual": "⚠️",
    }
    icon = angle_icons.get(triple["angle"], "⚪")
    s_icon = status_icons.get(triple["status"], "⏳")
    intensity_label = "◆ TERSE" if triple["intensity"] == "acute" else "◇ detailed"
    tid = triple["id"]

    with st.container(border=True):
        header_col, status_col = st.columns([3, 1])
        with header_col:
            st.markdown(
                f"{icon} **{triple['angle'].replace('_', ' ').upper()}**  ·  {intensity_label}"
            )
        with status_col:
            st.markdown(f"{s_icon} `{triple['status'].upper()}`")

        # Quality indicator — shown before editable fields
        violations = validate_triple(triple)
        if violations:
            st.warning("Quality issues:\n" + "\n".join(f"• {v}" for v in violations))

        new_system = st.text_area(
            "SYSTEM", value=triple.get("system_prompt") or "",
            key=f"sys_{tid}", height=80,
        )
        new_user = st.text_area(
            "USER", value=triple.get("user_message") or "",
            key=f"usr_{tid}", height=65,
        )
        new_assistant = st.text_area(
            "ASSISTANT", value=triple.get("assistant_response") or "",
            key=f"ast_{tid}", height=100,
        )

        # Auto-save on change
        if (
            new_system != (triple.get("system_prompt") or "")
            or new_user != (triple.get("user_message") or "")
            or new_assistant != (triple.get("assistant_response") or "")
        ):
            update_triple(tid, new_system, new_user, new_assistant)

        think = triple.get("thinking_trace") or ""
        if think.strip():
            with st.expander("Reasoning trace"):
                st.caption(think)

        tags_raw = triple.get("tags_json") or "[]"
        try:
            tags_list = json.loads(tags_raw)
        except Exception:
            tags_list = []
        new_tags = st.text_input(
            "Tags", value=", ".join(tags_list),
            key=f"tag_{tid}", placeholder="comma-separated tags",
        )
        if new_tags != ", ".join(tags_list):
            update_triple_tags(tid, new_tags)

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("✓ Approve", key=f"app_{tid}", type="primary"):
                set_triple_status(tid, "approved")
                auto_backup()
                st.rerun()
        with b2:
            if st.button("✗ Reject", key=f"rej_{tid}"):
                set_triple_status(tid, "rejected")
                st.rerun()
        with b3:
            if st.button("↺ Regen", key=f"reg_{tid}"):
                host = st.session_state.settings["ollama_host"]
                model = st.session_state.settings["model"]
                analysis_data = get_analysis()
                ctx = analysis_data.get("compressed_context", "") if analysis_data else ""
                try:
                    prompt_dict = build_generation_prompt(
                        chunk_content=chunk["content"],
                        angle=triple["angle"],
                        intensity=triple["intensity"],
                        compressed_context=ctx,
                    )
                    result = _generate_triple_json(
                        prompt_dict["system"],
                        model=model,
                        host=host,
                    )
                    violations = validate_triple(result)
                    new_status = "needs_manual" if violations else "pending"
                    update_triple(
                        tid,
                        result.get("system", ""),
                        result.get("user", ""),
                        result.get("assistant", ""),
                    )
                    set_triple_status(tid, new_status)
                    st.rerun()
                except Exception as e:
                    st.error(f"Regeneration failed: {e}")
        with b4:
            if st.button("+ Variant", key=f"var_{tid}"):
                new_id = f"triple_{uuid.uuid4().hex[:12]}"
                conn = get_session_db()
                conn.execute(
                    "INSERT INTO triples (id, chunk_id, angle, intensity, status) VALUES (?,?,?,?,?)",
                    (new_id, triple["chunk_id"], triple["angle"], triple["intensity"], "needs_manual"),
                )
                conn.commit()
                conn.close()
                st.rerun()


def render_review_phase():
    chunks = get_chunks()
    if not chunks:
        st.warning("No chunks found. Return to processing.")
        if st.button("← Back to Processing"):
            st.session_state.phase = "chunk"
            st.rerun()
        return

    analysis = get_analysis()
    stats = get_triple_stats()

    nav1, nav2, nav3, nav4, nav5, nav6, nav7 = st.columns([2, 2, 2, 2, 2, 2, 2])
    with nav1:
        prev_col, next_col = st.columns(2)
        with prev_col:
            if st.button("◀", key="prev_chunk"):
                st.session_state.current_chunk_index = max(
                    0, st.session_state.current_chunk_index - 1
                )
                st.rerun()
        with next_col:
            if st.button("▶", key="next_chunk"):
                st.session_state.current_chunk_index = min(
                    len(chunks) - 1, st.session_state.current_chunk_index + 1
                )
                st.rerun()
    with nav2:
        jump = st.number_input(
            "Chunk", min_value=1, max_value=len(chunks),
            value=st.session_state.current_chunk_index + 1,
            key="chunk_jump", label_visibility="collapsed",
        )
        if jump - 1 != st.session_state.current_chunk_index:
            st.session_state.current_chunk_index = jump - 1
            st.rerun()
    with nav3:
        filter_options = ["all", "pending", "approved", "rejected", "needs_manual"]
        new_filter = st.selectbox(
            "Filter", filter_options,
            index=filter_options.index(st.session_state.filter_mode),
            label_visibility="collapsed",
        )
        if new_filter != st.session_state.filter_mode:
            st.session_state.filter_mode = new_filter
            st.rerun()
    with nav4:
        all_tags = _collect_all_tags()
        tag_options = ["all tags"] + sorted(set(all_tags))
        current_tag = st.session_state.get("tag_filter", "")
        tag_index = tag_options.index(current_tag) if current_tag in tag_options else 0
        selected_tag = st.selectbox(
            "Tag filter",
            tag_options,
            index=tag_index,
            key="tag_filter_select",
            label_visibility="collapsed",
        )
        new_tag_filter = "" if selected_tag == "all tags" else selected_tag
        if new_tag_filter != st.session_state.get("tag_filter", ""):
            st.session_state.tag_filter = new_tag_filter
            st.rerun()
    with nav5:
        st.metric("Approved", f"{stats['approved']}/{stats['total']}")
    with nav6:
        pct = int(stats["approved"] / max(1, stats["total"]) * 100)
        st.metric("Progress", f"{pct}%")
    with nav7:
        if st.button("Export JSONL", type="primary"):
            render_export_dialog()

    st.caption(
        f"Chunk {st.session_state.current_chunk_index + 1} of {len(chunks)}  ·  "
        f"pending: {stats['pending']}  ·  rejected: {stats['rejected']}  ·  "
        f"needs manual: {stats['needs_manual']}"
    )
    st.divider()

    left_col, right_col = st.columns([2, 3])
    idx = st.session_state.current_chunk_index
    chunk = chunks[idx]

    with left_col:
        st.markdown("**Source passage**")
        st.info(chunk["content"])
        st.caption(
            f"Chunk {idx+1} of {len(chunks)}  ·  ~page {int(chunk.get('page_estimate', 0)+1)}"
        )
        if analysis:
            matches = find_taxonomy_matches(chunk["content"], analysis.get("taxonomy", []))
            if matches:
                with st.expander(f"{len(matches)} concept matches"):
                    for term in matches:
                        st.markdown(f"**{term.get('name')}**  ·  {term.get('definition', '')}")

    with right_col:
        triples = get_triples_for_chunk(
            chunk["id"],
            st.session_state.filter_mode,
            tag_filter=st.session_state.get("tag_filter") or None,
        )
        if not triples:
            st.caption(f"No triples with status '{st.session_state.filter_mode}' for this chunk.")
        else:
            for triple in triples:
                render_triple_card(triple, chunk, analysis)


def render_export_dialog():
    st.divider()
    st.markdown("### Export Dataset")

    conn = get_session_db()
    approved = [dict(r) for r in conn.execute(
        "SELECT * FROM triples WHERE status='approved'"
    ).fetchall()]
    conn.close()

    if not approved:
        st.warning("No approved triples to export.")
        return

    def token_overlap(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(len(ta), len(tb))

    dupes = []
    for i in range(len(approved)):
        for j in range(i + 1, len(approved)):
            sim = token_overlap(
                approved[i].get("user_message", ""),
                approved[j].get("user_message", ""),
            )
            if sim > 0.8:
                dupes.append((i, j, sim))

    col1, col2, col3 = st.columns(3)
    col1.metric("Approved triples", len(approved))
    col2.metric("Potential duplicates", len(dupes))
    sid = st.session_state.session_id or "unknown"
    col3.metric("Session", sid[:16] + "..." if len(sid) > 16 else sid)

    if dupes:
        with st.expander(f"Review {len(dupes)} duplicate pairs"):
            for i, j, sim in dupes[:20]:
                st.caption(f"{int(sim*100)}% overlap — triple {i+1} vs {j+1}")
                st.text(approved[i].get("user_message", "")[:100])
                st.text(approved[j].get("user_message", "")[:100])
                st.divider()

    lines = []
    for t in approved:
        obj = {
            "messages": [
                {"role": "system", "content": t.get("system_prompt", "")},
                {"role": "user", "content": t.get("user_message", "")},
                {"role": "assistant", "content": t.get("assistant_response", "")},
            ]
        }
        lines.append(json.dumps(obj))
    jsonl_content = "\n".join(lines)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dataset_{date_str}.jsonl"

    angle_counts = Counter(t["angle"] for t in approved)
    intensity_counts = Counter(t["intensity"] for t in approved)
    all_tags = []
    for t in approved:
        try:
            all_tags.extend(json.loads(t.get("tags_json", "[]")))
        except Exception:
            pass
    tag_counts = Counter(all_tags)

    metadata = {
        "export_date": date_str,
        "document": st.session_state.document_name,
        "total_approved": len(approved),
        "duplicate_pairs_detected": len(dupes),
        "triples_by_angle": dict(angle_counts),
        "triples_by_intensity": dict(intensity_counts),
        "tags_breakdown": dict(tag_counts.most_common(20)),
    }

    dl_col, meta_col = st.columns(2)
    with dl_col:
        st.download_button(
            label=f"⬇ Download {filename}",
            data=jsonl_content,
            file_name=filename,
            mime="application/jsonl",
            type="primary",
        )
    with meta_col:
        st.download_button(
            label="⬇ Download metadata.json",
            data=json.dumps(metadata, indent=2),
            file_name=f"metadata_{date_str}.json",
            mime="application/json",
        )
    with st.expander("Export statistics"):
        st.json(metadata)


# ── SECTION 5: SIDEBAR ───────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚗️ The Forge")
        st.markdown("##### document → dataset")
        st.divider()

        st.markdown("**Settings**")
        st.session_state.settings["ollama_host"] = st.text_input(
            "Ollama host", value=st.session_state.settings["ollama_host"]
        )
        st.session_state.settings["model"] = st.text_input(
            "Model", value=st.session_state.settings["model"]
        )
        st.session_state.settings["threshold"] = st.slider(
            "Chunk sensitivity", 0.5, 0.95,
            value=st.session_state.settings["threshold"],
            step=0.05,
            help="Lower = more chunks",
        )
        st.session_state.settings["cross_reference"] = st.toggle(
            "Cross-reference pass",
            value=st.session_state.settings["cross_reference"],
        )

        st.divider()

        phase_labels = {
            "upload": "1 · Upload",
            "analyze": "2 · Analyze",
            "chunk": "3 · Chunk",
            "generate": "4 · Generate",
            "review": "5 · Review",
        }
        current = st.session_state.phase
        for phase, label in phase_labels.items():
            if phase == current:
                st.markdown(f"**→ {label}**")
            else:
                st.markdown(f"  {label}")

        st.divider()

        if st.session_state.phase == "review" and st.session_state.session_db_path:
            stats = get_triple_stats()
            st.markdown("**Session stats**")
            st.metric("Approved", stats["approved"])
            st.metric("Pending", stats["pending"])
            st.metric("Rejected", stats["rejected"])
            db_size = os.path.getsize(st.session_state.session_db_path) // 1024
            st.caption(f"DB size: {db_size} KB")

            st.divider()
            st.markdown("**Repair Pass**")
            repairable = stats["pending"] + stats.get("needs_manual", 0)
            st.caption(f"{repairable} triples eligible for repair")

            scope = st.selectbox(
                "Scope",
                ["needs_manual_only", "all_pending", "all_non_approved"],
                key="repair_scope",
            )
            if st.button("⚙ Run Repair Pass", type="secondary"):
                host = st.session_state.settings["ollama_host"]
                model = st.session_state.settings["model"]
                render_repair_pass(scope, host, model)

            if st.button("Review Repaired Triples"):
                st.session_state.filter_mode = "pending"
                st.session_state.tag_filter = "repaired"
                st.session_state.current_chunk_index = 0
                st.rerun()

        if st.session_state.phase != "upload":
            st.divider()
            if st.button("← New Session"):
                keep = {"settings"}
                for key in [k for k in st.session_state.keys() if k not in keep]:
                    del st.session_state[key]
                st.rerun()

        st.divider()
        if st.button("🗑 Clear all sessions", help="Delete all session databases from disk"):
            sessions_dir = Path("sessions")
            if sessions_dir.exists():
                import shutil
                shutil.rmtree(sessions_dir)
                sessions_dir.mkdir()
            # Clear active session state too
            for key in ["session_id", "session_db_path", "document_id",
                        "document_text", "document_name", "analysis_id"]:
                st.session_state[key] = None
            st.session_state.phase = "upload"
            st.rerun()


# ── SECTION 6: MAIN ROUTER ───────────────────────────────────────────────────

def main():
    render_sidebar()
    phase = st.session_state.phase
    if phase == "upload":
        render_upload_phase()
    elif phase == "analyze":
        render_analyze_phase()
    elif phase == "chunk":
        render_chunk_phase()
    elif phase == "generate":
        render_generate_phase()
    elif phase == "review":
        render_review_phase()
    else:
        st.error(f"Unknown phase: {phase}")
        if st.button("Reset"):
            st.session_state.phase = "upload"
            st.rerun()


main()
