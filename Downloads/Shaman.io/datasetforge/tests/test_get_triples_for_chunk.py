"""Unit tests for get_triples_for_chunk() tag filter in datasetforge.py."""
import sys
import json
import sqlite3
import types
import pytest
from unittest.mock import MagicMock, patch


def _import_module():
    st_mock = MagicMock()
    st_mock.set_page_config = MagicMock()
    sys.modules.setdefault('streamlit', st_mock)
    for mod in ['httpx', 'numpy', 'pypdf']:
        sys.modules.setdefault(mod, MagicMock())
    backend_mock = types.ModuleType('backend')
    sys.modules['backend'] = backend_mock
    for mod in ['backend.phases', 'backend.phases.phase0_analysis',
                'backend.phases.semantic_chunker', 'backend.phases.phase2_generation']:
        sys.modules.setdefault(mod, MagicMock())
    personas_mock = MagicMock()
    personas_mock.GUIDE_PERSONA = "GP"
    personas_mock.GUIDE_INTEGRATION_VOICE = "GIV"
    personas_mock.TRAVELER_PERSONA = "TP"
    personas_mock.ANGLE_DEFINITIONS = {k: {"name": k, "description": "d", "guide_orientation": "g"} for k in ["first_encounter","identification","maximum_resistance","yielding","integration"]}
    personas_mock.INTENSITY_DEFINITIONS = {k: {"name": k, "description": "d"} for k in ["acute","moderate"]}
    sys.modules['backend.personas'] = personas_mock
    backend_mock.personas = personas_mock
    ollama_mock = types.ModuleType('backend.ollama_client')
    ollama_mock.generate_sync = MagicMock(return_value='{}')
    sys.modules['backend.ollama_client'] = ollama_mock
    backend_mock.ollama_client = ollama_mock
    import importlib
    if 'datasetforge' in sys.modules:
        return sys.modules['datasetforge']
    return importlib.import_module('datasetforge')


df = _import_module()
get_triples_for_chunk = df.get_triples_for_chunk


def _make_in_memory_db():
    """Create an in-memory SQLite DB with the triples table schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE triples (
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
            created_at TEXT,
            updated_at TEXT
        )
    """)
    return conn


def _insert_triple(conn, id, chunk_id, status="pending", tags=None):
    tags_json = json.dumps(tags or [])
    conn.execute(
        "INSERT INTO triples (id, chunk_id, angle, intensity, system_prompt, user_message, assistant_response, status, tags_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (id, chunk_id, "yielding", "moderate", "sys", "user msg", "I am here.", status, tags_json)
    )
    conn.commit()


def test_tag_filter_returns_only_matching_triples():
    conn = _make_in_memory_db()
    _insert_triple(conn, "t1", "chunk1", tags=["repaired"])
    _insert_triple(conn, "t2", "chunk1", tags=["reviewed"])
    _insert_triple(conn, "t3", "chunk1", tags=[])

    with patch.object(df, 'get_session_db', return_value=conn):
        result = get_triples_for_chunk("chunk1", tag_filter="repaired")

    assert len(result) == 1
    assert result[0]["id"] == "t1"


def test_tag_filter_none_returns_all_triples():
    conn = _make_in_memory_db()
    _insert_triple(conn, "t1", "chunk1", tags=["repaired"])
    _insert_triple(conn, "t2", "chunk1", tags=[])
    _insert_triple(conn, "t3", "chunk1", tags=["reviewed"])

    with patch.object(df, 'get_session_db', return_value=conn):
        result = get_triples_for_chunk("chunk1", tag_filter=None)

    assert len(result) == 3


def test_tag_filter_nonexistent_returns_empty_list():
    conn = _make_in_memory_db()
    _insert_triple(conn, "t1", "chunk1", tags=["repaired"])
    _insert_triple(conn, "t2", "chunk1", tags=["reviewed"])

    with patch.object(df, 'get_session_db', return_value=conn):
        result = get_triples_for_chunk("chunk1", tag_filter="nonexistent")

    assert result == []


def test_tag_filter_combined_with_filter_mode():
    conn = _make_in_memory_db()
    _insert_triple(conn, "t1", "chunk1", status="pending", tags=["repaired"])
    _insert_triple(conn, "t2", "chunk1", status="approved", tags=["repaired"])
    _insert_triple(conn, "t3", "chunk1", status="pending", tags=[])

    with patch.object(df, 'get_session_db', return_value=conn):
        result = get_triples_for_chunk("chunk1", filter_mode="pending", tag_filter="repaired")

    assert len(result) == 1
    assert result[0]["id"] == "t1"


def test_malformed_tags_json_excluded_from_tag_filtered_results():
    conn = _make_in_memory_db()
    conn.execute(
        "INSERT INTO triples (id, chunk_id, angle, intensity, system_prompt, user_message, assistant_response, status, tags_json) VALUES (?,?,?,?,?,?,?,?,?)",
        ("t1", "chunk1", "yielding", "moderate", "sys", "user", "I am here.", "pending", "not valid json")
    )
    conn.commit()

    with patch.object(df, 'get_session_db', return_value=conn):
        result = get_triples_for_chunk("chunk1", tag_filter="repaired")

    assert result == []
