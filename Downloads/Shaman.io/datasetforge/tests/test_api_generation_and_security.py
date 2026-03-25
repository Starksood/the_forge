"""Regression tests for generation persistence and API security/session scoping."""
import json
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import api, database
from backend.database import DatabaseManager, create_session_database
from backend.models import Chunk, Document, Triple
import backend.session_manager as session_manager_module


def _create_session_with_doc_and_chunk(tmpdir: str):
    original_sessions_dir = database.SESSIONS_DIR
    original_sm_sessions_dir = session_manager_module.SESSIONS_DIR
    database.SESSIONS_DIR = tmpdir
    session_manager_module.SESSIONS_DIR = tmpdir
    try:
        session_id = "test_session_generation"
        db_path = create_session_database(session_id, "Test Doc")
        db_manager = DatabaseManager(db_path)

        document_id = "doc_test_1"
        db_manager.create_document(
            document_id=document_id,
            session_id=session_id,
            filename="doc.txt",
            content="A short passage for testing generation persistence.",
            page_count=1,
        )
        db_manager.create_framework_analysis(
            analysis_id="analysis_test_1",
            document_id=document_id,
            taxonomy=[],
            framework_summary="Test framework summary",
            relationships=[{"from": "A", "to": "B", "connection": "links"}],
            raw_analysis="raw",
        )
        db_manager.create_chunk(
            chunk_id="chunk_test_1",
            document_id=document_id,
            sequence_number=0,
            start_char=0,
            end_char=40,
            content="Chunk test content",
            page_estimate=1.0,
        )
        return session_id, db_path, db_manager, original_sessions_dir, original_sm_sessions_dir
    except Exception:
        database.SESSIONS_DIR = original_sessions_dir
        session_manager_module.SESSIONS_DIR = original_sm_sessions_dir
        raise


def test_generate_stream_persists_triples():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_id, db_path, db_manager, original_sessions_dir, original_sm_sessions_dir = _create_session_with_doc_and_chunk(tmpdir)
        api.current_session_db = None  # force query-param scoped path
        client = TestClient(api.app)

        async def fake_phase2_streaming(*args, **kwargs):
            yield "data: " + json.dumps({
                "type": "triple",
                "chunk_index": 0,
                "angle": "first_encounter",
                "intensity": "acute",
                "triple": {
                    "system_text": "sys",
                    "user_text": "usr",
                    "assistant_text": "ast",
                    "think_block": "think",
                },
            }) + "\n\n"
            yield "data: " + json.dumps({"type": "done"}) + "\n\n"

        try:
            with patch("backend.api.phase2_generation.run_phase2_streaming", fake_phase2_streaming):
                resp = client.post(f"/api/documents/generate?session_id={session_id}")
                assert resp.status_code == 200

            with db_manager.get_session() as db_session:
                chunk = db_session.query(Chunk).filter(Chunk.id == "chunk_test_1").first()
                assert chunk is not None
                triples = db_session.query(Triple).filter(Triple.chunk_id == chunk.id).all()
                assert len(triples) == 1
                assert triples[0].system_prompt == "sys"
                assert triples[0].status in ("pending", "needs_manual")
        finally:
            api.current_session_db = None
            database.SESSIONS_DIR = original_sessions_dir
            session_manager_module.SESSIONS_DIR = original_sm_sessions_dir


def test_cross_reference_stream_persists_triples():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_id, _, db_manager, original_sessions_dir, original_sm_sessions_dir = _create_session_with_doc_and_chunk(tmpdir)
        api.current_session_db = None
        client = TestClient(api.app)

        async def fake_phase3_streaming(*args, **kwargs):
            yield "data: " + json.dumps({
                "type": "triple",
                "triple": {
                    "system_text": "xref sys",
                    "user_text": "xref usr",
                    "assistant_text": "xref ast",
                    "think_block": "xref think",
                },
            }) + "\n\n"
            yield "data: " + json.dumps({"type": "done"}) + "\n\n"

        try:
            with db_manager.get_session() as db_session:
                doc = db_session.query(Document).first()
                assert doc is not None
                document_id = doc.id

            with patch("backend.api.phase3_crossref.run_phase3_streaming", fake_phase3_streaming):
                resp = client.post(f"/api/documents/{document_id}/cross-reference?session_id={session_id}")
                assert resp.status_code == 200

            with db_manager.get_session() as db_session:
                triples = db_session.query(Triple).filter(Triple.is_cross_reference.is_(True)).all()
                assert len(triples) == 1
                assert triples[0].system_prompt == "xref sys"
        finally:
            api.current_session_db = None
            database.SESSIONS_DIR = original_sessions_dir
            session_manager_module.SESSIONS_DIR = original_sm_sessions_dir


def test_api_key_protects_non_public_routes():
    client = TestClient(api.app)
    original_api_key = api.API_KEY
    api.API_KEY = "secret-key"
    try:
        unauthorized = client.get("/api/sessions")
        assert unauthorized.status_code == 401

        authorized = client.get("/api/sessions", headers={"x-api-key": "secret-key"})
        # Endpoint may succeed or fail for other reasons, but should not be unauthorized
        assert authorized.status_code != 401
    finally:
        api.API_KEY = original_api_key
