"""Unit tests for repair_triple_with_ollama() in datasetforge.py."""
import sys
import json
import types
import pytest
from unittest.mock import MagicMock, patch


def _import_repair_triple():
    """Import repair_triple_with_ollama by mocking streamlit and heavy deps."""
    st_mock = MagicMock()
    st_mock.set_page_config = MagicMock()
    sys.modules.setdefault('streamlit', st_mock)

    for mod in ['httpx', 'numpy', 'pypdf']:
        sys.modules.setdefault(mod, MagicMock())

    # Build a proper backend package mock so sub-module imports work
    backend_mock = types.ModuleType('backend')
    sys.modules['backend'] = backend_mock

    for mod in [
        'backend.phases', 'backend.phases.phase0_analysis',
        'backend.phases.semantic_chunker', 'backend.phases.phase2_generation',
    ]:
        sys.modules.setdefault(mod, MagicMock())

    # Mock backend.personas with the real constants needed by datasetforge.py
    personas_mock = MagicMock()
    personas_mock.GUIDE_PERSONA = "GUIDE_PERSONA"
    personas_mock.GUIDE_INTEGRATION_VOICE = "GUIDE_INTEGRATION_VOICE"
    personas_mock.TRAVELER_PERSONA = "TRAVELER_PERSONA"
    personas_mock.ANGLE_DEFINITIONS = {
        "first_encounter": {"name": "First Encounter", "description": "...", "guide_orientation": "..."},
        "identification": {"name": "Identification", "description": "...", "guide_orientation": "..."},
        "maximum_resistance": {"name": "Maximum Resistance", "description": "...", "guide_orientation": "..."},
        "yielding": {"name": "Yielding", "description": "...", "guide_orientation": "..."},
        "integration": {"name": "Integration", "description": "...", "guide_orientation": "..."},
    }
    personas_mock.INTENSITY_DEFINITIONS = {
        "acute": {"name": "Acute", "description": "..."},
        "moderate": {"name": "Moderate", "description": "..."},
    }
    sys.modules['backend.personas'] = personas_mock
    backend_mock.personas = personas_mock

    # Mock backend.ollama_client with a generate_sync placeholder
    ollama_mock = types.ModuleType('backend.ollama_client')
    ollama_mock.generate_sync = MagicMock(return_value='{}')
    sys.modules['backend.ollama_client'] = ollama_mock
    backend_mock.ollama_client = ollama_mock

    import importlib
    if 'datasetforge' in sys.modules:
        df = sys.modules['datasetforge']
    else:
        df = importlib.import_module('datasetforge')
    return df.repair_triple_with_ollama


repair_triple_with_ollama = _import_repair_triple()

# A minimal valid triple that passes validate_triple
VALID_TRIPLE = {
    "id": "triple_abc123",
    "chunk_id": "chunk_001",
    "angle": "yielding",
    "intensity": "moderate",
    "system_prompt": "You are SHAMAN.OS, a calm psychedelic guide.",
    "user_message": "Something is dissolving and I cannot hold on.",
    "assistant_response": "I am here with you. Stay with the breath. This will pass.",
    "thinking_trace": "",
    "status": "needs_manual",
    "tags_json": "[]",
    "is_cross_reference": 0,
}

# A triple with a non-first-person assistant response (will fail validation)
INVALID_TRIPLE = {
    **VALID_TRIPLE,
    "assistant_response": "He felt calm and the storm passed slowly away.",
}


def _make_valid_repair_response() -> str:
    """Return a JSON string that passes validate_triple."""
    return json.dumps({
        "system": "You are SHAMAN.OS, a calm psychedelic guide.",
        "user": "Something is dissolving and I cannot hold on.",
        "assistant": "I am here with you. Stay with the breath. This will pass.",
    })


def _make_invalid_repair_response() -> str:
    """Return a JSON string whose assistant field fails first-person check."""
    return json.dumps({
        "system": "You are SHAMAN.OS, a calm psychedelic guide.",
        "user": "Something is dissolving and I cannot hold on.",
        "assistant": "He felt calm and the storm passed slowly away.",
    })


# ── Test 1: valid repair response → status='pending', 'repaired' in tags ─────

def test_valid_repair_sets_pending_and_repaired_tag():
    """
    When Ollama returns a valid repair response, the result should have
    status='pending' and 'repaired' in tags_json.

    Validates: Requirements 4.2, 4.3
    """
    with patch('datasetforge.generate_sync', return_value=_make_valid_repair_response()):
        import datasetforge as df
        result = df.repair_triple_with_ollama(
            dict(INVALID_TRIPLE), host="http://localhost:11434", model="gemma3:4b"
        )

    assert result['status'] == 'pending', f"Expected 'pending', got '{result['status']}'"
    tags = json.loads(result['tags_json'])
    assert 'repaired' in tags, f"Expected 'repaired' in tags, got {tags}"


# ── Test 2: invalid repair response → status='needs_manual', 'repaired' in tags

def test_invalid_repair_sets_needs_manual_and_repaired_tag():
    """
    When Ollama returns a repair response that still fails validation,
    the result should have status='needs_manual' and 'repaired' in tags_json.

    Validates: Requirements 4.2, 4.4
    """
    with patch('datasetforge.generate_sync', return_value=_make_invalid_repair_response()):
        import datasetforge as df
        result = df.repair_triple_with_ollama(
            dict(INVALID_TRIPLE), host="http://localhost:11434", model="gemma3:4b"
        )

    assert result['status'] == 'needs_manual', f"Expected 'needs_manual', got '{result['status']}'"
    tags = json.loads(result['tags_json'])
    assert 'repaired' in tags, f"Expected 'repaired' in tags, got {tags}"


# ── Test 3: network error → original content preserved, needs_manual, repaired tag

def test_network_error_preserves_original_and_sets_needs_manual():
    """
    When Ollama raises a network error, the original triple content must be
    preserved, status set to 'needs_manual', and 'repaired' added to tags.

    Validates: Requirements 4.1, 4.4
    """
    with patch('datasetforge.generate_sync', side_effect=ConnectionError("network failure")):
        import datasetforge as df
        original = dict(INVALID_TRIPLE)
        result = df.repair_triple_with_ollama(
            original, host="http://localhost:11434", model="gemma3:4b"
        )

    # Original content preserved
    assert result['system_prompt'] == INVALID_TRIPLE['system_prompt']
    assert result['user_message'] == INVALID_TRIPLE['user_message']
    assert result['assistant_response'] == INVALID_TRIPLE['assistant_response']

    # Status and tag
    assert result['status'] == 'needs_manual', f"Expected 'needs_manual', got '{result['status']}'"
    tags = json.loads(result['tags_json'])
    assert 'repaired' in tags, f"Expected 'repaired' in tags, got {tags}"


# ── Additional edge cases ─────────────────────────────────────────────────────

def test_repaired_tag_not_duplicated_if_already_present():
    """'repaired' should appear exactly once even if already in tags_json."""
    triple_with_tag = {**INVALID_TRIPLE, "tags_json": '["repaired"]'}
    with patch('datasetforge.generate_sync', side_effect=ConnectionError("fail")):
        import datasetforge as df
        result = df.repair_triple_with_ollama(
            dict(triple_with_tag), host="http://localhost:11434", model="gemma3:4b"
        )
    tags = json.loads(result['tags_json'])
    assert tags.count('repaired') == 1


def test_json_parse_error_falls_back_to_needs_manual():
    """When Ollama returns malformed JSON, fall back to needs_manual + repaired tag."""
    with patch('datasetforge.generate_sync', return_value="not valid json {{{{"):
        import datasetforge as df
        result = df.repair_triple_with_ollama(
            dict(INVALID_TRIPLE), host="http://localhost:11434", model="gemma3:4b"
        )
    assert result['status'] == 'needs_manual'
    tags = json.loads(result['tags_json'])
    assert 'repaired' in tags
