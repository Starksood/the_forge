"""Unit tests for validate_triple() in datasetforge.py."""
import sys
import types
import pytest
from unittest.mock import MagicMock, patch


def _import_validate_triple():
    """Import validate_triple by mocking streamlit and other heavy deps."""
    # Mock streamlit before importing datasetforge
    st_mock = MagicMock()
    st_mock.set_page_config = MagicMock()
    sys.modules.setdefault('streamlit', st_mock)

    # Mock other potentially-missing deps
    for mod in ['httpx', 'numpy', 'pypdf']:
        sys.modules.setdefault(mod, MagicMock())

    # Build a proper backend package mock so sub-module imports work
    import types as _types
    if 'backend' not in sys.modules or not isinstance(sys.modules['backend'], _types.ModuleType):
        backend_mock = _types.ModuleType('backend')
        sys.modules['backend'] = backend_mock
    else:
        backend_mock = sys.modules['backend']

    # Mock backend sub-modules
    for mod in [
        'backend.phases', 'backend.phases.phase0_analysis',
        'backend.phases.semantic_chunker', 'backend.phases.phase2_generation',
    ]:
        sys.modules.setdefault(mod, MagicMock())

    # Mock backend.personas
    if 'backend.personas' not in sys.modules:
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

    # Mock backend.ollama_client
    if 'backend.ollama_client' not in sys.modules:
        ollama_mock = _types.ModuleType('backend.ollama_client')
        ollama_mock.generate_sync = MagicMock(return_value='{}')
        sys.modules['backend.ollama_client'] = ollama_mock
        backend_mock.ollama_client = ollama_mock

    # Now import
    import importlib
    if 'datasetforge' in sys.modules:
        df = sys.modules['datasetforge']
    else:
        df = importlib.import_module('datasetforge')
    return df.validate_triple, df.ANALYSIS_PHRASES


validate_triple, ANALYSIS_PHRASES = _import_validate_triple()


# ── Required fields ──────────────────────────────────────────────────────────

def test_empty_triple_returns_violations_for_all_three_fields():
    violations = validate_triple({})
    messages = " ".join(violations)
    assert "system_prompt" in messages
    assert "user_message" in messages
    assert "assistant_response" in messages


def test_missing_system_prompt_returns_violation():
    triple = {
        "user_message": "I feel lost.",
        "assistant_response": "I am here with you, steady and present in this moment.",
    }
    violations = validate_triple(triple)
    assert any("system_prompt" in v for v in violations)


def test_empty_system_prompt_returns_violation():
    triple = {
        "system_prompt": "   ",
        "user_message": "I feel lost.",
        "assistant_response": "I am here with you, steady and present in this moment.",
    }
    violations = validate_triple(triple)
    assert any("system_prompt" in v for v in violations)


def test_missing_user_message_returns_violation():
    triple = {
        "system_prompt": "You are a guide.",
        "assistant_response": "I am here with you, steady and present in this moment.",
    }
    violations = validate_triple(triple)
    assert any("user_message" in v for v in violations)


def test_missing_assistant_response_returns_violation():
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "Something is happening.",
    }
    violations = validate_triple(triple)
    assert any("assistant_response" in v for v in violations)


# ── Analysis language detection ───────────────────────────────────────────────

def test_user_message_with_this_passage_returns_violation():
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "This passage describes a difficult experience.",
        "assistant_response": "I am here with you, steady and present in this moment.",
    }
    violations = validate_triple(triple)
    assert any("user_message" in v for v in violations)


def test_user_message_with_the_text_returns_violation():
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "The text says something about fear.",
        "assistant_response": "I am here with you, steady and present in this moment.",
    }
    violations = validate_triple(triple)
    assert any("user_message" in v for v in violations)


def test_user_message_analysis_language_case_insensitive():
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "THE AUTHOR wrote about this.",
        "assistant_response": "I am here with you, steady and present in this moment.",
    }
    violations = validate_triple(triple)
    assert any("user_message" in v for v in violations)


def test_user_message_without_analysis_language_no_violation():
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "Something is dissolving and I cannot hold on.",
        "assistant_response": "I am here with you, steady and present in this moment.",
    }
    violations = validate_triple(triple)
    assert not any("user_message" in v for v in violations)


# ── First-person present tense ────────────────────────────────────────────────

def test_assistant_response_non_first_person_returns_violation():
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "Something is dissolving.",
        "assistant_response": "He felt calm and the storm passed slowly away.",
    }
    violations = validate_triple(triple)
    assert any("assistant_response" in v and "first-person" in v for v in violations)


def test_assistant_response_starting_with_I_space_passes():
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "Something is dissolving.",
        "assistant_response": "I see you there, and I am not going anywhere right now.",
    }
    violations = validate_triple(triple)
    assert not any("first-person" in v for v in violations)


def test_assistant_response_with_first_person_verb_passes():
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "Something is dissolving.",
        "assistant_response": "Stay with me. I feel the weight of this moment with you.",
    }
    violations = validate_triple(triple)
    assert not any("first-person" in v for v in violations)


# ── Length bounds ─────────────────────────────────────────────────────────────

def test_assistant_response_too_short_returns_violation():
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "Something is dissolving.",
        "assistant_response": "I",  # 1 char
    }
    violations = validate_triple(triple)
    assert any("short" in v or "minimum" in v or "chars" in v for v in violations)


def test_assistant_response_exactly_39_chars_returns_violation():
    response = "I " + "x" * 37  # 39 chars total
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "Something is dissolving.",
        "assistant_response": response,
    }
    violations = validate_triple(triple)
    assert any("short" in v or "minimum" in v or "chars" in v for v in violations)


def test_assistant_response_exactly_40_chars_passes_length():
    response = "I " + "x" * 38  # 40 chars total
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "Something is dissolving.",
        "assistant_response": response,
    }
    violations = validate_triple(triple)
    assert not any("short" in v or "minimum" in v for v in violations)


def test_assistant_response_too_long_returns_violation():
    response = "I " + "x" * 601  # 603 chars
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "Something is dissolving.",
        "assistant_response": response,
    }
    violations = validate_triple(triple)
    assert any("long" in v or "maximum" in v or "chars" in v for v in violations)


def test_assistant_response_exactly_600_chars_passes_length():
    response = "I " + "x" * 598  # 600 chars
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "Something is dissolving.",
        "assistant_response": response,
    }
    violations = validate_triple(triple)
    assert not any("long" in v or "maximum" in v for v in violations)


# ── Valid triple ──────────────────────────────────────────────────────────────

def test_valid_triple_returns_empty_violations():
    triple = {
        "system_prompt": "You are SHAMAN.OS, a calm psychedelic guide.",
        "user_message": "Something is dissolving and I cannot hold on.",
        "assistant_response": "I am here with you. Stay with the breath. This will pass.",
    }
    violations = validate_triple(triple)
    assert violations == []


# ── Alternate field name variants ─────────────────────────────────────────────

def test_alternate_field_names_system_user_assistant():
    """validate_triple must handle system/user/assistant field names too."""
    triple = {
        "system": "You are SHAMAN.OS.",
        "user": "Something is dissolving.",
        "assistant": "I am here with you. Stay with the breath. This will pass.",
    }
    violations = validate_triple(triple)
    assert violations == []


def test_never_raises_on_unexpected_input():
    """validate_triple must never raise regardless of input."""
    for bad_input in [{}, {"foo": "bar"}, {"system_prompt": None}, {"assistant_response": 123}]:
        try:
            result = validate_triple(bad_input)
            assert isinstance(result, list)
        except Exception as e:
            pytest.fail(f"validate_triple raised unexpectedly: {e}")
