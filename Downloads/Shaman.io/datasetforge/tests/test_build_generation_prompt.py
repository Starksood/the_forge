"""Unit tests for build_generation_prompt() in datasetforge.py."""
import sys
import pytest
from unittest.mock import MagicMock


def _import_module():
    st_mock = MagicMock()
    st_mock.set_page_config = MagicMock()
    sys.modules.setdefault('streamlit', st_mock)
    for mod in ['httpx', 'numpy', 'pypdf']:
        sys.modules.setdefault(mod, MagicMock())
    for mod in [
        'backend.phases', 'backend.phases.phase0_analysis',
        'backend.phases.semantic_chunker', 'backend.phases.phase2_generation',
    ]:
        sys.modules.setdefault(mod, MagicMock())
    import importlib
    if 'datasetforge' in sys.modules:
        return sys.modules['datasetforge']
    return importlib.import_module('datasetforge')


df = _import_module()
build_generation_prompt = df.build_generation_prompt

from backend.personas import (
    ANGLE_DEFINITIONS, INTENSITY_DEFINITIONS,
    GUIDE_PERSONA, GUIDE_INTEGRATION_VOICE,
)

VALID_ANGLES = list(ANGLE_DEFINITIONS.keys())
VALID_INTENSITIES = list(INTENSITY_DEFINITIONS.keys())


def test_returns_dict_with_four_keys_for_all_combinations():
    for angle in VALID_ANGLES:
        for intensity in VALID_INTENSITIES:
            result = build_generation_prompt("chunk text", angle, intensity, "context")
            assert isinstance(result, dict)
            assert set(result.keys()) == {"system", "user", "assistant", "thinking"}


def test_thinking_is_always_empty_string():
    result = build_generation_prompt("chunk", "first_encounter", "acute", "ctx")
    assert result["thinking"] == ""


def test_system_contains_compressed_context():
    ctx = "unique_context_string_xyz"
    result = build_generation_prompt("chunk", "first_encounter", "moderate", ctx)
    assert ctx in result["system"]


def test_system_contains_angle_description():
    result = build_generation_prompt("chunk", "maximum_resistance", "acute", "ctx")
    assert ANGLE_DEFINITIONS["maximum_resistance"]["name"] in result["system"]


def test_system_contains_intensity_description():
    result = build_generation_prompt("chunk", "first_encounter", "acute", "ctx")
    assert INTENSITY_DEFINITIONS["acute"]["name"] in result["system"]


def test_integration_angle_uses_integration_voice():
    result = build_generation_prompt("chunk", "integration", "moderate", "ctx")
    # GUIDE_INTEGRATION_VOICE content should appear — check for its distinctive header
    assert "SYNTHESIS VOICE" in result["system"] or "synthesis" in result["system"].lower()


def test_non_integration_angle_uses_guide_persona():
    result = build_generation_prompt("chunk", "first_encounter", "moderate", "ctx")
    assert "THE ASSISTANT" in result["system"]


def test_invalid_angle_raises_value_error():
    with pytest.raises(ValueError):
        build_generation_prompt("chunk", "unknown_angle", "moderate", "ctx")


def test_invalid_intensity_raises_value_error():
    with pytest.raises(ValueError):
        build_generation_prompt("chunk", "first_encounter", "unknown_intensity", "ctx")


def test_chunk_content_truncated_to_3000_chars():
    long_chunk = "x" * 5000
    result = build_generation_prompt(long_chunk, "first_encounter", "moderate", "ctx")
    assert long_chunk not in result["system"]
    assert "x" * 3000 in result["system"]
    assert "x" * 3001 not in result["system"]


def test_system_contains_translation_instruction():
    result = build_generation_prompt("chunk", "yielding", "moderate", "ctx")
    system_lower = result["system"].lower()
    assert "do not" in system_lower or "construct a new scene" in system_lower
