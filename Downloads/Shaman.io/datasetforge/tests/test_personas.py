"""Unit tests for backend/personas.py — persona and angle/intensity constants."""

import pytest
from backend.personas import (
    GUIDE_PERSONA,
    GUIDE_INTEGRATION_VOICE,
    TRAVELER_PERSONA,
    ANGLE_DEFINITIONS,
    INTENSITY_DEFINITIONS,
)


def test_guide_persona_exists_and_non_empty():
    assert isinstance(GUIDE_PERSONA, str)
    assert len(GUIDE_PERSONA.strip()) > 0


def test_traveler_persona_exists_and_non_empty():
    assert isinstance(TRAVELER_PERSONA, str)
    assert len(TRAVELER_PERSONA.strip()) > 0


def test_guide_integration_voice_exists_and_non_empty():
    assert isinstance(GUIDE_INTEGRATION_VOICE, str)
    assert len(GUIDE_INTEGRATION_VOICE.strip()) > 0


def test_guide_integration_voice_differs_from_guide_persona():
    assert GUIDE_INTEGRATION_VOICE != GUIDE_PERSONA


def test_angle_definitions_has_exactly_five_keys():
    expected_keys = {"first_encounter", "identification", "maximum_resistance", "yielding", "integration"}
    assert set(ANGLE_DEFINITIONS.keys()) == expected_keys


def test_angle_definitions_values_are_non_empty_dicts():
    for key, value in ANGLE_DEFINITIONS.items():
        assert isinstance(value, dict), f"ANGLE_DEFINITIONS[{key!r}] should be a dict"
        assert value.get("name"), f"ANGLE_DEFINITIONS[{key!r}]['name'] should be non-empty"
        assert value.get("description"), f"ANGLE_DEFINITIONS[{key!r}]['description'] should be non-empty"
        assert value.get("guide_orientation"), f"ANGLE_DEFINITIONS[{key!r}]['guide_orientation'] should be non-empty"


def test_intensity_definitions_has_exactly_two_keys():
    expected_keys = {"acute", "moderate"}
    assert set(INTENSITY_DEFINITIONS.keys()) == expected_keys


def test_intensity_definitions_values_are_non_empty_dicts():
    for key, value in INTENSITY_DEFINITIONS.items():
        assert isinstance(value, dict), f"INTENSITY_DEFINITIONS[{key!r}] should be a dict"
        assert value.get("name"), f"INTENSITY_DEFINITIONS[{key!r}]['name'] should be non-empty"
        assert value.get("description"), f"INTENSITY_DEFINITIONS[{key!r}]['description'] should be non-empty"


def test_import_has_no_side_effects():
    """Importing the module should not raise or mutate anything unexpected."""
    import importlib
    import sys
    import types
    module_name = "backend.personas"
    module = sys.modules.get(module_name)
    # Only reload if it's a real module (not a mock or stub)
    if isinstance(module, types.ModuleType):
        reloaded = importlib.reload(module)
        assert reloaded.GUIDE_PERSONA == GUIDE_PERSONA
        assert reloaded.TRAVELER_PERSONA == TRAVELER_PERSONA
        assert reloaded.GUIDE_INTEGRATION_VOICE == GUIDE_INTEGRATION_VOICE
    else:
        # Constants are accessible from the already-imported module at top of file
        assert GUIDE_PERSONA
        assert TRAVELER_PERSONA
        assert GUIDE_INTEGRATION_VOICE
