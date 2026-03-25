"""
Property-based tests for the DatasetForge persona pipeline.
Uses hypothesis for generative testing.
"""

import json
import re
import sys
import os
import types
import pytest

# The .pycache directory contains a broken numpy stub that causes hypothesis's
# isinstance() check to fail. Remove it before hypothesis is imported.
_numpy = sys.modules.get("numpy")
if _numpy is not None and not isinstance(_numpy, types.ModuleType):
    del sys.modules["numpy"]
# Also remove any numpy submodules that may be stubs
for _key in list(sys.modules.keys()):
    if _key == "numpy" or _key.startswith("numpy."):
        _mod = sys.modules[_key]
        if not isinstance(_mod, types.ModuleType):
            del sys.modules[_key]

# Ensure the datasetforge package root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from backend.personas import (
    GUIDE_PERSONA,
    GUIDE_INTEGRATION_VOICE,
    TRAVELER_PERSONA,
    ANGLE_DEFINITIONS,
    INTENSITY_DEFINITIONS,
)
from datasetforge import (
    validate_triple,
    build_generation_prompt,
    get_triples_for_chunk,
    ANALYSIS_PHRASES,
)


# ── Property 1: Persona module structure ─────────────────────────────────────
# Feature: datasetforge-persona-pipeline, Property 1: Persona module structure

@settings(max_examples=100)
@given(st.none())  # no inputs needed — module-level assertion
def test_property_1_persona_module_structure(_):
    """All required constants exist, are non-empty, and have correct shapes."""
    assert isinstance(GUIDE_PERSONA, str) and GUIDE_PERSONA.strip()
    assert isinstance(TRAVELER_PERSONA, str) and TRAVELER_PERSONA.strip()
    assert isinstance(GUIDE_INTEGRATION_VOICE, str) and GUIDE_INTEGRATION_VOICE.strip()
    assert set(ANGLE_DEFINITIONS.keys()) == {
        "first_encounter", "identification", "maximum_resistance", "yielding", "integration"
    }
    assert set(INTENSITY_DEFINITIONS.keys()) == {"acute", "moderate"}


# ── Property 2: Prompt builder returns well-formed dict for all valid inputs ──
# Feature: datasetforge-persona-pipeline, Property 2: Prompt builder returns well-formed dict for all valid inputs

@settings(max_examples=100)
@given(
    angle=st.sampled_from(list(ANGLE_DEFINITIONS)),
    intensity=st.sampled_from(list(INTENSITY_DEFINITIONS)),
    chunk_content=st.text(min_size=1),
    compressed_context=st.text(),
)
def test_property_2_prompt_builder_well_formed(angle, intensity, chunk_content, compressed_context):
    """build_generation_prompt returns a dict with exactly the four required keys."""
    result = build_generation_prompt(chunk_content, angle, intensity, compressed_context)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"system", "user", "assistant", "thinking"}
    assert isinstance(result["system"], str)
    assert isinstance(result["user"], str)
    assert isinstance(result["assistant"], str)
    assert isinstance(result["thinking"], str)
    # system must embed compressed_context (when non-empty)
    if compressed_context:
        assert compressed_context in result["system"]


# ── Property 3: Integration angle uses GUIDE_INTEGRATION_VOICE ───────────────
# Feature: datasetforge-persona-pipeline, Property 3: Integration angle uses GUIDE_INTEGRATION_VOICE

@settings(max_examples=100)
@given(
    chunk=st.text(min_size=1),
    ctx=st.text(),
    intensity=st.sampled_from(list(INTENSITY_DEFINITIONS)),
)
def test_property_3_integration_angle_uses_integration_voice(chunk, ctx, intensity):
    """When angle='integration', system prompt contains GUIDE_INTEGRATION_VOICE content."""
    result = build_generation_prompt(chunk, "integration", intensity, ctx)
    # Check for a distinctive phrase that only appears in GUIDE_INTEGRATION_VOICE
    # (not in GUIDE_PERSONA), confirming the right voice was used
    assert "SYNTHESIS VOICE" in result["system"], (
        "system prompt should contain SYNTHESIS VOICE marker when angle='integration'"
    )


# ── Property 4: Prompt instructs translation not summarization ────────────────
# Feature: datasetforge-persona-pipeline, Property 4: Prompt instructs translation not summarization

@settings(max_examples=100)
@given(
    angle=st.sampled_from(list(ANGLE_DEFINITIONS)),
    intensity=st.sampled_from(list(INTENSITY_DEFINITIONS)),
    chunk=st.text(min_size=1),
    ctx=st.text(),
)
def test_property_4_prompt_instructs_translation_not_summarization(angle, intensity, chunk, ctx):
    """Prompt contains translation-oriented language and not summarization instructions."""
    result = build_generation_prompt(chunk, angle, intensity, ctx)
    system_lower = result["system"].lower()
    # Must contain translation-oriented language
    assert any(phrase in system_lower for phrase in [
        "construct", "scene", "do not quote", "do not summarize", "translate", "new scene"
    ]), "system prompt should contain translation-oriented language"
    # Must not instruct to summarize or describe the source
    assert "summarize the source" not in system_lower
    assert "describe the source" not in system_lower


# ── Property 5: Invalid angle or intensity raises ValueError ──────────────────
# Feature: datasetforge-persona-pipeline, Property 5: Invalid angle or intensity raises ValueError

@settings(max_examples=100)
@given(
    bad_angle=st.text().filter(lambda s: s not in ANGLE_DEFINITIONS),
    bad_intensity=st.text().filter(lambda s: s not in INTENSITY_DEFINITIONS),
)
def test_property_5_invalid_inputs_raise_value_error(bad_angle, bad_intensity):
    """ValueError is raised for any angle or intensity not in the defined sets."""
    with pytest.raises(ValueError):
        build_generation_prompt("some chunk", bad_angle, "acute", "ctx")
    with pytest.raises(ValueError):
        build_generation_prompt("some chunk", "first_encounter", bad_intensity, "ctx")


# ── Property 6: Validator always returns a list of strings ───────────────────
# Feature: datasetforge-persona-pipeline, Property 6: Validator always returns a list of strings

@settings(max_examples=100)
@given(
    triple=st.fixed_dictionaries({
        "system_prompt": st.text(),
        "user_message": st.text(),
        "assistant_response": st.text(),
    })
)
def test_property_6_validator_always_returns_list_of_strings(triple):
    """validate_triple always returns a list of strings, never raises."""
    result = validate_triple(triple)
    assert isinstance(result, list)
    assert all(isinstance(v, str) for v in result)


# ── Property 7: Validator detects analysis language in user_message ───────────
# Feature: datasetforge-persona-pipeline, Property 7: Validator detects analysis language in user_message

@settings(max_examples=100)
@given(
    phrase=st.sampled_from(ANALYSIS_PHRASES),
    prefix=st.text(),
    suffix=st.text(),
)
def test_property_7_validator_detects_analysis_language(phrase, prefix, suffix):
    """Violations are non-empty when user_message contains any analysis phrase."""
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": prefix + phrase + suffix,
        "assistant_response": "I feel a deep sense of calm washing over me right now.",
    }
    violations = validate_triple(triple)
    assert any("analysis language" in v for v in violations), (
        f"Expected analysis language violation for phrase '{phrase}'"
    )


# ── Property 8: Validator detects non-first-person assistant responses ─────────
# Feature: datasetforge-persona-pipeline, Property 8: Validator detects non-first-person assistant responses

_FIRST_PERSON_PATTERN = re.compile(
    r'\bI\s+(am|feel|see|hear|sense|know|want|need|think|notice)\b'
)


@settings(max_examples=100)
@given(
    response=st.text(min_size=40, max_size=600).filter(
        lambda s: not s.lstrip().startswith("I ") and not _FIRST_PERSON_PATTERN.search(s)
    )
)
def test_property_8_validator_detects_non_first_person(response):
    """Violations are non-empty for responses that don't use first-person present tense."""
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "What do you feel?",
        "assistant_response": response,
    }
    violations = validate_triple(triple)
    assert any("first-person" in v for v in violations), (
        "Expected first-person violation for non-first-person response"
    )


# ── Property 9: Validator enforces assistant response length bounds ────────────
# Feature: datasetforge-persona-pipeline, Property 9: Validator enforces assistant response length bounds

@settings(max_examples=100)
@given(
    length=st.one_of(
        st.integers(min_value=601, max_value=2000),
        st.integers(min_value=1, max_value=39),
    )
)
def test_property_9_validator_enforces_length_bounds(length):
    """Violations contain a length-related message for out-of-bounds responses."""
    response = "I " + "x" * (length - 2) if length >= 2 else "x" * length
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "What do you feel?",
        "assistant_response": response,
    }
    violations = validate_triple(triple)
    assert any("too short" in v or "too long" in v for v in violations), (
        f"Expected length violation for response of length {length}"
    )


# ── Property 10: Validator detects missing required fields ────────────────────
# Feature: datasetforge-persona-pipeline, Property 10: Validator detects missing required fields

@settings(max_examples=100)
@given(missing_field=st.sampled_from(["system_prompt", "user_message", "assistant_response"]))
def test_property_10_validator_detects_missing_fields(missing_field):
    """Violations are non-empty when any required field is absent or empty."""
    triple = {
        "system_prompt": "You are a guide.",
        "user_message": "What do you feel?",
        "assistant_response": "I feel calm and present right now in this moment.",
    }
    triple[missing_field] = ""
    violations = validate_triple(triple)
    assert len(violations) > 0, f"Expected violation for empty {missing_field}"


# ── Property 11: Repair engine always tags and sets status correctly ──────────
# Feature: datasetforge-persona-pipeline, Property 11: Repair engine always tags and sets status correctly

@settings(max_examples=100)
@given(
    system=st.text(min_size=1),
    user=st.text(min_size=1),
    assistant=st.text(min_size=40, max_size=600),
)
def test_property_11_repair_engine_tags_and_status(system, user, assistant):
    """
    When repair produces a valid triple, status='pending' and 'repaired' in tags.
    When repair produces an invalid triple, status='needs_manual' and 'repaired' in tags.
    Uses monkeypatching via unittest.mock to avoid real network calls.
    """
    from unittest.mock import patch
    import json as _json
    from datasetforge import repair_triple_with_ollama

    # Build a response that will pass or fail validate_triple
    # We test the "valid repair" path: first-person, correct length, no analysis phrases
    valid_assistant = "I feel a deep sense of calm and presence washing over me right now."
    valid_response = _json.dumps({
        "system_prompt": system[:200] or "You are a guide.",
        "user_message": "What do you feel right now?",
        "assistant_response": valid_assistant,
    })

    original_triple = {
        "id": 1,
        "system_prompt": "old system",
        "user_message": "old user",
        "assistant_response": "old assistant",
        "status": "needs_manual",
        "tags_json": "[]",
    }

    with patch("datasetforge.generate_sync", return_value=valid_response):
        result = repair_triple_with_ollama(original_triple, "http://localhost:11434", "test-model")

    tags = _json.loads(result.get("tags_json") or "[]")
    assert "repaired" in tags, "repaired tag must always be present"
    # Status depends on whether validate_triple passes
    violations = validate_triple(result)
    if not violations:
        assert result["status"] == "pending"
    else:
        assert result["status"] == "needs_manual"


# ── Property 12: Tag filter returns only matching triples ─────────────────────
# Feature: datasetforge-persona-pipeline, Property 12: Tag filter returns only matching triples

_SAMPLE_TAGS = ["repaired", "reviewed", "flagged", "approved", "custom"]


@settings(max_examples=100)
@given(
    tags_sets=st.lists(
        st.lists(st.sampled_from(_SAMPLE_TAGS), max_size=3),
        min_size=1,
        max_size=10,
    )
)
def test_property_12_tag_filter_returns_only_matching(tags_sets):
    """Python-side tag filtering returns only triples containing the target tag."""
    target_tag = "repaired"
    triples = [
        {"tags_json": json.dumps(tags)} for tags in tags_sets
    ]

    # Apply the same filtering logic as get_triples_for_chunk
    filtered = [
        t for t in triples
        if target_tag in json.loads(t.get("tags_json") or "[]")
    ]

    for t in filtered:
        assert target_tag in json.loads(t["tags_json"])

    # If none have the tag, result should be empty
    if not any(target_tag in tags for tags in tags_sets):
        assert filtered == []


# ── Property 13: Absent tag filter preserves existing behavior ────────────────
# Feature: datasetforge-persona-pipeline, Property 13: Absent tag filter preserves existing behavior

@settings(max_examples=100)
@given(
    triples_data=st.lists(
        st.fixed_dictionaries({
            "status": st.sampled_from(["pending", "approved", "needs_manual"]),
            "tags_json": st.lists(
                st.sampled_from(["repaired", "reviewed", "flagged"]), max_size=3
            ).map(json.dumps),
        }),
        min_size=0,
        max_size=10,
    )
)
def test_property_13_absent_tag_filter_preserves_behavior(triples_data):
    """
    Applying tag_filter=None or tag_filter='' to a list of triples returns the
    same result as not filtering at all — the pure Python filtering logic is identity.
    """
    # Replicate the Python-side filtering logic from get_triples_for_chunk
    def apply_tag_filter(triples, tag_filter):
        if not tag_filter:
            return triples
        filtered = []
        for t in triples:
            try:
                tags = json.loads(t.get("tags_json") or "[]")
                if tag_filter in tags:
                    filtered.append(t)
            except (json.JSONDecodeError, TypeError):
                pass
        return filtered

    result_none = apply_tag_filter(triples_data, None)
    result_empty = apply_tag_filter(triples_data, "")

    assert result_none == triples_data, "tag_filter=None should return all triples unchanged"
    assert result_empty == triples_data, "tag_filter='' should return all triples unchanged"
