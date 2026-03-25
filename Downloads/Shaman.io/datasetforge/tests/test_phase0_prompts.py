"""Unit tests for Phase 0 prompt content in backend/phases/phase0_analysis.py."""

import inspect
import os
import sys
import pytest

# Read the source file directly to avoid stale .pyc cache issues
_SOURCE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "backend", "phases", "phase0_analysis.py"
)
with open(_SOURCE_PATH) as _f:
    _SOURCE = _f.read()

# Also import the live module for signature checking
from backend.phases.phase0_analysis import (
    CHUNKED_ANALYSIS_PROMPT,
    SYNTHESIS_PROMPT,
    call_ollama_blocking,
)

# Override with source-parsed values to bypass stale .pyc
import ast as _ast
_tree = _ast.parse(_SOURCE)
for _node in _ast.walk(_tree):
    if isinstance(_node, _ast.Assign):
        for _t in _node.targets:
            if isinstance(_t, _ast.Name) and _t.id == "CHUNKED_ANALYSIS_PROMPT":
                CHUNKED_ANALYSIS_PROMPT = _ast.literal_eval(_node.value)
            if isinstance(_t, _ast.Name) and _t.id == "SYNTHESIS_PROMPT":
                SYNTHESIS_PROMPT = _ast.literal_eval(_node.value)


def test_chunked_analysis_prompt_contains_guidance_relevance():
    assert "guidance_relevance" in CHUNKED_ANALYSIS_PROMPT


def test_chunked_analysis_prompt_contains_guidance_framing():
    lower = CHUNKED_ANALYSIS_PROMPT.lower()
    assert any(phrase in lower for phrase in [
        "psychedelic", "guidance", "guide", "traveler",
    ]), "CHUNKED_ANALYSIS_PROMPT should contain guidance-oriented framing"


def test_synthesis_prompt_contains_imperative_instruction():
    lower = SYNTHESIS_PROMPT.lower()
    assert any(phrase in lower for phrase in [
        "imperative", "instructional", "instruct", "briefing", "register",
    ]), "SYNTHESIS_PROMPT should instruct the model to use imperative/instructional register"


def test_synthesis_prompt_contains_compressed_context_field():
    assert "compressed_context" in SYNTHESIS_PROMPT


def test_call_ollama_blocking_signature_unchanged():
    # Use AST to verify the signature from source, bypassing any stale .pyc
    import ast
    for node in ast.walk(ast.parse(_SOURCE)):
        if isinstance(node, ast.FunctionDef) and node.name == "call_ollama_blocking":
            args = [a.arg for a in node.args.args]
            assert args == ["prompt", "host", "model"], (
                f"call_ollama_blocking signature changed: {args}"
            )
            # Verify return annotation is 'dict'
            if node.returns:
                assert ast.unparse(node.returns) == "dict"
            return
    pytest.fail("call_ollama_blocking not found in source")
