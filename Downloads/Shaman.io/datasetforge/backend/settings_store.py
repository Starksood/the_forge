"""Local settings persistence for single-process DatasetForge app."""
import json
import os
from typing import Any, Dict


SETTINGS_FILE = "settings/local_settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "ollama_host": "http://localhost:11434",
    # llama3.2:3b is the recommended baseline; use smaller models only if RAM is extremely limited.
    "model_name": "gemma3:4b",
    "triples_per_chunk": 10,
    "min_chunk_length": 150,
    "cross_reference_enabled": True,
    # Cap characters sent for framework analysis (full book + prompt can OOM the runner).
    "ollama_num_ctx": 4096,
}


def _ensure_settings_dir() -> None:
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)


def get_settings() -> Dict[str, Any]:
    _ensure_settings_dir()
    merged = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                merged.update({k: v for k, v in loaded.items() if k in DEFAULT_SETTINGS})
        except Exception:
            pass
    # Env overrides (single-process local use)
    if os.getenv("DATASETFORGE_OLLAMA_NUM_CTX"):
        try:
            merged["ollama_num_ctx"] = int(os.getenv("DATASETFORGE_OLLAMA_NUM_CTX", "0"))
        except ValueError:
            pass
    return merged


def update_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    current = get_settings()
    for key, value in patch.items():
        if key in DEFAULT_SETTINGS and value is not None:
            current[key] = value
    _ensure_settings_dir()
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
    return current
