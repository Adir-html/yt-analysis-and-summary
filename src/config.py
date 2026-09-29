"""Configuration handling for the project.

Loads environment variables (optionally from a .env file) and exposes typed
helpers for the settings used by the pipeline.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present at the repository root (one directory above src/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_env(name: str, *, default: str | None = None, required: bool = False) -> str | None:
    """Retrieve an environment variable.

    Raises:
        RuntimeError: If ``required`` is set and the variable is missing.
    """
    value = os.getenv(name, default)
    if required and value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _require_env(name: str) -> str:
    """Return a required environment variable, raising if absent."""
    value = get_env(name, required=True)
    # required=True guarantees the value is present, so this narrowing is safe.
    assert value is not None
    return value


def get_api_key() -> str:
    """Get the OpenRouter API key, raising an error only when actually needed.

    This defers validation to usage time so users can run ``--help`` without
    an API key.
    """
    return _require_env("OPENROUTER_API_KEY")


def transcribe_model() -> str:
    """Model used for audio transcription (env-overridable)."""
    return get_env("OPENROUTER_TRANSCRIBE_MODEL") or "mistralai/voxtral-mini-transcribe"


def summarise_model() -> str:
    """Model used for transcript summarisation (env-overridable)."""
    return get_env("OPENROUTER_SUMMARISE_MODEL") or "anthropic/claude-3.5-sonnet"


def api_timeout() -> float:
    """Timeout in seconds for OpenRouter requests (env-overridable)."""
    return float(get_env("OPENROUTER_TIMEOUT") or 60.0)


def max_tokens() -> int:
    """Maximum tokens for summarisation responses (env-overridable)."""
    return int(get_env("OPENROUTER_MAX_TOKENS") or 1024)


def chunk_duration() -> int:
    """Maximum seconds per audio chunk for ``--split`` (env-overridable)."""
    return int(get_env("OPENROUTER_CHUNK_SECONDS") or 590)


def chunk_overlap() -> int:
    """Seconds of tail overlap between consecutive ``--split`` chunks.

    Each chunk starts this many seconds *before* the previous chunk ends, so
    speech straddling a boundary is heard in full by at least one chunk instead
    of being cut mid-word. The transcription becomes partially self-overlapping;
    the summary prompt is told about the overlap so duplicated seams are folded
    away rather than double-counted.
    """
    return int(get_env("OPENROUTER_CHUNK_OVERLAP") or 3)
