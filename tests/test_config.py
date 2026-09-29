"""Tests for src.config settings helpers."""

import pytest

from src import config


def test_get_api_key_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        config.get_api_key()


def test_transcribe_model_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_TRANSCRIBE_MODEL", raising=False)
    assert config.transcribe_model() == "mistralai/voxtral-mini-transcribe"


def test_transcribe_model_overridable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_TRANSCRIBE_MODEL", "custom/model")
    assert config.transcribe_model() == "custom/model"


def test_summarise_model_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_SUMMARISE_MODEL", raising=False)
    assert config.summarise_model() == "anthropic/claude-3.5-sonnet"


def test_summarise_model_overridable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_SUMMARISE_MODEL", "custom/summariser")
    assert config.summarise_model() == "custom/summariser"


def test_api_timeout_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_TIMEOUT", raising=False)
    assert config.api_timeout() == 60.0


def test_max_tokens_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MAX_TOKENS", raising=False)
    assert config.max_tokens() == 1024


def test_chunk_duration_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_CHUNK_SECONDS", raising=False)
    assert config.chunk_duration() == 590


def test_chunk_duration_overridable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_CHUNK_SECONDS", "120")
    assert config.chunk_duration() == 120


def test_chunk_overlap_default(monkeypatch):
    """The overlap default must match the CLI-documented default of 3."""
    monkeypatch.delenv("OPENROUTER_CHUNK_OVERLAP", raising=False)
    assert config.chunk_overlap() == 3


def test_chunk_overlap_overridable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_CHUNK_OVERLAP", "5")
    assert config.chunk_overlap() == 5
