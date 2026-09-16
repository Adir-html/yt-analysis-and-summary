"""Tests for src.transcription.client — especially transcribe_split."""

import pytest

from src.transcription.client import (
    RETRYABLE_STATUSES,
    TransientTranscriptionError,
    TranscriptionError,
    transcribe_split,
)


class TestTranscribeSplit:
    """Tests for the transcribe_split function."""

    def test_multiple_chunks_concatenated_in_order(
        self, audio_file, chunk_files, transcription_ctx
    ):
        """Each chunk is transcribed and the results concatenated with headers."""
        audio = audio_file("long.mp3")
        _, chunks = chunk_files(3)
        transcription_ctx.split_audio.return_value = chunks
        transcription_ctx.transcribe.side_effect = [f"text-{i}" for i in range(1, 4)]

        result = transcribe_split(audio)

        assert (
            result.index("--- Part 1 ---\ntext-1")
            < result.index("--- Part 2 ---\ntext-2")
            < result.index("--- Part 3 ---\ntext-3")
        )

    def test_single_chunk_calls_transcribe_directly(self, audio_file, transcription_ctx):
        """A file that is not split is transcribed directly and not cleaned up."""
        audio = audio_file("short.mp3")
        transcription_ctx.split_audio.return_value = [audio]
        transcription_ctx.transcribe.return_value = "Full transcript"

        result = transcribe_split(audio)

        assert result == "Full transcript"
        transcription_ctx.transcribe.assert_called_once_with(audio)
        transcription_ctx.cleanup_chunks.assert_not_called()

    def test_empty_text_chunks_are_skipped(self, audio_file, chunk_files, transcription_ctx):
        """Empty transcripts from a chunk do not produce an empty section."""
        audio = audio_file("video.mp3")
        _, chunks = chunk_files(2)
        transcription_ctx.split_audio.return_value = chunks
        transcription_ctx.transcribe.side_effect = ["Some text", ""]

        result = transcribe_split(audio)

        assert "Some text" in result
        assert "--- Part 2 ---" not in result

    def test_default_chunk_duration(self, audio_file, transcription_ctx, monkeypatch):
        """When no duration is given, the configured env value is used."""
        audio = audio_file("video.mp3")
        transcription_ctx.split_audio.return_value = [audio]
        monkeypatch.setenv("OPENROUTER_CHUNK_SECONDS", "42")

        transcribe_split(audio)

        assert transcription_ctx.split_audio.call_args.kwargs["chunk_duration"] == 42

    def test_default_overlap_from_config(self, audio_file, transcription_ctx, monkeypatch):
        """When no overlap is given, the configured env value is used."""
        audio = audio_file("video.mp3")
        transcription_ctx.split_audio.return_value = [audio]
        monkeypatch.setenv("OPENROUTER_CHUNK_OVERLAP", "7")

        transcribe_split(audio)

        assert transcription_ctx.split_audio.call_args.kwargs["overlap_seconds"] == 7

    def test_explicit_overlap_overrides_config(self, audio_file, transcription_ctx, monkeypatch):
        """An explicitly passed overlap wins over the env value."""
        audio = audio_file("video.mp3")
        transcription_ctx.split_audio.return_value = [audio]
        monkeypatch.setenv("OPENROUTER_CHUNK_OVERLAP", "7")

        transcribe_split(audio, overlap_seconds=11)

        assert transcription_ctx.split_audio.call_args.kwargs["overlap_seconds"] == 11

    def test_cleanup_called_after_split(self, audio_file, chunk_files, transcription_ctx):
        """cleanup_chunks is invoked with the produced chunks when splitting."""
        audio = audio_file("video.mp3")
        _, chunks = chunk_files(2)
        transcription_ctx.split_audio.return_value = chunks
        transcription_ctx.transcribe.side_effect = ["a", "b"]

        transcribe_split(audio)

        transcription_ctx.cleanup_chunks.assert_called_once_with(chunks)

    def test_cleanup_runs_even_if_chunk_fails_disabled(
        self, audio_file, chunk_files, transcription_ctx
    ):
        """A transient chunk failure still cleans up (transcribe_split is not retried)."""
        audio = audio_file("video.mp3")
        _, chunks = chunk_files(2)
        transcription_ctx.split_audio.return_value = chunks
        transcription_ctx.transcribe.side_effect = TransientTranscriptionError("boom")

        with pytest.raises(TransientTranscriptionError):
            transcribe_split(audio)

        transcription_ctx.cleanup_chunks.assert_called_once_with(chunks)


class TestTransientStatus:
    """Transient vs permanent status classification."""

    def test_retryable_statuses_are_classified_transient(self):
        for status in RETRYABLE_STATUSES:
            assert status in {408, 429, 500, 502, 503, 504}

    def test_permanent_status_not_in_transient(self):
        assert 401 not in RETRYABLE_STATUSES
        assert 400 not in RETRYABLE_STATUSES


def test_imports_are_reachable():
    """Sanity check for exception exports used across modules."""
    assert issubclass(TranscriptionError, RuntimeError)
