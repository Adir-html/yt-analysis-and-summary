"""Tests for src.audio.splitter module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.audio.splitter import _get_audio_duration, cleanup_chunks, split_audio


def _extract_windows(mock_run) -> list[tuple[float, float]]:
    """Return the ``(start, length)`` window parsed from each ffmpeg call's args."""
    windows = []
    for call in mock_run.call_args_list:
        cmd = call.args[0]
        start = float(cmd[cmd.index("-ss") + 1])
        length = float(cmd[cmd.index("-t") + 1])
        windows.append((start, length))
    return windows


class TestSplitAudio:
    """Tests for the split_audio function."""

    def test_short_file_returns_single(self, tmp_path):
        """A file shorter than the chunk duration should not be split."""
        audio = tmp_path / "short.mp3"
        audio.write_bytes(b"")

        with patch("src.audio.splitter._get_audio_duration", return_value=30.0):
            chunks = split_audio(audio, chunk_duration=590)

        assert len(chunks) == 1
        assert chunks[0] == audio

    def test_long_file_splits_with_ffmpeg(self, tmp_path):
        """A long file should be split via ffmpeg into multiple chunks."""
        audio = tmp_path / "long.mp3"
        audio.write_bytes(b"")

        with (
            patch("src.audio.splitter._get_audio_duration", return_value=1500.0),
            patch("src.audio.splitter.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("src.audio.splitter.subprocess.run") as mock_run,
            patch("src.audio.splitter.tempfile.mkdtemp", return_value=str(tmp_path / "tmp")),
        ):
            (tmp_path / "tmp").mkdir(exist_ok=True)
            chunks = split_audio(audio, chunk_duration=590)

        assert [c.name for c in chunks] == [
            "part_000.mp3",
            "part_001.mp3",
            "part_002.mp3",
        ]
        # One ffmpeg cut per chunk, back-to-back when overlap is 0.
        assert mock_run.call_count == 3
        windows = _extract_windows(mock_run)
        assert windows == [(0.0, 590.0), (590.0, 590.0), (1180.0, 320.0)]

    def test_overlap_rewinds_chunk_start(self, tmp_path):
        """With an overlap, each chunk starts earlier so seams repeat speech."""
        audio = tmp_path / "long.mp3"
        audio.write_bytes(b"")

        with (
            patch("src.audio.splitter._get_audio_duration", return_value=1180.0),
            patch("src.audio.splitter.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("src.audio.splitter.subprocess.run") as mock_run,
            patch("src.audio.splitter.tempfile.mkdtemp", return_value=str(tmp_path / "tmp")),
        ):
            (tmp_path / "tmp").mkdir(exist_ok=True)
            chunks = split_audio(audio, chunk_duration=590, overlap_seconds=3)

        assert len(chunks) == 3
        windows = _extract_windows(mock_run)
        # Chunk 2 would start at 590.0 back-to-back; with 3s overlap it starts at 587.0.
        assert windows[0] == (0.0, 590.0)
        assert windows[1][0] == 587.0
        assert windows[1][1] == 590.0

    def test_no_ffmpeg_fallback(self, tmp_path):
        """If ffmpeg is unavailable, return the original file unsplit."""
        audio = tmp_path / "video.mp3"
        audio.write_bytes(b"")

        with (
            patch("src.audio.splitter._get_audio_duration", return_value=1500.0),
            patch("src.audio.splitter.shutil.which", return_value=None),
        ):
            chunks = split_audio(audio, chunk_duration=590)

        assert chunks == [audio]

    def test_ffmpeg_failure_fallback(self, tmp_path):
        """If ffmpeg fails, return the original file unsplit."""
        import subprocess

        audio = tmp_path / "video.mp3"
        audio.write_bytes(b"")

        with (
            patch("src.audio.splitter._get_audio_duration", return_value=1500.0),
            patch("src.audio.splitter.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch(
                "src.audio.splitter.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
            ),
            patch("src.audio.splitter.tempfile.mkdtemp", return_value=str(tmp_path / "tmp")),
        ):
            chunks = split_audio(audio, chunk_duration=590)

        assert chunks == [audio]

    def test_unknown_duration_splits_until_eof(self, tmp_path):
        """Without duration info, the segment muxer cuts until EOF — no dropped audio."""
        audio = tmp_path / "unknown.mp3"
        audio.write_bytes(b"")

        with (
            patch("src.audio.splitter._get_audio_duration", return_value=None),
            patch("src.audio.splitter.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("src.audio.splitter.subprocess.run") as mock_run,
            patch("src.audio.splitter.tempfile.mkdtemp", return_value=str(tmp_path / "tmp")),
        ):
            # The muxer decides the chunk count itself at EOF; simulate it
            # producing two chunks.
            (tmp_path / "tmp").mkdir(exist_ok=True)
            for name in ("part_000.mp3", "part_001.mp3"):
                (tmp_path / "tmp" / name).write_bytes(b"")
            mock_run.return_value = MagicMock(returncode=0)

            chunks = split_audio(audio, chunk_duration=590)

        # Everything the muxer produced is returned — nothing truncated to a
        # single 590s window.
        assert [c.name for c in chunks] == ["part_000.mp3", "part_001.mp3"]
        # A single ffmpeg invocation, of the segment muxer (no per-chunk
        # -ss/-t window loop that would need the total duration).
        cmd = mock_run.call_args.args[0]
        assert cmd[cmd.index("-f") + 1] == "segment"

    def test_unknown_duration_ffmpeg_failure_fallback(self, tmp_path):
        """A failing segment muxer falls back to the original file unsplit."""
        import subprocess

        audio = tmp_path / "unknown.mp3"
        audio.write_bytes(b"")

        with (
            patch("src.audio.splitter._get_audio_duration", return_value=None),
            patch("src.audio.splitter.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch(
                "src.audio.splitter.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
            ),
            patch("src.audio.splitter.tempfile.mkdtemp", return_value=str(tmp_path / "tmp")),
        ):
            chunks = split_audio(audio, chunk_duration=590)

        assert chunks == [audio]


class TestCleanupChunks:
    """Tests for the cleanup_chunks function."""

    def test_removes_chunks_and_temp_dir(self, tmp_path):
        """cleanup_chunks removes splitter-owned chunk files and the temp dir."""
        chunk_dir = tmp_path / "yt-split-abc"
        chunk_dir.mkdir()
        f1 = chunk_dir / "part_000.mp3"
        f2 = chunk_dir / "part_001.mp3"
        f1.write_bytes(b"")
        f2.write_bytes(b"")

        cleanup_chunks([f1, f2])

        assert not f1.exists()
        assert not f2.exists()
        assert not chunk_dir.exists()

    def test_empty_list_is_noop(self):
        """cleanup_chunks with an empty list should not raise."""
        cleanup_chunks([])

    def test_missing_file_silent(self, tmp_path):
        """cleanup_chunks should not raise if a file is already gone."""
        chunk_dir = tmp_path / "yt-split-abc"
        chunk_dir.mkdir()
        existing = chunk_dir / "part_000.mp3"
        existing.write_bytes(b"")
        already_gone = chunk_dir / "part_001.mp3"

        cleanup_chunks([existing, already_gone])

        assert not existing.exists()
        assert not chunk_dir.exists()

    def test_unsplit_original_is_never_removed(self, tmp_path):
        """A user's original audio file must be left untouched."""
        audio = tmp_path / "my-video.mp3"
        audio.write_bytes(b"data")

        cleanup_chunks([audio])

        assert audio.exists()
        assert audio.read_bytes() == b"data"


class TestGetAudioDuration:
    """Tests for the _get_audio_duration helper."""

    def test_returns_none_without_ffprobe(self):
        """Should return None when ffprobe is not installed."""
        with patch("src.audio.splitter.shutil.which", return_value=None):
            assert _get_audio_duration(Path("/fake/audio.mp3")) is None

    def test_returns_duration(self, tmp_path):
        """Should return the parsed duration from ffprobe output."""
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"")

        with (
            patch("src.audio.splitter.shutil.which", return_value="/usr/bin/ffprobe"),
            patch("src.audio.splitter.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="42.5\n", stderr="")
            result = _get_audio_duration(audio)

        assert result == 42.5

    def test_returns_none_on_error(self, tmp_path):
        """Should return None if ffprobe fails."""
        import subprocess

        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"")

        with (
            patch("src.audio.splitter.shutil.which", return_value="/usr/bin/ffprobe"),
            patch(
                "src.audio.splitter.subprocess.run",
                side_effect=subprocess.SubprocessError("ffprobe failed"),
            ),
        ):
            result = _get_audio_duration(audio)

        assert result is None
