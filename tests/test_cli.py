"""Tests for src.cli — URL validation, file parsing, and exit codes."""

import sys

import pytest

from src.cli import _options_from_args, main, parse_args, read_urls_from_file, validate_youtube_url


class TestValidateYoutubeUrl:
    def test_accepts_watch_url(self):
        validate_youtube_url("https://www.youtube.com/watch?v=abc123DEF-_")

    def test_accepts_short_url(self):
        validate_youtube_url("https://youtu.be/abc123DEF-_")

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            validate_youtube_url("https://example.com/not-a-video")

    def test_rejects_wrong_shaped_watch_url(self):
        with pytest.raises(ValueError):
            validate_youtube_url("https://www.youtube.com/watch?v=abc")


class TestReadUrlsFromFile:
    def test_ignores_comments_and_blanks(self, tmp_path):
        path = tmp_path / "urls.txt"
        path.write_text("# comment\n\nhttps://youtu.be/abc123DEF-_\n", encoding="utf-8")
        assert read_urls_from_file(path) == ["https://youtu.be/abc123DEF-_"]

    def test_raises_on_only_comments(self, tmp_path):
        path = tmp_path / "urls.txt"
        path.write_text("# only comments\n", encoding="utf-8")
        with pytest.raises(ValueError):
            read_urls_from_file(path)


class TestOverlapSecondsFlag:
    """Parsing of the --overlap-seconds CLI flag."""

    def test_default_is_none(self, monkeypatch):
        """Without the flag, Options carries None so config supplies the default."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--url", "https://youtu.be/abc123DEF-_", "--split"],
        )
        opts = _options_from_args(parse_args())
        assert opts.overlap_seconds is None

    def test_flag_value_parsed(self, monkeypatch):
        """The flag value flows through to Options."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--url", "https://youtu.be/abc123DEF-_", "--split", "--overlap-seconds", "8"],
        )
        opts = _options_from_args(parse_args())
        assert opts.overlap_seconds == 8

    def test_flag_rejects_non_int(self, monkeypatch):
        """argparse rejects a non-integer value."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--url",
                "https://youtu.be/abc123DEF-_",
                "--split",
                "--overlap-seconds",
                "lots",
            ],
        )
        with pytest.raises(SystemExit):
            parse_args()


class TestMainExitCodes:
    def test_invalid_single_url_returns_nonzero(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--url", "not-a-url", "--out-dir", str(tmp_path)],
        )
        assert main() == 1

    def test_batch_all_fail_returns_nonzero(self, monkeypatch, tmp_path):
        batch = tmp_path / "urls.txt"
        batch.write_text("https://youtu.be/abc123DEF-_", encoding="utf-8")
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--batch-file", str(batch), "--out-dir", str(tmp_path)],
        )
        monkeypatch.setattr("src.cli.process_single_url", lambda *a, **k: False)
        assert main() == 1

    def test_batch_all_succeed_returns_zero(self, monkeypatch, tmp_path):
        batch = tmp_path / "urls.txt"
        batch.write_text("https://youtu.be/abc123DEF-_", encoding="utf-8")
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--batch-file", str(batch), "--out-dir", str(tmp_path)],
        )
        monkeypatch.setattr("src.cli.process_single_url", lambda *a, **k: True)
        assert main() == 0
