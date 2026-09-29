# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common development commands

- **Setup a Python environment**
  ```
  python -m venv .venv
  .venv\Scripts\activate  # PowerShell: .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  pip install -r requirements-dev.txt  # for development tools
  ```
- **Run the main script**
  ```
  python -m src.cli --url <youtube-url>
  ```
  Additional flags: `--no-summary`, `--out-dir`, `--force`, `--split`,
  `--overlap-seconds <n>`, `--batch-file <file>`, `--verbose`, `--quiet`.
  `--overlap-seconds` sets the tail overlap between split chunks (defaults to
  the `OPENROUTER_CHUNK_OVERLAP` env var, else 3s).
- **Run a single test**
  ```
  pytest tests/test_module.py::test_name
  ```
- **Run the full test suite** (network tests are skipped by default)
  ```
  pytest
  ```
- **Run the network tests explicitly**
  ```
  pytest -m network
  ```
- **Lint / format**
  ```
  ruff check src/ tests/
  black src/ tests/
  ```
- **CI** – a `ci.yml` workflow runs ruff, black, and pytest on every push/PR.

## High-level architecture & workflow

The repository provides a small end-to-end pipeline for extracting, transcribing, and summarising YouTube video content.

1. **Entry point** – `src/cli.py` that accepts a YouTube URL (or batch file)
2. **Audio extraction** – `src/yt/downloader.py` uses `yt-dlp` to download the best audio track
3. **Audio splitting** – `src/audio/splitter.py` splits long audio into <10-minute chunks (`--split`) via ffmpeg
4. **Transcription** – `src/transcription/client.py` calls OpenRouter's speech-to-text API
5. **Summarisation** – `src/summarisation/summariser.py` generates bullet-point summaries
6. **Utilities** – shared modules in `src/utils/` (logging, retry) and `src/config.py` (env settings)

The overall data flow is:

```
YouTube URL → yt-dlp → audio file → Transcribe (OpenRouter) → transcript.txt → Summarise (OpenRouter) → summary.txt
```

## Project structure

```
src/
├── __init__.py
├── cli.py              # CLI entry point
├── config.py           # Environment variable settings (lazy API key validation)
├── yt/
│   ├── __init__.py
│   ├── downloader.py   # Audio download with ffmpeg conversion
│   └── utils.py        # Slugify / clean_title helpers
├── audio/
│   ├── __init__.py
│   └── splitter.py     # ffmpeg-based chunking for the --split flag
├── transcription/
│   ├── __init__.py
│   └── client.py       # OpenRouter API client (transcribe & summarise)
├── summarisation/
│   ├── __init__.py
│   └── summariser.py   # Summary generation + save-to-disk wrapper
└── utils/
    ├── __init__.py
    ├── logging.py      # stdlib logging with a rich console handler
    └── retry.py        # Tenacity retry decorator (transient-only) + RetryableError

tests/
├── conftest.py         # Shared fixtures (audio/chunk factories, mock helpers)
├── test_audio_splitter.py
├── test_cli.py
├── test_client.py
├── test_config.py
├── test_downloader.py
├── test_retry.py
├── test_transcription.py
└── test_youtube_api_access.py   # live-network tests, marked `network`
```

## Key patterns

- **Lazy API key validation**: `config.py` defers validation until API calls are made, allowing `--help` without a key; `get_api_key()` is the *only* caller of the key env var.
- **Centralised settings**: `config.py` exposes typed helpers (`transcribe_model()`, `summarise_model()`, `api_timeout()`, `max_tokens()`, `chunk_duration()`) instead of scattered `os.getenv` calls.
- **Transient-only retries**: `utils/retry.py` retries only `RetryableError` subclasses; the API client raises transient subclasses for retryable HTTP statuses (429/5xx) and network errors, and permanent ones otherwise.
- **ProgressManager class**: encapsulates `tqdm` bar state in `src/yt/downloader.py`.
- **Shared test fixtures**: `tests/conftest.py` provides `audio_file`, `chunk_files`, and `transcription_ctx` (a patched split/cleanup/transcribe helper) to avoid repeating mock boilerplate; network tests are marked `network` and skipped by default.
- **Output naming**: transcripts go to `<stem>_transcript.txt`, summaries to `<stem>_summary.txt` (the only place that builds these names is `_save_transcript`/`_save_summary` in `cli.py` and `summariser.py`).