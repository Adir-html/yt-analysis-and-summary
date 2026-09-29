# yt-analysis-and-summary

A small Python CLI that downloads a YouTube video's audio, transcribes it with OpenRouter, and optionally produces a concise summary. The tool shows live progress bars for download and conversion and logs each step with colourful console output. Tested with German-language content.

## Features
- **Audio download** using `yt-dlp` with progress displayed via `tqdm`
- **FFmpeg conversion** to MP3 with a conversion progress indicator
- **Transcription** via OpenRouter's speech-to-text model
- **Audio splitting** (`--split`) — automatically splits long videos into <10-minute chunks before transcription to work around the OpenRouter API's input-duration limit
- **Summarisation** using the same model with a bullet-point prompt
- **Batch processing** – process multiple videos from a text file
- **Rich console logging** for clear, colour-coded messages
- **Resume support** – skips download if audio file already exists (use `--force` to re-download)
- **Retry with visibility** – automatic retries with progress logging for transient API errors
- **Better error messages** – user-friendly hints for common HTTP errors (502, 429, 401, etc.)
- **Standard development workflow** – lint (`ruff`), format (`black`), and test (`pytest`)

## Prerequisites
- Python **3.11+**
- **FFmpeg** installed and available on the system `PATH`
- An OpenRouter API key (set `OPENROUTER_API_KEY` in a `.env` file or the environment)

## Installation
```bash
# Clone the repository
git clone <repo-url> && cd yt-analysis-and-summary

# Create a virtual environment
python -m venv .venv
source .venv\Scripts\activate  # PowerShell: .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for development
```

## Usage
```bash
# Basic run – downloads, transcribes, and summarises
python -m src.cli --url "https://www.youtube.com/watch?v=example"

# Skip summarisation (only transcript)
python -m src.cli --url "..." --no-summary

# Specify a custom output directory
python -m src.cli --url "..." --out-dir ./data

# Force re-download even if audio file exists
python -m src.cli --url "..." --force

# Split long videos into <10-minute chunks before transcribing
# (useful for videos longer than the OpenRouter API limit)
python -m src.cli --url "..." --split

# Adjust log verbosity
python -m src.cli --url "..." --verbose   # debug output
python -m src.cli --url "..." --quiet     # warnings and errors only

# Batch processing – process multiple videos from a file
python -m src.cli --batch-file urls.txt --out-dir ./data
```

**Exit codes:** `0` on success. A failed single run, or a batch run where any
video fails, returns `1` — so the CLI works as a gate in scripts and CI.

### Batch file format

Create a text file with one YouTube URL per line. Lines starting with `#` are treated as comments, and empty lines are ignored:

```
# My video list
https://www.youtube.com/watch?v=video1
https://youtu.be/video2
https://www.youtube.com/watch?v=video3
```

The CLI will process each video in sequence, logging errors without stopping the batch.

The CLI will display `tqdm` bars for the download and conversion steps and log the transcription and summarisation stages with `rich`.

### Audio splitting (`--split`)

Long videos (over ~10 minutes) can exceed OpenRouter's audio transcription input limit and fail. The `--split` flag splits the downloaded audio into consecutive <10-minute chunks (590 seconds each), transcribes each chunk separately, and automatically concatenates the transcripts in order before summarisation. The combined transcript is saved just like a normal run.

```bash
python -m src.cli --url "...long-video..." --split
python -m src.cli --url "...long-video..." --split --overlap-seconds 5
```

Chunks overlap slightly by default (`--overlap-seconds`, default `3`, env `OPENROUTER_CHUNK_OVERLAP`): each chunk starts a few seconds before the previous one ends, so speech that straddles a boundary is heard in full by at least one chunk instead of being cut mid-word. The summariser prompt is aware of the overlap so repeated seam text is not double-counted.

Chunk files are temporary and cleaned up automatically after transcription.

## Development
```bash
# Lint the code
ruff check src/ tests/

# Auto-format
black src/ tests/

# Run the test suite
pytest
```

## Environment variables
- `OPENROUTER_API_KEY` – required for any OpenRouter request
- Optional overrides (copy `.env.example` to `.env`):
  - `OPENROUTER_TRANSCRIBE_MODEL` – transcription model (default `mistralai/voxtral-mini-transcribe`)
  - `OPENROUTER_SUMMARISE_MODEL` – summarisation model (default `anthropic/claude-3.5-sonnet`)
  - `OPENROUTER_TIMEOUT` – request timeout in seconds (default `60.0`)
  - `OPENROUTER_MAX_TOKENS` – max tokens for summaries (default `1024`)
  - `OPENROUTER_CHUNK_SECONDS` – max seconds per audio chunk for `--split` (default `590`)
  - `OPENROUTER_CHUNK_OVERLAP` – seconds of tail overlap between consecutive chunks (default `3`; set `0` for back-to-back chunks)

## License
MIT License updated.

## Author
Guybrush Threepwood