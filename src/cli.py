"""Command-line entry point for the YouTube analysis tool.

Usage example:
    python -m src.cli --url "https://www.youtube.com/watch?v=..."

The CLI orchestrates:
1. Downloading audio with progress bars.
2. Transcribing the audio via OpenRouter.
3. Optionally summarising the transcript.

All steps are logged with rich-styled output.
"""

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from .summarisation.summariser import summarise_and_save
from .transcription.client import transcribe, transcribe_split
from .utils.logging import configure, logger
from .yt.downloader import download_audio

console = Console()

_YOUTUBE_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
)


def validate_youtube_url(url: str) -> None:
    """Validate that the URL is a valid YouTube video URL.

    Args:
        url: The URL to validate

    Raises:
        ValueError: If the URL is not a valid YouTube video URL
    """
    if not _YOUTUBE_URL_PATTERN.match(url):
        raise ValueError(
            "Invalid YouTube URL. Expected format: "
            "https://www.youtube.com/watch?v=... or https://youtu.be/..."
        )


def read_urls_from_file(path: Path) -> list[str]:
    """Read YouTube URLs from a text file, one per line.

    Args:
        path: Path to the file containing URLs

    Returns:
        List of validated URL strings

    Raises:
        ValueError: If any URL in the file is invalid
    """
    urls = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        validate_youtube_url(line)
        urls.append(line)
    if not urls:
        raise ValueError(f"No valid URLs found in {path}")
    return urls


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download, transcribe, and optionally summarise YouTube videos."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="YouTube video URL to process")
    group.add_argument("--batch-file", type=Path, help="Text file with one YouTube URL per line")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Directory where audio, transcript and summary files will be stored (default: ./output)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the summarisation step and only produce a transcript",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download audio even if file already exists",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help=(
            "Split audio into <10-minute chunks before transcribing. "
            "Useful for long videos that exceed the transcription API limit."
        ),
    )
    parser.add_argument(
        "--overlap-seconds",
        type=int,
        help=(
            "Seconds of tail overlap between consecutive split chunks, so speech "
            "straddling a boundary is heard in full. Only relevant with --split. "
            "Must be >= 0 and smaller than the chunk duration. "
            "Defaults to the OPENROUTER_CHUNK_OVERLAP env var (else 3)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug-level log output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show warnings and errors",
    )
    return parser.parse_args()


@dataclass
class Options:
    """Runtime options shared across single and batch processing."""

    no_summary: bool
    force: bool
    split: bool
    overlap_seconds: int | None = None


def _options_from_args(args: argparse.Namespace) -> Options:
    return Options(
        no_summary=args.no_summary,
        force=args.force,
        split=args.split,
        overlap_seconds=args.overlap_seconds,
    )


def process_single_url(url: str, out_dir: Path, opts: Options) -> bool:
    """Process a single URL: download, transcribe, and optionally summarise.

    Args:
        url: YouTube video URL to process
        out_dir: Output directory for files
        opts: Processing options

    Returns:
        True on success, False on error
    """
    try:
        console.print(f"[bold cyan]Processing:[/] {url}")
        audio_path = download_audio(url, out_dir, force=opts.force)
        console.print(f"[green]Audio ready at:[/] {audio_path}")

        console.print("[bold cyan]Transcribing audio…[/]")
        transcript = (
            transcribe_split(audio_path, overlap_seconds=opts.overlap_seconds)
            if opts.split
            else transcribe(audio_path)
        )

        base_name = audio_path.stem
        _save_transcript(transcript, out_dir, base_name)

        if not opts.no_summary:
            console.print("[bold cyan]Generating summary…[/]")
            _save_summary(transcript, out_dir, base_name)

        console.print("[bold green]Complete![/]")
        return True
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        raise
    except Exception as exc:
        logger.error(f"Error: {exc}")
        return False


def main() -> int:
    """Run the CLI, returning a process exit code."""
    args = parse_args()
    configure(logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = _options_from_args(args)

    if args.url:
        try:
            validate_youtube_url(args.url)
        except ValueError as exc:
            logger.error(f"Error: {exc}")
            return 1
        return 1 if not process_single_url(args.url, out_dir, opts) else 0

    # Batch mode.
    try:
        urls = read_urls_from_file(args.batch_file)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Batch file error: {exc}")
        return 1

    console.print(f"[bold cyan]Processing {len(urls)} video(s)…[/]")
    successes = 0
    for url in urls:
        if process_single_url(url, out_dir, opts):
            successes += 1
        else:
            console.print("[yellow]Skipping to next video…[/]")

    console.print(f"[bold green]Batch complete:[/] {successes}/{len(urls)} successful")
    return 0 if successes == len(urls) else 1


def _save_transcript(transcript: str, out_dir: Path, base_name: str) -> None:
    """Save the transcript to ``<base_name>_transcript.txt``."""
    transcript_path = out_dir / f"{base_name}_transcript.txt"
    transcript_path.write_text(transcript, encoding="utf-8")
    console.print(f"[green]Transcript written to:[/] {transcript_path}")


def _save_summary(transcript: str, out_dir: Path, base_name: str) -> None:
    """Generate and save a summary, using the path returned by the summariser."""
    summary_path = summarise_and_save(transcript, out_dir, base_name)
    console.print(f"[green]Summary written to:[/] {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
