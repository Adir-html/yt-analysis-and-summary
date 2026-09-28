"""Audio file splitting utilities.

Splits large audio files into consecutive chunks of at most
``DEFAULT_CHUNK_DURATION`` seconds. Each chunk is produced with its own
ffmpeg ``-ss``/``-t`` cut so a configurable number of seconds at the tail of
chunk N is *repeated* at the head of chunk N+1 (the overlap). This way speech
that straddles a boundary is heard in full by at least one chunk instead of
being cut mid-word.

When ``overlap_seconds`` is 0 the chunks are back-to-back, exactly like the
old ``-f segment`` behaviour.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from .. import config
from ..utils.logging import logger

# YouTube videos longer than ~10 minutes can exceed the OpenRouter
# transcription API's input limit. We split at 590 seconds (just under
# 10 minutes) to stay safely within the limit. This constant is the single
# source of truth: :func:`src.config.chunk_duration` falls back to it when
# ``OPENROUTER_CHUNK_SECONDS`` is unset.
DEFAULT_CHUNK_DURATION = 590

# Seconds of tail overlap between consecutive chunks when none is configured.
# Each chunk starts this many seconds before the previous one ends. This is the
# single source of truth for the overlap default (3s): :func:`src.config.chunk_overlap`
# falls back to it when ``OPENROUTER_CHUNK_OVERLAP`` is unset, so direct callers
# of :func:`split_audio` get the same default as the CLI.
DEFAULT_CHUNK_OVERLAP = 3

# Prefix for the temporary directory that split_audio creates. cleanup_chunks
# uses it to recognise (and only ever remove) temp dirs it owns, never a
# directory that happens to contain a user's files.
_TEMP_DIR_PREFIX = "yt-split-"


def split_audio(
    audio_path: Path,
    chunk_duration: int | None = None,
    overlap_seconds: int | None = None,
) -> list[Path]:
    """Split *audio_path* into overlapping chunks of at most *chunk_duration* seconds.

    Chunk k covers the window ``[k*step, k*step + chunk_duration]`` where
    ``step = chunk_duration - overlap_seconds``. Because each successive chunk
    starts ``overlap_seconds`` before its predecessor ends, the tail of one
    chunk is repeated at the head of the next. This keeps boundary-straddling
    speech intact in at least one chunk; the overlap is later folded away by the
    transcription/summarisation step.

    If the file is shorter than *chunk_duration* or ffmpeg is unavailable,
    a single-element list containing *audio_path* is returned (no split).

    Args:
        audio_path: Path to the input audio file (any ffmpeg-supported format).
        chunk_duration: Length of each chunk's window in seconds. The *new*
            audio each chunk adds is ``chunk_duration - overlap_seconds`` (with
            a non-zero overlap the on-disk chunk repeats that many seconds of
            the previous chunk's tail). Defaults to
            :func:`src.config.chunk_duration` (``OPENROUTER_CHUNK_SECONDS`` env
            var, else :data:`DEFAULT_CHUNK_DURATION`).
        overlap_seconds: Seconds of tail overlap shared between consecutive
            chunks. 0 produces back-to-back chunks (the historical behaviour).
            Must satisfy ``0 <= overlap_seconds < chunk_duration``. Defaults to
            :func:`src.config.chunk_overlap` (``OPENROUTER_CHUNK_OVERLAP`` env
            var, else :data:`DEFAULT_CHUNK_OVERLAP`).

    Returns:
        Ordered list of chunk file paths. The first element is the
        beginning of the file and the last element is the end.

    Raises:
        ValueError: If *overlap_seconds* is negative or not smaller than
            *chunk_duration*.
    """
    if chunk_duration is None:
        chunk_duration = config.chunk_duration()
    if overlap_seconds is None:
        overlap_seconds = config.chunk_overlap()

    # An overlap of 0 is back-to-back; a negative one would create gaps between
    # chunks, and one >= the chunk duration would make every chunk re-start
    # before the previous one, yielding hundreds of tiny chunks. Reject both
    # rather than silently warping the chunk layout.
    if overlap_seconds < 0:
        raise ValueError(f"overlap_seconds must be >= 0, got {overlap_seconds}")
    if overlap_seconds >= chunk_duration:
        raise ValueError(
            f"overlap_seconds ({overlap_seconds}) must be smaller than "
            f"chunk_duration ({chunk_duration})"
        )

    # Short-enough files are returned unchanged — no need to split.
    duration = _get_audio_duration(audio_path)
    if duration is not None and duration <= chunk_duration:
        logger.info(
            f"File is {duration:.1f}s — no splitting needed " f"(limit: {chunk_duration}s)."
        )
        return [audio_path]

    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        logger.warning("ffmpeg not found — cannot split audio file. " "Returning unsplit file.")
        return [audio_path]

    # The amount of *new* audio each chunk adds. When there's no overlap this is
    # just the chunk duration (identical to the old segment-muxer behaviour).
    # Validation above guarantees 1 <= step (overlap < chunk_duration).
    step = chunk_duration - overlap_seconds

    # Create a temporary directory for the chunk files.
    temp_dir = Path(tempfile.mkdtemp(prefix=_TEMP_DIR_PREFIX))

    if duration is None:
        # Duration couldn't be probed (e.g. ffprobe missing or failing). The
        # -ss/-t window path needs the total duration to know where the file
        # ends, so fall back to ffmpeg's segment muxer instead: it cuts until
        # EOF on its own, dropping no audio. The tail overlap is not applied
        # on this path (it needs absolute window starts, which need the total
        # duration); chunks are back-to-back like an overlap of 0.
        try:
            chunks = _split_unknown_duration(ffmpeg_exe, audio_path, temp_dir, chunk_duration)
        except subprocess.SubprocessError as exc:
            logger.warning(f"Split failed ({exc}); returning unsplit file.")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return [audio_path]
    else:
        logger.info(
            f"Splitting {audio_path.name} into {chunk_duration}s chunks "
            f"({overlap_seconds}s overlap)..."
        )

        chunks: list[Path] = []

        index = 0
        start = 0.0
        while True:
            stop = min(start + chunk_duration, duration)
            chunk_path = temp_dir / f"part_{index:03d}.mp3"
            try:
                _extract_window(ffmpeg_exe, audio_path, chunk_path, start, stop)
            except subprocess.SubprocessError as exc:
                logger.warning(f"Chunk {index} split failed ({exc}); returning unsplit file.")
                cleanup_chunks(chunks)
                shutil.rmtree(temp_dir, ignore_errors=True)
                return [audio_path]
            chunks.append(chunk_path)
            index += 1

            start += step
            if start >= duration:
                break

    if not chunks:
        logger.warning("ffmpeg produced no chunks; returning unsplit file.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return [audio_path]

    logger.info(f"Split into {len(chunks)} chunk(s).")
    return chunks


def _extract_window(
    ffmpeg_exe: str, audio_path: Path, out_path: Path, start: float, stop: float
) -> None:
    """Cut *audio_path* to the closed ``[start, stop]`` window into *out_path*.

    ``-ss`` is given as an *input* option (before ``-i``) so ffmpeg seeks
    accurately instead of streaming from the start; ``-t`` then caps the output
    length. The audio is re-encoded (no ``-c copy``) so cuts can land on any
    sample boundary — required for sub-second overlapping windows, which the
    key-frame-copy only segment muxer could not do.
    """
    length = stop - start
    cmd = [
        ffmpeg_exe,
        "-y",  # overwrite outputs
        "-ss",
        f"{start:.3f}",
        "-i",
        str(audio_path),
        "-t",
        f"{length:.3f}",
        "-vn",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _split_unknown_duration(
    ffmpeg_exe: str,
    audio_path: Path,
    temp_dir: Path,
    chunk_duration: int,
) -> list[Path]:
    """Split *audio_path* into chunks without knowing its total duration.

    Used when probing fails: ffmpeg's segment muxer cuts repeatedly until EOF
    by itself, so unknown-length inputs lose no audio (the ``-ss``/``-t``
    window path needs the total duration to place the final window). Unlike
    the old segment-muxer implementation, chunks are re-encoded (no ``-c
    copy``) so boundaries land on sample positions rather than keyframes.

    Returns:
        Produced chunk files sorted by name (the temp dir is searched for
        ``part_*.mp3``); an empty list means nothing was written.
    """
    cmd = [
        ffmpeg_exe,
        "-y",  # overwrite outputs
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_duration),
        "-reset_timestamps",
        "1",
        "-vn",
        str(temp_dir / "part_%03d.mp3"),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(temp_dir.glob("part_*.mp3"))


def cleanup_chunks(chunks: list[Path]) -> None:
    """Remove temporary chunk files and the temp dir that produced them.

    Only directories created by :func:`split_audio` (identified by their
    ``yt-split-`` prefix) are removed. If *chunks* contains an unsplit original
    file or files elsewhere on disk, they are left untouched.

    Args:
        chunks: List of chunk file paths (as returned by ``split_audio``).
    """
    if not chunks:
        return

    # Only touch files that live inside a splitter-owned temp dir, so a user's
    # original audio file (e.g. the unsplit result) is never removed.
    temp_dirs = {chunk.parent for chunk in chunks if chunk.parent.name.startswith(_TEMP_DIR_PREFIX)}

    for chunk in chunks:
        if chunk.parent in temp_dirs:
            try:
                chunk.unlink()
            except OSError:
                pass  # already gone — nothing to clean up

    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _get_audio_duration(path: Path) -> float | None:
    """Return the duration of *path* in seconds, or None if it cannot be determined.

    Uses ffprobe (shipped with ffmpeg) for an accurate reading without
    decoding the file.
    """
    ffprobe_exe = shutil.which("ffprobe")
    if not ffprobe_exe:
        return None

    try:
        result = subprocess.run(
            [
                ffprobe_exe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return None
