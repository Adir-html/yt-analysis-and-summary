"""OpenRouter HTTP client for transcription and summarisation.

Provides three high-level functions:
- ``transcribe`` – send an audio file and receive a transcript string.
- ``transcribe_split`` – split a long audio file and transcribe each chunk.
- ``summarise`` – send a transcript and receive a concise summary.

Only the leaf network calls are retried, and only for transient failures;
``transcribe_split`` itself is never retried so a late failure does not
re-split and re-transcribe already-finished chunks.
"""

import base64
import re
from pathlib import Path

import httpx

from .. import config
from ..audio.splitter import cleanup_chunks, split_audio
from ..config import api_timeout, get_api_key
from ..utils.logging import logger
from ..utils.retry import RetryableError, retry


class TranscriptionError(RuntimeError):
    """Raised when audio transcription fails due to API or network issues."""


class SummarisationError(RuntimeError):
    """Raised when summarisation fails due to API or network issues."""


class TransientTranscriptionError(TranscriptionError, RetryableError):
    """Transient transcription failure that is safe to retry."""


class TransientSummarisationError(SummarisationError, RetryableError):
    """Transient summarisation failure that is safe to retry."""


# HTTP statuses worth retrying after a short back-off; everything else is fatal.
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}

# OpenRouter endpoint for audio transcriptions (expects JSON with base64 audio).
_AUDIO_API_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
# Chat endpoint for summarisation.
_CHAT_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Bullet-point summary prompt, with the transcript substituted in.
_SUMMARY_PROMPT = (
    "Summarise the following transcript in bullet points, focusing on the "
    "main ideas, arguments, and conclusions. Leave out any appearing advertising or promotion."
    "Write detailed but concise summaries. Finally, write 1 sentence summarising the key takeaway. "
    "Use concise language."
    "\n\n"
    "The transcript may have been stitched together from multiple audio parts; "
    "sections are separated by '--- Part ... ---' markers. Consecutive parts may "
    "overlap slightly, so a short stretch of text can appear twice at a seam. "
    "Treat the parts as one continuous source: read across the markers, cover "
    "each idea exactly once, and do not double-count a repeated seam or treat "
    "repeated wording as emphasis."
    "\n\n Transcript:\n\n"
    "```\n{transcript}\n```"
)


def _get_headers() -> dict[str, str]:
    """Get headers with API key, evaluated lazily."""
    return {"Authorization": f"Bearer {get_api_key()}"}


def _post(
    url: str,
    payload: dict,
    permanent: type,
    transient: type,
) -> dict:
    """POST a JSON *payload* to an OpenRouter endpoint and return the JSON body.

    Transport-level failures raise ``transient`` (safe to retry); HTTP errors
    are converted to ``permanent`` or ``transient`` based on the status code.
    """
    try:
        response = httpx.post(
            url,
            headers={**_get_headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=api_timeout(),
        )
    except httpx.TransportError as exc:
        raise transient(f"Network error contacting OpenRouter: {exc}") from exc

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        _raise_with_details(exc, permanent, transient)

    try:
        return response.json()
    except ValueError as exc:
        raise permanent(f"OpenRouter returned a non-JSON response: {exc}") from exc


def _raise_with_details(exc: httpx.HTTPStatusError, permanent: type, transient: type) -> None:
    """Raise a user-friendly error from an ``httpx.HTTPStatusError``.

    Follows OpenRouter's error format: ``{"error": {"code", "message", ...}}``
    See: https://openrouter.ai/docs/errors-and-debugging
    """
    response = exc.response
    status = response.status_code
    content_type = response.headers.get("content-type", "")
    response_text = response.text

    # Cloudflare error pages come back as HTML with a short title.
    is_html = (
        "text/html" in content_type
        or response_text.strip().startswith("<!DOCTYPE html>")
        or response_text.strip().startswith("<html")
    )

    if is_html:
        title_match = re.search(r"<title>([^<]+)</title>", response_text)
        error_msg = (
            title_match.group(1)
            if title_match
            else (response.reason_phrase or f"HTTP {status} error")
        )
    else:
        try:
            error_obj = response.json().get("error", {})
            error_msg = error_obj.get("message", "") or response_text
        except ValueError:
            error_msg = response_text or ""

    if not error_msg:
        error_msg = response.reason_phrase or f"HTTP {status} error"

    # Friendly hints for common statuses (following OpenRouter docs).
    status_hints = {
        400: "Invalid request - check your parameters",
        401: "Authentication failed - check your OPENROUTER_API_KEY",
        402: "Payment required - add more credits to your OpenRouter account",
        403: "Access forbidden - check permissions or content policy",
        404: "Resource not found",
        408: "Request timeout - try again",
        429: "Rate limit exceeded - wait before retrying",
        502: "Bad gateway - provider returned invalid response",
        503: "Service unavailable - no providers available",
        504: "Gateway timeout - provider took too long to respond",
    }
    hint = status_hints.get(status)
    full_msg = f"{error_msg} ({hint})" if hint else f"{error_msg} (HTTP {status})"

    error_cls = transient if status in RETRYABLE_STATUSES else permanent
    raise error_cls(full_msg) from exc


def _post_audio_json(audio_path: Path, model: str) -> dict:
    """Build a base64-encoded audio payload and POST it to the transcription endpoint.

    Follows the OpenRouter example:
    {
        "model": "mistralai/voxtral-mini-transcribe",
        "input_audio": {"data": "<base64-encoded-audio>", "format": "wav"}
    }
    """
    audio_bytes = audio_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    # Determine format from file extension; default to "wav" if unknown.
    fmt = audio_path.suffix.lstrip(".").lower() or "wav"

    payload = {"model": model, "input_audio": {"data": audio_b64, "format": fmt}}
    return _post(_AUDIO_API_URL, payload, TranscriptionError, TransientTranscriptionError)


def _post_chat(payload: dict) -> dict:
    """POST a JSON *payload* to the chat endpoint and return the parsed JSON."""
    return _post(_CHAT_API_URL, payload, SummarisationError, TransientSummarisationError)


@retry()
def transcribe(audio_path: Path) -> str:
    """Transcribe *audio_path* using OpenRouter's audio transcription endpoint.

    The model defaults to the ``OPENROUTER_TRANSCRIBE_MODEL`` environment
    variable, falling back to Voxtral Mini (see :func:`src.config.transcribe_model`).
    """
    model = config.transcribe_model()
    logger.info(f"Transcribing audio file {audio_path} with model {model}")
    data = _post_audio_json(audio_path, model)
    # The response follows OpenAI's Whisper schema: {"text": "..."}
    return data.get("text", "").strip()


def transcribe_split(audio_path: Path, chunk_duration: int | None = None) -> str:
    """Transcribe *audio_path*, splitting into chunks if too long.

    If the audio exceeds *chunk_duration* seconds, it is split into
    consecutive parts, each transcribed separately, and the transcripts
    are concatenated in order. This works around OpenRouter's ~10-minute
    input limit for long audio files.

    Args:
        audio_path: Path to the audio file to transcribe.
        chunk_duration: Maximum seconds per chunk (defaults to the
            ``OPENROUTER_CHUNK_SECONDS`` env var, else
            ``audio.splitter.DEFAULT_CHUNK_DURATION``).

    Returns:
        The combined transcript string.
    """
    if chunk_duration is None:
        chunk_duration = config.chunk_duration()
    chunks = split_audio(
        audio_path,
        chunk_duration=chunk_duration,
        overlap_seconds=config.chunk_overlap(),
    )

    logger.info(f"Transcribing {len(chunks)} chunk(s) from {audio_path.name}")

    try:
        if len(chunks) == 1 and chunks[0].resolve() == audio_path.resolve():
            # No splitting occurred — transcribe the original file directly.
            return transcribe(audio_path)

        parts = []
        for i, chunk_path in enumerate(chunks, start=1):
            logger.info(f"Transcribing chunk {i}/{len(chunks)}: {chunk_path.name}")
            text = transcribe(chunk_path)
            if text:
                parts.append(f"--- Part {i} ---\n{text}")

        return "\n".join(parts).strip()
    finally:
        if len(chunks) > 1:
            cleanup_chunks(chunks)


@retry()
def summarise(transcript: str) -> str:
    """Summarise *transcript* using OpenRouter's chat endpoint.

    Sends a concise prompt that asks for a bullet-point summary.
    """
    logger.info(f"Summarising transcript (length={len(transcript)} chars)")
    payload = {
        "model": config.summarise_model(),
        "messages": [{"role": "user", "content": _SUMMARY_PROMPT.format(transcript=transcript)}],
        "max_tokens": config.max_tokens(),
    }
    data = _post_chat(payload)
    choices = data.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "").strip()
