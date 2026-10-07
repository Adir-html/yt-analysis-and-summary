from pathlib import Path
import re

TIMESTAMP_PATTERN = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")


def extract_timestamp(line: str) -> str | None:
    """Extract a timestamp from a transcript line if one is present."""
    match = TIMESTAMP_PATTERN.search(line)

    if match:
        return match.group(1)

    return None


def search_transcripts(output_dir: Path, keyword: str) -> list[dict]:
    """Search transcript files for a keyword."""
    results = []

    for transcript_path in sorted(output_dir.glob("*_transcript.txt")):
        matches = []

        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            if keyword.lower() in line.lower():
                matches.append(
                    {
                        "snippet": line,
                        "timestamp": extract_timestamp(line),
                    }
                )

        if matches:
            results.append(
                {
                    "filename": transcript_path.name,
                    "matches": matches,
                }
            )

    return results
