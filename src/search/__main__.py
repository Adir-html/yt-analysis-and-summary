import argparse
from pathlib import Path

from .search import search_transcripts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search YouTube transcripts for a keyword."
    )
    parser.add_argument("keyword", help="Keyword to search for")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory containing transcript files (default: ./output)",
    )

    args = parser.parse_args()

    results = search_transcripts(args.output_dir, args.keyword)

    if not results:
        print(f'No results found for "{args.keyword}".')
        return 0

    print(f'Found "{args.keyword}" in {len(results)} video(s):\n')

    for result in results:
        print(f"=== {result['filename']} ===")

        for match in result["matches"]:
            if match["timestamp"]:
                print(f"  [{match['timestamp']}] {match['snippet']}")
            else:
                print(f"  {match['snippet']}")

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())