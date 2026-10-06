
from src.search.search import search_transcripts


def test_search_transcripts_finds_matching_lines(tmp_path):
    transcript = tmp_path / "python_video_transcript.txt"
    transcript.write_text(
        "Today we talk about JavaScript.\n"
        "Python is great for AI development.\n"
        "We also discuss APIs.\n",
        encoding="utf-8",
    )

    results = search_transcripts(tmp_path, "python")

    assert len(results) == 1
    assert results[0]["filename"] == "python_video_transcript.txt"
    assert results[0]["matches"] == [
        {
            "snippet": "Python is great for AI development.",
            "timestamp": None,
        }
    ]


def test_search_transcripts_searches_multiple_videos(tmp_path):
    first = tmp_path / "first_video_transcript.txt"
    first.write_text(
        "Python is useful for AI.\n"
        "JavaScript is useful for web development.\n",
        encoding="utf-8",
    )

    second = tmp_path / "second_video_transcript.txt"
    second.write_text(
        "We are learning about Python today.\n"
        "APIs are useful for connecting services.\n",
        encoding="utf-8",
    )

    results = search_transcripts(tmp_path, "python")

    assert len(results) == 2
    assert results[0]["filename"] == "first_video_transcript.txt"
    assert results[1]["filename"] == "second_video_transcript.txt"


def test_search_transcripts_groups_multiple_matches_in_one_video(tmp_path):
    transcript = tmp_path / "python_video_transcript.txt"
    transcript.write_text(
        "Python is useful for AI.\n"
        "JavaScript is useful for web development.\n"
        "Python is also easy to learn.\n",
        encoding="utf-8",
    )

    results = search_transcripts(tmp_path, "python")

    assert len(results) == 1
    assert results[0]["filename"] == "python_video_transcript.txt"
    assert results[0]["matches"] == [
        {
            "snippet": "Python is useful for AI.",
            "timestamp": None,
        },
        {
            "snippet": "Python is also easy to learn.",
            "timestamp": None,
        },
    ]


def test_search_transcripts_detects_timestamp(tmp_path):
    transcript = tmp_path / "timestamped_video_transcript.txt"
    transcript.write_text(
        "[00:42] Python is useful for AI.\n"
        "[01:12:30] Python can automate tasks.\n",
        encoding="utf-8",
    )

    results = search_transcripts(tmp_path, "python")

    assert len(results) == 1
    assert results[0]["matches"] == [
        {
            "snippet": "[00:42] Python is useful for AI.",
            "timestamp": "00:42",
        },
        {
            "snippet": "[01:12:30] Python can automate tasks.",
            "timestamp": "01:12:30",
        },
    ]


