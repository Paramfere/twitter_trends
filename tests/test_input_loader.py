import csv
from pathlib import Path

import pandas as pd

from generation_engine.input_loader import load_trend_contexts


def _write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_load_trend_contexts(tmp_path: Path) -> None:
    """Basic happy-path integration test."""
    # Build fake session dir structure
    session_dir = tmp_path / "session_test"
    processed_dir = session_dir / "processed"
    raw_dir = session_dir / "raw" / "kaito_data"

    # Tech topics CSV
    topics_csv = processed_dir / "2025-07-01_12-00-00_tech_topics.csv"
    _write_csv(
        topics_csv,
        [
            {
                "topic": "LayerZero",
                "region": "US",
                "significance_score": "8.7",
                "tweet_volume": 120000,
            },
        ],
    )

    # Kaito summary CSV
    summary_csv = raw_dir / "kaito_tech_summary_2025-07-01_12-00-00.csv"
    _write_csv(
        summary_csv,
        [
            {"trend_topic": "LayerZero", "tweet_text": "LayerZero just flipped everything 🚀"},
            {"trend_topic": "LayerZero", "tweet_text": "Why $ZRO is pumping today?"},
        ],
    )

    contexts = load_trend_contexts(session_dir)
    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx.topic == "LayerZero"
    assert ctx.region == "US"
    assert ctx.significance == 8.7
    assert ctx.tweet_volume == 120000
    assert len(ctx.exemplar_tweets) == 2 