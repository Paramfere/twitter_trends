"""Utilities for loading a FERE session into TrendContext objects."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .models import TrendContext


def _latest_csv(glob_pattern: str) -> Path | None:  # noqa: D401
    """Return the newest CSV path matching *glob_pattern* or *None* if none found."""
    matches = sorted(Path().glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def load_trend_contexts(session_dir: Path, max_exemplars: int = 3) -> List[TrendContext]:
    """Merge tech-topics CSV and Kaito summary into :class:`TrendContext` objects.

    Parameters
    ----------
    session_dir:
        Path to a single FERE session directory (e.g. ``data/session_01-Jul-2025_20-46-42``).
    max_exemplars:
        Maximum number of exemplar tweets to attach per topic.
    """

    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(session_dir)

    # Locate files – be resilient to timestamp naming
    topics_csv = _latest_csv(str(session_dir / "processed" / "*_tech_topics.csv"))
    summary_csv = _latest_csv(str(session_dir / "raw" / "kaito_data" / "kaito_tech_summary_*.csv"))

    if topics_csv is None:
        raise FileNotFoundError("No *_tech_topics.csv found in " + str(session_dir))
    if summary_csv is None:
        # Exemplar tweets optional – allow empty list
        summary_df = pd.DataFrame()
    else:
        summary_df = pd.read_csv(summary_csv)

    topics_df = pd.read_csv(topics_csv)

    trend_contexts: List[TrendContext] = []
    for _, row in topics_df.iterrows():
        topic = str(row["topic"])
        region = str(row.get("region", "US"))
        significance_raw = row.get("significance_score", 0.0)
        significance = float(significance_raw) if significance_raw is not None else 0.0  # type: ignore[arg-type]

        tweet_vol_raw = row.get("tweet_volume")
        if tweet_vol_raw is None or (isinstance(tweet_vol_raw, float) and pd.isna(tweet_vol_raw)):
            tweet_volume_int = None
        else:
            try:
                tweet_volume_int = int(tweet_vol_raw)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                tweet_volume_int = None

        if not summary_df.empty:
            tweets_for_topic = summary_df[summary_df.get("trend_topic", "") == topic]
            exemplars = (
                tweets_for_topic["tweet_text"].astype(str).head(max_exemplars).tolist()  # type: ignore[index]
                if "tweet_text" in tweets_for_topic.columns
                else []
            )
        else:
            exemplars = []

        trend_contexts.append(
            TrendContext(
                topic=topic,
                region=region,
                significance=significance,
                tweet_volume=tweet_volume_int,
                exemplar_tweets=exemplars,
            ),
        )

    # Sort by significance desc
    return sorted(trend_contexts, key=lambda tc: tc.significance, reverse=True) 