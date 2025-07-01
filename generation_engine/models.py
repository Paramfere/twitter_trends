"""Pydantic data models used across the generation engine."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class TrendContext(BaseModel):
    """Aggregated information about a hot trend used for idea generation."""

    topic: str = Field(..., description="Trending topic name, e.g. 'Apple Vision Pro'")
    region: str = Field(..., description="Region where the topic is trending (US, SG, etc.)")
    significance: float = Field(..., ge=0, le=10, description="FERE significance score 0-10")
    tweet_volume: int | None = Field(None, description="Tweet volume count if known")
    exemplar_tweets: List[str] = Field(default_factory=list, description="High-engagement tweets (raw text)")

    model_config = {
        "frozen": True,
    } 