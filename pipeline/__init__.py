from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

@dataclass
class Item:
    title: str
    url: str
    date: str            # ISO 8601, e.g. "2026-08-17"
    body: str
    source: str          # e.g. "hf-papers", "hn", "batch"

@dataclass
class RankedItem(Item):
    score: float         # 0.0 - 1.0, judge's rubric score
    judge_reason: str = ""

@dataclass
class StoryGroup:
    """A dedup group: near-duplicate Items describing the same underlying story.

    distil collapses same-story Items into one group and returns *every* group;
    selection happens later in rank.
    """
    title: str           # canonical story title
    content: str         # representative body (richest of the group)
    urls: list[str]      # distinct URLs across the group
    sources: set[str]    # distinct source tags, e.g. {"hf-papers", "rss:import-ai"}
    first_date: str      # earliest ISO date in the group
    consensus: int       # = len(sources) — how many distinct channels covered it
    # representative item kept for provenance
    rep_url: str = ""

@dataclass
class Episode:
    audio_path: str      # path to the generated .mp3
    manifest: list[RankedItem]
    created_at: str      # ISO 8601 timestamp
