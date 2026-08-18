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
    source: str          # e.g. "hf-papers", "arxiv", "hn", "rss:openai"

@dataclass
class RankedItem(Item):
    score: float         # 0.0 - 1.0, judge's rubric score
    judge_reason: str = ""

@dataclass
class Episode:
    audio_path: str      # path to the generated .mp3
    manifest: list[RankedItem]
    created_at: str      # ISO 8601 timestamp
