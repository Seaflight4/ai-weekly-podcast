from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Item:
    title: str
    url: str
    date: str            # ISO 8601, e.g. "2026-08-17"
    body: str
    source: str          # e.g. "hn", "arxiv"

@dataclass
class RankedItem(Item):
    score: float         # 0.0 - 1.0, judge's rubric score (importance)
    judge_reason: str = ""
    # Phase 2 personalization. Defaults keep Phase 1 behaviour:
    # final_score == score, no personal pass applied.
    personal_score: float = 0.0    # 0.0-1.0, personal-match pass
    personal_reason: str = ""      # one-sentence match rationale
    alpha: float = 1.0             # blend weight used for this item's final score
    final_score: float = 0.0       # alpha*score + (1-alpha)*personal_score

@dataclass
class Episode:
    audio_path: str      # path to the generated .mp3
    manifest: list[RankedItem]
    created_at: str       # ISO 8601 timestamp
