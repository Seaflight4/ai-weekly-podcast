from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Item:
    title: str
    url: str
    date: str            # ISO 8601, e.g. "2026-08-17"
    body: str
    source: str          # e.g. "hn", "arxiv"
    # Relevance score (0.0-1.0) from the collect-stage small-model gate
    # (Mistral-Small). Used by the rank stage to prefilter the big-LLM input.
    # 0.0 when the item predates the score field or the gate did not score it.
    gate_score: float = 0.0
    # Hacker News popularity signal: the highest point score among HN stories
    # collected this week that point at this item (HN is a link-only source —
    # it adds attention, not content). Set on the HN item itself at collect;
    # when a cross-source dedup rule drops an HN twin, its points are folded
    # onto the surviving item so the rank judge can weigh community attention.
    # 0 when no HN story referenced the item.
    hn_points: int = 0

@dataclass
class RankedItem(Item):
    score: float = 0.0      # 0.0 - 1.0, judge's rubric score (importance)
    judge_reason: str = ""

@dataclass
class Episode:
    audio_path: str      # path to the generated .mp3
    manifest: list[RankedItem]
    created_at: str       # ISO 8601 timestamp
    # Resolved RunConfig (dict form) that produced this episode, for
    # reproducibility. Empty dict when the run used all defaults.
    config: dict = field(default_factory=dict)
