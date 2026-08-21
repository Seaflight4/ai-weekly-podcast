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
    source: str          # e.g. "hn", "arxiv"

@dataclass
class RankedItem(Item):
    score: float         # 0.0 - 1.0, judge's rubric score
    judge_reason: str = ""
    kind: str = ""       # informational tag ("deep" | "brief"); no longer the
                         # episode's top-level structure (topics drive that now)

@dataclass
class Topic:
    """A group of ranked items describing the same theme/event.

    Built by `cluster`: MiniLM candidate grouping -> LLM label + cross-cluster
    merge. `members` is a list of RankedItem; `primary_url` names the anchor
    member; `aggregate_score` = max member score (the importance signal for
    topic-level selection).
    """
    id: str
    title: str = ""           # 2-5 word LLM label (filled by cluster)
    why: str = ""             # one-sentence "why this matters" (filled by cluster)
    members: list[RankedItem] = field(default_factory=list)
    primary_url: str = ""
    aggregate_score: float = 0.0

@dataclass
class Segment:
    """One topic's slot in the narrative plan."""
    topic_id: str
    minutes: float            # budget-allocated minutes
    opening: str = ""         # one-sentence setup
    signposts: list[str] = field(default_factory=list)  # listener wayfinding cues
    transition_out: str = ""  # bridge to the next topic
    speakers: list[str] = field(default_factory=list)  # which host drives it

@dataclass
class EpisodePlan:
    """The narrative planner's output; `generate` renders it into a brief."""
    date: str
    target_minutes: float
    hook: str
    motif: str
    segments: list[Segment] = field(default_factory=list)
    outro: str = ""

@dataclass
class Episode:
    audio_path: str      # path to the generated .mp3
    manifest: list[RankedItem]
    created_at: str       # ISO 8601 timestamp
