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
    score: float         # 0.0 - 1.0, judge's rubric score (importance)
    judge_reason: str = ""
    kind: str = ""       # informational tag ("deep" | "brief"); no longer the
                         # episode's top-level structure (topics drive that now)
    # Phase 2 personalization (tickets 04/05). Defaults keep Phase 1 behaviour:
    # final_score == score, no personal pass applied.
    personal_score: float = 0.0    # 0.0-1.0, personal-match pass (ticket 04)
    personal_reason: str = ""      # one-sentence match rationale
    alpha: float = 1.0             # blend weight used for this item's final score
    final_score: float = 0.0       # alpha*score + (1-alpha)*personal_score (ticket 05)

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
    editorial_group: str = ""  # label of the editorial meta-group (ticket 13)

@dataclass
class EditorialGroup:
    """An editorial meta-group of topics, proposed by the plan stage (ticket 13).

    Groups topics by their editorial role in the week's narrative — e.g. "3
    model releases", "2 alerting developments", "This week's surprises". This
    is podcast narratology, not topic similarity: a release group can hold
    semantically-distant items because they're all "this week's model drops".
    """
    label: str                # human-readable group name
    role: str                 # one of: release|method|incident|evaluation|industry|analysis|standalone
    topic_ids: list[str] = field(default_factory=list)
    opener: str = ""          # one-sentence group-level setup

@dataclass
class EpisodePlan:
    """The narrative planner's output; `generate` renders it into a brief."""
    date: str
    target_minutes: float
    hook: str
    motif: str
    segments: list[Segment] = field(default_factory=list)
    outro: str = ""
    editorial_groups: list[EditorialGroup] = field(default_factory=list)  # ticket 13

@dataclass
class Episode:
    audio_path: str      # path to the generated .mp3
    manifest: list[RankedItem]
    created_at: str       # ISO 8601 timestamp
