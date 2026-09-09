"""Per-run customizable config for the podcast pipeline.

A single ``RunConfig`` carries the user-facing knobs — time window, audience
level, familiar topics, podcast length, per-topic depth — from which the actual
targets (window dates, derived source count, depth factor) are computed.

Values are resolved in order: code defaults < config file < CLI overrides.

Format is YAML (PyYAML is already a dependency of the vendored podcastfy
stack; there is no TOML library in the dependency set).
"""
from __future__ import annotations

import dataclasses
import datetime
from dataclasses import dataclass, field
from pathlib import Path

# --- option vocabularies ----------------------------------------------------

PODCAST_LENGTH_CHOICES = ("short", "medium", "long")
TOPIC_DEPTH_CHOICES = ("brief", "deep-dive")
AUDIENCE_CHOICES = ("researcher", "intermediate", "beginner")

# Target duration (minutes) for each podcast-length option.
LENGTH_MINUTES = {"short": 10.0, "medium": 17.5, "long": 30.0}

# Target duration (minutes) per source for each depth option. This is the
# source-count driver: depth = how long each topic is given, length = how
# many topics fit (see RunConfig.budget).
DEPTH_MINUTES = {"brief": 1.0, "deep-dive": 2.0}

# Input reference fetch scale per depth — controls how much source text the
# LLM sees (per_paper_chars etc.). It does NOT scale the output word budget;
# duration is derived separately (WPM * length).
DEPTH_FACTOR = {"brief": 0.8, "deep-dive": 1.6}

# Spoken words-per-minute used to convert length presets into a total word
# budget. Fixed (measured deep-dive output runs ~165-190 wpm, so 165 gives a
# small safety margin).
WPM = 165.0

# Intro + recap together take this fraction of the total length; the rest is
# split evenly across sources.
INTRO_RECAP_FRACTION = 0.20
# Within the intro/recap slice, the intro gets this share (recap = 1 - share).
INTRO_SHARE = 0.40
# Word cap for the cross-episode MEMORY part: a brief "prior coverage sync"
# that connects the episode to prior coverage without becoming a topic.
MEMORY_WORDS = 90

MIN_SOURCES = 4
MAX_SOURCES = 30
MAX_WINDOW_DAYS = 14   # bound arXiv fetch cost (a week is ~1000 papers)
DEFAULT_WINDOW_DAYS = 7

# Cross-episode memory retention, in multiples of the episode's own window:
# an episode may reference prior episodes whose window-end falls within
# ``mem_windows * window_days`` of the current window-end (default 2 windows
# ~= the last 2 episodes for the default 7-day window).
MEM_WINDOWS_DEFAULT = 2
MEM_WINDOWS_MIN = 1
MEM_WINDOWS_MAX = 4

# --- interest steering ("item selection in the rank stage") ------------------
# final_score = (1 - alpha) * importance + alpha * personal_match. alpha is
# user-configurable (0 disables steering); the defaults below are safe/sane.
STEERING_ALPHA = 0.3

# Baseline audience sentence used by the transcript LLM. Researcher wording is
# the historical default; the others trade explanation depth for accessibility.
AUDIENCE_PROMPTS = {
    "researcher": (
        "AI researchers at a leading software consulting firm. They know ML "
        "fundamentals (RL, transformers, MoE, quantization, diffusion, "
        "autoregressive decoding, tokenization). Explain ONLY what is novel "
        "to each item. Do not define basics they already know."
    ),
    "intermediate": (
        "Practitioners with solid ML fundamentals but not specialists. Define "
        "specialized terms where helpful and keep explanations concrete, "
        "self-contained, and tied to practical impact."
    ),
    "beginner": (
        "Curious listeners who are new to AI and ML. Define core concepts "
        "(transformers, RL, quantization, etc.) in plain language, build "
        "intuition before detail, and avoid unexplained jargon."
    ),
}


@dataclass
class RunConfig:
    """Resolved user-facing config for one podcast episode.

    ``window_start`` / ``window_end`` are ISO dates (YYYY-MM-DD). When unset
    they default to ``end - 7 days`` and ``today`` respectively.
    """
    window_start: str | None = None
    window_end: str | None = None
    audience_level: str = "researcher"
    familiar_topics: list[str] = field(default_factory=list)
    # Topics the user is interested in (taxonomy ids). Empty = steering off.
    # Drives the rank-stage personal score: personal = cosine(item_topics,
    # profile), final = (1-alpha)*importance + alpha*personal.
    topic_prefs: list[str] = field(default_factory=list)
    steering_alpha: float = STEERING_ALPHA
    length: str = "medium"
    depth: str = "deep-dive"
    # Cross-episode memory retention in episode-windows (see MEM_WINDOWS_*).
    mem_windows: int = MEM_WINDOWS_DEFAULT

    # ------------------------------------------------------------------ utils

    def resolve_window(self) -> tuple[datetime.date, datetime.date]:
        """Return (start, end) dates, applying defaults and validating."""
        end_s = self.window_end or datetime.date.today().isoformat()
        end = _parse_date(end_s, "window.end")
        if self.window_start:
            start = _parse_date(self.window_start, "window.start")
        else:
            start = end - datetime.timedelta(days=DEFAULT_WINDOW_DAYS)
        if start > end:
            raise ValueError(
                f"window.start {start.isoformat()} is after window.end {end.isoformat()}"
            )
        if (end - start).days > MAX_WINDOW_DAYS:
            raise ValueError(
                f"window spans {(end - start).days} days, max {MAX_WINDOW_DAYS} "
                f"(arXiv volume grows ~150 papers/day)"
            )
        return start, end

    def budget(self) -> dict:
        """The episode's word budget, derived from length and per-source depth.

        Length fixes the TOTAL spoken word count (paced at ``WPM``); depth
        fixes the per-source time and therefore words per source. Source
        count = the topic budget (length minus the 20% intro/recap slice)
        divided by per-source words, so the produced duration conforms to the
        length preset instead of drifting.
        """
        total_words = round(LENGTH_MINUTES[self.length] * WPM)
        ir_words = round(total_words * INTRO_RECAP_FRACTION)
        intro_words = round(ir_words * INTRO_SHARE)
        recap_words = ir_words - intro_words
        per_source_words = round(DEPTH_MINUTES[self.depth] * WPM)
        topic_words = total_words - ir_words
        n = max(MIN_SOURCES, min(MAX_SOURCES, round(topic_words / per_source_words)))
        return {
            "total_words": total_words,
            "intro_words": intro_words,
            "recap_words": recap_words,
            "per_source_words": per_source_words,
            "num_sources": n,
        }

    def num_sources(self) -> int:
        """Source count derived from length / per-source depth."""
        return self.budget()["num_sources"]

    def depth_factor(self) -> float:
        return DEPTH_FACTOR[self.depth]

    def audience_prompt(self) -> str:
        return AUDIENCE_PROMPTS[self.audience_level]

    def familiar_clause(self) -> str:
        if not self.familiar_topics:
            return "none"
        return ", ".join(self.familiar_topics)

    def target_minutes(self) -> float:
        """Duration the budget should land on: intro/recap + sources × depth."""
        b = self.budget()
        return (LENGTH_MINUTES[self.length] * INTRO_RECAP_FRACTION
                + b["num_sources"] * DEPTH_MINUTES[self.depth])

    def podcastfy_overrides(self) -> dict:
        """Knobs threaded into the vendored podcastfy transcript generator.

        ``depth_factor`` + ``max_num_chunks`` scale how much source text the
        LLM sees; the word-budget keys set each part's output cap (so the
        transcript conforms to the length preset).
        """
        b = self.budget()
        return {
            "depth_factor": self.depth_factor(),
            "max_num_chunks": b["num_sources"],
            "per_source_words": b["per_source_words"],
            "intro_words": b["intro_words"],
            "recap_words": b["recap_words"],
            "memory_words": MEMORY_WORDS,
            "audience_prompt": self.audience_prompt(),
            "familiar_clause": self.familiar_clause(),
        }

    # ------------------------------------------------------------- persistence

    def to_dict(self) -> dict:
        return {
            "window": {
                "start": self.window_start,
                "end": self.window_end,
            },
            "audience": {
                "level": self.audience_level,
                "familiar_topics": list(self.familiar_topics),
                "topic_prefs": list(self.topic_prefs),
                "steering_alpha": self.steering_alpha,
            },
            "podcast": {
                "length": self.length,
                "depth": self.depth,
                "mem_windows": self.mem_windows,
                "num_sources": self.num_sources(),
                "target_minutes": self.target_minutes(),
                "word_budget": self.budget(),
            },
        }

    def to_yaml(self) -> str:
        import yaml
        body = {
            "window": {"start": self.window_start, "end": self.window_end},
            "audience": {
                "level": self.audience_level,
                "familiar_topics": list(self.familiar_topics),
                "topic_prefs": list(self.topic_prefs),
                "steering_alpha": self.steering_alpha,
            },
            "podcast": {"length": self.length, "depth": self.depth,
                        "mem_windows": self.mem_windows},
        }
        return yaml.safe_dump(body, sort_keys=False)


# --- resolution -------------------------------------------------------------

def resolve(config_path: str | Path | None = None, *,
            date: str | None = None,
            window_start: str | None = None,
            window_end: str | None = None,
            audience_level: str | None = None,
            familiar_topics: list[str] | None = None,
            topic_prefs: list[str] | None = None,
            steering_alpha: float | None = None,
            length: str | None = None,
            depth: str | None = None,
            mem_windows: int | None = None) -> RunConfig:
    """Build a ``RunConfig`` from defaults < config file < explicit overrides.

    ``date`` is a back-compat alias for ``window_end`` (the run anchor/end
    date). Applies only when ``window_end`` is not otherwise set.
    """
    cfg = RunConfig()
    if config_path is not None:
        cfg = _apply_file(cfg, config_path)
    # --date back-compat: it is the window end / run anchor. Only applies when
    # it is an actual ISO date — a run id (DD-MM-YYYY-HHMMSS, the unique run
    # folder the stage re-runs target) must never be parsed as a window date.
    if window_end is None and date is not None and _looks_iso(date):
        window_end = date
    for name, value in (
        ("window_start", window_start),
        ("window_end", window_end),
        ("audience_level", audience_level),
        ("familiar_topics", familiar_topics),
        ("topic_prefs", topic_prefs),
        ("steering_alpha", steering_alpha),
        ("length", length),
        ("depth", depth),
        ("mem_windows", mem_windows),
    ):
        if value is not None:
            setattr(cfg, name, value)
    _validate(cfg)
    return cfg


def _validate(cfg: RunConfig) -> None:
    if cfg.audience_level not in AUDIENCE_CHOICES:
        raise ValueError(f"audience level {cfg.audience_level!r} not in {AUDIENCE_CHOICES}")
    if cfg.length not in PODCAST_LENGTH_CHOICES:
        raise ValueError(f"podcast length {cfg.length!r} not in {PODCAST_LENGTH_CHOICES}")
    if cfg.depth not in TOPIC_DEPTH_CHOICES:
        raise ValueError(f"depth {cfg.depth!r} not in {TOPIC_DEPTH_CHOICES}")
    if not 0.0 <= cfg.steering_alpha <= 1.0:
        raise ValueError(f"steering_alpha {cfg.steering_alpha!r} must be 0..1")
    if not MEM_WINDOWS_MIN <= cfg.mem_windows <= MEM_WINDOWS_MAX:
        raise ValueError(
            f"mem_windows {cfg.mem_windows!r} must be {MEM_WINDOWS_MIN}..{MEM_WINDOWS_MAX}")
    from .topics import TAXONOMY_BY_ID
    bad = [t for t in cfg.topic_prefs if t not in TAXONOMY_BY_ID]
    if bad:
        raise ValueError(f"unknown topic_prefs {bad!r} — valid ids: {sorted(TAXONOMY_BY_ID)}")
    cfg.resolve_window()


def _apply_file(cfg: RunConfig, path: str | Path) -> RunConfig:
    import yaml
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--config: {p} not found")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file {p} must contain a YAML mapping")
    out = dataclasses.replace(cfg)
    window = raw.get("window") or {}
    if "start" in window and window["start"] is not None:
        out.window_start = str(window["start"])
    if "end" in window and window["end"] is not None:
        out.window_end = str(window["end"])
    audience = raw.get("audience") or {}
    if "level" in audience and audience["level"] is not None:
        out.audience_level = str(audience["level"])
    if "familiar_topics" in audience and audience["familiar_topics"] is not None:
        ft = audience["familiar_topics"]
        out.familiar_topics = [str(t) for t in (ft if isinstance(ft, list) else [ft])]
    if "topic_prefs" in audience and audience["topic_prefs"] is not None:
        tp = audience["topic_prefs"]
        out.topic_prefs = [str(t) for t in (tp if isinstance(tp, list) else [tp])]
    if "steering_alpha" in audience and audience["steering_alpha"] is not None:
        try:
            out.steering_alpha = float(audience["steering_alpha"])
        except (TypeError, ValueError):
            pass
    podcast = raw.get("podcast") or {}
    if "length" in podcast and podcast["length"] is not None:
        out.length = str(podcast["length"])
    if "depth" in podcast and podcast["depth"] is not None:
        out.depth = str(podcast["depth"])
    if "mem_windows" in podcast and podcast["mem_windows"] is not None:
        try:
            out.mem_windows = int(podcast["mem_windows"])
        except (TypeError, ValueError):
            pass
    return out


def _parse_date(s: str, label: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(s.strip())
    except ValueError:
        raise ValueError(f"{label} must be an ISO date YYYY-MM-DD, got {s!r}")


def _looks_iso(s: str) -> bool:
    try:
        datetime.date.fromisoformat(s.strip())
        return True
    except ValueError:
        return False
