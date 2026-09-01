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

# Target duration (minutes) per source for each depth option.
DEPTH_MINUTES = {"brief": 1.0, "deep-dive": 2.0}

# Per-source transcript budget multiplier vs the pipeline's historical
# ~1.25 min/source baseline (brief trims it, deep-dive extends it).
DEPTH_FACTOR = {"brief": 0.8, "deep-dive": 1.6}

MIN_SOURCES = 4
MAX_SOURCES = 30
MAX_WINDOW_DAYS = 14   # bound arXiv fetch cost (a week is ~1000 papers)
DEFAULT_WINDOW_DAYS = 7

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
    length: str = "medium"
    depth: str = "deep-dive"

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

    def num_sources(self) -> int:
        """Derived source count from podcast length / per-topic depth."""
        n = round(LENGTH_MINUTES[self.length] / DEPTH_MINUTES[self.depth])
        return max(MIN_SOURCES, min(MAX_SOURCES, n))

    def depth_factor(self) -> float:
        return DEPTH_FACTOR[self.depth]

    def audience_prompt(self) -> str:
        return AUDIENCE_PROMPTS[self.audience_level]

    def familiar_clause(self) -> str:
        if not self.familiar_topics:
            return "none"
        return ", ".join(self.familiar_topics)

    def target_minutes(self) -> float:
        """Estimated episode duration = num_sources * per-topic minutes."""
        return self.num_sources() * DEPTH_MINUTES[self.depth]

    def podcastfy_overrides(self) -> dict:
        """Knobs threaded into the vendored podcastfy transcript generator."""
        return {
            "depth_factor": self.depth_factor(),
            "max_num_chunks": self.num_sources(),
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
            },
            "podcast": {
                "length": self.length,
                "depth": self.depth,
                "num_sources": self.num_sources(),
                "target_minutes": self.target_minutes(),
            },
        }

    def to_yaml(self) -> str:
        import yaml
        body = {
            "window": {"start": self.window_start, "end": self.window_end},
            "audience": {
                "level": self.audience_level,
                "familiar_topics": list(self.familiar_topics),
            },
            "podcast": {"length": self.length, "depth": self.depth},
        }
        return yaml.safe_dump(body, sort_keys=False)


# --- resolution -------------------------------------------------------------

def resolve(config_path: str | Path | None = None, *,
            date: str | None = None,
            window_start: str | None = None,
            window_end: str | None = None,
            audience_level: str | None = None,
            familiar_topics: list[str] | None = None,
            length: str | None = None,
            depth: str | None = None) -> RunConfig:
    """Build a ``RunConfig`` from defaults < config file < explicit overrides.

    ``date`` is a back-compat alias for ``window_end`` (the run anchor/end
    date). Applies only when ``window_end`` is not otherwise set.
    """
    cfg = RunConfig()
    if config_path is not None:
        cfg = _apply_file(cfg, config_path)
    # --date back-compat: it is the window end / run anchor.
    if window_end is None and date is not None:
        window_end = date
    for name, value in (
        ("window_start", window_start),
        ("window_end", window_end),
        ("audience_level", audience_level),
        ("familiar_topics", familiar_topics),
        ("length", length),
        ("depth", depth),
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
    podcast = raw.get("podcast") or {}
    if "length" in podcast and podcast["length"] is not None:
        out.length = str(podcast["length"])
    if "depth" in podcast and podcast["depth"] is not None:
        out.depth = str(podcast["depth"])
    return out


def _parse_date(s: str, label: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(s.strip())
    except ValueError:
        raise ValueError(f"{label} must be an ISO date YYYY-MM-DD, got {s!r}")
