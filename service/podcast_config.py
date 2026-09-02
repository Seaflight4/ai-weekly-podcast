"""Persistent podcast configuration: ``data/podcast_config.yaml``.

A single config file with two sections:

``user``     — who the podcast is for. Applies to EVERY generation:
               knowledge level, familiar topics. Used by manual runs and
               brief-edit re-renders alike.
``podcast``  — episode-shape defaults. These are the pre-filled defaults of
               the "Generate new episode" dialog (which the user confirms
               per run); they are not written back when a run is started.

  user:
    audience: researcher          # researcher | intermediate | beginner
    familiar_topics: []           # topics the audience already knows
  podcast:
    window_days: 7                # rolling news window (max MAX_WINDOW_DAYS)
    length: medium                # short | medium | long
    depth: deep-dive              # brief | deep-dive

Created on first use (the UI shows a setup dialog with these defaults);
afterwards it only changes via the Settings dialog. Editor notes / "generate
new episode" confirmations never write it back.

``resolved_run_config`` merges the file with per-run podcast overrides into
the flat dict ``jobs.full_run_cmd`` turns into ``pipeline`` CLI flags. The
pipeline never reads this file — the service is its only owner.
"""
from __future__ import annotations

import datetime
import pathlib

import yaml

from pipeline import config as config_mod

CONFIG_PATH = pathlib.Path("data/podcast_config.yaml")

# First-run defaults, per spec: past week, researcher, no familiar topics,
# medium (15-20 min), deep-dive (~2 min/source).
DEFAULTS: dict = {
    "user": {
        "audience": "researcher",
        "familiar_topics": [],
    },
    "podcast": {
        "window_days": 7,
        "length": "medium",
        "depth": "deep-dive",
    },
}

_MAX_WINDOW_DAYS = config_mod.MAX_WINDOW_DAYS


def exists() -> bool:
    return CONFIG_PATH.exists()


def load() -> dict:
    """Current config merged over the defaults. A missing or malformed file
    reads as the defaults (the GET endpoint reports that as ``first_run``)."""
    cfg = _deep_copy(DEFAULTS)
    if not CONFIG_PATH.exists():
        return cfg
    try:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return cfg
    if not isinstance(raw, dict):
        return cfg
    user = raw.get("user") or {}
    podcast = raw.get("podcast") or {}
    if user.get("audience"):
        cfg["user"]["audience"] = str(user["audience"])
    if user.get("familiar_topics") is not None:
        ft = user["familiar_topics"]
        cfg["user"]["familiar_topics"] = [
            str(t) for t in (ft if isinstance(ft, list) else [ft]) if str(t).strip()
        ]
    try:
        if podcast.get("window_days") is not None:
            cfg["podcast"]["window_days"] = int(podcast["window_days"])
    except (TypeError, ValueError):
        pass
    if podcast.get("length"):
        cfg["podcast"]["length"] = str(podcast["length"])
    if podcast.get("depth"):
        cfg["podcast"]["depth"] = str(podcast["depth"])
    return cfg


def save(cfg: dict) -> dict:
    """Validate and persist a user/podcast config dict. Returns the stored
    (normalized) config. Raises ValueError on invalid values."""
    stored = _validate(cfg)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(stored, sort_keys=False), encoding="utf-8")
    return stored


def resolved_run_config(podcast: dict | None = None) -> dict:
    """Merge the persistent config with per-run podcast overrides into the
    flat knob dict ``jobs.full_run_cmd`` understands.

    ``podcast`` (from the generate dialog) keys: window_start, window_end,
    length, depth. Any missing key falls back to the config file; without
    overrides the rolling window resolves at call time:
    end = today, start = today - window_days.

    The merged result is run through ``pipeline.config.resolve`` so dates,
    vocabularies and the max window span are validated the same way the
    pipeline CLI would validate them. Raises ValueError on bad input.
    """
    cfg = load()
    over = podcast or {}
    end = over.get("window_end") or datetime.date.today().isoformat()
    start = over.get("window_start")
    if not start:
        start = (datetime.date.fromisoformat(end)
                 - datetime.timedelta(days=int(cfg["podcast"]["window_days"]))
                 ).isoformat()
    run = config_mod.resolve(
        window_start=str(start),
        window_end=str(end),
        audience_level=cfg["user"]["audience"],
        familiar_topics=[str(t) for t in cfg["user"]["familiar_topics"]],
        length=over.get("length") or cfg["podcast"]["length"],
        depth=over.get("depth") or cfg["podcast"]["depth"],
    )
    return {
        "window_start": run.window_start,
        "window_end": run.window_end,
        "audience_level": run.audience_level,
        "familiar_topics": list(run.familiar_topics),
        "length": run.length,
        "depth": run.depth,
        "window_days": (run.resolve_window()[1] - run.resolve_window()[0]).days,
        "num_sources": run.num_sources(),
    }


def _validate(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        raise ValueError("config must be an object with 'user' and 'podcast'")
    user = cfg.get("user") or {}
    podcast = cfg.get("podcast") or {}
    audience = str(user.get("audience") or "").strip()
    if audience not in config_mod.AUDIENCE_CHOICES:
        raise ValueError(f"audience {audience!r} not in {config_mod.AUDIENCE_CHOICES}")
    ft = user.get("familiar_topics") or []
    if not isinstance(ft, list):
        raise ValueError("familiar_topics must be a list of strings")
    try:
        window_days = int(podcast.get("window_days", DEFAULTS["podcast"]["window_days"]))
    except (TypeError, ValueError) as e:
        raise ValueError(f"window_days must be an integer, got {podcast.get('window_days')!r}") from e
    if not 1 <= window_days <= _MAX_WINDOW_DAYS:
        raise ValueError(f"window_days must be 1..{_MAX_WINDOW_DAYS}, got {window_days}")
    length = str(podcast.get("length") or "").strip()
    if length not in config_mod.PODCAST_LENGTH_CHOICES:
        raise ValueError(f"length {length!r} not in {config_mod.PODCAST_LENGTH_CHOICES}")
    depth = str(podcast.get("depth") or "").strip()
    if depth not in config_mod.TOPIC_DEPTH_CHOICES:
        raise ValueError(f"depth {depth!r} not in {config_mod.TOPIC_DEPTH_CHOICES}")
    return {
        "user": {
            "audience": audience,
            "familiar_topics": [str(t).strip() for t in ft if str(t).strip()],
        },
        "podcast": {
            "window_days": window_days,
            "length": length,
            "depth": depth,
        },
    }


def _deep_copy(cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        out[k] = {ik: list(iv) if isinstance(iv, list) else iv
                  for ik, iv in v.items()}
    return out
