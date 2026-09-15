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

import yaml

from pipeline import config as config_mod
from pipeline import topics as topics_mod

from . import DATA_ROOT as APP_DATA_ROOT

CONFIG_PATH = APP_DATA_ROOT / "podcast_config.yaml"

# First-run defaults, per spec: past week, researcher, no familiar topics,
# medium (15-20 min), deep-dive (~2 min/source).
DEFAULTS: dict = {
    "user": {
        "audience": "researcher",
        "familiar_topics": [],
        "topic_prefs": [],
        "steering_alpha": config_mod.STEERING_ALPHA,
    },
    "podcast": {
        "window_days": 7,
        "length": "medium",
        "depth": "deep-dive",
        "mem_windows": config_mod.MEM_WINDOWS_DEFAULT,
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
    if user.get("topic_prefs") is not None:
        tp = user["topic_prefs"]
        cfg["user"]["topic_prefs"] = [
            str(t) for t in (tp if isinstance(tp, list) else [tp]) if str(t).strip()
        ]
    try:
        if user.get("steering_alpha") is not None:
            cfg["user"]["steering_alpha"] = float(user["steering_alpha"])
    except (TypeError, ValueError):
        pass
    try:
        if podcast.get("window_days") is not None:
            cfg["podcast"]["window_days"] = int(podcast["window_days"])
    except (TypeError, ValueError):
        pass
    if podcast.get("length"):
        cfg["podcast"]["length"] = str(podcast["length"])
    if podcast.get("depth"):
        cfg["podcast"]["depth"] = str(podcast["depth"])
    try:
        if podcast.get("mem_windows") is not None:
            cfg["podcast"]["mem_windows"] = int(podcast["mem_windows"])
    except (TypeError, ValueError):
        pass
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
        topic_prefs=[str(t) for t in cfg["user"]["topic_prefs"]],
        steering_alpha=float(cfg["user"]["steering_alpha"]),
        length=over.get("length") or cfg["podcast"]["length"],
        depth=over.get("depth") or cfg["podcast"]["depth"],
        mem_windows=int(cfg["podcast"]["mem_windows"]),
    )
    return {
        "window_start": run.window_start,
        "window_end": run.window_end,
        "audience_level": run.audience_level,
        "familiar_topics": list(run.familiar_topics),
        "topic_prefs": list(run.topic_prefs),
        "steering_alpha": run.steering_alpha,
        "length": run.length,
        "depth": run.depth,
        "mem_windows": run.mem_windows,
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
    tp = user.get("topic_prefs") or []
    if not isinstance(tp, list):
        raise ValueError("topic_prefs must be a list of strings")
    bad = [str(t) for t in tp if str(t).strip() and str(t).strip() not in topics_mod.TAXONOMY_BY_ID]
    if bad:
        raise ValueError(f"unknown topic_prefs {bad!r} — valid ids: "
                         f"{sorted(topics_mod.TAXONOMY_BY_ID)}")
    try:
        alpha = float(user.get("steering_alpha", config_mod.STEERING_ALPHA))
    except (TypeError, ValueError):
        raise ValueError("steering_alpha must be a number 0..1")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"steering_alpha must be 0..1, got {alpha}")
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
    mem_min = config_mod.MEM_WINDOWS_MIN
    mem_max = config_mod.MEM_WINDOWS_MAX
    try:
        mem_windows = int(podcast.get("mem_windows", config_mod.MEM_WINDOWS_DEFAULT))
    except (TypeError, ValueError) as e:
        raise ValueError(f"mem_windows must be an integer, got {podcast.get('mem_windows')!r}") from e
    if not mem_min <= mem_windows <= mem_max:
        raise ValueError(f"mem_windows must be {mem_min}..{mem_max}, got {mem_windows}")
    return {
        "user": {
            "audience": audience,
            "familiar_topics": [str(t).strip() for t in ft if str(t).strip()],
            "topic_prefs": [str(t).strip() for t in tp if str(t).strip()],
            "steering_alpha": alpha,
        },
        "podcast": {
            "window_days": window_days,
            "length": length,
            "depth": depth,
            "mem_windows": mem_windows,
        },
    }


def _deep_copy(cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        out[k] = {ik: list(iv) if isinstance(iv, list) else iv
                  for ik, iv in v.items()}
    return out
