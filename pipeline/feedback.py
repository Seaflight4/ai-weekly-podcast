"""Feedback log: binary kept/skipped marks per top-N item, aggregated across
runs with recency weighting (last 4 runs).

The personal-match pass (`rank.py`) reads the aggregated log so "items you
keep" rise and "items you skip" fall over time. Per-run marks live at
`data/<run>/feedback.json`; the aggregate lives at `feedback_log.json` at the
repo root.

Binary only (no 1-5 rating, no free-text skip reason) — the simplest signal
that still lets the personal pass learn "less of this, more of that" across
runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import datetime, json, pathlib

from . import store

FEEDBACK_LOG_PATH = pathlib.Path("feedback_log.json")
RECENT_RUNS = 4   # recency window: last 4 runs kept at full weight, older dropped


@dataclass
class FeedbackLog:
    kept_urls: set[str] = field(default_factory=set)
    skipped_urls: set[str] = field(default_factory=set)


def run_feedback_path(date: str | None = None) -> pathlib.Path:
    return store.run_dir(date) / "feedback.json"


def mark(url: str, kept: bool, date: str | None = None) -> pathlib.Path:
    """Append a kept/skipped mark for `url` to the current run's feedback.json,
    then rebuild the aggregate. Idempotent within a run: re-marking the same
    url overwrites the previous mark.
    """
    path = run_feedback_path(date)
    entry = {"url": url, "kept": bool(kept)}
    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = {"run": path.parent.name, "marks": []}
    # overwrite prior mark for the same url within this run
    data["marks"] = [m for m in data.get("marks", []) if m.get("url") != url]
    data["marks"].append(entry)
    data["run"] = path.parent.name
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    rebuild_aggregate()
    return path


def load_feedback(log_path: pathlib.Path | str | None = None) -> FeedbackLog:
    """Read the aggregated feedback log, recency-weighted (last RECENT_RUNS
    runs). Returns an empty FeedbackLog if the file is absent.

    `log_path` defaults to the current module-level `FEEDBACK_LOG_PATH`
    (resolved at call time so tests can monkeypatch it).
    """
    p = pathlib.Path(log_path) if log_path is not None else FEEDBACK_LOG_PATH
    if not p.exists():
        return FeedbackLog()
    data = json.loads(p.read_text())
    runs = data.get("runs", [])
    recent = runs[-RECENT_RUNS:]
    kept: set[str] = set()
    skipped: set[str] = set()
    for run in recent:
        for m in run.get("marks", []):
            url = m.get("url", "")
            if not url:
                continue
            if m.get("kept"):
                kept.add(url)
            else:
                skipped.add(url)
    return FeedbackLog(kept_urls=kept, skipped_urls=skipped)


def rebuild_aggregate() -> pathlib.Path:
    """Scan the data dir for `feedback.json` files, sort by run date, write the
    aggregate to `feedback_log.json` at the repo root. Idempotent. Uses
    `store.ROOT` so tests that monkeypatch the data dir work.
    """
    root = store.ROOT
    runs: list[dict] = []
    if root.exists():
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            fb = d / "feedback.json"
            if not fb.exists():
                continue
            try:
                data = json.loads(fb.read_text())
            except json.JSONDecodeError:
                continue
            data["run"] = d.name
            runs.append(data)
    payload = {"rebuilt_at": datetime.date.today().isoformat(),
                "recent_runs": RECENT_RUNS, "runs": runs}
    FEEDBACK_LOG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return FEEDBACK_LOG_PATH
