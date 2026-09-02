"""Scan data/history/ for run folders and load their artifacts.

The filesystem is the source of truth — no DB. A run folder is named
``DD-MM-YYYY`` (the format the pipeline writes). Each folder may contain:

    collect.json     — every collected Item
    rank.json         — the full scored pool (RankedItem)
    podcast_brief.md — the source-grouped digest fed to the audio backend
    episode.json     — manifest + audio/transcript metadata
    episode.mp3      — the produced audio (absent if generation failed)
    transcript.md    — ground-truth (backend) or whisper transcript

One shared root for every episode, however it was created (scheduled
auto-run, manual generate, or a brief-edit re-render that replaces a date's
episode in place).
"""
from __future__ import annotations

import datetime
import json
import pathlib
import shutil

DATA_ROOT = pathlib.Path("data/history")
DATE_FORMAT = "%d-%m-%Y"


def _normalize_date(date: str) -> str:
    """Accept ISO (YYYY-MM-DD) or DD-MM-YYYY; return DD-MM-YYYY (folder format)."""
    for fmt in ("%Y-%m-%d", DATE_FORMAT):
        try:
            return datetime.datetime.strptime(date, fmt).strftime(DATE_FORMAT)
        except ValueError:
            continue
    raise ValueError(f"bad date {date!r} — expected YYYY-MM-DD or DD-MM-YYYY")


def _is_run_folder(name: str) -> bool:
    try:
        datetime.datetime.strptime(name, DATE_FORMAT)
        return True
    except ValueError:
        return False


def _list_runs(root: pathlib.Path) -> list[dict]:
    """All run folders under ``root``, newest first, with a status badge."""
    if not root.exists():
        return []
    runs: list[tuple[datetime.date, dict]] = []
    for p in root.iterdir():
        if not p.is_dir() or not _is_run_folder(p.name):
            continue
        d = datetime.datetime.strptime(p.name, DATE_FORMAT).date()
        runs.append((d, _run_summary(p)))
    runs.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in runs]


def _run_summary(folder: pathlib.Path) -> dict:
    ep = _load_json(folder / "episode.json")
    audio = folder / "episode.mp3"
    if ep is None:
        status = "empty"
    elif not audio.exists():
        status = "draft"
    else:
        status = "ready"
    return {
        "date": folder.name,
        "status": status,
        "has_audio": audio.exists(),
        "has_transcript": (folder / "transcript.md").exists(),
        "has_brief": (folder / "podcast_brief.md").exists(),
        "items": len(ep["manifest"]) if ep else 0,
        "selection_source": ep.get("selection_source") if ep else None,
        "created_at": ep.get("created_at") if ep else None,
    }


def _get_run(root: pathlib.Path, date: str) -> dict | None:
    """Load every artifact for one run folder. Returns None if not found."""
    folder = root / _normalize_date(date)
    if not folder.is_dir():
        return None
    summary = _run_summary(folder)
    summary["brief"] = _read_text(folder / "podcast_brief.md")
    summary["transcript"] = _read_text(folder / "transcript.md")
    summary["episode"] = _load_json(folder / "episode.json")
    summary["rank"] = _load_json(folder / "rank.json")
    return summary


def _load_json(path: pathlib.Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_text(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


# --- public API -----------------------------------------------------------

def list_runs() -> list[dict]:
    return _list_runs(DATA_ROOT)


def get_run(date: str) -> dict | None:
    return _get_run(DATA_ROOT, date)


def audio_path(date: str) -> pathlib.Path | None:
    folder = DATA_ROOT / _normalize_date(date)
    audio = folder / "episode.mp3"
    return audio if audio.exists() else None


def brief_path(date: str) -> pathlib.Path | None:
    folder = DATA_ROOT / _normalize_date(date)
    p = folder / "podcast_brief.md"
    return p if p.exists() else None


def transcript_path(date: str) -> pathlib.Path | None:
    folder = DATA_ROOT / _normalize_date(date)
    p = folder / "transcript.md"
    return p if p.exists() else None


def run_folder(date: str) -> pathlib.Path:
    """Return (creating if needed) the run folder for a date."""
    folder = DATA_ROOT / _normalize_date(date)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def delete_run(date: str) -> bool:
    """Delete the whole history entry for ``date`` — folder and artifacts.

    Returns True if a folder existed and was removed, False if there was
    nothing to delete. Raises ValueError for an unparseable date.
    """
    folder = DATA_ROOT / _normalize_date(date)
    if not folder.is_dir():
        return False
    shutil.rmtree(folder)
    return True
