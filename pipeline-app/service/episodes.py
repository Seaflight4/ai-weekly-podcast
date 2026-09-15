"""Scan data/history/ for run folders and load their artifacts.

The filesystem is the source of truth — no DB. A run folder is named
``DD-MM-YYYY`` (legacy) or ``DD-MM-YYYY-HHMMSS`` (time-stamped run id, the
default for new episodes). Each folder may contain:

    collect.json     — every collected Item
    rank.json         — the full scored pool (RankedItem)
    podcast_brief.md — the source-grouped digest fed to the audio backend
    episode.json     — manifest + audio/transcript metadata
    episode.mp3      — the produced audio (absent if generation failed)
    transcript.md    — ground-truth (backend) or whisper transcript

One shared root for every episode, however it was created (a UI-generated
run, a CLI run, or a brief-edit re-render that replaces a date's episode in
place).
"""
from __future__ import annotations

import datetime
import json
import pathlib
import shutil

from . import DATA_ROOT as APP_DATA_ROOT

DATA_ROOT = APP_DATA_ROOT / "history"
DATE_FORMAT = "%d-%m-%Y"
# Run folders are time-stamped (DD-MM-YYYY-HHMMSS) so a second episode
# generated the same day gets its own folder instead of overwriting the first.
# Legacy date-only folders (DATE_FORMAT) remain readable.
RUN_ID_FORMAT = "%d-%m-%Y-%H%M%S"


def _normalize_date(date: str) -> str:
    """Accept ISO (YYYY-MM-DD), DD-MM-YYYY, or a run id DD-MM-YYYY-HHMMSS;
    return the folder name (ISO -> DD-MM-YYYY; other forms unchanged)."""
    for fmt in ("%Y-%m-%d", RUN_ID_FORMAT, DATE_FORMAT):
        try:
            parsed = datetime.datetime.strptime(date, fmt)
        except ValueError:
            continue
        return parsed.strftime(DATE_FORMAT if fmt == "%Y-%m-%d" else fmt)
    raise ValueError(f"bad date {date!r} — expected "
                     f"YYYY-MM-DD, DD-MM-YYYY, or DD-MM-YYYY-HHMMSS")


def _parse_run_id(name: str) -> datetime.datetime | None:
    """Parse a run folder name (DD-MM-YYYY or DD-MM-YYYY-HHMMSS) to a datetime,
    or None if it isn't a run folder."""
    for fmt in (RUN_ID_FORMAT, DATE_FORMAT):
        try:
            return datetime.datetime.strptime(name, fmt)
        except ValueError:
            continue
    return None


def make_run_id(window_end_iso: str,
                now: datetime.datetime | str | None = None) -> str:
    """Build the unique id for a new episode: the window-end date plus a
    user's local creation time (DD-MM-YYYY-HHMMSS), so a second episode
    generated the same day gets its own folder instead of overwriting the
    first.

    ``now`` is the client's local wall-clock time — a datetime or an ISO-ish
    string like ``YYYY-MM-DDTHH:MM:SS``. It falls back to the server's local
    time only when ``now`` is missing or unparseable (the server may run UTC,
    so a client-supplied local time keeps the stamp in the user's timezone).
    """
    d = datetime.date.fromisoformat(window_end_iso)
    if now is None:
        now = datetime.datetime.now()
    elif isinstance(now, str):
        try:
            now = datetime.datetime.fromisoformat(now.strip())
        except ValueError:
            now = datetime.datetime.now()
    return f"{d.strftime(DATE_FORMAT)}-{now.strftime('%H%M%S')}"


def _date_label(run_id: str) -> str:
    """Human label for a run id: 'DD-MM-YYYY' when it has no time component,
    else 'DD-MM-YYYY HH:MM'."""
    dt = _parse_run_id(run_id)
    if dt is None:
        return run_id
    if dt.hour == dt.minute == dt.second == 0:
        return dt.strftime(DATE_FORMAT)
    return dt.strftime(DATE_FORMAT + " %H:%M")


def _is_run_folder(name: str) -> bool:
    return _parse_run_id(name) is not None


def _list_runs(root: pathlib.Path) -> list[dict]:
    """All run folders under ``root``, newest first, with a status badge."""
    if not root.exists():
        return []
    runs: list[tuple[datetime.datetime, dict]] = []
    for p in root.iterdir():
        if not p.is_dir() or not _is_run_folder(p.name):
            continue
        dt = _parse_run_id(p.name)
        runs.append((dt, _run_summary(p)))
    runs.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in runs]


def _run_summary(folder: pathlib.Path) -> dict:
    ep = _load_json(folder / "episode.json")
    audio = folder / "episode.mp3"
    labels = _load_json(folder / "labels.json")
    if ep is None:
        status = "empty"
    elif not audio.exists():
        status = "draft"
    else:
        status = "ready"
    return {
        "date": folder.name,
        "date_label": _date_label(folder.name),
        "status": status,
        "has_audio": audio.exists(),
        "has_transcript": (folder / "transcript.md").exists(),
        "has_brief": (folder / "podcast_brief.md").exists(),
        "items": len(ep["manifest"]) if ep else 0,
        "selection_source": ep.get("selection_source") if ep else None,
        "created_at": ep.get("created_at") if ep else None,
        "duration_sec": ep.get("duration_sec") if ep else None,
        "transcript_words": ep.get("transcript_words") if ep else None,
        "topics": (labels or {}).get("episode_topics") or [],
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
    summary["labels"] = _load_json(folder / "labels.json")
    summary["memory"] = _load_json(folder / "memory.json")
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

def list_runs(topics: list[str] | None = None) -> list[dict]:
    """All runs, newest first. With ``topics``, keep only runs whose topic
    labels cover any of the given topic ids (OR-match)."""
    runs = _list_runs(DATA_ROOT)
    if topics:
        wanted = {t.strip().lower() for t in topics if t.strip()}
        if wanted:
            runs = [r for r in runs if _covers(r, wanted)]
    return runs


def _covers(run: dict, wanted: set[str]) -> bool:
    ids = {t.get("topic", "").lower() for t in (run.get("topics") or [])}
    return bool(ids & wanted)


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


def labels_path(date: str) -> pathlib.Path | None:
    folder = DATA_ROOT / _normalize_date(date)
    p = folder / "labels.json"
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
