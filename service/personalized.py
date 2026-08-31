"""Personalized renders.

A single shared namespace under ``data/personalized/``. Each personalized
render for a given date holds: an edited brief + the re-generated audio +
transcript + a copy of the default run's rank.json. Default episodes
(``data/default/<date>/``) are never touched by personalized renders.

No per-user identity: the app runs on a colleague's local machine, so there
is one shared personalized library.
"""
from __future__ import annotations

import pathlib

from . import episodes
from .episodes import _normalize_date, _list_runs, _get_run, DATE_FORMAT

PERSONALIZED_ROOT = pathlib.Path("data/personalized")


def list_runs() -> list[dict]:
    """All personalized renders, newest first (same shape as
    episodes.list_runs)."""
    return _list_runs(PERSONALIZED_ROOT)


def get_run(date: str) -> dict | None:
    """Load every artifact for one personalized render."""
    return _get_run(PERSONALIZED_ROOT, date)


def audio_path(date: str) -> pathlib.Path | None:
    folder = PERSONALIZED_ROOT / _normalize_date(date)
    audio = folder / "episode.mp3"
    return audio if audio.exists() else None


def brief_path(date: str) -> pathlib.Path | None:
    folder = PERSONALIZED_ROOT / _normalize_date(date)
    p = folder / "podcast_brief.md"
    return p if p.exists() else None


def transcript_path(date: str) -> pathlib.Path | None:
    folder = PERSONALIZED_ROOT / _normalize_date(date)
    p = folder / "transcript.md"
    return p if p.exists() else None


def prepare_render(date: str, brief_markdown: str) -> pathlib.Path:
    """Create the personalized run dir for ``date``, copy in the default
    run's rank.json (so the subprocess can read it from its data root), and
    write the edited brief. Returns the run dir path.

    The default ``data/default/<date>/`` is never written — only read from
    (rank.json + the source brief). The render is self-contained under the
    personalized dir.
    """
    import shutil
    folder = PERSONALIZED_ROOT / _normalize_date(date)
    folder.mkdir(parents=True, exist_ok=True)
    # The pipeline's generate stage reads rank.json from its data root; copy
    # the default run's rank.json so the personalized subprocess can see it.
    default_rank = episodes.DATA_ROOT / _normalize_date(date) / "rank.json"
    if default_rank.exists():
        shutil.copy2(default_rank, folder / "rank.json")
    (folder / "podcast_brief.md").write_text(brief_markdown, encoding="utf-8")
    return folder
