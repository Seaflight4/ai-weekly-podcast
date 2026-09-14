"""Async job store for podcast generation.

A single-worker thread pool serializes generation (it's heavy: LLM + TTS).
Jobs are persisted in a sqlite database so status survives restarts.
"""
from __future__ import annotations

import pathlib
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from podcast_engine import EngineConfig, Source, generate_podcast


_DB_PATH = pathlib.Path("data/.mcp_jobs.db")
_LOCK = threading.Lock()
_EXECUTOR: ThreadPoolExecutor | None = None

# Minimum time between resource reads for a non-terminal job, and the delay
# before the first read is allowed. Keeps a polling client to ~1 check/minute
# instead of hammering the job immediately after launch.
POLL_INTERVAL_SECONDS = 60.0


def _get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=1)
    return _EXECUTOR


def _get_db() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id            TEXT PRIMARY KEY,
            status        TEXT NOT NULL DEFAULT 'pending',
            stage         TEXT,
            error         TEXT,
            run_dir       TEXT,
            transcript    TEXT,
            audio_path    TEXT,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            next_poll_at  TEXT
        )
    """)
    # Lightweight migration: add the pacing column to DBs created before
    # it existed (CREATE TABLE IF NOT EXISTS never alters existing tables).
    cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "next_poll_at" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN next_poll_at TEXT")
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _next_poll_at(job_id: str) -> datetime | None:
    """The earliest time a non-terminal job may be read again (or None if unknown)."""
    with _LOCK:
        conn = _get_db()
        row = conn.execute(
            "SELECT next_poll_at FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        conn.close()
    return _parse_ts(row[0] if row else None)


def submit(sources: list[Source], config: EngineConfig) -> str:
    """Create a job and submit it to the background worker.

    Returns the job ID immediately. The worker calls
    :func:`podcast_engine.generate_podcast` and updates the job row.
    """
    job_id = uuid.uuid4().hex
    with _LOCK:
        conn = _get_db()
        created = _now()
        conn.execute(
            "INSERT INTO jobs (id, status, created_at, updated_at, next_poll_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (job_id, "pending", created, created,
             (_now_dt() + timedelta(seconds=POLL_INTERVAL_SECONDS)).isoformat()),
        )
        conn.commit()
        conn.close()

    _get_executor().submit(_run_job, job_id, sources, config)
    return job_id


def _run_job(job_id: str, sources: list[Source], config: EngineConfig) -> None:
    _update(job_id, status="running", stage="fetching sources")
    try:
        def on_part(idx: int, text: str) -> None:
            _update(job_id, stage=f"synthesizing part {idx + 1}")

        result = generate_podcast(
            sources=sources,
            config=config,
            on_part=on_part,
        )

        transcript_text = ""
        if result.transcript_path and result.transcript_path.exists():
            transcript_text = result.transcript_path.read_text(encoding="utf-8")

        _update(
            job_id,
            status="completed",
            stage="done",
            run_dir=str(result.run_dir),
            transcript=transcript_text,
            audio_path=str(result.audio_path) if result.audio_path else "",
        )
    except Exception as e:
        _update(job_id, status="failed", stage="error", error=str(e))


def _update(job_id: str, **fields) -> None:
    fields["updated_at"] = _now()
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values())
    with _LOCK:
        conn = _get_db()
        conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", vals + [job_id])
        conn.commit()
        conn.close()


def get(job_id: str) -> dict:
    """Return the current state of a job as a dict."""
    with _LOCK:
        conn = _get_db()
        row = conn.execute(
            "SELECT id, status, stage, error, run_dir, transcript, audio_path, "
            "created_at, updated_at FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        conn.close()
    if row is None:
        return {"status": "not_found", "error": f"job {job_id} not found"}
    return {
        "job_id": row[0],
        "status": row[1],
        "stage": row[2],
        "error": row[3],
        "run_dir": row[4],
        "transcript": row[5] if row[5] else "",
        "audio_path": row[6] if row[6] else "",
        "created_at": row[7],
        "updated_at": row[8],
    }


def poll(job_id: str, interval: float = POLL_INTERVAL_SECONDS) -> dict:
    """Return a job's state, pacing reads so they land ~once per ``interval``.

    Used by the ``podcast://jobs/{job_id}`` resource. Terminal jobs
    (completed/failed/not_found) return immediately without waiting. For an
    in-flight job the call blocks until its ``next_poll_at`` (set to ~one
    interval after submission, then advanced by one interval per read), so
    the first read never comes right after launch and the client cannot
    poll faster than one check per interval.

    ``interval`` is overridable so tests can run with a tiny value.
    """
    result = get(job_id)
    if result["status"] in ("completed", "failed", "not_found"):
        return result

    next_read = _next_poll_at(job_id)
    if next_read is not None:
        wait = (next_read - _now_dt()).total_seconds()
        if wait > 0:
            time.sleep(wait)

    with _LOCK:
        conn = _get_db()
        conn.execute(
            "UPDATE jobs SET next_poll_at = ? WHERE id = ?",
            ((_now_dt() + timedelta(seconds=interval)).isoformat(), job_id),
        )
        conn.commit()
        conn.close()

    return get(job_id)
