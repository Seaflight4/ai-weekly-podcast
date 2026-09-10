"""Async job store for podcast generation.

A single-worker thread pool serializes generation (it's heavy: LLM + TTS).
Jobs are persisted in a sqlite database so status survives restarts.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from podcast_engine import EngineConfig, Source, generate_podcast


_DB_PATH = pathlib.Path("data/.mcp_jobs.db")
_LOCK = threading.Lock()
_EXECUTOR: ThreadPoolExecutor | None = None


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
            id          TEXT PRIMARY KEY,
            status      TEXT NOT NULL DEFAULT 'pending',
            stage       TEXT,
            error       TEXT,
            run_dir     TEXT,
            transcript  TEXT,
            audio_path  TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def submit(sources: list[Source], config: EngineConfig) -> str:
    """Create a job and submit it to the background worker.

    Returns the job ID immediately. The worker calls
    :func:`podcast_engine.generate_podcast` and updates the job row.
    """
    job_id = uuid.uuid4().hex
    with _LOCK:
        conn = _get_db()
        conn.execute(
            "INSERT INTO jobs (id, status, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (job_id, "pending", _now(), _now()),
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
