"""APScheduler wrapper: auto-run a fresh episode on a cron trigger.

Default: every Monday 09:00 (configurable via the SCHEDULE_CRON env var, a
5-field cron string: "min hour day month day-of-week"). On fire it enqueues
a full ``pipeline run`` into the jobs registry; if a run is already active
it skips (no queue) and logs the skip.

Config is persisted to ``data/.schedule.json`` so runtime toggles survive
restarts. The FastAPI app starts/stops the scheduler on its lifespan.
"""
from __future__ import annotations

import json
import os
import pathlib
import threading

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from . import jobs

CONFIG_PATH = pathlib.Path("data/.schedule.json")
DEFAULT_CRON = os.environ.get("SCHEDULE_CRON", "0 9 * * fri")
DEFAULT_ENABLED = os.environ.get("SCHEDULE_ENABLED", "true").lower() in ("1", "true", "yes", "on")
JOB_ID = "weekly-episode"

_scheduler: BackgroundScheduler | None = None
_lock = threading.Lock()
_state = {"enabled": DEFAULT_ENABLED, "cron": DEFAULT_CRON, "next_fire": None}


def _parse_cron(expr: str) -> CronTrigger:
    """5-field cron: minute hour day month day-of-week.

    day-of-week accepts the same names APSscheduler does: mon, tue, ... sun
    (case-insensitive) or numeric 0-6 where 0=Monday (APScheduler convention,
    NOT standard cron where 0=Sunday). Prefer names for clarity.
    """
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(f"cron must have 5 fields, got {len(parts)}: {expr!r}")
    return CronTrigger(
        minute=parts[0], hour=parts[1], day=parts[2],
        month=parts[3], day_of_week=parts[4],
    )


def _fire():
    """Scheduler callback: enqueue a full run unless one is active.

    The run uses the full persistent config (``data/podcast_config.yaml``):
    user-wise knobs always, podcast-wise resolved at fire time with a fresh
    rolling window (end = today, start = today - window_days).
    """
    from . import app  # noqa: avoid circular import at module load
    from . import episodes
    from . import podcast_config
    env = {"PIPELINE_DATA_ROOT": str(episodes.DATA_ROOT)}
    cmd = jobs.full_run_cmd(config=podcast_config.resolved_run_config())
    job, err = jobs.submit("full", cmd, date=None, env=env)
    if err == "busy":
        print("[scheduler] skipped — a run is already active")
        app.broadcast({"type": "schedule-skip", "reason": "run-in-progress"})
        return
    if job is not None:
        app.broadcast({"type": "job-started", "job": job.to_dict()})


def state() -> dict:
    """Current scheduler state, including the next fire time."""
    next_fire = None
    if _scheduler is not None:
        try:
            next_fire = _scheduler.get_job(JOB_ID).next_run_time
        except Exception:
            next_fire = None
    return {
        "enabled": _state["enabled"],
        "cron": _state["cron"],
        "next_fire": next_fire.isoformat() if next_fire else None,
    }


def configure(enabled: bool | None = None, cron: str | None = None) -> dict:
    """Update enabled flag and/or cron expression, persist, and reschedule."""
    with _lock:
        if cron is not None:
            _parse_cron(cron)  # validate before saving
            _state["cron"] = cron
        if enabled is not None:
            _state["enabled"] = enabled
        _persist()
    _reschedule()
    return state()


def _reschedule():
    if _scheduler is None:
        return
    existing = _scheduler.get_job(JOB_ID)
    if existing is not None:
        _scheduler.remove_job(JOB_ID)
    if _state["enabled"]:
        trigger = _parse_cron(_state["cron"])
        _scheduler.add_job(_fire, trigger, id=JOB_ID, replace_existing=True)


def _persist():
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(_state), encoding="utf-8")


def _load():
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            _state["enabled"] = saved.get("enabled", DEFAULT_ENABLED)
            _state["cron"] = saved.get("cron", DEFAULT_CRON)
        except (json.JSONDecodeError, OSError):
            pass


def start():
    global _scheduler
    _load()
    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.start()
    _reschedule()


def shutdown():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
