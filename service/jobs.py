"""Background job runner for pipeline runs.

One run at a time, process-wide. A run spawns ``python -m pipeline run ...``
as a subprocess (decoupled from the API — a pipeline crash can't take down
the service) and tails its stdout/stderr into an in-memory ring buffer plus a
log file under ``data/.jobs/<job_id>.log``.

Two job kinds share the same lock:
  - full run:  ``python -m pipeline run [--date D] [--no-audio] [config flags]``
  - re-render: ``python -m pipeline run --only generate --date D --brief-in
                <edited brief> [--config <run config.yaml>] [user overrides]``

Both produce a finished episode (or fail and leave the brief).

Progress + ETA use fixed per-stage estimates (no self-calibration):
  stage 1 (collect)  ~120s
  stage 2 (rank)     ~90s
  stage 3 (generate) ~300s
Stage transitions are detected by scanning the subprocess log for the
``[N/3]`` markers the orchestrator prints.
"""
from __future__ import annotations

import datetime
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
import uuid

LOG_DIR = pathlib.Path("data/.jobs")
LOG_TAIL_LINES = 200

# Default per-stage duration estimates (seconds), 1-indexed. Jobs get
# config-scaled estimates via estimate_stages() at submit time (based on
# source count and window size); these static values are a sane fallback.
STAGE_ESTIMATES = [0.0, 120.0, 90.0, 300.0]
STAGE_LABELS = ["", "collecting", "ranking", "generating"]
STAGE_COUNT = 3
_TOTAL_ESTIMATE = sum(STAGE_ESTIMATES)


def estimate_stages(kind: str, *, num_sources: int,
                    window_days: int = 0) -> list[float]:
    """Per-stage duration estimates scaled by this episode's config.

    ``num_sources`` is the derived source count (length/depth via
    ``RunConfig.budget()``); ``window_days`` is the collect/rank span. Both
    drive the two dominant costs: window-sized collect/rank and source-count
    of LLM/TTS parts. Coefficients are calibrated against short/medium
    benchmark runs (short+1d ≈180s, short+3d ≈253s, medium+1d ≈298s) — they
    are estimates for the ETA, not exact timings.

    A ``generate`` re-render only runs stage [3/3], so collect/rank are 0.
    """
    collect = 30.0 + 10.0 * window_days
    rank = 20.0 + 5.0 * window_days + 2.0 * num_sources
    generate = 40.0 + 26.0 * num_sources
    if kind == "generate":
        return [0.0, 0.0, 0.0, generate]
    return [0.0, collect, rank, generate]

# Matches the orchestrator's `[1/3] collecting...` style markers.
_STAGE_MARKER = re.compile(r"^\[(\d)/\d\]")


class Job:
    def __init__(self, kind: str, date: str | None, cmd: list[str],
                 env: dict | None = None,
                 stage_estimates: list[float] | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind            # "full" | "generate"
        self.date = date
        self.cmd = cmd
        self.env = env              # extra env vars for the subprocess
        self.stage_estimates = stage_estimates or list(STAGE_ESTIMATES)
        self.status = "queued"      # queued | running | done | failed
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.log_lines: list[str] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._start_mono: float = time.monotonic()
        self._end_mono: float | None = None
        # monotonic time each stage marker first appeared: {1: t, 2: t, 3: t}
        self._stage_times: dict[int, float] = {}

    def _progress(self) -> dict:
        """Derive progress from log markers + fixed stage estimates.

        Returns ``{stage_index, stage_count, stage, elapsed_sec, eta_sec,
        fraction}`` where ``fraction`` is in [0, 1] (capped at 0.99 while
        running so the bar never reads 100% before the job is done).
        """
        end = self._end_mono if self._end_mono is not None else time.monotonic()
        elapsed = max(0.0, end - self._start_mono)
        done = self.status in ("done", "failed")
        est = self.stage_estimates
        total_estimate = sum(est)

        if not self._stage_times:
            # No stage marker yet: subprocess is starting up / pre-collect.
            if done:
                return {"stage_index": STAGE_COUNT, "stage_count": STAGE_COUNT,
                        "stage": STAGE_LABELS[STAGE_COUNT] if not self.log_lines
                                else "finished",
                        "elapsed_sec": elapsed, "eta_sec": 0.0, "fraction": 1.0}
            eta = total_estimate
            return {"stage_index": 0, "stage_count": STAGE_COUNT,
                    "stage": "starting", "elapsed_sec": elapsed,
                    "eta_sec": eta, "fraction": min(0.99, elapsed / (elapsed + eta))}

        current = max(self._stage_times)
        completed_duration = sum(est[1:current])
        stage_elapsed = elapsed - completed_duration
        stage_remaining = max(0.0, est[current] - stage_elapsed)
        future_stages = sum(est[current + 1:STAGE_COUNT + 1])
        eta = stage_remaining + future_stages
        fraction = (completed_duration + min(stage_elapsed, est[current])) / total_estimate

        if done:
            return {"stage_index": STAGE_COUNT, "stage_count": STAGE_COUNT,
                    "stage": "finished", "elapsed_sec": elapsed,
                    "eta_sec": 0.0, "fraction": 1.0}
        return {"stage_index": current, "stage_count": STAGE_COUNT,
                "stage": STAGE_LABELS[current],
                "elapsed_sec": elapsed, "eta_sec": eta,
                "fraction": min(0.99, fraction)}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "date": self.date,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self._progress(),
            "log_tail": "\n".join(self.log_lines[-LOG_TAIL_LINES:]),
        }


_lock = threading.Lock()
_active: Job | None = None
_recent: list[Job] = []
_recent_lock = threading.Lock()


def active_job() -> Job | None:
    return _active


def list_jobs(limit: int = 20) -> list[dict]:
    with _recent_lock:
        return [j.to_dict() for j in _recent[:limit]]


def submit(kind: str, cmd: list[str], date: str | None = None,
           env: dict | None = None,
           stage_estimates: list[float] | None = None
           ) -> tuple[Job | None, str]:
    """Try to start a job. Returns (job, error).

    A second submit while a run is active returns (None, "busy"). The caller
    maps that to HTTP 409. ``env`` is merged into the subprocess environment.
    ``stage_estimates`` scales the progress ETA to this job's config.
    """
    global _active
    if not _lock.acquire(blocking=False):
        return None, "busy"
    try:
        if _active is not None and _active.status in ("queued", "running"):
            return None, "busy"
        job = Job(kind, date, cmd, env=env, stage_estimates=stage_estimates)
        _active = job
        job.status = "running"
        job.started_at = _now()
        job._thread = threading.Thread(target=_run, args=(job,), daemon=True)
        job._thread.start()
        return job, ""
    finally:
        _lock.release()


def _run(job: Job) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job.id}.log"
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"=== job {job.id} {job.kind} date={job.date} ===\n")
            logf.write("cmd: " + " ".join(job.cmd) + "\n")
            if job.env:
                logf.write("env: " + ", ".join(f"{k}={v}" for k, v in job.env.items()) + "\n")
            logf.flush()
            run_env = None
            if job.env:
                run_env = dict(os.environ)
                run_env.update(job.env)
            job._proc = subprocess.Popen(
                job.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=run_env,
            )
            assert job._proc.stdout is not None
            for line in job._proc.stdout:
                line = line.rstrip("\n")
                job.log_lines.append(line)
                logf.write(line + "\n")
                logf.flush()
                m = _STAGE_MARKER.match(line)
                if m is not None:
                    idx = int(m.group(1))
                    if 1 <= idx <= STAGE_COUNT and idx not in job._stage_times:
                        job._stage_times[idx] = time.monotonic()
            rc = job._proc.wait()
            if rc == 0:
                job.status = "done"
            else:
                job.status = "failed"
                job.log_lines.append(f"[exit code {rc}]")
    except Exception as e:
        job.status = "failed"
        job.log_lines.append(f"[runner error: {e}]")
    finally:
        job._end_mono = time.monotonic()
        job.finished_at = _now()
        with _recent_lock:
            _recent.insert(0, job)
            del _recent[50:]


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def full_run_cmd(date: str | None = None, no_audio: bool = False,
                 config: dict | None = None,
                 config_path: str | None = None) -> list[str]:
    """Build the CLI argv for a full run.

    ``config`` is a dict of user-facing knobs (keys: window_start, window_end,
    audience_level, familiar_topics, length, depth) turned into CLI flags; an
    explicit ``config_path`` (a resolved config.yaml) is appended as --config.
    """
    cmd = [sys.executable, "-m", "pipeline", "run"]
    if date:
        cmd += ["--date", date]
    if no_audio:
        cmd += ["--no-audio"]
    if config_path:
        cmd += ["--config", str(config_path)]
    for key, flag in (("window_start", "--window-start"),
                      ("window_end", "--window-end"),
                      ("audience_level", "--audience"),
                      ("length", "--length"),
                      ("depth", "--depth")):
        value = (config or {}).get(key)
        if value:
            cmd += [flag, str(value)]
    familiar = (config or {}).get("familiar_topics")
    if familiar:
        cmd += ["--familiar", ",".join(str(t) for t in familiar)]
    return cmd


def generate_cmd(date: str, brief_in: pathlib.Path,
                 config: dict | None = None,
                 config_path: str | None = None) -> list[str]:
    """Build the CLI argv for a generate-only re-render.

    ``config_path`` points at the target run's stored config.yaml, restoring
    that episode's original podcast-wise knobs (length/depth/window) for the
    LLM/TTS; ``config`` carries user-wise overrides (audience_level,
    familiar_topics from the persistent config) applied on top as CLI flags.
    Pipeline resolution order is code defaults < config file < CLI flags, so
    the user's current knowledge level always wins.
    """
    cmd = [sys.executable, "-m", "pipeline", "run",
           "--only", "generate", "--date", date, "--brief-in", str(brief_in)]
    if config_path:
        cmd += ["--config", str(config_path)]
    value = (config or {}).get("audience_level")
    if value:
        cmd += ["--audience", str(value)]
    familiar = (config or {}).get("familiar_topics")
    if familiar:
        cmd += ["--familiar", ",".join(str(t) for t in familiar)]
    return cmd
