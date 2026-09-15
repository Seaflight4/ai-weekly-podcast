"""Background job runner for pipeline runs.

Serialized FIFO queue, process-wide: at most one pipeline runs at a time, but
additional submissions are queued instead of rejected. A run spawns
``python -m pipeline run ...`` as a subprocess in its own process group
(decoupled from the API — a pipeline crash can't take down the service) and
tails its stdout/stderr into an in-memory ring buffer plus a log file under
``data/.jobs/<job_id>.log``.

Two job kinds share the one queue:
  - full run:  ``python -m pipeline run [--date D] [--no-audio] [config flags]``
  - re-render: ``python -m pipeline run --only generate --date D --brief-in
                <edited brief> [--config <run config.yaml>] [user overrides]``

Both produce a finished episode (or fail and leave the brief).

Cancellation: a queued job is dequeued and dropped; a running job gets its
process group SIGTERM'd (SIGKILL after a grace) and its intermediate results
are cleaned up — a ``full`` run's folder is deleted, a ``generate`` re-render
is rolled back to a pre-run snapshot of the episode files (the user's edited
brief is kept).

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
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

from . import events
from . import episodes as episodes_mod
from . import DATA_ROOT as APP_DATA_ROOT

LOG_DIR = APP_DATA_ROOT / ".jobs"
LOG_TAIL_LINES = 200

# Where pre-render snapshots of an episode's files live until the job settles.
BACKUP_ROOT = APP_DATA_ROOT / ".rerender-backups"
# Files a ``generate`` re-render rewrites; the edited podcast_brief.md is
# intentionally NOT here, so the user's brief edit survives a cancel.
SNAPSHOT_FILES = ("transcript.md", "memory.json", "labels.json",
                  "config.yaml", "episode.json", "episode.mp3")

# How long SIGTERM gets to make the process group exit before SIGKILL.
KILL_GRACE_SECONDS = 5.0

# Cadence of the live progress pushes sent over the SSE stream while a job is
# running (replaces the old 1.5s browser poll).
PROGRESS_INTERVAL = 1.0

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
                 stage_estimates: list[float] | None = None,
                 window_start: str | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind            # "full" | "generate"
        self.date = date
        self.window_start = window_start   # ISO window start, for the name
        self.cmd = cmd
        self.env = env              # extra env vars for the subprocess
        self.stage_estimates = stage_estimates or list(STAGE_ESTIMATES)
        self.status = "queued"      # queued | running | done | failed | cancelled
        self.submitted_at: str = _now()
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.error: str | None = None   # short human reason when status == failed
        self.log_lines: list[str] = []
        self.backup_dir: pathlib.Path | None = None
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._cancel_requested = threading.Event()
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
        done = self.status in ("done", "failed", "cancelled")
        est = self.stage_estimates
        total_estimate = sum(est)

        if self.status == "queued":
            return {"stage_index": 0, "stage_count": STAGE_COUNT,
                    "stage": "queued", "elapsed_sec": 0.0,
                    "eta_sec": total_estimate, "fraction": 0.0}

        if not self._stage_times:
            # No stage marker yet: queued->started / subprocess starting up.
            if done:
                stage = "cancelled" if self.status == "cancelled" else (
                    STAGE_LABELS[STAGE_COUNT] if not self.log_lines else "finished")
                return {"stage_index": STAGE_COUNT, "stage_count": STAGE_COUNT,
                        "stage": stage, "elapsed_sec": elapsed,
                        "eta_sec": 0.0, "fraction": 1.0}
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
            stage = "cancelled" if self.status == "cancelled" else "finished"
            return {"stage_index": STAGE_COUNT, "stage_count": STAGE_COUNT,
                    "stage": stage, "elapsed_sec": elapsed,
                    "eta_sec": 0.0, "fraction": 1.0}
        return {"stage_index": current, "stage_count": STAGE_COUNT,
                "stage": STAGE_LABELS[current],
                "elapsed_sec": elapsed, "eta_sec": eta,
                "fraction": min(0.99, fraction)}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": _job_name(self),
            "date": self.date,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "progress": self._progress(),
            "log_tail": "\n".join(self.log_lines[-LOG_TAIL_LINES:]),
        }


def _job_name(job: Job) -> str:
    """A human-readable label for the jobs panel (what the episode is about),
    not the raw job id. Uses the coverage window when known — e.g.
    'New episode — 02–05 Sep 2026' — falling back to the run-date label."""
    label = None
    if job.date:
        label = (episodes_mod.window_label(job.window_start, job.date)
                 or episodes_mod._date_label(job.date))
    if job.kind == "generate":
        return f"Re-render — {label}" if label else "Re-render"
    return f"New episode — {label}" if label else "New episode"


_lock = threading.Lock()
_cond = threading.Condition(_lock)
_queue: list[Job] = []
_active: Job | None = None
_recent: list[Job] = []
_recent_lock = threading.Lock()

# daemon housekeeping, started lazily on first submit
_house_lock = threading.Lock()
_dispatcher: threading.Thread | None = None
_ticker: threading.Thread | None = None


def active_job() -> Job | None:
    with _lock:
        return _active


def list_jobs(limit: int = 20) -> list[dict]:
    """Active job, then queued (FIFO), then finished (newest first)."""
    with _lock:
        out: list[Job] = []
        if _active is not None:
            out.append(_active)
        out.extend(j for j in _queue if j is not _active)
        seen = {id(j) for j in out}
    with _recent_lock:
        out.extend(j for j in _recent if id(j) not in seen)
    return [j.to_dict() for j in out[:limit]]


def find_job(job_id: str) -> Job | None:
    with _lock:
        for j in ([_active] if _active is not None else []) + list(_queue):
            if j.id == job_id:
                return j
    with _recent_lock:
        for j in _recent:
            if j.id == job_id:
                return j
    return None


# --- housekeeping threads ----------------------------------------------------

def _ensure_housekeeping() -> None:
    global _dispatcher, _ticker
    with _house_lock:
        if _dispatcher is None or not _dispatcher.is_alive():
            _dispatcher = threading.Thread(target=_dispatcher_loop, daemon=True)
            _dispatcher.start()
        if _ticker is None or not _ticker.is_alive():
            _ticker = threading.Thread(target=_ticker_loop, daemon=True)
            _ticker.start()


def _dispatcher_loop() -> None:
    """Pop queued jobs one at a time; a finished job frees the slot for the
    next. Runs the tail-thread per job so callers/tests can join it."""
    global _active
    while True:
        with _cond:
            while not _queue:
                _cond.wait()
            job = _queue.pop(0)
            _active = job
            job.started_at = _now()
            job._thread = threading.Thread(target=_run, args=(job,), daemon=True)
            job._thread.start()
            # "running" only once the tail thread is actually started, so
            # submit()'s wait (below) is a reliable "thread exists + started".
            job.status = "running"
        job._thread.join()
        with _lock:
            _active = None


def _ticker_loop() -> None:
    """Persistent live-progress pusher: broadcasts ``run_progress`` for the
    current active job at ``PROGRESS_INTERVAL``; idles when nothing runs."""
    while True:
        time.sleep(PROGRESS_INTERVAL)
        job = active_job()
        if job is None or job.status not in ("queued", "running"):
            continue
        try:
            events.broadcast({"type": "run_progress", "job": job.to_dict()})
        except Exception:
            pass


# --- submit / run / cancel ---------------------------------------------------

def submit(kind: str, cmd: list[str], date: str | None = None,
           env: dict | None = None,
           stage_estimates: list[float] | None = None,
           window_start: str | None = None) -> Job:
    """Enqueue a new job and return it (status ``queued``).

    A ``generate`` re-render snapshots the existing episode's files first so a
    later cancel can roll it back. No more ``"busy"`` — submissions queue.
    ``env`` is merged into the subprocess environment. ``stage_estimates``
    scales the progress ETA to this job's config. ``window_start`` (ISO) lets
    the job's name show the coverage window, not just the run date.
    """
    job = Job(kind, date, cmd, env=env, stage_estimates=stage_estimates,
              window_start=window_start)
    _ensure_housekeeping()
    with _lock:
        # If a job is already active/queued this one is queued behind it; the
        # caller sees "queued" right away. Otherwise the dispatcher starts it
        # immediately — wait for that so the caller can join its thread.
        occupied = _active is not None or bool(_queue)
        _queue.append(job)
        _cond.notify()
    try:
        # Every client learns about the new job at once (not just the
        # submitter), so the multi-job panel stays in sync across tabs.
        events.broadcast({"type": "run_queued", "job": job.to_dict()})
    except Exception:
        pass
    if not occupied:
        # Nothing else runs: the dispatcher should pick this up immediately.
        # Wait until the tail thread is started (status leaves "queued" only
        # after start) so the caller can join it right after submit.
        deadline = time.monotonic() + 2.0
        while job.status == "queued" and time.monotonic() < deadline:
            time.sleep(0.005)
    return job


def cancel(job_id: str) -> tuple[Job | None, str]:
    """Cancel a queued or running job. Returns (job, error) with error "" on
    success, "not_found" for an unknown id, "not_running" when the job is
    already done/failed/cancelled."""
    with _lock:
        queued = None
        for j in list(_queue):
            if j.id == job_id:
                _queue.remove(j)
                queued = j
                break
        if queued is not None:
            if queued.status != "queued":
                return queued, "not_running"
            queued._cancel_requested.set()
            queued.status = "cancelled"
            queued._end_mono = time.monotonic()
            queued.finished_at = _now()
            _recent_insert(queued)
            if queued.backup_dir is not None:
                shutil.rmtree(queued.backup_dir, ignore_errors=True)
            to_broadcast = queued
        else:
            active = _active
            if active is not None and active.id == job_id:
                if active.status not in ("queued", "running"):
                    return active, "not_running"
                active._cancel_requested.set()
                active.status = "cancelled"
                if active._proc is not None and active._proc.poll() is None:
                    _terminate_async(active)
                to_broadcast = None   # ``_run`` broadcasts once the proc is dead
            else:
                with _recent_lock:
                    for j in _recent:
                        if j.id == job_id:
                            return j, "not_running"
                return None, "not_found"
    if to_broadcast is not None:
        try:
            events.broadcast({"type": "run_finished", "job": to_broadcast.to_dict()})
        except Exception:
            pass
    return (to_broadcast if to_broadcast is not None else active), ""


def _terminate_proc(job: Job) -> None:
    """SIGTERM the whole process group and wait; escalate to SIGKILL after a
    grace if the group ignores the first signal."""
    if job._proc is None or job._proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(job._proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, AttributeError):
        return
    try:
        job._proc.wait(timeout=KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(job._proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, AttributeError):
            pass


def _terminate_async(job: Job) -> None:
    """Terminate the process group without blocking the caller."""
    threading.Thread(target=_terminate_proc, args=(job,), daemon=True).start()


def _run(job: Job) -> None:
    # Snapshot the episode's files at START (not submit): a re-render that was
    # queued behind another re-render of the same episode must restore to the
    # state when it begins, not to an older submit-time state. The snapshot
    # happens before any Popen so a cancel that raced startup still restores.
    if job.kind == "generate" and job.date and job.backup_dir is None:
        job.backup_dir = _snapshot_rerender(job.date, job.id)
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
                start_new_session=True,   # own process group -> cancel kills the tree
            )
            assert job._proc.stdout is not None
            # A cancel may have raced subprocess startup: the flag is set but
            # there was no _proc to signal yet — terminate it as soon as it
            # exists so a cancelled job never runs to completion.
            if job._cancel_requested.is_set():
                _terminate_proc(job)
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
            if job._cancel_requested.is_set():
                job.status = "cancelled"
            elif rc == 0:
                job.status = "done"
            else:
                job.status = "failed"
                job.error = _failure_reason(job.log_lines, rc)
                job.log_lines.append(f"[exit code {rc}]")
                logf.write(f"[exit code {rc}]\n")
                logf.flush()
    except Exception as e:
        job.status = "failed"
        job.error = f"runner error: {e}"
        job.log_lines.append(f"[runner error: {e}]")
        try:
            with log_path.open("a", encoding="utf-8") as logf:
                logf.write(f"[runner error: {e}]\n")
        except OSError:
            pass
    finally:
        job._end_mono = time.monotonic()
        job.finished_at = _now()
        _recent_insert(job)
        try:
            # Completion edge: browsers refresh the episode list / detail from
            # this single push instead of polling.
            events.broadcast({"type": "run_finished", "job": job.to_dict()})
        except Exception:
            pass
        _cleanup(job)


def _failure_reason(log_lines: list[str], rc: int) -> str:
    """Short, human reason for a failed run, for the jobs panel.

    Prefers the loud error lines the pipeline now prints (ERROR:/traceback /
    runner error); falls back to the bare exit code.
    """
    for line in reversed(log_lines):
        s = line.strip()
        if (s.startswith("ERROR:") or s.startswith("[runner error")
                or s.startswith("Traceback")):
            return s
    return f"pipeline exited with code {rc}"


def _recent_insert(job: Job) -> None:
    with _recent_lock:
        _recent.insert(0, job)
        del _recent[50:]


# --- cancellation cleanup ----------------------------------------------------

def _cleanup(job: Job) -> None:
    """Intermediate-result cleanup once the process is (guaranteed) dead."""
    if job.status == "cancelled":
        if job.kind == "full" and job.date:
            _delete_run_folder_retry(job.date)
        elif job.kind == "generate":
            _restore_rerender(job)
    elif job.backup_dir is not None:
        # done/failed with a re-render snapshot: the fresh episode won, drop it.
        shutil.rmtree(job.backup_dir, ignore_errors=True)


def _delete_run_folder_retry(date: str) -> None:
    """Delete a full run's folder, retrying briefly (the process may just have
    exited and released its file handles)."""
    for attempt in range(4):
        try:
            episodes_mod.delete_run(date)
            return
        except OSError:
            time.sleep(0.2 if attempt < 3 else 0)


# --- re-render snapshot / restore --------------------------------------------

def _snapshot_rerender(date: str, job_id: str) -> pathlib.Path | None:
    """Back up the files a re-render rewrites so a cancel can roll them back.

    ``podcast_brief.md`` is intentionally excluded — the user's edited brief
    is the reason for the re-render and must survive a cancel. Returns None
    (and removes any partial backup) when the folder has nothing to back up.
    """
    try:
        folder = episodes_mod.DATA_ROOT / episodes_mod._normalize_date(date)
    except ValueError:
        return None
    if not folder.is_dir():
        return None
    backup = BACKUP_ROOT / job_id
    try:
        backup.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    copied = False
    for name in SNAPSHOT_FILES:
        src = folder / name
        if src.exists():
            try:
                shutil.copy2(src, backup / name)
                copied = True
            except OSError:
                continue
    if not copied:
        shutil.rmtree(backup, ignore_errors=True)
        return None
    return backup


def _restore_rerender(job: Job) -> None:
    """Roll a cancelled re-render back to its pre-run snapshot, then drop the
    backup and any partial cache the aborted run left behind."""
    if not job.backup_dir or not job.backup_dir.exists():
        shutil.rmtree(job.backup_dir, ignore_errors=True)
        return
    try:
        folder = episodes_mod.DATA_ROOT / episodes_mod._normalize_date(job.date)
        if folder.is_dir():
            for name in SNAPSHOT_FILES:
                src = job.backup_dir / name
                if src.exists():
                    try:
                        shutil.copy2(src, folder / name)
                    except OSError:
                        pass
            cache = folder / ".podcastfy-cache"
            if cache.exists():
                shutil.rmtree(cache, ignore_errors=True)
    finally:
        shutil.rmtree(job.backup_dir, ignore_errors=True)


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
                      ("depth", "--depth"),
                      ("steering_alpha", "--steering-alpha"),
                      ("mem_windows", "--mem-windows")):
        value = (config or {}).get(key)
        if value is not None and value != "":
            cmd += [flag, str(value)]
    familiar = (config or {}).get("familiar_topics")
    if familiar:
        cmd += ["--familiar", ",".join(str(t) for t in familiar)]
    topic_prefs = (config or {}).get("topic_prefs")
    if topic_prefs:
        cmd += ["--topics", ",".join(str(t) for t in topic_prefs)]
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
    mem_windows = (config or {}).get("mem_windows")
    if mem_windows is not None:
        cmd += ["--mem-windows", str(mem_windows)]
    return cmd
