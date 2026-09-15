"""HTTP service over the pipeline.

Serves existing episodes, lets a user trigger a full run, and exposes a
personalization path (edit the brief, re-generate audio). An APScheduler
background job auto-runs a fresh episode on a configurable cron (default:
every Monday 09:00).

Runtime state lives under this app's own ``data/`` folder (anchored to the
app directory, not the process CWD), so it works whether the service runs
from cmd or in Docker (where ``/app/data`` is the bind mount).
"""
from __future__ import annotations

import pathlib

APP_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_ROOT = APP_ROOT / "data"
