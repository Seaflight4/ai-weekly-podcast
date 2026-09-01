"""FastAPI service over the pipeline.

Serves existing episodes (read from data/), lets a user trigger a full run
or a personalized re-render, and exposes the scheduler config. The static
frontend is served at /.
"""
from __future__ import annotations

import collections
import pathlib
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import episodes, jobs, personalized, scheduler

STATIC_DIR = pathlib.Path(__file__).resolve().parent.parent / "static"

# Recent broadcast events (scheduler skips, etc.). Bounded deque.
_events: collections.deque = collections.deque(maxlen=100)


def broadcast(event: dict) -> None:
    _events.append(event)


app = FastAPI(title="AI Weekly Podcast")


@app.on_event("startup")
def _startup():
    scheduler.start()


@app.on_event("shutdown")
def _shutdown():
    scheduler.shutdown()


# --- episodes ---------------------------------------------------------------

@app.get("/api/episodes")
def list_episodes():
    return episodes.list_runs()


@app.get("/api/episodes/{date}")
def get_episode(date: str):
    run = episodes.get_run(date)
    if run is None:
        raise HTTPException(404, f"no run for {date}")
    return run


@app.get("/api/episodes/{date}/audio")
def get_audio(date: str):
    p = episodes.audio_path(date)
    if p is None:
        raise HTTPException(404, "no audio for this run")
    return FileResponse(p, media_type="audio/mpeg", filename="episode.mp3")


@app.get("/api/episodes/{date}/brief")
def get_brief(date: str):
    p = episodes.brief_path(date)
    if p is None:
        raise HTTPException(404, "no brief for this run")
    return PlainTextResponse(p.read_text(encoding="utf-8"), media_type="text/markdown")


@app.get("/api/episodes/{date}/transcript")
def get_transcript(date: str):
    p = episodes.transcript_path(date)
    if p is None:
        raise HTTPException(404, "no transcript for this run")
    return PlainTextResponse(p.read_text(encoding="utf-8"), media_type="text/markdown")


# --- runs -------------------------------------------------------------------

@app.get("/api/runs")
def list_runs():
    active = jobs.active_job()
    out = jobs.list_jobs()
    if active is not None:
        out = [active.to_dict()] + [j for j in out if j["id"] != active.id]
    return out


@app.post("/api/runs")
def submit_run(payload: dict = Body(default={})):
    date = payload.get("date")
    no_audio = bool(payload.get("no_audio", False))
    config = payload.get("config") or {}
    cmd = jobs.full_run_cmd(date=date, no_audio=no_audio, config=config)
    env = {"PIPELINE_DATA_ROOT": str(episodes.DATA_ROOT)}
    job, err = jobs.submit("full", cmd, date=date, env=env)
    if err == "busy":
        active = jobs.active_job()
        return JSONResponse(
            {"detail": "a run is already active",
             "active_job_id": active.id if active else None},
            status_code=409,
        )
    return job.to_dict()


@app.post("/api/runs/{date}/generate")
def submit_generate(date: str, payload: dict = Body(default={})):
    """Re-render audio from an edited brief into the shared personalized
    library.

    Body: {"brief_markdown": "..."} — the user's edited brief. We write it
    + a copy of rank.json into ``data/personalized/<date>/``, then run
    ``pipeline run --only generate --brief-in ...`` with
    ``PIPELINE_DATA_ROOT=data/personalized`` so the subprocess writes
    episode.mp3 + episode.json into the personalized library. The default
    ``data/default/<date>/`` is never touched.
    """
    brief_md = payload.get("brief_markdown")
    if not brief_md:
        raise HTTPException(400, "body must include 'brief_markdown'")
    try:
        run_dir = personalized.prepare_render(date, brief_md)
    except ValueError as e:
        raise HTTPException(400, str(e))
    brief_path = run_dir / "podcast_brief.md"
    cmd = jobs.generate_cmd(date, brief_path)
    env = {"PIPELINE_DATA_ROOT": str(personalized.PERSONALIZED_ROOT)}
    job, err = jobs.submit("generate", cmd, date=date, env=env)
    if err == "busy":
        active = jobs.active_job()
        return JSONResponse(
            {"detail": "a run is already active",
             "active_job_id": active.id if active else None},
            status_code=409,
        )
    return job.to_dict()


@app.get("/api/runs/{job_id}")
def get_run(job_id: str):
    active = jobs.active_job()
    if active is not None and active.id == job_id:
        return active.to_dict()
    for j in jobs.list_jobs():
        if j["id"] == job_id:
            return j
    raise HTTPException(404, "no such job")


# --- personalized renders (shared library) --------------------------------

@app.get("/api/personalized")
def list_personalized():
    return personalized.list_runs()


@app.get("/api/personalized/{date}")
def get_personalized(date: str):
    run = personalized.get_run(date)
    if run is None:
        raise HTTPException(404, f"no render for {date}")
    return run


@app.get("/api/personalized/{date}/audio")
def get_personalized_audio(date: str):
    p = personalized.audio_path(date)
    if p is None:
        raise HTTPException(404, "no audio for this render")
    return FileResponse(p, media_type="audio/mpeg", filename="episode.mp3")


@app.get("/api/personalized/{date}/brief")
def get_personalized_brief(date: str):
    p = personalized.brief_path(date)
    if p is None:
        raise HTTPException(404, "no brief for this render")
    return PlainTextResponse(p.read_text(encoding="utf-8"), media_type="text/markdown")


@app.get("/api/personalized/{date}/transcript")
def get_personalized_transcript(date: str):
    p = personalized.transcript_path(date)
    if p is None:
        raise HTTPException(404, "no transcript for this render")
    return PlainTextResponse(p.read_text(encoding="utf-8"), media_type="text/markdown")


@app.delete("/api/personalized/{date}")
def delete_personalized(date: str):
    """Delete a personalized render. The default episode for the same date
    is never affected."""
    try:
        removed = personalized.delete_run(date)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not removed:
        raise HTTPException(404, f"no render for {date}")
    return {"deleted": date}


# --- schedule --------------------------------------------------------------

@app.get("/api/schedule")
def get_schedule():
    return scheduler.state()


@app.post("/api/schedule")
def set_schedule(payload: dict = Body(default={})):
    enabled = payload.get("enabled")
    cron = payload.get("cron")
    try:
        return scheduler.configure(enabled=enabled, cron=cron)
    except ValueError as e:
        raise HTTPException(400, str(e))


# --- events (scheduler broadcasts) -----------------------------------------

@app.get("/api/events")
def get_events():
    return list(_events)


# --- static frontend --------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
