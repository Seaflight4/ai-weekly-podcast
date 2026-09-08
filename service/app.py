"""FastAPI service over the pipeline.

Serves the episode history (read from data/history/), lets a user trigger a
full run or a brief-edit re-render (which replaces a date's episode in
place), and manages the persistent podcast config (data/podcast_config.yaml).
The static frontend is served at /.
"""
from __future__ import annotations

import pathlib

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline import config as config_mod

from . import episodes, jobs, podcast_config

STATIC_DIR = pathlib.Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="AI Weekly Podcast")


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


@app.delete("/api/episodes/{date}")
def delete_episode(date: str):
    """Delete a history entry — folder and all artifacts. Cannot be undone."""
    try:
        removed = episodes.delete_run(date)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not removed:
        raise HTTPException(404, f"no episode for {date}")
    return {"deleted": date}


# --- podcast config ---------------------------------------------------------

@app.get("/api/config")
def get_config():
    """The persistent podcast config. ``first_run`` is true until the user
    saves it once (the UI shows a setup dialog pre-filled with defaults)."""
    return {"first_run": not podcast_config.exists(),
            "config": podcast_config.load()}


@app.put("/api/config")
def put_config(payload: dict = Body(default={})):
    try:
        stored = podcast_config.save(payload)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"first_run": False, "config": stored}


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
    """Start a full collect → rank → generate run.

    Body: {"date": ..., "no_audio": ..., "podcast": {"window_start",
    "window_end", "length", "depth"}} — the podcast section is the user's
    confirmed dialog values. User-wise knobs (audience, familiar topics)
    come from the persistent config, never from the request.
    """
    date = payload.get("date")
    no_audio = bool(payload.get("no_audio", False))
    try:
        config = podcast_config.resolved_run_config(payload.get("podcast"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    # The run anchors on the resolved window end. Unless the body pins an
    # explicit folder date, build a unique time-stamped id (DD-MM-YYYY-HHMMSS)
    # from the window end so a second episode generated the same day gets its
    # own folder instead of overwriting the first. The job carries that id (the
    # UI auto-selects it on completion); the pipeline still gets --window-end
    # for the actual collection window.
    date = date or episodes.make_run_id(config["window_end"])
    cmd = jobs.full_run_cmd(date=date, no_audio=no_audio, config=config)
    env = {"PIPELINE_DATA_ROOT": str(episodes.DATA_ROOT)}
    stage_estimates = jobs.estimate_stages(
        "full",
        num_sources=int(config["num_sources"]),
        window_days=int(config["window_days"]),
    )
    job, err = jobs.submit("full", cmd, date=date, env=env,
                           stage_estimates=stage_estimates)
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
    """Re-render a history episode from an edited brief, replacing it in
    place.

    Body: {"brief_markdown": "..."} — the user's edited brief. We write it
    over ``data/history/<date>/podcast_brief.md`` (rank.json already lives
    in that folder), then run ``pipeline run --only generate --brief-in ...``
    with the run's stored config.yaml for podcast-wise knobs and CLI
    overrides for the persistent config's user-wise knobs. The date's
    episode.mp3 + episode.json are regenerated; ``selection_source`` flips
    to "customized".
    """
    brief_md = payload.get("brief_markdown")
    if not brief_md:
        raise HTTPException(400, "body must include 'brief_markdown'")
    folder = episodes.DATA_ROOT / episodes._normalize_date(date)
    if not (folder / "rank.json").exists():
        raise HTTPException(404, f"no ranked pool for {date} — cannot re-render")
    brief_path = folder / "podcast_brief.md"
    brief_path.write_text(brief_md, encoding="utf-8")
    user = podcast_config.load()["user"]
    run_cfg = folder / "config.yaml"
    cmd = jobs.generate_cmd(
        date, brief_path,
        config={"audience_level": user["audience"],
                "familiar_topics": user["familiar_topics"]},
        config_path=run_cfg if run_cfg.exists() else None,
    )
    # Scale the ETA to this episode's generate-only work: restore its
    # length/depth from the stored run config (fall back to the persistent
    # config). collect/rank don't run, so window_days is 0.
    est_cfg = None
    if run_cfg.exists():
        try:
            est_cfg = config_mod.resolve(config_path=str(run_cfg))
        except (ValueError, FileNotFoundError):
            est_cfg = None
    if est_cfg is None:
        pc = podcast_config.load()["podcast"]
        est_cfg = config_mod.RunConfig(length=pc["length"], depth=pc["depth"])
    stage_estimates = jobs.estimate_stages(
        "generate", num_sources=est_cfg.num_sources(), window_days=0)
    env = {"PIPELINE_DATA_ROOT": str(episodes.DATA_ROOT)}
    job, err = jobs.submit("generate", cmd, date=date, env=env,
                           stage_estimates=stage_estimates)
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


# --- static frontend --------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
