"""Tests for the HTTP service layer.

No network, no LLM, no real subprocess. The pipeline subprocess is either
mocked (jobs) or replaced with a trivial command that exits immediately.
"""
import json, pathlib, sys, time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from service import episodes, jobs, podcast_config, scheduler


# --- episodes scan + status badges -----------------------------------------

def test_episodes_list_status_badges(tmp_path, monkeypatch):
    monkeypatch.setattr(episodes, "DATA_ROOT", tmp_path)
    # ready: episode.json + episode.mp3
    r = tmp_path / "31-08-2026"; r.mkdir()
    (r / "episode.json").write_text(json.dumps({
        "manifest": [{"url": "u"}], "created_at": "2026-08-31T09:00:00Z",
        "selection_source": "personalized"}))
    (r / "episode.mp3").write_bytes(b"")
    # draft: episode.json but no mp3
    d = tmp_path / "24-08-2026"; d.mkdir()
    (d / "episode.json").write_text(json.dumps({"manifest": []}))
    # empty: no episode.json
    e = tmp_path / "17-08-2026"; e.mkdir()
    # non-date folder ignored
    (tmp_path / ".jobs").mkdir()

    runs = episodes.list_runs()
    assert [x["date"] for x in runs] == ["31-08-2026", "24-08-2026", "17-08-2026"]
    by_date = {x["date"]: x for x in runs}
    assert by_date["31-08-2026"]["status"] == "ready"
    assert by_date["31-08-2026"]["selection_source"] == "personalized"
    assert by_date["24-08-2026"]["status"] == "draft"
    assert by_date["17-08-2026"]["status"] == "empty"


def test_episodes_get_run_loads_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(episodes, "DATA_ROOT", tmp_path)
    r = tmp_path / "31-08-2026"; r.mkdir()
    (r / "episode.json").write_text(json.dumps({"manifest": [{"url": "u"}]}))
    (r / "podcast_brief.md").write_text("# brief")
    (r / "transcript.md").write_text("hello")
    (r / "rank.json").write_text(json.dumps([{"url": "u", "score": 0.9}]))
    run = episodes.get_run("2026-08-31")  # ISO accepted
    assert run is not None
    assert run["brief"] == "# brief"
    assert run["transcript"] == "hello"
    assert run["rank"][0]["score"] == 0.9


def test_episodes_get_run_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(episodes, "DATA_ROOT", tmp_path)
    assert episodes.get_run("2026-01-01") is None


# --- jobs: single-run lock -------------------------------------------------

@pytest.fixture
def reset_jobs():
    """Reset jobs module-global state between tests."""
    jobs._active = None
    jobs._recent.clear()
    yield
    # wait for any lingering thread to finish
    a = jobs._active
    if a is not None and a._thread is not None and a._thread.is_alive():
        if a._proc is not None and a._proc.poll() is None:
            a._proc.terminate()
            a._proc.wait(timeout=5)
        a._thread.join(timeout=5)
    jobs._active = None
    jobs._recent.clear()


def test_jobs_second_submit_while_active_returns_busy(reset_jobs):
    # A sleeping command keeps the job active long enough to attempt a second.
    cmd = [sys.executable, "-c", "import time; time.sleep(2)"]
    job, err = jobs.submit("full", cmd, date=None)
    assert err == "" and job is not None and job.status == "running"
    job2, err2 = jobs.submit("full", cmd, date=None)
    assert err2 == "busy" and job2 is None
    # cleanup: let it finish
    job._thread.join(timeout=10)
    assert job.status in ("done", "failed")


def test_jobs_command_builder():
    cmd = jobs.full_run_cmd(date="2026-08-31", no_audio=True)
    assert "--date" in cmd and "2026-08-31" in cmd and "--no-audio" in cmd
    cmd = jobs.generate_cmd("31-08-2026", pathlib.Path("data/31-08-2026/podcast_brief.md"))
    assert "--only" in cmd and cmd[cmd.index("--only") + 1] == "generate"
    assert "--brief-in" in cmd
    # config_path (run's stored config.yaml) + user-wise overrides thread
    # through as pipeline CLI flags.
    cmd = jobs.generate_cmd(
        "31-08-2026", pathlib.Path("data/31-08-2026/podcast_brief.md"),
        config={"audience_level": "beginner", "familiar_topics": ["RLHF", "MoE"]},
        config_path=pathlib.Path("data/31-08-2026/config.yaml"))
    assert "--config" in cmd and "data/31-08-2026/config.yaml" in cmd
    assert "--audience" in cmd and cmd[cmd.index("--audience") + 1] == "beginner"
    assert "--familiar" in cmd and cmd[cmd.index("--familiar") + 1] == "RLHF,MoE"
    # full_run_cmd flattens the resolved knob dict into flags.
    cmd = jobs.full_run_cmd(config={
        "window_start": "2026-08-24", "window_end": "2026-08-31",
        "audience_level": "intermediate", "familiar_topics": [],
        "length": "long", "depth": "brief"})
    assert cmd[cmd.index("--window-start") + 1] == "2026-08-24"
    assert cmd[cmd.index("--audience") + 1] == "intermediate"
    assert cmd[cmd.index("--length") + 1] == "long"
    assert cmd[cmd.index("--depth") + 1] == "brief"


# --- scheduler: cron parse + persist ---------------------------------------

def test_scheduler_parse_cron_valid():
    trig = scheduler._parse_cron("0 9 * * 1")
    assert trig is not None


def test_scheduler_parse_cron_rejects_bad_field_count():
    with pytest.raises(ValueError):
        scheduler._parse_cron("0 9 *")
    with pytest.raises(ValueError):
        scheduler._parse_cron("not-a-cron at all")


def test_scheduler_persist_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "CONFIG_PATH", tmp_path / "schedule.json")
    scheduler._state.update({"enabled": True, "cron": "30 7 * * 2"})
    scheduler._persist()
    scheduler._state.update({"enabled": True, "cron": scheduler.DEFAULT_CRON})
    scheduler._load()
    assert scheduler._state["cron"] == "30 7 * * 2"
    assert scheduler._state["enabled"] is True


def test_scheduler_configure_persists_and_validates(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "CONFIG_PATH", tmp_path / "schedule.json")
    # _scheduler is None here (not started) so _reschedule is a no-op.
    scheduler._scheduler = None
    with pytest.raises(ValueError):
        scheduler.configure(cron="bad")
    st = scheduler.configure(enabled=False, cron="0 10 * * 5")
    assert st["enabled"] is False
    assert st["cron"] == "0 10 * * 5"


def test_scheduler_default_cron_is_friday():
    # Friday auto-run: the default cron expression uses the 'fri' weekday.
    assert "fri" in scheduler.DEFAULT_CRON.lower()


# --- podcast_config: persistent data/podcast_config.yaml --------------------

@pytest.fixture
def cfg_root(tmp_path, monkeypatch):
    """Isolate the config file under a tmp path."""
    monkeypatch.setattr(podcast_config, "CONFIG_PATH", tmp_path / "podcast_config.yaml")
    return tmp_path


def test_podcast_config_load_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(podcast_config, "CONFIG_PATH", tmp_path / "podcast_config.yaml")
    assert podcast_config.exists() is False
    cfg = podcast_config.load()
    assert cfg == podcast_config.DEFAULTS
    assert cfg["user"]["audience"] == "researcher"
    assert cfg["podcast"]["window_days"] == 7
    assert cfg["podcast"]["length"] == "medium"
    assert cfg["podcast"]["depth"] == "deep-dive"
    assert cfg["user"]["familiar_topics"] == []


def test_podcast_config_save_round_trip_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(podcast_config, "CONFIG_PATH", tmp_path / "podcast_config.yaml")
    saved = podcast_config.save({
        "user": {"audience": "beginner", "familiar_topics": [" RLHF ", "MoE", ""]},
        "podcast": {"window_days": 10, "length": "long", "depth": "brief"},
    })
    # normalized: familiar topics trimmed, empties dropped
    assert saved["user"]["familiar_topics"] == ["RLHF", "MoE"]
    assert podcast_config.exists() is True
    assert podcast_config.load() == saved
    # invalid values are rejected before persisting
    for bad in (
        {"user": {"audience": "wizard", "familiar_topics": []},
         "podcast": {"window_days": 7, "length": "medium", "depth": "deep-dive"}},
        {"user": {"audience": "beginner", "familiar_topics": []},
         "podcast": {"window_days": 30, "length": "medium", "depth": "deep-dive"}},
        {"user": {"audience": "beginner", "familiar_topics": []},
         "podcast": {"window_days": 7, "length": "huge", "depth": "deep-dive"}},
        {"user": {"audience": "beginner", "familiar_topics": []},
         "podcast": {"window_days": 7, "length": "medium", "depth": "nope"}},
        {"user": {"audience": "beginner", "familiar_topics": []},
         "podcast": {"window_days": "soon", "length": "medium", "depth": "deep-dive"}},
    ):
        with pytest.raises(ValueError):
            podcast_config.save(bad)


def test_podcast_config_resolved_run_config_merges_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(podcast_config, "CONFIG_PATH", tmp_path / "podcast_config.yaml")
    podcast_config.save({
        "user": {"audience": "intermediate", "familiar_topics": ["RLHF"]},
        "podcast": {"window_days": 5, "length": "medium", "depth": "deep-dive"},
    })
    # Explicit podcast-wise overrides win; user-wise always from the file.
    cfg = podcast_config.resolved_run_config({
        "window_start": "2026-08-01", "window_end": "2026-08-08",
        "length": "long", "depth": "brief"})
    assert cfg["window_start"] == "2026-08-01"
    assert cfg["window_end"] == "2026-08-08"
    assert cfg["length"] == "long"
    assert cfg["depth"] == "brief"
    assert cfg["audience_level"] == "intermediate"
    assert cfg["familiar_topics"] == ["RLHF"]
    # Missing override keys fall back to the file; no overrides (auto-run)
    # resolves the rolling window at call time.
    cfg = podcast_config.resolved_run_config()
    assert cfg["length"] == "medium" and cfg["depth"] == "deep-dive"
    assert cfg["window_start"] is not None and cfg["window_end"] is not None
    import datetime
    today = datetime.date.today()
    assert cfg["window_end"] == today.isoformat()
    assert cfg["window_start"] == (today - datetime.timedelta(days=5)).isoformat()
    # bad window span in the request is caught by pipeline validation
    with pytest.raises(ValueError):
        podcast_config.resolved_run_config({
            "window_start": "2026-01-01", "window_end": "2026-08-31"})


# --- episodes: delete a history entry ---------------------------------------

def test_episodes_delete_run(tmp_path, monkeypatch):
    monkeypatch.setattr(episodes, "DATA_ROOT", tmp_path)
    r = tmp_path / "31-08-2026"; r.mkdir()
    (r / "episode.json").write_text("{}")
    assert episodes.delete_run("2026-08-31") is True
    assert not r.exists()
    assert episodes.delete_run("2026-08-31") is False
    with pytest.raises(ValueError):
        episodes.delete_run("not-a-date")


# --- jobs: env threading ---------------------------------------------------

def test_jobs_submit_passes_env_to_subprocess(tmp_path, reset_jobs):
    # A trivial command that prints its PIPELINE_DATA_ROOT and exits.
    cmd = [sys.executable, "-c",
           "import os, sys; sys.stdout.write(os.environ.get('PIPELINE_DATA_ROOT','unset'))"]
    job, err = jobs.submit("generate", cmd, date="2026-08-31",
                           env={"PIPELINE_DATA_ROOT": "/tmp/whatever"})
    assert err == ""
    job._thread.join(timeout=10)
    assert job.status == "done"
    assert "/tmp/whatever" in "\n".join(job.log_lines)
