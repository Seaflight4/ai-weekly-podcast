"""Tests for the HTTP service layer.

No network, no LLM, no real subprocess. The pipeline subprocess is either
mocked (jobs) or replaced with a trivial command that exits immediately.
"""
import json, pathlib, sys, time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from service import episodes, jobs, personalized, scheduler


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


# --- personalized: shared library filesystem ops ----------------------------

@pytest.fixture
def pers_data(tmp_path, monkeypatch):
    """Isolate personalized + episodes under a tmp data root."""
    monkeypatch.setattr(episodes, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(personalized, "PERSONALIZED_ROOT", tmp_path / "personalized")
    # A default run with rank.json + brief, so prepare_render can copy rank.json.
    default_run = tmp_path / "31-08-2026"
    default_run.mkdir()
    (default_run / "rank.json").write_text(
        json.dumps([{"title": "P", "url": "https://arxiv.org/abs/2601.1",
                     "source": "arxiv", "score": 0.9, "judge_reason": "imp"}]))
    (default_run / "podcast_brief.md").write_text("# brief\n")
    return tmp_path


def test_personalized_prepare_render_copies_rank_and_writes_brief(pers_data):
    folder = personalized.prepare_render("2026-08-31", "# edited\n- [P](https://arxiv.org/abs/2601.1)\n")
    assert folder == pers_data / "personalized" / "31-08-2026"
    assert (folder / "rank.json").exists()       # copied from default
    assert (folder / "podcast_brief.md").read_text().startswith("# edited")
    # default run untouched
    assert (pers_data / "31-08-2026" / "rank.json").exists()


def test_personalized_list_and_get(pers_data):
    personalized.prepare_render("2026-08-31", "# brief\n")
    personalized.prepare_render("2026-08-24", "# brief\n")
    runs = personalized.list_runs()
    assert [r["date"] for r in runs] == ["31-08-2026", "24-08-2026"]  # newest first
    run = personalized.get_run("2026-08-31")
    assert run is not None and run["brief"] == "# brief\n"


def test_personalized_empty_when_none(pers_data):
    assert personalized.list_runs() == []


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
