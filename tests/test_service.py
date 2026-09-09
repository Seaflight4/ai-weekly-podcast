"""Tests for the HTTP service layer.

No network, no LLM, no real subprocess. The pipeline subprocess is either
mocked (jobs) or replaced with a trivial command that exits immediately.
"""
import json, pathlib, sys, time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from service import episodes, jobs, podcast_config


# --- episodes scan + status badges -----------------------------------------

def test_episodes_list_status_badges(tmp_path, monkeypatch):
    monkeypatch.setattr(episodes, "DATA_ROOT", tmp_path)
    # ready: episode.json + episode.mp3
    r = tmp_path / "31-08-2026"; r.mkdir()
    (r / "episode.json").write_text(json.dumps({
        "manifest": [{"url": "u"}], "created_at": "2026-08-31T09:00:00Z",
        "selection_source": "personalized",
        "duration_sec": 754.5, "transcript_words": 2000}))
    (r / "episode.mp3").write_bytes(b"")
    # ready, time-stamped: a second episode generated the same day
    r2 = tmp_path / "31-08-2026-154500"; r2.mkdir()
    (r2 / "episode.json").write_text(json.dumps({
        "manifest": [{"url": "u2"}], "created_at": "2026-08-31T15:45:00Z",
        "selection_source": "auto"}))
    (r2 / "episode.mp3").write_bytes(b"")
    # draft: episode.json but no mp3
    d = tmp_path / "24-08-2026"; d.mkdir()
    (d / "episode.json").write_text(json.dumps({"manifest": []}))
    # empty: no episode.json
    e = tmp_path / "17-08-2026"; e.mkdir()
    # non-date folder ignored
    (tmp_path / ".jobs").mkdir()

    runs = episodes.list_runs()
    assert [x["date"] for x in runs] == [
        "31-08-2026-154500", "31-08-2026", "24-08-2026", "17-08-2026"]
    by_date = {x["date"]: x for x in runs}
    assert by_date["31-08-2026"]["status"] == "ready"
    assert by_date["31-08-2026"]["date_label"] == "31-08-2026"
    assert by_date["31-08-2026"]["selection_source"] == "personalized"
    assert by_date["31-08-2026"]["duration_sec"] == 754.5
    assert by_date["31-08-2026"]["transcript_words"] == 2000
    assert by_date["31-08-2026-154500"]["status"] == "ready"
    assert by_date["31-08-2026-154500"]["date_label"] == "31-08-2026 15:45"
    assert by_date["24-08-2026"]["status"] == "draft"
    assert by_date["17-08-2026"]["status"] == "empty"


def test_episodes_get_run_accepts_time_stamped_id(tmp_path, monkeypatch):
    monkeypatch.setattr(episodes, "DATA_ROOT", tmp_path)
    r = tmp_path / "31-08-2026-154500"; r.mkdir()
    (r / "episode.json").write_text(json.dumps({"manifest": [{"url": "u"}]}))
    (r / "podcast_brief.md").write_text("# brief")
    (r / "rank.json").write_text(json.dumps([{"url": "u", "score": 0.9}]))
    run = episodes.get_run("31-08-2026-154500")
    assert run is not None
    assert run["brief"] == "# brief"
    assert run["rank"][0]["score"] == 0.9


def test_make_run_id_is_time_stamped_and_same_day_unique():
    import datetime
    a = episodes.make_run_id("2026-08-31", datetime.datetime(2026, 8, 31, 9, 0, 0))
    b = episodes.make_run_id("2026-08-31", datetime.datetime(2026, 8, 31, 15, 45, 30))
    assert a == "31-08-2026-090000"
    assert b == "31-08-2026-154530"
    assert a != b


def test_make_run_id_accepts_client_local_time_string():
    # A client-supplied local time string (e.g. from the browser) is used
    # verbatim, so a UTC server still stamps the user's local time.
    assert episodes.make_run_id("2026-08-31",
                                "2026-08-31T08:39:12") == "31-08-2026-083912"
    # Unparseable input falls back to server time (never crashes).
    id_from_bad = episodes.make_run_id("2026-08-31", "garbage")
    assert len(id_from_bad) == len("31-08-2026-000000")


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


def test_jobs_estimate_stages_scales_with_config():
    # Full run: bigger window (collect/rank) and more sources (generate) both
    # push the total up monotonically.
    small = jobs.estimate_stages("full", num_sources=4, window_days=1)
    big_window = jobs.estimate_stages("full", num_sources=4, window_days=7)
    many_sources = jobs.estimate_stages("full", num_sources=12, window_days=1)
    assert sum(small) < sum(big_window) < sum(many_sources)
    assert len(small) == jobs.STAGE_COUNT + 1 and small[0] == 0.0
    # sanity: medium run (7 sources, 1d window) lands in the measured ~5 min
    # range (medium+1d benchmark ~298s).
    med = jobs.estimate_stages("full", num_sources=7, window_days=1)
    assert 240 <= sum(med) <= 480
    # Re-render only runs the generate stage: collect/rank are zeroed.
    gen = jobs.estimate_stages("generate", num_sources=7, window_days=7)
    assert gen[:3] == [0.0, 0.0, 0.0]
    assert gen[3] > 0
    assert sum(gen) < sum(med)


# --- podcast_config: persistent data/podcast_config.yaml --------------------

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
    r2 = tmp_path / "31-08-2026-154500"; r2.mkdir()
    (r2 / "episode.json").write_text("{}")
    assert episodes.delete_run("31-08-2026-154500") is True
    assert not r2.exists()
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


# --- topic labels: summary, filtering, config --------------------------------

def test_episodes_summary_includes_topics_and_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(episodes, "DATA_ROOT", tmp_path)
    for name, labels in [
        ("01-09-2026", {"episode_topics": [{"topic": "model_release", "weight": 0.9}]}),
        ("02-09-2026", {"episode_topics": [{"topic": "post_training", "weight": 0.9}]}),
    ]:
        d = tmp_path / name
        d.mkdir()
        (d / "episode.json").write_text(json.dumps(
            {"manifest": [{"url": "u"}], "created_at": "2026-09-01T09:00:00Z"}))
        (d / "episode.mp3").write_bytes(b"")
        (d / "labels.json").write_text(json.dumps(labels))

    runs = episodes.list_runs()
    assert len(runs) == 2
    by_date = {r["date"]: r for r in runs}
    assert by_date["01-09-2026"]["topics"] == [{"topic": "model_release", "weight": 0.9}]

    only = episodes.list_runs(topics=["post_training"])
    assert [r["date"] for r in only] == ["02-09-2026"]
    both = episodes.list_runs(topics=["model_release", "post_training"])
    assert len(both) == 2
    assert episodes.list_runs(topics=["does_not_exist"]) == []
    assert episodes.list_runs(topics=[""]) == runs


def test_episodes_get_run_includes_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(episodes, "DATA_ROOT", tmp_path)
    d = tmp_path / "01-09-2026"
    d.mkdir()
    (d / "episode.json").write_text(json.dumps({"manifest": [{"url": "u"}]}))
    (d / "labels.json").write_text(json.dumps(
        {"episode_topics": [{"topic": "post_training", "weight": 0.9}],
         "topics": {"post_training": 1.0}}))
    run = episodes.get_run("01-09-2026")
    assert run["labels"]["episode_topics"][0]["topic"] == "post_training"


def test_podcast_config_validates_topic_prefs_and_alpha(tmp_path, monkeypatch):
    monkeypatch.setattr(podcast_config, "CONFIG_PATH", tmp_path / "podcast_config.yaml")
    base = {"user": {"audience": "researcher"},
            "podcast": {"window_days": 7, "length": "short", "depth": "brief"}}
    with pytest.raises(ValueError):
        podcast_config.save({**base, "user": {"audience": "researcher",
                                              "topic_prefs": ["not_a_topic"]}})
    with pytest.raises(ValueError):
        podcast_config.save({**base, "user": {"audience": "researcher",
                                              "steering_alpha": 1.7}})

    stored = podcast_config.save({
        **base,
        "user": {"audience": "researcher",
                 "topic_prefs": ["post_training", "agents_tool_use"],
                 "steering_alpha": 0.4},
    })
    assert stored["user"]["topic_prefs"] == ["post_training", "agents_tool_use"]
    assert stored["user"]["steering_alpha"] == 0.4
    # default alpha when unspecified
    defaulted = podcast_config.save({**base, "user": {"audience": "researcher"}})
    assert defaulted["user"]["steering_alpha"] == 0.3


def test_resolved_run_config_carries_topic_prefs(tmp_path, monkeypatch):
    monkeypatch.setattr(podcast_config, "CONFIG_PATH", tmp_path / "podcast_config.yaml")
    podcast_config.save({
        "user": {"audience": "researcher",
                 "topic_prefs": ["post_training"], "steering_alpha": 0.5},
        "podcast": {"window_days": 2, "length": "short", "depth": "brief"},
    })
    rc = podcast_config.resolved_run_config({})
    assert rc["topic_prefs"] == ["post_training"]
    assert rc["steering_alpha"] == 0.5


def test_full_run_cmd_carries_steering_flags():
    cmd = jobs.full_run_cmd(config={
        "topic_prefs": ["post_training"],
        "steering_alpha": 0.4,
    })
    assert "--topics" in cmd
    assert cmd[cmd.index("--topics") + 1] == "post_training"
    assert "--steering-alpha" in cmd
    assert cmd[cmd.index("--steering-alpha") + 1] == "0.4"
    # no prefs -> no steering flags at all
    clean = jobs.full_run_cmd(config={"length": "short"})
    assert "--topics" not in clean
    assert "--steering-alpha" not in clean
