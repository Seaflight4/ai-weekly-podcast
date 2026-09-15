"""Tests for the HTTP service layer.

No network, no LLM, no real subprocess. The pipeline subprocess is either
mocked (jobs) or replaced with a trivial command that exits immediately.
"""
import asyncio, json, pathlib, queue, sys, time

import pytest

_FILE = pathlib.Path(__file__).resolve()
# service lives in this app folder (parents[1]); podcast_engine is shared in
# podcast-engine/ at the repo root (parents[2]).
sys.path.insert(0, str(_FILE.parents[1]))
sys.path.insert(0, str(_FILE.parents[2]))
sys.path.insert(0, str(_FILE.parents[2] / "podcast-engine"))

from service import episodes, events, jobs, podcast_config


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


def test_episodes_window_label_formats():
    wl = episodes.window_label
    assert wl("2026-09-05", "2026-09-05") == "05 Sep 2026"            # single day
    assert wl(None, "2026-09-05") == "05 Sep 2026"                     # unknown start
    assert wl("2026-09-02", "2026-09-05") == "02\u201305 Sep 2026"      # same month
    assert wl("2026-08-29", "2026-09-05") == "29 Aug \u2013 05 Sep 2026"   # cross month
    assert wl("2026-12-30", "2027-01-05") == "30 Dec 2026 \u2013 05 Jan 2027"  # cross year
    assert wl("2026-01-10", "01-02-2026") == "10 Jan \u2013 01 Feb 2026"  # mixed formats
    assert wl(None, None) is None
    # a run id as the end boundary still resolves to its date part
    assert wl("2026-09-02", "05-09-2026-124555") == "02\u201305 Sep 2026"


def test_episodes_run_summary_coverage_window(tmp_path, monkeypatch):
    monkeypatch.setattr(episodes, "DATA_ROOT", tmp_path)
    r = tmp_path / "05-09-2026-124555"; r.mkdir()
    (r / "episode.json").write_text(json.dumps({
        "manifest": [{"url": "u"}],
        "config": {"window": {"start": "2026-09-02", "end": "2026-09-05"}},
    }))
    run = episodes.get_run("05-09-2026-124555")
    assert run["window_label"] == "02\u201305 Sep 2026"
    assert run["window_start"] == "2026-09-02" and run["window_end"] == "2026-09-05"
    assert run["created_label"] == "12:45"
    # an empty run (no episode.json) still labels by the window-end in its id
    e = tmp_path / "12-09-2026-100000"; e.mkdir()
    empty = episodes.get_run("12-09-2026-100000")
    assert empty["status"] == "empty"
    assert empty["window_label"] == "12 Sep 2026"
    assert empty["created_label"] == "10:00"


def test_episodes_list_runs_suffixes_second_run_of_same_window(tmp_path, monkeypatch):
    monkeypatch.setattr(episodes, "DATA_ROOT", tmp_path)
    for name in ("05-09-2026-100000", "05-09-2026-140000"):
        d = tmp_path / name
        d.mkdir()
    runs = episodes.list_runs()
    assert [r["date"] for r in runs] == ["05-09-2026-140000", "05-09-2026-100000"]
    assert runs[0]["window_label"] == "05 Sep 2026 #2"   # newer of the window
    assert runs[1]["window_label"] == "05 Sep 2026"      # oldest keeps plain label


# --- jobs: single-run lock -------------------------------------------------

@pytest.fixture
def reset_jobs():
    """Reset job module-global state between tests (queue, active, recent).

    The daemon dispatcher/ticker threads are left alive (they idle safely),
    so never re-created ; only the job state they work on is cleared.
    """

    def _reset():
        with jobs._lock:
            jobs._queue.clear()
            active = jobs._active
            jobs._active = None
        if active is not None and active._proc is not None and active._proc.poll() is None:
            try:
                active._proc.kill()
            except OSError:
                pass
        if active is not None and active._thread is not None and active._thread.is_alive():
            active._thread.join(timeout=10)
        with jobs._recent_lock:
            jobs._recent.clear()

    _reset()
    yield
    _reset()


def test_jobs_queues_second_submit(reset_jobs):
    # A second submit while one is active is QUEUED, not rejected.
    slow = [sys.executable, "-c", "import time; time.sleep(1.2)"]
    job1 = jobs.submit("full", slow, date=None)
    assert job1.status == "running"
    quick = [sys.executable, "-c", "print('ok')"]
    job2 = jobs.submit("full", quick, date=None)
    assert job2.status == "queued"
    # it runs after the first finishes and both are tracked
    job1._thread.join(timeout=10)
    assert job1.status == "done"
    # the dispatcher may only now have picked the second job up — wait for it
    deadline = time.monotonic() + 10
    while job2.status == "queued" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job2.status != "queued"
    job2._thread.join(timeout=10)
    assert job2.status == "done"
    ids = [j["id"] for j in jobs.list_jobs()]
    assert job1.id in ids and job2.id in ids


def test_jobs_submit_broadcasts_run_queued_and_names_jobs(reset_jobs, monkeypatch):
    sent = []
    monkeypatch.setattr(jobs.events, "broadcast", sent.append)
    job = jobs.submit("full", [sys.executable, "-c", "print('ok')"], date="31-08-2026")
    d = job.to_dict()
    assert d["name"] == "New episode — 31 Aug 2026"
    assert d["submitted_at"] and "T" in d["submitted_at"]
    queued = [p for p in sent if p["type"] == "run_queued"]
    assert len(queued) == 1
    assert queued[0]["job"]["name"] == "New episode — 31 Aug 2026"
    job._thread.join(timeout=10)


def test_job_names_cover_rerender_and_dateless():
    assert jobs._job_name(jobs.Job("generate", "31-08-2026-100000", [])) == \
        "Re-render — 31 Aug 2026"
    assert jobs._job_name(jobs.Job("generate", "31-08-2026", [])) == \
        "Re-render — 31 Aug 2026"
    assert jobs._job_name(jobs.Job("full", None, [])) == "New episode"
    # A known window start makes the name show the coverage range.
    with_ws = jobs.Job("full", "05-09-2026-124555", [], window_start="2026-09-02")
    assert jobs._job_name(with_ws) == "New episode — 02–05 Sep 2026"


def test_jobs_failure_writes_exit_code_to_disk_and_error_field(tmp_path, monkeypatch, reset_jobs):
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    fail = [sys.executable, "-c", "import sys; sys.exit(3)"]
    job = jobs.submit("full", fail, date=None)
    job._thread.join(timeout=10)
    assert job.status == "failed"
    assert job.error == "pipeline exited with code 3"
    assert "[exit code 3]" in "\n".join(job.log_lines)
    disk = (tmp_path / f"{job.id}.log").read_text()
    assert "[exit code 3]" in disk          # parity: on-disk log shows it too
    d = job.to_dict()
    assert d["status"] == "failed" and d["error"] == "pipeline exited with code 3"


def test_jobs_failure_prefers_pipeline_error_line(tmp_path, monkeypatch, reset_jobs):
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path)
    cmd = [sys.executable, "-c",
           "print('ERROR: rank judge batch 4 failed: APIConnectionError: boom'); import sys; sys.exit(1)"]
    job = jobs.submit("full", cmd, date=None)
    job._thread.join(timeout=10)
    assert job.status == "failed"
    assert job.error == "ERROR: rank judge batch 4 failed: APIConnectionError: boom"
    disk = (tmp_path / f"{job.id}.log").read_text()
    assert "ERROR: rank judge batch 4 failed" in disk


def test_jobs_cancel_running_terminates_and_cleans_folder(tmp_path, monkeypatch, reset_jobs):
    # Cancel mid-run: the process group is killed and the full run's folder is
    # removed (intermediate results cleanup).
    monkeypatch.setattr(jobs.episodes_mod, "DATA_ROOT", tmp_path)
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    job = jobs.submit("full", cmd, date="31-08-2026")
    assert job.status == "running"
    folder = tmp_path / "31-08-2026"
    folder.mkdir()
    (folder / "collect.json").write_text("{}")
    got, err = jobs.cancel(job.id)
    assert err == "" and got is not None
    job._thread.join(timeout=10)
    assert job.status == "cancelled"
    assert not folder.exists()


def test_jobs_cancel_queued_drops_it(reset_jobs):
    slow = [sys.executable, "-c", "import time; time.sleep(1.2)"]
    job1 = jobs.submit("full", slow, date=None)
    queued = jobs.submit("full", [sys.executable, "-c", "print('ok')"], date="2026-08-31")
    assert queued.status == "queued"
    got, err = jobs.cancel(queued.id)
    assert err == "" and got is not None and got.status == "cancelled"
    job1._thread.join(timeout=10)
    assert job1.status == "done"
    # the cancelled job never ran; not in the active/queued listing
    jobs_present = [j["id"] for j in jobs.list_jobs() if j["status"] in ("queued", "running")]
    assert queued.id not in jobs_present


def test_jobs_cancel_unknown_or_finished(reset_jobs):
    got, err = jobs.cancel("does-not-exist")
    assert got is None and err == "not_found"
    done = jobs.submit("full", [sys.executable, "-c", "print('ok')"], date=None)
    done._thread.join(timeout=10)
    assert done.status == "done"
    got, err = jobs.cancel(done.id)
    assert got is not None and err == "not_running"


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
        "length": "long", "depth": "brief", "mem_windows": 3})
    assert cmd[cmd.index("--window-start") + 1] == "2026-08-24"
    assert cmd[cmd.index("--audience") + 1] == "intermediate"
    assert cmd[cmd.index("--length") + 1] == "long"
    assert cmd[cmd.index("--depth") + 1] == "brief"
    assert cmd[cmd.index("--mem-windows") + 1] == "3"


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
    assert cfg["podcast"]["mem_windows"] == 2
    assert cfg["user"]["familiar_topics"] == []


def test_podcast_config_save_round_trip_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(podcast_config, "CONFIG_PATH", tmp_path / "podcast_config.yaml")
    saved = podcast_config.save({
        "user": {"audience": "beginner", "familiar_topics": [" RLHF ", "MoE", ""]},
        "podcast": {"window_days": 10, "length": "long", "depth": "brief",
                    "mem_windows": 3},
    })
    # normalized: familiar topics trimmed, empties dropped
    assert saved["user"]["familiar_topics"] == ["RLHF", "MoE"]
    assert saved["podcast"]["mem_windows"] == 3
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
        {"user": {"audience": "beginner", "familiar_topics": []},
         "podcast": {"window_days": 7, "length": "medium", "depth": "deep-dive",
                     "mem_windows": 6}},
        {"user": {"audience": "beginner", "familiar_topics": []},
         "podcast": {"window_days": 7, "length": "medium", "depth": "deep-dive",
                     "mem_windows": "soon"}},
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
    assert cfg["mem_windows"] == 2  # from the saved config (default)
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
    job = jobs.submit("generate", cmd, date="2026-08-31",
                      env={"PIPELINE_DATA_ROOT": "/tmp/whatever"})
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


def test_taxonomy_endpoint_shape():
    """GET /api/taxonomy exposes the active taxonomy (id/label/description)
    so the UI renders the Sources-of-interest checkboxes from it — after a
    runtime refresh adopts a new artifact, this reflects the new set."""
    from service.app import get_taxonomy
    items = get_taxonomy()
    ids = {t["id"] for t in items}
    assert "other" in ids and "agents" in ids
    assert len(ids) == len(items)  # unique ids
    for t in items:
        assert all(k in t for k in ("id", "label", "description"))


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
                 "topic_prefs": ["post_training", "ai_for_science"],
                 "steering_alpha": 0.4},
    })
    assert stored["user"]["topic_prefs"] == ["post_training", "ai_for_science"]
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


# --- future-window + malformed-input safety nets ------------------------------

def test_resolved_run_config_rejects_future_window(tmp_path, monkeypatch):
    monkeypatch.setattr(podcast_config, "CONFIG_PATH", tmp_path / "podcast_config.yaml")
    import datetime
    with pytest.raises(ValueError):
        podcast_config.resolved_run_config({
            "window_start": "2099-01-01", "window_end": "2099-01-08",
            "length": "medium", "depth": "deep-dive"})


def test_resolved_run_config_allows_client_ahead_timezone(tmp_path, monkeypatch):
    # A user east of the UTC server may legitimately pick local "today" which
    # is the server date + 1. Their supplied ``now`` is the reference, so this
    # window-ending "tomorrow" (server time) is valid.
    monkeypatch.setattr(podcast_config, "CONFIG_PATH", tmp_path / "podcast_config.yaml")
    import datetime
    today = datetime.date.today()
    client_today = today + datetime.timedelta(days=1)
    cfg = podcast_config.resolved_run_config({
        "window_start": (client_today - datetime.timedelta(days=6)).isoformat(),
        "window_end": client_today.isoformat(),
        "length": "short", "depth": "brief"},
        now=f"{client_today.isoformat()}T10:00:00")
    assert cfg["window_end"] == client_today.isoformat()


def test_resolved_run_config_clamps_absurd_client_clock(tmp_path, monkeypatch):
    # A client clock claiming year 2099 is clamped to the server date, so a
    # truly future window is still rejected even when ``now`` is bogus.
    monkeypatch.setattr(podcast_config, "CONFIG_PATH", tmp_path / "podcast_config.yaml")
    with pytest.raises(ValueError):
        podcast_config.resolved_run_config({
            "window_start": "2099-01-01", "window_end": "2099-01-08"},
            now="2099-01-01T10:00:00")


def test_bad_date_path_params_are_400():
    from fastapi import HTTPException
    from service.app import get_audio, get_brief, get_episode, get_transcript, submit_generate
    for fn, args in (
        (get_episode, ("not-a-date",)),
        (get_audio, ("not-a-date",)),
        (get_brief, ("not-a-date",)),
        (get_transcript, ("not-a-date",)),
        (submit_generate, ("not-a-date", {"brief_markdown": "x"})),
    ):
        with pytest.raises(HTTPException) as e:
            fn(*args)
        assert e.value.status_code == 400


def test_submit_generate_requires_string_brief():
    from fastapi import HTTPException
    from service.app import submit_generate
    with pytest.raises(HTTPException) as e:
        submit_generate("31-08-2026", {"brief_markdown": [1, 2]})
    assert e.value.status_code == 400
    with pytest.raises(HTTPException) as e:
        submit_generate("31-08-2026", {"brief_markdown": "   "})
    assert e.value.status_code == 400


def test_submit_run_rejects_future_window(tmp_path, monkeypatch):
    monkeypatch.setattr(podcast_config, "CONFIG_PATH", tmp_path / "podcast_config.yaml")
    from fastapi import HTTPException
    from service.app import submit_run
    with pytest.raises(HTTPException) as e:
        submit_run({"date": None, "no_audio": True, "podcast": {
            "window_start": "2099-01-01", "window_end": "2099-01-08",
            "length": "medium", "depth": "deep-dive"}})
    assert e.value.status_code == 400


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


# --- events: SSE push hub ----------------------------------------------------

@pytest.fixture
def reset_events():
    with events._subs_lock:
        events._subs.clear()
    yield
    with events._subs_lock:
        events._subs.clear()


def test_events_broadcast_reaches_every_subscriber(reset_events):
    q1 = events.subscribe()
    q2 = events.subscribe()
    try:
        events.broadcast({"type": "run_progress", "job": {"id": "a"}})
        assert q1.get_nowait() == {"type": "run_progress", "job": {"id": "a"}}
        assert q2.get_nowait() == {"type": "run_progress", "job": {"id": "a"}}
        # unsubscribe stops delivery to that client only.
        events.unsubscribe(q1)
        events.broadcast({"type": "episodes_changed"})
        assert q2.get_nowait()["type"] == "episodes_changed"
        with pytest.raises(queue.Empty):
            q1.get_nowait()
    finally:
        for q in (q1, q2):
            events.unsubscribe(q)


def test_events_broadcast_drops_for_full_queue_not_crash(reset_events):
    q = events.subscribe()
    try:
        # Fill to the cap, then one more: the oldest is dropped, newest lands.
        for i in range(events._MAX_QUEUED + 5):
            events.broadcast({"type": "episodes_changed", "i": i})
        payload = None
        while True:
            try:
                payload = q.get_nowait()
            except queue.Empty:
                break
        assert payload is not None
        assert payload["i"] in range(events._MAX_QUEUED, events._MAX_QUEUED + 5)
    finally:
        events.unsubscribe(q)


def test_events_frames_formats_sse_event(reset_events):
    q = events.subscribe()
    try:
        events.broadcast({"type": "run_finished", "job": {"id": "x"}})
        frame = _first_frame(q)
        assert frame == (
            'event: run_finished\n'
            'data: {"type": "run_finished", "job": {"id": "x"}}\n\n')
    finally:
        events.unsubscribe(q)


def _first_frame(q: queue.Queue) -> str:
    async def _g():
        agen = events.frames(q)
        return await agen.__anext__()
    return asyncio.run(_g())


def test_sse_endpoint_streams_events(tmp_path, monkeypatch, reset_events):
    # NOTE: starlette's TestClient transport buffers the whole body, so an
    # infinite SSE stream cannot be consumed via client.stream. Drive the
    # endpoint's StreamingResponse generator directly instead.
    from service.app import sse_events

    resp = asyncio.run(sse_events())
    assert resp.media_type == "text/event-stream"
    assert resp.headers["cache-control"] == "no-cache"

    async def _drive():
        events.broadcast({"type": "episodes_changed"})
        frame = await resp.body_iterator.__anext__()
        await resp.body_iterator.aclose()  # triggers the unsubscribe path
        return frame

    frame = asyncio.run(_drive())
    assert 'event: episodes_changed\n' in frame
    assert '"type": "episodes_changed"' in frame
    # The ended stream left no subscriber behind.
    assert events._subs == []


# --- jobs: SSE broadcast on completion + progress ticker ---------------------

def test_jobs_broadcasts_run_finished_on_completion(tmp_path, reset_jobs, monkeypatch):
    sent = []
    monkeypatch.setattr(jobs.events, "broadcast", sent.append)
    cmd = [sys.executable, "-c", "print('ok')"]
    job = jobs.submit("full", cmd, date="2026-08-31")
    job._thread.join(timeout=10)
    assert job.status == "done"
    finished = [p for p in sent if p["type"] == "run_finished"]
    assert len(finished) == 1
    assert finished[0]["job"]["status"] == "done"
    assert finished[0]["job"]["date"] == "2026-08-31"


def test_jobs_ticker_pushes_progress_then_finish(tmp_path, reset_jobs, monkeypatch):
    sent = []
    monkeypatch.setattr(jobs.events, "broadcast", sent.append)
    monkeypatch.setattr(jobs, "PROGRESS_INTERVAL", 0.05)
    cmd = [sys.executable, "-c", "import time; time.sleep(0.4)"]
    job = jobs.submit("full", cmd, date=None)
    job._thread.join(timeout=10)
    assert job.status == "done"
    progress = [p for p in sent if p["type"] == "run_progress"]
    finished = [p for p in sent if p["type"] == "run_finished"]
    assert len(progress) >= 1
    assert len(finished) == 1
    assert all(p["job"]["id"] == job.id for p in progress)


# --- re-render snapshot / restore --------------------------------------------

def test_snapshot_happens_and_cancel_restores_prior_episode(tmp_path, monkeypatch, reset_jobs):
    # A ``generate`` re-render snapshots the episode files; cancelling in the
    # middle restores them, and keeps the user's edited brief (not snapshotted).
    monkeypatch.setattr(jobs.episodes_mod, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(jobs, "BACKUP_ROOT", tmp_path / ".backups")
    folder = tmp_path / "31-08-2026"
    folder.mkdir()
    (folder / "episode.json").write_text(json.dumps(
        {"selection_source": "auto", "manifest": [{"url": "old"}]}))
    (folder / "episode.mp3").write_bytes(b"OLDMP3")
    (folder / "transcript.md").write_text("OLD TRANSCRIPT")
    brief = folder / "podcast_brief.md"
    brief.write_text("OLD BRIEF")
    (folder / "rank.json").write_text("[]")   # required for a re-render

    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    job = jobs.submit("generate", cmd, date="31-08-2026")
    assert job.status == "running"
    # snapshot is taken when the run starts (before any mutation) — wait for it
    deadline = time.monotonic() + 5
    while job.backup_dir is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.backup_dir is not None and job.backup_dir.exists()
    assert (job.backup_dir / "episode.json").exists()
    # user already edited the brief before submitting
    brief.write_text("EDITED BRIEF")
    # simulate the run partially rewriting the episode, then cancel it
    (folder / "episode.json").write_text(json.dumps(
        {"selection_source": "customized", "manifest": [{"url": "new"}]}))
    (folder / "transcript.md").write_text("NEW PARTIAL TRANSCRIPT")
    got, err = jobs.cancel(job.id)
    assert err == ""
    job._thread.join(timeout=10)
    assert job.status == "cancelled"
    # restored to the snapshot; the edited brief survives
    assert folder.exists()
    assert json.loads((folder / "episode.json").read_text())["selection_source"] == "auto"
    assert (folder / "episode.mp3").read_bytes() == b"OLDMP3"
    assert (folder / "transcript.md").read_text() == "OLD TRANSCRIPT"
    assert brief.read_text() == "EDITED BRIEF"
    assert not job.backup_dir.exists()          # backup dropped
    assert not (folder / ".podcastfy-cache").exists()


def test_rerender_success_drops_snapshot(tmp_path, monkeypatch, reset_jobs):
    monkeypatch.setattr(jobs.episodes_mod, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(jobs, "BACKUP_ROOT", tmp_path / ".backups")
    folder = tmp_path / "31-08-2026"
    folder.mkdir()
    (folder / "episode.json").write_text("{}")
    cmd = [sys.executable, "-c", "print('ok')"]
    job = jobs.submit("generate", cmd, date="31-08-2026")
    job._thread.join(timeout=10)
    assert job.status == "done"
    assert job.backup_dir is not None and not job.backup_dir.exists()


# --- cancel endpoint wiring --------------------------------------------------

def test_cancel_endpoint_returns_404_unknown_and_409_finished(reset_jobs):
    from fastapi import HTTPException
    from service.app import cancel_run
    with pytest.raises(HTTPException) as e:
        cancel_run("nope")
    assert e.value.status_code == 404
    done = jobs.submit("full", [sys.executable, "-c", "print('ok')"], date=None)
    done._thread.join(timeout=10)
    assert done.status == "done"
    with pytest.raises(HTTPException) as e:
        cancel_run(done.id)
    assert e.value.status_code == 409
