"""Contract tests for the podcast-quality pipeline.

These check JSON *shapes and invariants* per stage on synthetic data (no
network, no LLM). Quality is reviewed via the inspection reports; these tests
catch regressions in the contract.
"""
import json, pathlib, tempfile
import pytest

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipeline import Item, RankedItem, Topic, EpisodePlan, Segment
from pipeline import rank, plan, inspect
from pipeline import profile as profile_mod
from pipeline import feedback as feedback_mod


# --- Phase 2: profile + RankedItem schema --------------------------------

def test_profile_parses_frontmatter(tmp_path):
    p = tmp_path / "profile.md"
    p.write_text(
        "---\n"
        "topics:\n"
        "  - agent evals\n"
        "  - inference cost\n"
        "anti_topics:\n"
        "  - pure scaling\n"
        "---\n"
        "## What I'm working on\n"
        "Agent reliability this quarter.\n",
        encoding="utf-8",
    )
    prof = profile_mod.load_profile(p)
    assert prof.topics == ["agent evals", "inference cost"]
    assert prof.anti_topics == ["pure scaling"]
    assert "Agent reliability" in prof.body


def test_profile_missing_returns_empty(tmp_path):
    prof = profile_mod.load_profile(tmp_path / "nonexistent.md")
    assert prof.topics == []
    assert prof.anti_topics == []
    assert prof.body == ""


def test_rankeditem_round_trips_new_fields():
    r = RankedItem(title="t", url="u", date="d", body="b", source="arxiv",
                   score=0.8, judge_reason="imp", kind="deep",
                   personal_score=0.6, personal_reason="match",
                   alpha=0.7, final_score=0.74)
    d = r.__dict__
    r2 = RankedItem(**d)
    assert r2.personal_score == 0.6
    assert r2.alpha == 0.7
    assert r2.final_score == 0.74


def test_rankeditem_legacy_load_uses_defaults():
    # A Phase 1 rank.json item (no personalization fields) still loads.
    legacy = {"title": "t", "url": "u", "date": "d", "body": "b",
              "source": "arxiv", "score": 0.8, "judge_reason": "r", "kind": "deep"}
    r = RankedItem(**legacy)
    assert r.personal_score == 0.0
    assert r.alpha == 1.0
    assert r.final_score == 0.0


# --- Phase 2: feedback log ------------------------------------------------

def test_feedback_mark_and_aggregate(tmp_path, monkeypatch):
    # isolate data/ root and the aggregate log to tmp_path. Each run lives in
    # a dated subdir under data/, mirroring store.run_dir's real shape.
    data_root = tmp_path / "data"
    data_root.mkdir()
    run_dir = data_root / "24-08-2026"
    run_dir.mkdir()
    monkeypatch.setattr(feedback_mod.store, "run_dir", lambda date=None: run_dir)
    monkeypatch.setattr(feedback_mod.store, "ROOT", data_root)
    monkeypatch.setattr(feedback_mod, "FEEDBACK_LOG_PATH", tmp_path / "feedback_log.json")
    monkeypatch.setattr(feedback_mod, "RECENT_RUNS", 2)

    feedback_mod.mark("u1", kept=True)
    feedback_mod.mark("u2", kept=False)
    log = feedback_mod.load_feedback()
    assert log.kept_urls == {"u1"}
    assert log.skipped_urls == {"u2"}


def test_feedback_recency_weights_last_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_mod, "FEEDBACK_LOG_PATH", tmp_path / "feedback_log.json")
    monkeypatch.setattr(feedback_mod, "RECENT_RUNS", 2)
    # hand-build an aggregate with 3 runs; only the last 2 should count
    payload = {"runs": [
        {"run": "01-01-2026", "marks": [{"url": "old", "kept": True}]},
        {"run": "02-01-2026", "marks": [{"url": "recent1", "kept": True}]},
        {"run": "03-01-2026", "marks": [{"url": "recent2", "kept": False}]},
    ]}
    (tmp_path / "feedback_log.json").write_text(json.dumps(payload))
    log = feedback_mod.load_feedback()
    assert "old" not in log.kept_urls          # decayed out
    assert log.kept_urls == {"recent1"}
    assert log.skipped_urls == {"recent2"}


def test_feedback_missing_log_returns_empty(tmp_path):
    log = feedback_mod.load_feedback(tmp_path / "nonexistent.json")
    assert log.kept_urls == set()
    assert log.skipped_urls == set()


# --- Phase 2: personal pass + blend (tickets 04/05) -----------------------

def test_personal_pass_noop_without_profile(tmp_path, monkeypatch):
    # No profile.md -> final_score == score (Phase 1 behaviour)
    items = [Item(title=f"t{i}", url=f"u{i}", date="d", body="b", source="arxiv")
             for i in range(3)]
    rank._judge_batch = lambda groups, rubric: [(0.9 - i * 0.1, "r") for i in range(len(groups))]
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(rank.profile_mod, "PROFILE_PATH", tmp_path / "nonexistent.md")
    rank.RUBRIC_MODE = "unified"
    out = rank.rank(items)
    for r in out:
        assert r.final_score == r.score
        assert r.personal_score == 0.0


def test_personal_pass_noop_when_alpha_is_one(tmp_path, monkeypatch):
    # ALPHA=1.0 -> final_score == score even with a profile
    p = tmp_path / "profile.md"
    p.write_text("---\ntopics:\n  - x\n---\nbody\n")
    items = [Item(title=f"t{i}", url=f"u{i}", date="d", body="b", source="arxiv")
             for i in range(3)]
    rank._judge_batch = lambda groups, rubric: [(0.9 - i * 0.1, "r") for i in range(len(groups))]
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(rank, "ALPHA", 1.0)
    monkeypatch.setattr(rank.profile_mod, "PROFILE_PATH", p)
    rank.RUBRIC_MODE = "unified"
    out = rank.rank(items)
    for r in out:
        assert r.final_score == r.score


def test_blend_formula(tmp_path, monkeypatch):
    # ALPHA=0.7, importance=0.8, personal=0.6 -> final = 0.7*0.8 + 0.3*0.6 = 0.74
    p = tmp_path / "profile.md"
    p.write_text("---\ntopics:\n  - x\n---\nbody\n")
    items = [Item(title="t", url="u", date="d", body="b", source="arxiv")]
    def fake_judge(groups, rubric):
        if "core interest" in rubric:
            return [(0.6, "match")] * len(groups)
        return [(0.8, "imp")] * len(groups)
    rank._judge_batch = fake_judge
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(rank, "ALPHA", 0.7)
    monkeypatch.setattr(rank.profile_mod, "PROFILE_PATH", p)
    monkeypatch.setattr(rank.feedback_mod, "FEEDBACK_LOG_PATH", tmp_path / "fb.json")
    rank.RUBRIC_MODE = "unified"
    out = rank.rank(items)
    assert out[0].score == 0.8
    assert out[0].personal_score == 0.6
    assert out[0].alpha == 0.7
    assert abs(out[0].final_score - 0.74) < 1e-6


def test_blend_pure_personal(tmp_path, monkeypatch):
    # ALPHA=0.0 -> final_score == personal_score
    p = tmp_path / "profile.md"
    p.write_text("---\ntopics:\n  - x\n---\nbody\n")
    items = [Item(title="t", url="u", date="d", body="b", source="arxiv")]
    def fake_judge(groups, rubric):
        if "core interest" in rubric:
            return [(0.9, "match")] * len(groups)
        return [(0.3, "imp")] * len(groups)
    rank._judge_batch = fake_judge
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(rank, "ALPHA", 0.0)
    monkeypatch.setattr(rank.profile_mod, "PROFILE_PATH", p)
    monkeypatch.setattr(rank.feedback_mod, "FEEDBACK_LOG_PATH", tmp_path / "fb.json")
    rank.RUBRIC_MODE = "unified"
    out = rank.rank(items)
    assert abs(out[0].final_score - 0.9) < 1e-6


# --- Phase 2: --from-cache (ticket 06) ------------------------------------

def test_rank_from_cache_applies_personal(tmp_path, monkeypatch):
    # cache has 2 items with importance scores; personal pass adds personal_score
    p = tmp_path / "profile.md"
    p.write_text("---\ntopics:\n  - comics\n---\nbody\n")
    cache = tmp_path / "rank.json"
    cache.write_text(json.dumps([
        {"title": "a", "url": "u1", "date": "d", "body": "b", "source": "arxiv",
         "score": 0.8, "judge_reason": "imp", "kind": "deep",
         "personal_score": 0.0, "personal_reason": "", "alpha": 1.0, "final_score": 0.0},
        {"title": "b", "url": "u2", "date": "d", "body": "b", "source": "arxiv",
         "score": 0.5, "judge_reason": "imp", "kind": "deep",
         "personal_score": 0.0, "personal_reason": "", "alpha": 1.0, "final_score": 0.0},
    ]))
    # personal pass returns 0.9 for item 0, 0.3 for item 1
    def fake_judge(groups, rubric):
        return [(0.9 if i == 0 else 0.3, "match") for i in range(len(groups))]
    rank._judge_batch = fake_judge
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(rank, "ALPHA", 0.5)
    monkeypatch.setattr(rank.profile_mod, "PROFILE_PATH", p)
    monkeypatch.setattr(rank.feedback_mod, "FEEDBACK_LOG_PATH", tmp_path / "fb.json")
    out = rank.rank_from_cache(str(cache))
    assert len(out) == 2
    # item 0: 0.5*0.8 + 0.5*0.9 = 0.85; item 1: 0.5*0.5 + 0.5*0.3 = 0.4
    assert abs(out[0].final_score - 0.85) < 1e-6
    assert abs(out[1].final_score - 0.4) < 1e-6
    assert out[0].personal_score == 0.9
    assert out[0].alpha == 0.5


# --- Phase 2: --ab-personal A/B harness (ticket 08) -----------------------

def test_ab_personal_writes_both_arms_and_report(tmp_path, monkeypatch):
    p = tmp_path / "profile.md"
    p.write_text("---\ntopics:\n  - comics\n---\nbody\n")
    cache = tmp_path / "rank.json"
    cache.write_text(json.dumps([
        {"title": "a", "url": "u1", "date": "d", "body": "b", "source": "arxiv",
         "score": 0.8, "judge_reason": "imp", "kind": "deep",
         "personal_score": 0.0, "personal_reason": "", "alpha": 1.0, "final_score": 0.0},
        {"title": "b", "url": "u2", "date": "d", "body": "b", "source": "arxiv",
         "score": 0.5, "judge_reason": "imp", "kind": "deep",
         "personal_score": 0.0, "personal_reason": "", "alpha": 1.0, "final_score": 0.0},
    ]))
    # personal pass: item 0 gets 0.3 (off-topic), item 1 gets 0.9 (on-topic)
    def fake_judge(groups, rubric):
        return [(0.3 if i == 0 else 0.9, "match") for i in range(len(groups))]
    rank._judge_batch = fake_judge
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(inspect.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(rank, "ALPHA", 0.5)
    monkeypatch.setattr(rank.profile_mod, "PROFILE_PATH", p)
    monkeypatch.setattr(rank.feedback_mod, "FEEDBACK_LOG_PATH", tmp_path / "fb.json")
    results = rank.rank_ab_personal(str(cache), top_n=2)
    assert len(results["plain"]) == 2
    assert len(results["personal"]) == 2
    # plain arm: sorted by score -> a(0.8) then b(0.5)
    assert results["plain"][0].url == "u1"
    # personal arm: 0.5*0.8+0.5*0.3=0.55 vs 0.5*0.5+0.5*0.9=0.7 -> b wins
    assert results["personal"][0].url == "u2"
    # both arm files written
    assert (tmp_path / "rank_plain.json").exists()
    assert (tmp_path / "rank_personal.json").exists()
    # report written
    assert (tmp_path / "rank_report.md").exists()
    report = (tmp_path / "rank_report.md").read_text()
    assert "Personalization A/B report" in report
    assert "ALPHA" in report
    # with top_n=2 and only 2 items, both are in both arms' top — the win is
    # the reordering: u2 (low importance, high personal) ranks #1 under personal
    assert "0.90" in report  # personal score of u2 appears in the personal top-N table


# --- 01: arXiv gate (collect.json shape unchanged) -----------------------

def test_collect_shape_after_gate():
    items = [
        {"title": "t1", "url": "https://arxiv.org/abs/2601.00001", "date": "2026-08-21",
         "body": "abstract", "source": "arxiv"},
        {"title": "h1", "url": "https://news.ycombinator.com/item?id=1", "date": "2026-08-21",
         "body": "", "source": "hn"},
    ]
    # gate doesn't change the Item shape, just which items survive
    assert all({"title", "url", "date", "body", "source"} <= set(i) for i in items)


# --- 02/03: rank writes the full scored pool, sorted ---------------------

def test_rank_returns_full_pool_sorted(tmp_path, monkeypatch):
    items = [
        Item(title=f"t{i}", url=f"u{i}", date="2026-08-21", body="b", source="arxiv")
        for i in range(3)
    ]
    # monkeypatch the judge to avoid network; isolate store to a temp dir
    rank._judge_batch = lambda groups, rubric: [(0.9 - i * 0.1, "r") for i in range(len(groups))]
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)
    rank.RUBRIC_MODE = "unified"
    out = rank.rank(items)
    assert len(out) == 3                      # full pool, no top-N slice
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= r.score <= 1.0 for r in out)


def test_knee_finds_dropoff():
    ranked = [RankedItem(title="t", url=f"u{i}", date="d", body="b", source="arxiv",
                        score=s) for i, s in enumerate(
        [0.95, 0.93, 0.91, 0.89, 0.88, 0.87, 0.5, 0.49, 0.48, 0.3])]
    knee = rank._knee(ranked)
    assert knee is not None
    # the sharp drop is between index 5 (0.87) and 6 (0.5)
    assert 4 <= knee <= 6


def test_knee_flat_returns_none():
    ranked = [RankedItem(title="t", url=f"u{i}", date="d", body="b", source="arxiv",
                        score=0.5) for i in range(10)]
    assert rank._knee(ranked) is None


def test_select_pool_caps_at_max():
    ranked = [RankedItem(title="t", url=f"u{i}", date="d", body="b", source="arxiv",
                        score=1 - i * 0.001) for i in range(100)]
    pool, knee = rank.select_pool(ranked, max_n=50)
    assert len(pool) <= 50


# --- 04/05: cluster invariants --------------------------------------------

def test_topic_members_unique_and_nonempty():
    members = [
        RankedItem(title="p1", url="u1", date="d", body="b", source="arxiv", score=0.9),
        RankedItem(title="h1", url="u2", date="d", body="b", source="hn", score=0.8),
    ]
    t = Topic(id="t1", title="T", why="w", members=members, primary_url="u1",
              aggregate_score=0.9)
    assert len(t.members) >= 1
    urls = [m.url for m in t.members]
    assert len(urls) == len(set(urls))          # no member in two slots


def test_cluster_report_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(inspect.store, "run_dir", lambda: tmp_path)
    topics = [{"id": "t1", "title": "T", "why": "w", "primary_url": "u1",
               "aggregate_score": 0.9,
               "members": [{"title": "p", "url": "u1", "date": "d", "body": "b",
                            "source": "arxiv", "score": 0.9}]}]
    inspect.cluster_report(topics)
    assert (tmp_path / "cluster_report.md").exists()


# --- 06: plan budget invariants ------------------------------------------

def test_budget_fill_within_envelope():
    topics = []
    for i in range(20):
        m = RankedItem(title=f"t{i}", url=f"u{i}", date="d",
                       body="word " * 300, source="arxiv", score=0.9 - i * 0.05)
        topics.append(Topic(id=f"t{i}", title=f"T{i}", why="w", members=[m],
                            primary_url=f"u{i}", aggregate_score=m.score))
    minutes = plan._budget_fill(topics)
    total = plan.INTRO_MIN + plan.OUTRO_MIN + sum(minutes) + plan.TRANSITION_MIN * max(len(minutes) - 1, 0)
    assert plan.MIN_MIN <= total <= plan.MAX_MIN or len(topics) < len(minutes)
    assert len(minutes) >= 1


def test_plan_contract(monkeypatch):
    topics = [Topic(id="t1", title="T1", why="w", members=[
        RankedItem(title="p", url="u1", date="d", body="b", source="arxiv", score=0.9)],
        primary_url="u1", aggregate_score=0.9)]
    # Merged call: editorial_groups + outline in one response.
    def fake_chat(msg, prompt, model):
        return json.dumps({
            "editorial_groups": [{
                "label": "This week's release", "role": "release",
                "topic_ids": ["t1"], "opener": "A release this week...",
            }],
            "hook": "h", "motif": "m",
            "segments": [{"opening": "o", "signposts": ["s1", "s2"],
                           "transition_out": "tr", "speakers": ["A", "B"]}],
            "outro": "e",
        })
    plan._chat = fake_chat
    monkeypatch.setattr(plan.store, "write", lambda *a, **k: None)
    monkeypatch.setattr(plan.inspect, "budget_report", lambda *a, **k: None)
    monkeypatch.setattr(plan.inspect, "plan_report", lambda *a, **k: None)
    ep = plan.plan(topics, date="2026-08-21")
    assert ep.motif == "m"
    assert len(ep.segments) == 1
    assert sum(s.minutes for s in ep.segments) > 0
    assert ep.segments[0].topic_id == "t1"
    # Ticket 13: editorial groups
    assert len(ep.editorial_groups) == 1
    assert ep.editorial_groups[0].role == "release"
    assert ep.segments[0].editorial_group == "This week's release"


def test_plan_editorial_groups_partition_topics(monkeypatch):
    # 3 topics; merged call groups t1+t2 as releases, t3 standalone
    topics = [
        Topic(id=f"t{i}", title=f"T{i}", why="w", members=[
            RankedItem(title=f"p{i}", url=f"u{i}", date="d", body="b",
                       source="arxiv", score=0.9 - i * 0.05)],
            primary_url=f"u{i}", aggregate_score=0.9 - i * 0.05)
        for i in range(3)
    ]
    def fake_chat(msg, prompt, model):
        return json.dumps({
            "editorial_groups": [
                {"label": "2 releases", "role": "release",
                 "topic_ids": ["t0", "t1"], "opener": "Two releases..."},
                {"label": "Also worth knowing", "role": "standalone",
                 "topic_ids": ["t2"], "opener": "And one more thing..."},
            ],
            "hook": "h", "motif": "m",
            "segments": [{"opening": "o", "signposts": ["s1", "s2"],
                           "transition_out": "tr", "speakers": ["A", "B"]}
                          for _ in range(3)],
            "outro": "e",
        })
    plan._chat = fake_chat
    monkeypatch.setattr(plan.store, "write", lambda *a, **k: None)
    monkeypatch.setattr(plan.inspect, "budget_report", lambda *a, **k: None)
    monkeypatch.setattr(plan.inspect, "plan_report", lambda *a, **k: None)
    ep = plan.plan(topics, date="2026-08-21")
    # every topic is in exactly one group, no overlap
    all_tids = [tid for g in ep.editorial_groups for tid in g.topic_ids]
    assert sorted(all_tids) == ["t0", "t1", "t2"]
    # each segment carries its group label
    seg_groups = {s.topic_id: s.editorial_group for s in ep.segments}
    assert seg_groups["t0"] == "2 releases"
    assert seg_groups["t1"] == "2 releases"
    assert seg_groups["t2"] == "Also worth knowing"


def test_plan_editorial_bad_response_falls_back(monkeypatch):
    topics = [Topic(id="t1", title="T1", why="w", members=[
        RankedItem(title="p", url="u1", date="d", body="b", source="arxiv", score=0.9)],
        primary_url="u1", aggregate_score=0.9)]
    # merged call returns garbage; fallback kicks in
    def fake_chat(msg, prompt, model):
        return "NOT JSON"
    plan._chat = fake_chat
    monkeypatch.setattr(plan.store, "write", lambda *a, **k: None)
    monkeypatch.setattr(plan.inspect, "budget_report", lambda *a, **k: None)
    monkeypatch.setattr(plan.inspect, "plan_report", lambda *a, **k: None)
    ep = plan.plan(topics, date="2026-08-21")
    # fallback: one standalone group holding all topics
    assert len(ep.editorial_groups) == 1
    assert ep.editorial_groups[0].role == "standalone"
    assert "t1" in ep.editorial_groups[0].topic_ids


# --- 07: brief links -----------------------------------------------------

def test_brief_has_link_per_member():
    from pipeline import generate
    plan_ep = EpisodePlan(date="2026-08-21", target_minutes=30, hook="h", motif="m",
                          segments=[Segment(topic_id="t1", minutes=2.0, opening="o",
                                            signposts=["s"], transition_out="t",
                                            speakers=["A", "B"])],
                          outro="e")
    topics = [Topic(id="t1", title="T", why="w", members=[
        RankedItem(title="p", url="https://arxiv.org/abs/2601.00001", date="d",
                   body="b", source="arxiv", score=0.9),
        RankedItem(title="h", url="https://hn.example/x", date="d", body="b",
                   source="hn", score=0.8)],
        primary_url="https://arxiv.org/abs/2601.00001", aggregate_score=0.9)]
    lines = generate._brief_lines(plan_ep, {t.id: t for t in topics})
    text = "\n".join(lines)
    assert "https://arxiv.org/abs/2601.00001" in text
    assert "https://hn.example/x" in text
    assert "https://arxiv.org/pdf/2601.00001" in text   # PDF link added
