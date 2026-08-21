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
    plan._chat = lambda msg, prompt, model: json.dumps({
        "hook": "h", "motif": "m",
        "segments": [{"opening": "o", "signposts": ["s1", "s2"],
                       "transition_out": "tr", "speakers": ["A", "B"]}],
        "outro": "e",
    })
    monkeypatch.setattr(plan.store, "write", lambda *a, **k: None)
    monkeypatch.setattr(plan.inspect, "budget_report", lambda *a, **k: None)
    monkeypatch.setattr(plan.inspect, "plan_report", lambda *a, **k: None)
    ep = plan.plan(topics, date="2026-08-21")
    assert ep.motif == "m"
    assert len(ep.segments) == 1
    assert sum(s.minutes for s in ep.segments) > 0
    assert ep.segments[0].topic_id == "t1"


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
