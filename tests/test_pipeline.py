"""Contract tests for the lean pipeline (collect -> rank -> generate).

No network, no LLM. The judge is monkeypatched; store is isolated to a tmp dir.
"""
import json, pathlib
import pytest

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipeline import Item, RankedItem
from pipeline import rank, store, profile as profile_mod, feedback as feedback_mod


# --- Profile ---------------------------------------------------------------

def test_profile_parses_frontmatter(tmp_path):
    p = tmp_path / "profile.md"
    p.write_text(
        "---\n"
        "topics:\n"
        "  - agent evals\n"
        "  - inference cost\n"
        "anti_topics:\n"
        "  - pure scaling\n"
        "knowledge_level: expert\n"
        "tone: conversational\n"
        "format: brief\n"
        "target_length: short\n"
        "top_n: 15\n"
        "---\n"
        "## What I'm working on\n"
        "Agent reliability this quarter.\n",
        encoding="utf-8",
    )
    prof = profile_mod.load_profile(p)
    assert prof.topics == ["agent evals", "inference cost"]
    assert prof.anti_topics == ["pure scaling"]
    assert "Agent reliability" in prof.body
    assert prof.knowledge_level == "expert"
    assert prof.tone == "conversational"
    assert prof.format == "brief"
    assert prof.target_length == "short"
    assert prof.top_n == 15


def test_profile_style_defaults_when_absent(tmp_path):
    p = tmp_path / "profile.md"
    p.write_text("---\ntopics:\n  - x\n---\nbody\n")
    prof = profile_mod.load_profile(p)
    assert prof.knowledge_level == "researcher"
    assert prof.tone == "dense"
    assert prof.format == "deep_dive"
    assert prof.target_length == "default"
    assert prof.intro_style == "theme-first"
    assert prof.outro_style == "links"
    assert prof.transition_style == "bridge"
    assert prof.top_n == profile_mod.DEFAULT_TOP_N


def test_profile_rejects_invalid_style_falls_back(tmp_path):
    p = tmp_path / "profile.md"
    p.write_text(
        "---\n"
        "topics:\n  - x\n"
        "tone: bogus\n"
        "format: not-a-format\n"
        "top_n: not-a-number\n"
        "---\nbody\n",
        encoding="utf-8",
    )
    prof = profile_mod.load_profile(p)
    assert prof.tone == "dense"               # fallback to default
    assert prof.format == "deep_dive"
    assert prof.top_n == profile_mod.DEFAULT_TOP_N


def test_profile_top_n_clamped_to_range(tmp_path):
    p = tmp_path / "profile.md"
    p.write_text("---\ntopics:\n  - x\ntop_n: 999\n---\nbody\n")
    prof = profile_mod.load_profile(p)
    assert prof.top_n == 20                    # out-of-range falls back to default


def test_profile_missing_returns_empty(tmp_path):
    prof = profile_mod.load_profile(tmp_path / "nonexistent.md")
    assert prof.topics == []
    assert prof.anti_topics == []
    assert prof.body == ""
    assert prof.format == "deep_dive"


# --- RankedItem -----------------------------------------------------------

def test_rankeditem_round_trips_personal_fields():
    r = RankedItem(title="t", url="u", date="d", body="b", source="arxiv",
                   score=0.8, judge_reason="imp",
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
              "source": "arxiv", "score": 0.8, "judge_reason": "r"}
    r = RankedItem(**legacy)
    assert r.personal_score == 0.0
    assert r.alpha == 1.0
    assert r.final_score == 0.0


# --- Feedback log ----------------------------------------------------------

def test_feedback_mark_and_aggregate(tmp_path, monkeypatch):
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
    payload = {"runs": [
        {"run": "01-01-2026", "marks": [{"url": "old", "kept": True}]},
        {"run": "02-01-2026", "marks": [{"url": "recent1", "kept": True}]},
        {"run": "03-01-2026", "marks": [{"url": "recent2", "kept": False}]},
    ]}
    (tmp_path / "feedback_log.json").write_text(json.dumps(payload))
    log = feedback_mod.load_feedback()
    assert "old" not in log.kept_urls
    assert log.kept_urls == {"recent1"}
    assert log.skipped_urls == {"recent2"}


def test_feedback_missing_log_returns_empty(tmp_path):
    log = feedback_mod.load_feedback(tmp_path / "nonexistent.json")
    assert log.kept_urls == set()
    assert log.skipped_urls == set()


# --- Personal pass + blend -------------------------------------------------

def test_personal_pass_noop_without_profile(tmp_path, monkeypatch):
    items = [Item(title=f"t{i}", url=f"u{i}", date="d", body="b", source="arxiv")
             for i in range(3)]
    rank._judge_batch = lambda groups, rubric: [(0.9 - i * 0.1, "r") for i in range(len(groups))]
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(rank.profile_mod, "PROFILE_PATH", tmp_path / "nonexistent.md")
    out = rank.rank(items)
    for r in out:
        assert r.final_score == r.score
        assert r.personal_score == 0.0


def test_personal_pass_noop_when_alpha_is_one(tmp_path, monkeypatch):
    p = tmp_path / "profile.md"
    p.write_text("---\ntopics:\n  - x\n---\nbody\n")
    items = [Item(title=f"t{i}", url=f"u{i}", date="d", body="b", source="arxiv")
             for i in range(3)]
    rank._judge_batch = lambda groups, rubric: [(0.9 - i * 0.1, "r") for i in range(len(groups))]
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(rank, "ALPHA", 1.0)
    monkeypatch.setattr(rank.profile_mod, "PROFILE_PATH", p)
    out = rank.rank(items)
    for r in out:
        assert r.final_score == r.score


def test_blend_formula(tmp_path, monkeypatch):
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
    out = rank.rank(items)
    assert out[0].score == 0.8
    assert out[0].personal_score == 0.6
    assert out[0].alpha == 0.7
    assert abs(out[0].final_score - 0.74) < 1e-6


def test_blend_pure_personal(tmp_path, monkeypatch):
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
    out = rank.rank(items)
    assert abs(out[0].final_score - 0.9) < 1e-6


# --- --from-cache ---------------------------------------------------------

def test_rank_from_cache_applies_personal(tmp_path, monkeypatch):
    p = tmp_path / "profile.md"
    p.write_text("---\ntopics:\n  - comics\n---\nbody\n")
    cache = tmp_path / "rank.json"
    cache.write_text(json.dumps([
        {"title": "a", "url": "u1", "date": "d", "body": "b", "source": "arxiv",
         "score": 0.8, "judge_reason": "imp",
         "personal_score": 0.0, "personal_reason": "", "alpha": 1.0, "final_score": 0.0},
        {"title": "b", "url": "u2", "date": "d", "body": "b", "source": "arxiv",
         "score": 0.5, "judge_reason": "imp",
         "personal_score": 0.0, "personal_reason": "", "alpha": 1.0, "final_score": 0.0},
    ]))
    def fake_judge(groups, rubric):
        return [(0.9 if i == 0 else 0.3, "match") for i in range(len(groups))]
    rank._judge_batch = fake_judge
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(rank, "ALPHA", 0.5)
    monkeypatch.setattr(rank.profile_mod, "PROFILE_PATH", p)
    monkeypatch.setattr(rank.feedback_mod, "FEEDBACK_LOG_PATH", tmp_path / "fb.json")
    out = rank.rank_from_cache(str(cache))
    assert len(out) == 2
    assert abs(out[0].final_score - 0.85) < 1e-6
    assert abs(out[1].final_score - 0.4) < 1e-6
    assert out[0].personal_score == 0.9
    assert out[0].alpha == 0.5


# --- rank writes the full scored pool, sorted -----------------------------

def test_rank_returns_full_pool_sorted(tmp_path, monkeypatch):
    items = [
        Item(title=f"t{i}", url=f"u{i}", date="2026-08-21", body="b", source="arxiv")
        for i in range(3)
    ]
    rank._judge_batch = lambda groups, rubric: [(0.9 - i * 0.1, "r") for i in range(len(groups))]
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(rank.profile_mod, "PROFILE_PATH", tmp_path / "nonexistent.md")
    out = rank.rank(items)
    assert len(out) == 3
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= r.score <= 1.0 for r in out)


# --- store.latest_dir date-aware sort --------------------------------------

def test_latest_dir_sorts_by_date_not_lexicographically(tmp_path, monkeypatch):
    # cross-year: 31-12-2026 is earlier than 02-01-2027 but sorts later
    # lexicographically; the date-aware sort must pick 02-01-2027.
    root = tmp_path / "data"
    root.mkdir()
    (root / "31-12-2026").mkdir()
    (root / "02-01-2027").mkdir()
    (root / "not-a-date").mkdir()  # must be skipped
    monkeypatch.setattr(store, "ROOT", root)
    latest = store.latest_dir()
    assert latest.name == "02-01-2027"


def test_latest_dir_raises_when_no_run_folders(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(store, "ROOT", root)
    with pytest.raises(SystemExit):
        store.latest_dir()


# --- generate instruction templating --------------------------------------

def test_generate_instructions_templates_profile():
    from pipeline import generate
    prof = profile_mod.Profile(
        tone="casual", knowledge_level="undergrad",
        intro_style="biggest-story", outro_style="teaser",
        transition_style="next",
    )
    chosen = [RankedItem(title="Paper A", url="u1", date="d", body="b",
                         source="arxiv", score=0.9, final_score=0.9)]
    text = generate._instructions(prof, chosen)
    assert "Casual" in text and "undergraduate" in text
    assert "biggest story" in text
    assert "teaser" in text
    assert "Next:" in text
    assert "Paper A" in text


def test_generate_brief_groups_by_source(tmp_path, monkeypatch):
    from pipeline import generate
    chosen = [
        RankedItem(title="Paper A", url="https://arxiv.org/abs/2601.00001",
                   date="d", body="abstract", source="arxiv",
                   score=0.9, final_score=0.9),
        RankedItem(title="HN Story", url="https://hn.example/x",
                   date="d", body="body", source="hn",
                   score=0.8, final_score=0.8),
    ]
    by_source = generate._group_by_source(chosen)
    assert {s: [i.title for i in items] for s, items in by_source.items()} == {
        "arxiv": ["Paper A"],
        "hn": ["HN Story"],
    }
    text = generate._brief_text(chosen, by_source, profile_mod.Profile())
    assert "arXiv papers" in text
    assert "Hacker News stories" in text
    assert "https://arxiv.org/pdf/2601.00001" in text   # PDF link added


def test_generate_picks_top_n_by_final_score(tmp_path, monkeypatch):
    from pipeline import generate
    ranked = [
        RankedItem(title=f"t{i}", url=f"u{i}", date="d", body="b", source="arxiv",
                   score=0.5, final_score=1.0 - i * 0.1)
        for i in range(5)
    ]
    prof = profile_mod.Profile(top_n=3)
    monkeypatch.setattr(store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(generate.store, "run_dir", lambda date=None: tmp_path)
    ep = generate.generate(ranked, make_audio=False, profile=prof)
    assert len(ep.manifest) == 3
    assert ep.manifest[0].title == "t0"
    assert ep.manifest[2].title == "t2"
