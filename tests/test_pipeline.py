"""Contract tests for the lean pipeline (collect -> rank -> generate).

No network, no LLM. The judge is monkeypatched; store is isolated to a tmp dir.
"""
import json, pathlib
import pytest

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipeline import Item, RankedItem
from pipeline import rank, store, generate, profile as profile_mod


# --- Profile ---------------------------------------------------------------

def test_profile_parses_frontmatter(tmp_path):
    p = tmp_path / "profile.md"
    p.write_text(
        "---\n"
        "knowledge_level: undergrad\n"
        "pace: brief\n"
        "format: brief\n"
        "target_length: short\n"
        "---\n",
        encoding="utf-8",
    )
    prof = profile_mod.load_profile(p)
    assert prof.knowledge_level == "undergrad"
    assert prof.pace == "brief"
    assert prof.format == "brief"
    assert prof.target_length == "short"


def test_profile_style_defaults_when_absent(tmp_path):
    p = tmp_path / "profile.md"
    p.write_text("---\n---\n")
    prof = profile_mod.load_profile(p)
    assert prof.knowledge_level == "researcher"
    assert prof.pace == "deep_dive"
    assert prof.format == "deep_dive"
    assert prof.target_length == "default"


def test_profile_rejects_invalid_style_falls_back(tmp_path):
    p = tmp_path / "profile.md"
    p.write_text(
        "---\n"
        "pace: bogus\n"
        "knowledge_level: expert\n"
        "---\n",
        encoding="utf-8",
    )
    prof = profile_mod.load_profile(p)
    assert prof.pace == "deep_dive"               # fallback to default
    assert prof.knowledge_level == "researcher"  # 'expert' not valid


def test_profile_missing_returns_defaults(tmp_path):
    prof = profile_mod.load_profile(tmp_path / "nonexistent.md")
    assert prof.pace == "deep_dive"
    assert prof.knowledge_level == "researcher"


# --- RankedItem -----------------------------------------------------------

def test_rankeditem_round_trips():
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
    legacy = {"title": "t", "url": "u", "date": "d", "body": "b",
              "source": "arxiv", "score": 0.8, "judge_reason": "r"}
    r = RankedItem(**legacy)
    assert r.personal_score == 0.0
    assert r.alpha == 1.0
    assert r.final_score == 0.0


# --- rank writes the full scored pool, sorted -----------------------------

def test_rank_returns_full_pool_sorted(tmp_path, monkeypatch):
    items = [
        Item(title=f"t{i}", url=f"u{i}", date="2026-08-21", body="b", source="arxiv")
        for i in range(3)
    ]
    rank._judge_batch = lambda groups, rubric: [(0.9 - i * 0.1, "r") for i in range(len(groups))]
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)
    out = rank.rank(items)
    assert len(out) == 3
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= r.score <= 1.0 for r in out)
    # final_score == score (pure importance, no personal pass)
    assert all(r.final_score == r.score for r in out)


def test_rank_from_cache_sets_final_score(tmp_path, monkeypatch):
    cache = tmp_path / "rank.json"
    cache.write_text(json.dumps([
        {"title": "a", "url": "u1", "date": "d", "body": "b", "source": "arxiv",
         "score": 0.8, "judge_reason": "imp",
         "personal_score": 0.0, "personal_reason": "", "alpha": 1.0, "final_score": 0.0},
        {"title": "b", "url": "u2", "date": "d", "body": "b", "source": "arxiv",
         "score": 0.5, "judge_reason": "imp",
         "personal_score": 0.0, "personal_reason": "", "alpha": 1.0, "final_score": 0.0},
    ]))
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)
    out = rank.rank_from_cache(str(cache))
    assert len(out) == 2
    assert out[0].final_score == 0.8
    assert out[1].final_score == 0.5
    assert out[0].url == "u1"  # higher score first


# --- store.latest_dir date-aware sort --------------------------------------

def test_latest_dir_sorts_by_date_not_lexicographically(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    (root / "31-12-2026").mkdir()
    (root / "02-01-2027").mkdir()
    (root / "not-a-date").mkdir()
    monkeypatch.setattr(store, "ROOT", root)
    latest = store.latest_dir()
    assert latest.name == "02-01-2027"


def test_latest_dir_raises_when_no_run_folders(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(store, "ROOT", root)
    with pytest.raises(SystemExit):
        store.latest_dir()


# --- generate: source selection by threshold + floor/cap ----------------

def test_select_sources_threshold_within_range():
    # 7 items >= 0.8, deep_dive floor=5 cap=10 -> all 7 qualify.
    ranked = [
        RankedItem(title=f"t{i}", url=f"u{i}", date="d", body="b", source="arxiv",
                   score=0.88 - i * 0.01, final_score=0.88 - i * 0.01)
        for i in range(7)
    ] + [
        RankedItem(title=f"low{i}", url=f"l{i}", date="d", body="b", source="arxiv",
                   score=0.5, final_score=0.5)
        for i in range(5)
    ]
    chosen = generate.select_sources(ranked, "deep_dive")
    assert len(chosen) == 7
    assert all(c.final_score >= 0.8 for c in chosen)


def test_select_sources_cap_on_strong_week():
    # 15 items >= 0.8, deep_dive cap=10 -> cut to 10 (top-scored).
    ranked = [
        RankedItem(title=f"t{i}", url=f"u{i}", date="d", body="b", source="arxiv",
                   score=0.95 - i * 0.01, final_score=0.95 - i * 0.01)
        for i in range(15)
    ] + [
        RankedItem(title=f"low{i}", url=f"l{i}", date="d", body="b", source="arxiv",
                   score=0.5, final_score=0.5)
        for i in range(5)
    ]
    chosen = generate.select_sources(ranked, "deep_dive")
    assert len(chosen) == 10
    assert chosen[0].final_score == 0.95
    assert chosen[-1].final_score == 0.86


def test_select_sources_floor_on_weak_week():
    # Only 2 items >= 0.8, deep_dive floor=5 -> pad to 5 with next-highest.
    ranked = [
        RankedItem(title="a", url="u1", date="d", body="b", source="arxiv",
                   score=0.9, final_score=0.9),
        RankedItem(title="b", url="u2", date="d", body="b", source="arxiv",
                   score=0.82, final_score=0.82),
        RankedItem(title="c", url="u3", date="d", body="b", source="arxiv",
                   score=0.6, final_score=0.6),
        RankedItem(title="d", url="u4", date="d", body="b", source="arxiv",
                   score=0.5, final_score=0.5),
        RankedItem(title="e", url="u5", date="d", body="b", source="arxiv",
                   score=0.4, final_score=0.4),
    ]
    chosen = generate.select_sources(ranked, "deep_dive")
    assert len(chosen) == 5
    assert chosen[0].final_score == 0.9
    assert chosen[-1].final_score == 0.4


def test_select_sources_brief_has_larger_cap():
    # 15 items >= 0.8, brief cap=20 -> all 15 qualify (within floor=10, cap=20).
    ranked = [
        RankedItem(title=f"t{i}", url=f"u{i}", date="d", body="b", source="arxiv",
                   score=0.85 - i * 0.001, final_score=0.85 - i * 0.001)
        for i in range(15)
    ] + [
        RankedItem(title=f"low{i}", url=f"l{i}", date="d", body="b", source="arxiv",
                   score=0.5, final_score=0.5)
        for i in range(5)
    ]
    chosen = generate.select_sources(ranked, "brief")
    assert len(chosen) == 15
    assert all(c.final_score >= 0.8 for c in chosen)


def test_select_sources_brief_floor():
    # Only 3 items >= 0.8, brief floor=10 -> pad to 10.
    ranked = [
        RankedItem(title=f"hi{i}", url=f"h{i}", date="d", body="b", source="arxiv",
                   score=0.85, final_score=0.85)
        for i in range(3)
    ] + [
        RankedItem(title=f"mid{i}", url=f"m{i}", date="d", body="b", source="arxiv",
                   score=0.5, final_score=0.5)
        for i in range(20)
    ]
    chosen = generate.select_sources(ranked, "brief")
    assert len(chosen) == 10
    assert chosen[0].final_score == 0.85


def test_select_sources_empty_pool():
    assert generate.select_sources([], "deep_dive") == []


def test_select_sources_small_pool_below_floor():
    # Only 3 items total, all below 0.8; floor=5 but pool=3 -> return all 3.
    ranked = [
        RankedItem(title=f"t{i}", url=f"u{i}", date="d", body="b", source="arxiv",
                   score=0.5, final_score=0.5)
        for i in range(3)
    ]
    chosen = generate.select_sources(ranked, "deep_dive")
    assert len(chosen) == 3


# --- generate: instruction templating --------------------------------------

def test_generate_instructions_templates_profile():
    prof = profile_mod.Profile(
        knowledge_level="undergrad", pace="brief",
    )
    chosen = [RankedItem(title="Paper A", url="u1", date="d", body="b",
                         source="arxiv", score=0.9, final_score=0.9)]
    text = generate._instructions(prof, chosen)
    # knowledge_level: undergrad -> explains jargon
    assert "attention mechanism in one sentence" in text
    # pace: brief -> cover more topics briefly
    assert "briefly" in text
    # fixed intro: overview with example
    assert "overview" in text
    assert "In today's" in text
    # fixed outro: summary
    assert "summary" in text
    # fixed transition: pause or explicit sentence
    assert "pause" in text
    assert "Now let's jump to the next topic" in text
    # item title included
    assert "Paper A" in text


def test_generate_instructions_researcher_skips_basics():
    prof = profile_mod.Profile(knowledge_level="researcher", pace="deep_dive")
    chosen = []
    text = generate._instructions(prof, chosen)
    assert "no need to define" in text
    assert "go straight to the substance" in text
    assert "Depth over breadth" in text


def test_generate_brief_groups_by_source(tmp_path, monkeypatch):
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
    assert "https://arxiv.org/pdf/2601.00001" in text


def test_generate_writes_episode(tmp_path, monkeypatch):
    # 3 items >= 0.8, rest below. deep_dive floor=5 cap=10 -> 3 qualify,
    # padded to floor=5.
    ranked = [
        RankedItem(title=f"top{i}", url=f"u{i}", date="d", body="b", source="arxiv",
                   score=0.9, final_score=0.9)
        for i in range(3)
    ] + [
        RankedItem(title=f"low{i}", url=f"l{i}", date="d", body="b", source="arxiv",
                   score=0.5, final_score=0.5)
        for i in range(10)
    ]
    prof = profile_mod.Profile(pace="deep_dive")
    monkeypatch.setattr(store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(generate.store, "run_dir", lambda date=None: tmp_path)
    ep = generate.generate(ranked, make_audio=False, profile=prof)
    assert len(ep.manifest) == 5
    assert ep.manifest[0].final_score == 0.9
    assert ep.manifest[-1].final_score == 0.5
