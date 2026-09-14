"""Contract tests for the lean pipeline (collect -> rank -> generate).

No network, no LLM. The judge is monkeypatched; store is isolated to a tmp dir.
"""
import json, math, pathlib, re, sys, types
import pytest

_FILE = pathlib.Path(__file__).resolve()
# pipeline/service live in this app folder (parents[1]); podcast_engine is
# shared in podcast-engine/ at the repo root (parents[2]).
sys.path.insert(0, str(_FILE.parents[1]))
sys.path.insert(0, str(_FILE.parents[2]))
sys.path.insert(0, str(_FILE.parents[2] / "podcast-engine"))

from pipeline import Item, RankedItem
from pipeline import rank, store, generate, transcribe
from pipeline import config as config_mod
from pipeline import topics
from pipeline import label as label_mod
from pipeline import memory as memory_mod
from podcast_engine import EpisodeResult


# --- RankedItem -----------------------------------------------------------

def test_rankeditem_round_trips():
    r = RankedItem(title="t", url="u", date="d", body="b", source="arxiv",
                   score=0.8, judge_reason="imp")
    d = r.__dict__
    r2 = RankedItem(**d)
    assert r2.score == 0.8
    assert r2.judge_reason == "imp"


def test_rankeditem_legacy_load_ignores_unknown_keys():
    # Old caches may carry fields the schema has never had (alpha) — the
    # tolerant loaders filter to known dataclass fields. personal_score /
    # final_score / topics are legitimate fields again (steering), so they
    # load with their stored values.
    legacy = {"title": "t", "url": "u", "date": "d", "body": "b",
              "source": "arxiv", "score": 0.8, "judge_reason": "r",
              "personal_score": 0.6, "alpha": 0.7, "final_score": 0.74}
    import dataclasses
    known = {f.name for f in dataclasses.fields(RankedItem)}
    r = RankedItem(**{k: v for k, v in legacy.items() if k in known})
    assert r.score == 0.8
    assert r.personal_score == 0.6
    assert r.final_score == 0.74
    assert not hasattr(r, "alpha")


# --- rank writes the full scored pool, sorted -----------------------------

def test_rank_returns_full_pool_sorted(tmp_path, monkeypatch):
    items = [
        Item(title=f"t{i}", url=f"u{i}", date="2026-08-21", body="b", source="arxiv")
        for i in range(3)
    ]
    monkeypatch.setattr(
        rank, "_judge_batch",
        lambda groups, rubric: [(0.9 - i * 0.1, "r", None)
                                for i in range(len(groups))])
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)
    out = rank.rank(items)
    assert len(out) == 3
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= r.score <= 1.0 for r in out)


def test_rank_from_cache_loads_legacy_keys(tmp_path, monkeypatch):
    # A saved rank.json with removed Phase-2 fields still loads via the
    # tolerant key filter; items sort by score.
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
    assert out[0].score == 0.8
    assert out[1].score == 0.5
    assert out[0].url == "u1"  # higher score first


# --- store.latest_dir date-aware sort --------------------------------------

def test_latest_dir_sorts_by_date_not_lexicographically(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    (root / "31-12-2026").mkdir()
    (root / "02-01-2027").mkdir()
    # a time-stamped run on the same day is newer than the bare-date folder
    (root / "02-01-2027-235959").mkdir()
    (root / "not-a-date").mkdir()
    monkeypatch.setattr(store, "ROOT", root)
    latest = store.latest_dir()
    assert latest.name == "02-01-2027-235959"


def test_store_run_dir_accepts_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    d = store.run_dir("31-08-2026-154500")
    assert d == tmp_path / "31-08-2026-154500"
    assert d.is_dir()
    # ISO still normalizes to the plain date folder.
    d2 = store.run_dir("2026-08-31")
    assert d2 == tmp_path / "31-08-2026"


def test_latest_dir_raises_when_no_run_folders(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(store, "ROOT", root)
    with pytest.raises(SystemExit):
        store.latest_dir()


def test_store_root_from_env(tmp_path, monkeypatch):
    """PIPELINE_DATA_ROOT redirects the pipeline's data root, so a run can
    write into a different history tree (the service points it at
    data/history). Read at module import time (per-subprocess)."""
    import importlib, os
    monkeypatch.setenv("PIPELINE_DATA_ROOT", str(tmp_path))
    # store reads the env at import; reload so the new value takes effect.
    import pipeline.store as s
    importlib.reload(s)
    try:
        assert s.ROOT == tmp_path
        d = s.run_dir("2026-08-31")
        assert d == tmp_path / "31-08-2026"
        assert d.is_dir()
    finally:
        # Restore the module so other tests aren't affected.
        monkeypatch.delenv("PIPELINE_DATA_ROOT", raising=False)
        importlib.reload(s)


# --- collect: source registry + data-driven cross-source dedup --------------

def test_collect_source_registry():
    """The collect stage is driven by a SOURCES registry; adding a source is
    registering a collector (plus a label + dedup rule), not editing branches."""
    from pipeline import collect
    assert list(collect.SOURCES) == ["hn", "arxiv"]
    assert all(callable(fn) for fn in collect.SOURCES.values())


def test_collect_joins_all_source_branches(tmp_path, monkeypatch):
    """collect() runs every registered branch and joins their items — the join
    must key its futures dict by source name (regression: a flipped key/value
    caused KeyError: 'hn' the moment any branch finished)."""
    from pipeline import collect
    import json

    def fake_hn(date, start):
        return [Item(title="hn-t", url="https://h.example/x", date="2026-09-03",
                     body="body", source="hn")]

    def fake_arxiv(date, start):
        return [Item(title="ax-t", url="https://arxiv.org/abs/2601.00002",
                     date="2026-09-03", body="abs", source="arxiv")]

    monkeypatch.setattr(collect, "SOURCES",
                        {"hn": fake_hn, "arxiv": fake_arxiv})
    monkeypatch.setattr(collect.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(collect.store, "write",
                        lambda name, payload, date=None: (tmp_path / name)
                        .write_text(json.dumps(payload, indent=2)))

    out = collect.collect("2026-09-04", window_start="2026-09-03")
    assert [i.source for i in out] == ["hn", "arxiv"]  # registry order
    assert json.loads((tmp_path / "collect.json").read_text())[0]["title"] == "hn-t"


def test_collect_degrades_when_a_source_fails(tmp_path, monkeypatch):
    """A source branch that raises (e.g. arXiv HTTP 429 throttling) is dropped
    with a warning; collect continues with the healthy sources instead of
    failing the whole run."""
    from pipeline import collect
    import json

    def fake_hn(date, start):
        return [Item(title="hn-t", url="https://h.example/x", date="2026-09-03",
                     body="body", source="hn")]

    def fake_arxiv(date, start):
        raise RuntimeError("HTTP 429 throttled")

    monkeypatch.setattr(collect, "SOURCES", {"hn": fake_hn, "arxiv": fake_arxiv})
    monkeypatch.setattr(collect.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(collect.store, "write",
                        lambda name, payload, date=None: (tmp_path / name)
                        .write_text(json.dumps(payload, indent=2)))

    out = collect.collect("2026-09-04", window_start="2026-09-03")
    assert [i.source for i in out] == ["hn"]  # the healthy branch survives
    assert json.loads((tmp_path / "collect.json").read_text())[0]["title"] == "hn-t"


def test_collect_raises_when_every_source_fails(tmp_path, monkeypatch):
    from pipeline import collect

    def boom(date, start):
        raise RuntimeError("all sources down")

    monkeypatch.setattr(collect, "SOURCES", {"hn": boom, "arxiv": boom})
    monkeypatch.setattr(collect.store, "run_dir", lambda date=None: tmp_path)
    # no source produced anything -> the empty-file guard still aborts
    with pytest.raises(SystemExit):
        collect.collect("2026-09-04")


def test_dedup_rules_hn_arxiv_twin_dropped():
    """An HN item pointing at an arXiv paper we already collected is dropped
    (the arXiv entry carries the full abstract)."""
    from pipeline import collect
    items = [
        Item(title="paper", url="https://arxiv.org/abs/2601.12345", date="d",
             body="abstract", source="arxiv"),
        Item(title="HN link", url="https://arxiv.org/abs/2601.12345", date="d",
             body="", source="hn"),
        Item(title="keep", url="https://other.example/x", date="d", body="",
             source="hn"),
    ]
    out = collect._dedup(items, collect.DEDUP_RULES)
    assert [i.source + ":" + i.title for i in out] == [
        "arxiv:paper", "hn:keep"]


def test_hn_emits_points_for_rank_signal(monkeypatch):
    """HN items carry their points (the popularity signal) through collect so
    the rank judge can weigh community attention — points are no longer
    discarded at selection time."""
    from pipeline import collect

    def fake_get_json(url):
        return {"hits": [
            {"objectID": "1", "title": "big story", "url": "https://a.example",
             "created_at": "2026-09-03T12:00:00Z", "points": 412},
            {"objectID": "2", "title": "smaller", "url": "https://b.example",
             "created_at": "2026-09-03T13:00:00Z", "points": 135},
        ], "nbPages": 1}

    monkeypatch.setattr(collect, "_get_json", fake_get_json)
    import datetime
    items = collect._hn("2026-09-04", datetime.date(2026, 9, 3))
    assert [i.hn_points for i in items] == [412, 135]


def test_dedup_carries_hn_points_to_winner():
    """Cross-source dedup DROPS the HN twin (it adds no content) but folds the
    twin's upvotes onto the surviving arXiv item so rank sees the HN signal."""
    from pipeline import collect
    items = [
        Item(title="paper", url="https://arxiv.org/abs/2601.12345", date="d",
             body="abstract", source="arxiv"),
        Item(title="HN link", url="https://arxiv.org/abs/2601.12345", date="d",
             body="", source="hn", hn_points=340),
    ]
    out = collect._dedup(items, collect.DEDUP_RULES)
    assert [i.source for i in out] == ["arxiv"]
    assert out[0].hn_points == 340


def test_dedup_leaves_winner_untouched_without_hn_points():
    """An HN twin without points never fabricates a boost on the winner."""
    from pipeline import collect
    items = [
        Item(title="paper", url="https://arxiv.org/abs/2601.99999", date="d",
             body="abstract", source="arxiv"),
        Item(title="HN link", url="https://arxiv.org/abs/2601.99999", date="d",
             body="", source="hn"),   # no hn_points recorded
    ]
    out = collect._dedup(items, collect.DEDUP_RULES)
    assert [i.source for i in out] == ["arxiv"]
    assert out[0].hn_points == 0


def test_story_to_dict_carries_hn_upvotes():
    """The judge payload exposes hn_upvotes only when present, so the rubric's
    HN-attention rule can fire without changing every item's shape."""
    from pipeline import rank as rank_mod
    plain = Item(title="t", url="u", date="d", body="b", source="arxiv")
    assert "hn_upvotes" not in rank_mod._story_to_dict(plain, 0)
    covered = Item(title="t2", url="u2", date="d", body="b", source="hn",
                   hn_points=233)
    assert rank_mod._story_to_dict(covered, 1)["hn_upvotes"] == 233


def test_hn_window_is_full_inclusive(monkeypatch):
    """HN honors a full-inclusive [start, end] window: the end day is included
    (previously `created_at < end-midnight` silently excluded all of it) and
    items beyond the window are dropped. The fake emulates Algolia's server-side
    numeric filtering on the created_at bounds the query actually sends."""
    from pipeline import collect
    from urllib.parse import urlparse, parse_qs
    import datetime

    def ts(s):
        return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

    all_hits = [
        {"objectID": "1", "title": "at-start", "url": "https://a.example",
         "created_at": "2026-09-03T00:00:00Z", "points": 150},
        {"objectID": "2", "title": "on-end-day", "url": "https://b.example",
         "created_at": "2026-09-04T23:59:59Z", "points": 150},
        {"objectID": "3", "title": "beyond-end", "url": "https://c.example",
         "created_at": "2026-09-05T00:00:00Z", "points": 150},
    ]

    def fake_get_json(url):
        nf = parse_qs(urlparse(url).query)["numericFilters"][0]
        lo = hi = None
        for f in nf.split(","):
            if f.startswith("created_at_i>="):
                lo = int(f.split(">=")[1])
            elif f.startswith("created_at_i<"):
                hi = int(f.split("<")[1])
        assert lo is not None and hi is not None
        return {"hits": [h for h in all_hits
                         if ts(h["created_at"]) >= lo and ts(h["created_at"]) < hi],
                "nbPages": 1}

    monkeypatch.setattr(collect, "_get_json", fake_get_json)
    items = collect._hn("2026-09-04", datetime.date(2026, 9, 3))
    assert [i.title for i in items] == ["at-start", "on-end-day"]


def test_hn_single_day_window_nonempty(monkeypatch):
    """A one-day window (start == end) returns that day's stories instead of an
    empty range — the degenerate case the old `created_at_i<{day_ts}` made
    impossible."""
    from pipeline import collect

    def fake_get_json(url):
        return {"hits": [
            {"objectID": "1", "title": "yesterday-only", "url": "https://d.example",
             "created_at": "2026-09-03T12:00:00Z", "points": 150},
        ], "nbPages": 1}

    monkeypatch.setattr(collect, "_get_json", fake_get_json)
    import datetime
    items = collect._hn("2026-09-03", datetime.date(2026, 9, 3))
    assert [i.title for i in items] == ["yesterday-only"]


# --- generate: source selection by top-N above a quality floor ------------

def test_select_sources_top_n_default():
    # Default target (medium + deep-dive config) = 7. 15 items >= 0.8 + 5 lows
    # -> top 7 by score all clear the floor.
    ranked = [
        RankedItem(title=f"t{i}", url=f"u{i}", date="d", body="b", source="arxiv",
                   score=0.90 - i * 0.005)
        for i in range(15)
    ] + [
        RankedItem(title=f"low{i}", url=f"l{i}", date="d", body="b", source="arxiv",
                   score=0.5)
        for i in range(5)
    ]
    chosen = generate.select_sources(ranked)
    assert len(chosen) == 7
    assert chosen[0].score == 0.90
    assert chosen[-1].score == 0.90 - 6 * 0.005
    assert all(c.score >= generate.MIN_SCORE_FLOOR for c in chosen)


def test_select_sources_respects_target_n():
    # Explicitly requested top-4 out of a strong pool.
    ranked = [
        RankedItem(title=f"t{i}", url=f"u{i}", date="d", body="b", source="arxiv",
                   score=0.95 - i * 0.005)
        for i in range(25)
    ] + [
        RankedItem(title=f"low{i}", url=f"l{i}", date="d", body="b", source="arxiv",
                   score=0.5)
        for i in range(5)
    ]
    chosen = generate.select_sources(ranked, target_n=4)
    assert len(chosen) == 4
    assert chosen[0].score == 0.95
    assert chosen[-1].score == 0.95 - 3 * 0.005


def test_select_sources_no_padding_on_weak_week():
    # Only 2 items >= 0.8 plus a handful at/above the floor: top-N never pads
    # with sub-floor junk to reach the target.
    ranked = [
        RankedItem(title="a", url="u1", date="d", body="b", source="arxiv",
                   score=0.9),
        RankedItem(title="b", url="u2", date="d", body="b", source="arxiv",
                   score=0.82),
    ] + [
        RankedItem(title=f"low{i}", url=f"l{i}", date="d", body="b",
                   source="arxiv", score=0.5)
        for i in range(5)
    ] + [
        RankedItem(title=f"junk{i}", url=f"j{i}", date="d", body="b",
                   source="arxiv", score=0.3 - i * 0.05)
        for i in range(5)
    ]
    chosen = generate.select_sources(ranked, target_n=9)
    assert len(chosen) == 7  # 0.9, 0.82 + 5 at 0.5; sub-floor junk excluded
    assert all(c.score >= generate.MIN_SCORE_FLOOR for c in chosen)


def test_select_sources_empty_pool():
    assert generate.select_sources([]) == []


def test_select_sources_small_pool_all_above_floor():
    # Only 3 items total, all at the floor -> return all 3 (no pad, no cut).
    ranked = [
        RankedItem(title=f"t{i}", url=f"u{i}", date="d", body="b", source="arxiv",
                   score=0.5)
        for i in range(3)
    ]
    chosen = generate.select_sources(ranked, target_n=9)
    assert len(chosen) == 3


def test_generate_brief_groups_by_source(tmp_path, monkeypatch):
    chosen = [
        RankedItem(title="Paper A", url="https://arxiv.org/abs/2601.00001",
                   date="d", body="abstract", source="arxiv", score=0.9,
                   topics={"post_training": 1.0}),
        RankedItem(title="HN Story", url="https://hn.example/x",
                   date="d", body="body", source="hn", score=0.8),
    ]
    by_source = generate._group_by_source(chosen)
    assert {s: [i.title for i in items] for s, items in by_source.items()} == {
        "arxiv": ["Paper A"],
        "hn": ["HN Story"],
    }
    text = generate._brief_text(chosen, by_source)
    assert "arXiv papers" in text
    assert "Hacker News stories" in text
    assert "https://arxiv.org/pdf/2601.00001" in text
    # Source order is registry/label order: arXiv -> HN.
    assert text.index("arXiv papers") < text.index("Hacker News stories")
    # Labeled items carry the topic id right after the score; unlabeled don't.
    assert "— score 0.90 · post_training" in text
    assert "hn.example/x) — score 0.80\n" in text


def test_generate_writes_episode(tmp_path, monkeypatch):
    # 3 items at 0.9 + 20 at the floor. Default target (medium+deep) = 7:
    # top 7 by score above the 0.5 floor -> 3 x 0.9 + 4 x 0.5, no padding.
    ranked = [
        RankedItem(title=f"top{i}", url=f"u{i}", date="d", body="b", source="arxiv",
                   score=0.9)
        for i in range(3)
    ] + [
        RankedItem(title=f"low{i}", url=f"l{i}", date="d", body="b", source="arxiv",
                   score=0.5)
        for i in range(20)
    ]
    monkeypatch.setattr(store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(generate.store, "run_dir", lambda date=None: tmp_path)
    ep = generate.generate(ranked, make_audio=False)
    assert len(ep.manifest) == 7
    assert ep.manifest[0].score == 0.9
    assert ep.manifest[-1].score == 0.5


# --- audio result ----------------------------------------------------------

def test_audio_result_dataclass():
    from podcast_engine import audio_backend as audio
    r = audio.AudioResult(
        audio_path=pathlib.Path("a.mp3"),
        transcript_path=pathlib.Path("t.md"),
        backend="podcastfy",
    )
    assert r.backend == "podcastfy"
    assert r.transcript_path is not None


def test_part_audio_ok_rejects_garbled_oversized_part(tmp_path):
    """The part-audio sanity check accepts duration consistent with the part's
    word count and rejects a part that is many multiples longer — the garbled
    TTS signature seen in production (e.g. part_006: 442s for ~350 words)."""
    from podcast_engine import audio_backend as audio
    from pydub import AudioSegment

    # ~20 words with ~2s of audio: well within the word-derived ceiling.
    short = tmp_path / "short.mp3"
    AudioSegment.silent(duration=2000).export(str(short), format="mp3")
    assert audio.part_audio_ok(" ".join(["word"] * 20), short) is True

    # ~100 words but 200s of audio: far longer than ~45s expected -> reject.
    long_p = tmp_path / "garbled.mp3"
    AudioSegment.silent(duration=200_000).export(str(long_p), format="mp3")
    assert audio.part_audio_ok(" ".join(["word"] * 100), long_p) is False

    # Unreadable/missing audio is never accepted.
    assert audio.part_audio_ok("some words", tmp_path / "missing.mp3") is False


def test_podcastfy_backend_generate_writes_audio_and_transcript(tmp_path, monkeypatch):
    """PodcastfyBackend.generate drives SimplePodcastGenerator (mocked) and
    writes episode.mp3 + transcript.md into the run dir, returning a result
    with transcript_path set so the transcribe stage can skip Whisper."""
    from podcast_engine import audio_backend as audio

    # Fake SimplePodcastGenerator that avoids the langchain/openai deps.
    class FakeGenerator:
        def __init__(self, papers_dir=None, web_dir=None, **kw):
            self.papers_dir = papers_dir
            self.web_dir = web_dir
            self.tts_model = kw.get("tts_model", "tng")
        def build_combined_content(self, sources, intro_text=""):
            return "=== INTRO ===\n=== END INTRO ==="
        def generate_transcript(self, combined, on_part=None):
            return "<Person1>hello</Person1>\n<Person2>world</Person2>"
        def generate_audio(self, transcript, output_path, temp_audio_dir=None):
            pathlib.Path(output_path).write_bytes(b"FAKE_MP3")

    # Inject a fake podcast_engine.generator module so the lazy import
    # inside PodcastfyBackend.generate picks up FakeGenerator without needing
    # the langchain/openai optional deps installed.
    fake_mod = types.ModuleType("podcast_engine.generator")
    fake_mod.SimplePodcastGenerator = FakeGenerator
    monkeypatch.setitem(sys.modules, "podcast_engine.generator", fake_mod)

    chosen = [RankedItem(title="t", url="u", date="d", body="b", source="arxiv",
                         score=0.9)]

    result = audio.PodcastfyBackend().generate(
        run_dir=tmp_path, chosen=chosen, sources=chosen,
    )
    assert result.backend == "podcastfy"
    assert result.audio_path == tmp_path / "episode.mp3"
    assert result.audio_path.exists() and result.audio_path.read_bytes()
    assert result.transcript_path == tmp_path / "transcript.md"
    assert "<Person1>hello</Person1>" in result.transcript_path.read_text()
    # Per-run cache dirs were created under the run dir.
    assert (tmp_path / ".podcastfy-cache").is_dir()


# --- transcript cache: fingerprint-gated reuse (fix stale-transcript reuse) --

def test_transcript_fingerprint_stable_and_sensitive(tmp_path):
    from podcast_engine import audio_backend as audio

    chosen = [
        RankedItem(title="A", url="https://arxiv.org/abs/1", date="d", body="b",
                   source="arxiv", score=0.9),
        RankedItem(title="B", url="https://arxiv.org/abs/2", date="d", body="b",
                   source="arxiv", score=0.8),
    ]
    cfg = {"per_source_words": 330, "depth_factor": 1.6}
    fp1 = audio.transcript_fingerprint(chosen, cfg, sources=chosen)
    # Deterministic: same inputs -> same fingerprint (url order is sorted).
    assert fp1 == audio.transcript_fingerprint(list(reversed(chosen)), cfg,
                                               sources=list(reversed(chosen)))
    # Sensitive: selection change or config change invalidate.
    other = [chosen[0]]
    assert audio.transcript_fingerprint(other, cfg, sources=other) != fp1
    assert audio.transcript_fingerprint(chosen, {**cfg, "familiar_clause": "x"},
                                        sources=chosen) != fp1


def test_podcastfy_backend_reuses_transcript_on_fingerprint_match(tmp_path, monkeypatch):
    """A cached transcript.md is reused only when its stored fingerprint still
    matches the current selection/config (fast TTS-retry path)."""
    from podcast_engine import audio_backend as audio

    transcript_text = "<Person1>cached content</Person1>"
    (tmp_path / "transcript.md").write_text(transcript_text, encoding="utf-8")

    chosen = [RankedItem(title="A", url="https://arxiv.org/abs/1", date="d",
                         body="b", source="arxiv", score=0.9)]
    cfg = {"per_source_words": 330, "depth_factor": 1.6}

    class FakeGenerator:
        def __init__(self, papers_dir=None, web_dir=None, **kw):
            self.papers_dir = papers_dir
            self.web_dir = web_dir
            self.tts_model = kw.get("tts_model", "tng")
        def build_combined_content(self, sources, intro_text=""):
            raise AssertionError("LLM path must not run for a matching cache")
        def generate_transcript(self, combined, on_part=None):
            raise AssertionError("LLM path must not run for a matching cache")
        def generate_audio(self, transcript, output_path, temp_audio_dir=None):
            pathlib.Path(output_path).write_bytes(b"FAKE_MP3")

    fake_mod = types.ModuleType("podcast_engine.generator")
    fake_mod.SimplePodcastGenerator = FakeGenerator
    monkeypatch.setitem(sys.modules, "podcast_engine.generator", fake_mod)

    # Write the marker matching the current inputs (as a real run would).
    (tmp_path / ".podcastfy-cache").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".podcastfy-cache" / "transcript.fingerprint").write_text(
        audio.transcript_fingerprint(chosen, cfg, sources=chosen), encoding="utf-8")

    result = audio.PodcastfyBackend().generate(run_dir=tmp_path,
                                               chosen=chosen, sources=chosen,
                                               config=cfg)
    assert result.audio_path.exists()
    # Transcript is untouched: the cached one was reused, not regenerated.
    assert result.transcript_path.read_text(encoding="utf-8") == transcript_text


def test_podcastfy_backend_regenerates_when_transcript_stale(tmp_path, monkeypatch):
    """A transcript.md without (or with a stale) fingerprint is REGENERATED —
    e.g. a fresh full run on the same date must not reuse audio for the old
    selection."""
    from podcast_engine import audio_backend as audio

    (tmp_path / "transcript.md").write_text("<Person1>STALE nine-topic transcript</Person1>")

    chosen = [RankedItem(title="A", url="https://arxiv.org/abs/1", date="d",
                         body="b", source="arxiv", score=0.9)]
    cfg = {"per_source_words": 330, "depth_factor": 1.6}

    calls = {"transcripts": 0}

    class FakeGenerator:
        def __init__(self, papers_dir=None, web_dir=None, **kw):
            self.papers_dir = papers_dir
            self.web_dir = web_dir
            self.tts_model = kw.get("tts_model", "tng")
        def build_combined_content(self, sources, intro_text=""):
            return "=== INTRO ===\n=== END INTRO ==="
        def generate_transcript(self, combined, on_part=None):
            calls["transcripts"] += 1
            return "<Person1>fresh short transcript</Person1>"

    fake_mod = types.ModuleType("podcast_engine.generator")
    fake_mod.SimplePodcastGenerator = FakeGenerator
    monkeypatch.setitem(sys.modules, "podcast_engine.generator", fake_mod)

    result = audio.PodcastfyBackend().generate(run_dir=tmp_path,
                                               chosen=chosen, sources=chosen,
                                               config=cfg)
    assert calls["transcripts"] == 1
    assert result.transcript_path.read_text(encoding="utf-8") == "<Person1>fresh short transcript</Person1>"
    # And a fingerprint marker is persisted for the next run.
    marker = tmp_path / ".podcastfy-cache" / "transcript.fingerprint"
    assert marker.read_text(encoding="utf-8") == audio.transcript_fingerprint(chosen, cfg, sources=chosen)


def test_podcastfy_backend_failure_returns_empty_audio(tmp_path, monkeypatch):
    """On exception, the backend returns an empty audio_path and any partial
    transcript."""
    from podcast_engine import audio_backend as audio

    class FakeGenerator:
        def __init__(self, papers_dir=None, web_dir=None, **kw):
            pass
        def build_combined_content(self, sources, intro_text=""):
            raise RuntimeError("boom")

    fake_mod = types.ModuleType("podcast_engine.generator")
    fake_mod.SimplePodcastGenerator = FakeGenerator
    monkeypatch.setitem(sys.modules, "podcast_engine.generator", fake_mod)

    chosen = [RankedItem(title="t", url="u", date="d", body="b", source="arxiv",
                         score=0.9)]

    result = audio.PodcastfyBackend().generate(
        run_dir=tmp_path, chosen=chosen, sources=chosen,
    )
    assert str(result.audio_path) == "."
    assert result.backend == "podcastfy"


# --- generate: backend wiring + episode.json transcript_source ------------

def test_generate_writes_transcript_source_for_podcastfy(tmp_path, monkeypatch):
    """generate() threads the chosen backend through and records
    transcript_source in episode.json."""
    ranked = [
        RankedItem(title="t", url="u", date="d", body="b", source="arxiv",
                   score=0.9)
    ]
    monkeypatch.setattr(store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(generate.store, "run_dir", lambda date=None: tmp_path)

    (tmp_path / "transcript.md").write_text("<Person1>hi</Person1>", encoding="utf-8")
    fake_result = EpisodeResult(
        run_dir=tmp_path,
        audio_path=tmp_path / "episode.mp3",
        transcript_path=tmp_path / "transcript.md",
        backend="podcastfy",
    )
    monkeypatch.setattr(generate, "_generate_podcast", lambda **kw: fake_result)

    ep = generate.generate(ranked, make_audio=True)
    assert ep.audio_path == str(tmp_path / "episode.mp3")
    manifest = json.loads((tmp_path / "episode.json").read_text())
    assert manifest["transcript_source"] == "generated"
    assert manifest["backend"] == "podcastfy"


def test_generate_no_audio_records_no_backend(tmp_path, monkeypatch):
    ranked = [
        RankedItem(title="t", url="u", date="d", body="b", source="arxiv",
                   score=0.9)
    ]
    monkeypatch.setattr(store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(generate.store, "run_dir", lambda date=None: tmp_path)

    ep = generate.generate(ranked, make_audio=False)
    manifest = json.loads((tmp_path / "episode.json").read_text())
    assert manifest["transcript_source"] == "whisper"
    assert manifest["backend"] is None
    # No audio produced -> no duration to record.
    assert manifest["duration_sec"] is None
    assert manifest["transcript_words"] is None


def test_generate_brief_in_skips_selection_and_uses_edited_brief(tmp_path, monkeypatch):
    """With brief_in set, generate skips select_sources, does NOT overwrite
    the brief, and reconstructs the manifest by URL-matching against ranked."""
    ranked = [
        RankedItem(title="Paper A", url="https://arxiv.org/abs/2601.00001",
                   date="d", body="b", source="arxiv", score=0.9),
        RankedItem(title="Paper B", url="https://arxiv.org/abs/2601.00002",
                   date="d", body="b", source="arxiv", score=0.5),
    ]
    # An edited brief that drops Paper B.
    brief_path = tmp_path / "edited_brief.md"
    edited = ("# AI News Digest\n\n"
              "## arXiv papers (1)\n\n"
              "- [Paper A](https://arxiv.org/abs/2601.00001) · "
              "[PDF](https://arxiv.org/pdf/2601.00001) — score 0.90\n")
    brief_path.write_text(edited, encoding="utf-8")

    monkeypatch.setattr(store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(generate.store, "run_dir", lambda date=None: tmp_path)

    seen_kw = {}
    (tmp_path / "transcript.md").write_text("<Person1>hi</Person1>", encoding="utf-8")
    fake_result = EpisodeResult(
        run_dir=tmp_path,
        audio_path=tmp_path / "episode.mp3",
        transcript_path=tmp_path / "transcript.md",
        backend="podcastfy",
    )
    def _fake_gen(**kw):
        seen_kw.update(kw)
        return fake_result
    monkeypatch.setattr(generate, "_generate_podcast", _fake_gen)

    # The run-dir brief must NOT pre-exist (so we can prove we don't write it).
    assert not (tmp_path / "podcast_brief.md").exists()
    ep = generate.generate(ranked, make_audio=True, brief_in=str(brief_path))
    # The run-dir brief was written with the edited content.
    assert (tmp_path / "podcast_brief.md").read_text() == edited
    # Manifest reconstructed from the brief: only Paper A.
    assert [r.url for r in ep.manifest] == ["https://arxiv.org/abs/2601.00001"]
    manifest = json.loads((tmp_path / "episode.json").read_text())
    assert manifest["selection_source"] == "customized"
    assert len(manifest["manifest"]) == 1


def test_generate_no_brief_in_records_auto_selection_source(tmp_path, monkeypatch):
    ranked = [
        RankedItem(title="t", url="u", date="d", body="b", source="arxiv", score=0.9)
    ]
    monkeypatch.setattr(store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(generate.store, "run_dir", lambda date=None: tmp_path)
    (tmp_path / "transcript.md").write_text("<Person1>hi</Person1>", encoding="utf-8")
    fake_result = EpisodeResult(
        run_dir=tmp_path,
        audio_path=tmp_path / "episode.mp3",
        transcript_path=tmp_path / "transcript.md",
        backend="podcastfy",
    )
    monkeypatch.setattr(generate, "_generate_podcast", lambda **kw: fake_result)
    generate.generate(ranked, make_audio=True)
    manifest = json.loads((tmp_path / "episode.json").read_text())
    assert manifest["selection_source"] == "auto"


def test_generate_records_duration_and_words(tmp_path, monkeypatch):
    """With audio + transcript produced, episode.json records a measured
    duration (display telemetry) and the spoken-word count."""
    ranked = [
        RankedItem(title="t", url="u", date="d", body="b", source="arxiv", score=0.9)
    ]
    monkeypatch.setattr(store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(generate.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(generate, "_audio_duration_sec", lambda p: 754.5)

    (tmp_path / "episode.mp3").write_bytes(b"x")
    (tmp_path / "transcript.md").write_text(
        "<Person1>Hello world</Person1>\n<Person2>How are you today</Person2>\n",
        encoding="utf-8")
    fake_result = EpisodeResult(
        run_dir=tmp_path,
        audio_path=tmp_path / "episode.mp3",
        transcript_path=tmp_path / "transcript.md",
        backend="podcastfy",
    )
    monkeypatch.setattr(generate, "_generate_podcast", lambda **kw: fake_result)

    generate.generate(ranked, make_audio=True)
    manifest = json.loads((tmp_path / "episode.json").read_text())
    assert manifest["duration_sec"] == 754.5
    assert manifest["transcript_words"] == 6


def test_transcript_word_count():
    from pipeline.generate import _transcript_words
    assert _transcript_words("<Person1>Hello world</Person1>\n<Person2>How are you</Person2>") == 5
    assert _transcript_words("no tags at all") == 4
    assert _transcript_words("") == 0


# --- transcribe: no-op when transcript already generated ------------------

def test_transcribe_noop_when_transcript_generated(tmp_path, monkeypatch):
    """When episode.json has transcript_source=generated and transcript.md
    exists, transcribe returns it without importing faster-whisper."""
    (tmp_path / "episode.json").write_text(json.dumps({
        "audio_path": str(tmp_path / "episode.mp3"),
        "transcript_source": "generated",
        "backend": "podcastfy",
    }))
    (tmp_path / "transcript.md").write_text("<Person1>hi</Person1>")
    monkeypatch.setattr(store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(transcribe.store, "run_dir", lambda date=None: tmp_path)

    out = transcribe.transcribe(date="2026-08-31")
    assert out == tmp_path / "transcript.md"
    assert "hi" in out.read_text()


def test_transcribe_noop_missing_transcript_falls_through(tmp_path, monkeypatch):
    """When transcript_source != "generated", transcribe proceeds to load
    Whisper (mocked here) instead of short-circuiting."""
    # audio_path points to a real (empty) file so _resolve_audio is happy.
    audio_file = tmp_path / "episode.mp3"
    audio_file.write_bytes(b"")
    (tmp_path / "episode.json").write_text(json.dumps({
        "audio_path": str(audio_file),
        "transcript_source": "whisper",
        "backend": "podcastfy",
    }))
    monkeypatch.setattr(store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(transcribe.store, "run_dir", lambda date=None: tmp_path)

    # Inject a fake faster_whisper so transcribe runs without a GPU/codec.
    class FakeSegments:
        def __iter__(self):
            return iter([])
    class FakeInfo:
        language = "en"
        language_probability = 0.99
        duration = 0.0
    class FakeModel:
        def __init__(self, *a, **kw):
            pass
        def transcribe(self, *a, **kw):
            return FakeSegments(), FakeInfo()
    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    out = transcribe.transcribe(date="2026-08-31")
    assert out == tmp_path / "transcript.md"
    assert "Transcript" in out.read_text()


# --- config budget: length + depth -> word budget / source count -----------

def test_config_budget_table():
    # num_sources = round((LENGTH - 20% intro/recap) / per-source minutes).
    expected = {
        ("short", "brief"): 8,
        ("short", "deep-dive"): 4,
        ("medium", "brief"): 14,
        ("medium", "deep-dive"): 7,
        ("long", "brief"): 24,
        ("long", "deep-dive"): 12,
    }
    targets = {"short": 10.0, "medium": 17.6, "long": 30.0}
    for (length, depth), n in expected.items():
        rc = config_mod.RunConfig(length=length, depth=depth)
        b = rc.budget()
        assert b["num_sources"] == n, (length, depth, b)
        # Last source lands the duration exactly on the length preset.
        assert rc.target_minutes() == targets[length], (length, depth)
        assert b["per_source_words"] == round(config_mod.DEPTH_MINUTES[depth] * config_mod.WPM)
        # Deep-dive double-inflation is gone: per-source words only depend on
        # depth, total words only on length.
        assert b["total_words"] == round(config_mod.LENGTH_MINUTES[length] * config_mod.WPM)


def test_config_budget_clamps_source_count(monkeypatch):
    monkeypatch.setattr(config_mod, "LENGTH_MINUTES",
                        {"short": 1.0, "medium": 1.0, "long": 1.0})
    assert config_mod.RunConfig(length="long", depth="brief").num_sources() == config_mod.MIN_SOURCES
    monkeypatch.setattr(config_mod, "LENGTH_MINUTES",
                        {"short": 1000.0, "medium": 1000.0, "long": 1000.0})
    assert config_mod.RunConfig(length="short", depth="deep-dive").num_sources() == config_mod.MAX_SOURCES


def test_collect_hn_drops_empty_body(monkeypatch):
    """HN items whose body fetch comes back empty are dropped at collect and
    never reach rank — a bodyless story would air as an empty topic."""
    from pipeline import collect
    import datetime
    items = [
        Item(title="a", url="https://a.example", date="2026-09-03", body="", source="hn"),
        Item(title="b", url="https://b.example", date="2026-09-03", body="", source="hn"),
        Item(title="c", url="https://c.example", date="2026-09-03", body="", source="hn"),
    ]
    monkeypatch.setattr(collect, "_hn", lambda date, start: items)
    monkeypatch.setattr(collect, "_hn_relevant", lambda its: its)
    monkeypatch.setattr(collect, "_fetch_body",
                        lambda url: "c body" if "c.example" in url else "")
    out = collect._collect_hn("2026-09-04", datetime.date(2026, 9, 3))
    assert [i.title for i in out] == ["c"]
    assert out[0].body == "c body"


def test_judge_model_is_fixed_not_env_configurable(monkeypatch):
    """The rank judge model is a pinned constant, not an env knob — a stray
    JUDGE_MODEL value must be ignored."""
    from pipeline import llm
    monkeypatch.setenv("SKAINET_API_KEY", "test-key")
    monkeypatch.setenv("JUDGE_MODEL", "some/other-model")
    assert llm._config()["default_model"] == "Qwen/Qwen3.8-27B"


def test_content_trim_enforces_word_cap():
    from podcast_engine.content_generator import ContentCleanerMixin
    txt = ("<Person1>One two three four. Five six seven eight nine ten.</Person1>\n"
           "<Person2>Hello world. This is a longer turn with several words indeed.</Person2>\n"
           "<Person1>Tail turn three.</Person1>")
    for cap in (5, 6, 8, 12, 20, 200):
        t = ContentCleanerMixin._trim_to_words(txt, cap)
        assert ContentCleanerMixin._word_count(t) <= cap, (cap, t)
        assert t.count("<Person1>") == t.count("</Person1>")
        assert t.count("<Person2>") == t.count("</Person2>")
        assert "<Person1>" in t  # keeps leading turn
    # Already within cap -> unchanged.
    assert ContentCleanerMixin._trim_to_words(txt, 10**6) == txt


# --- content generation: word-count band around the per-part target ---------

class _FakeChain:
    """Pops canned responses; the last one repeats if overrun."""
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def invoke(self, params):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r


def _dialogue(words: int) -> str:
    return f"<Person1>{'word ' * words}</Person1>"


def _generator(chain=None):
    from podcast_engine.content_generator import LongFormContentGenerator
    return LongFormContentGenerator(chain, None, {
        "max_num_chunks": 4, "per_source_words": 100,
        "intro_words": 40, "recap_words": 40})


def test_invoke_with_retry_accepts_in_band_and_retries_out():
    chain = _FakeChain([_dialogue(50), _dialogue(100)])   # 50 < 80 (out), 100 in band
    gen = _generator(chain)
    out = gen._invoke_with_retry({}, min_turns=0, target_words=100)
    assert chain.calls == 2
    assert f"<Person1>{'word ' * 100}</Person1>" == out


def test_invoke_with_retry_retries_overshoot_too():
    chain = _FakeChain([_dialogue(130), _dialogue(95)])   # 130 > 120 (out), 95 in band
    gen = _generator(chain)
    out = gen._invoke_with_retry({}, min_turns=0, target_words=100)
    assert chain.calls == 2
    assert f"<Person1>{'word ' * 95}</Person1>" == out


def test_invoke_with_retry_relaxed_tol_accepts_overshoot():
    # With tol=0.3 the band is [70, 130]; a 130-word part is now in-band and
    # accepted without a retry (the deterministic trim caps it afterwards).
    chain = _FakeChain([_dialogue(130)])
    gen = _generator(chain)
    out = gen._invoke_with_retry({}, min_turns=0, target_words=100, tol=0.3)
    assert chain.calls == 1
    assert f"<Person1>{'word ' * 130}</Person1>" == out


def test_call_llm_json_requests_json_object_response_format(monkeypatch):
    # The raw-client call must request response_format=json_object so DeepSeek
    # keeps its long reasoning out of `content`; otherwise complex parts hit the
    # token cap (finish_reason=length), yield no JSON, and get discarded.
    # The generator module imports langchain, which the host test venv lacks;
    # stub the only langchain symbol it needs so the module imports cleanly.
    lr = types.ModuleType("langchain_core")
    lr.runnables = types.ModuleType("langchain_core.runnables")
    lr.runnables.RunnableLambda = lambda fn: None
    monkeypatch.setitem(sys.modules, "langchain_core", lr)
    monkeypatch.setitem(sys.modules, "langchain_core.runnables", lr.runnables)

    from podcast_engine import generator as gen_mod

    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            payload = json.dumps({
                "dialogue": [{"speaker": "1",
                              "text": "hello world, this is a long enough spoken "
                                      "line to clear the parsing floor"}]})
            class _Msg:
                content = payload
                finish_reason = "stop"
            class _Choice:
                message = _Msg()
                finish_reason = "stop"
            class _Resp:
                choices = [_Choice()]
            return _Resp()

    class _Client:
        chat = type("_Chat", (), {"completions": _Completions()})()

    gen = object.__new__(gen_mod.SimplePodcastGenerator)
    gen._use_raw_client = True
    gen._raw_client = _Client()
    gen.model_name = "deepseek-ai/DeepSeek-V4-Flash-0731"
    gen._max_output_tokens = 16000
    gen._check_required_params = lambda prompt, params: None

    params = {
        "audience": "a", "familiar_topics": "none", "podcast_name": "p",
        "podcast_tagline": "t", "output_language": "English", "instruction": "i",
        "context": "c", "input_text": "x", "host1_name": "Brian",
        "host2_name": "Tina", "roles_person1": "AI researcher",
        "roles_person2": "AI researcher", "podcast_topic": "AI research",
    }
    out = gen._call_llm_json(params)
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert "<Person1>" in out


def test_invoke_with_retry_returns_closest_when_never_in_band():
    chain = _FakeChain([_dialogue(130), _dialogue(60)])   # never inside [80, 120]
    gen = _generator(chain)
    out = gen._invoke_with_retry({}, min_turns=0, target_words=100, max_attempts=2)
    assert chain.calls == 2
    # Closest to target (130 has dev 30; 60 has dev 40).
    assert f"<Person1>{'word ' * 130}</Person1>" == out


def test_invoke_with_retry_keeps_turn_floor():
    # Both attempts are inside the word band (90 words, band 80-120); the first
    # is retried because it has only 1 turn and the floor wants 2.
    two_turns = (f"<Person1>{'word ' * 40}</Person1>\n"
                 f"<Person2>{'more ' * 50}</Person2>")
    chain = _FakeChain([_dialogue(90), two_turns])
    gen = _generator(chain)
    out = gen._invoke_with_retry({}, min_turns=2, target_words=100)
    assert chain.calls == 2
    assert out == two_turns


def test_source_part_prompt_targets_words():
    gen = _generator()
    p = gen.enhance_prompt_params(
        {"host1_name": "A", "host2_name": "B"},
        part_idx=1, total_parts=3, chat_context="", chunk_len=500)
    assert "targeting about 100 words of dialogue" in p["instruction"]
    assert "at most 100 words" not in p["instruction"]


# --- topic taxonomy + steering helpers --------------------------------------

def test_personal_match_neutral_when_no_profile():
    # No profile => neutral 0.5 so the blend is a no-op (final == importance).
    assert topics.personal_match({"post_training": 0.9}, {}) == 0.5


def test_personal_match_cosine_overlap():
    vec = {"post_training": 0.9, "ai_for_science": 0.1}
    pref = {"post_training": 1.0}
    v = topics.personal_match(vec, pref)
    assert abs(v - (0.9 / (0.9 ** 2 + 0.1 ** 2) ** 0.5)) < 1e-9
    assert 0.0 < v <= 1.0
    assert topics.personal_match({"ai_for_science": 1.0}, pref) == 0.0
    assert topics.personal_match({}, pref) == 0.0


def test_final_score_interpolated_blend():
    assert abs(topics.final_score(0.8, 1.0, 0.3) - 0.86) < 1e-9
    assert topics.final_score(0.8, 0.9, 0.0) == 0.8
    assert topics.final_score(0.8, 0.9, 1.0) == 0.9


def test_episode_vector_spans_full_taxonomy_and_normalizes():
    vec = topics.episode_vector(
        [{"post_training": 0.9}, {"model_release": 0.8}], weights=[1.0, 1.0])
    assert set(vec) == set(topics.TAXONOMY_IDS)
    assert abs(sum(vec.values()) - 1.0) < 1e-9
    assert vec["post_training"] > 0 and vec["model_release"] > 0
    assert vec["safety_alignment"] == 0.0


def test_validate_topic_map():
    assert topics.validate_topic_map({"post_training": 0.9}) == []
    assert "bogus" in topics.validate_topic_map({"post_training": 0.9, "bogus": 1.0})
    assert topics.validate_topic_map("nope")

    ids = set(topics.TAXONOMY_IDS)
    assert len(ids) == len(topics.TAXONOMY)  # unique ids
    assert "other" in ids


# --- label pass --------------------------------------------------------------

def test_label_items_parses_and_filters_taxonomy(monkeypatch):
    monkeypatch.setattr(label_mod, "_chat", lambda payload, model: json.dumps({
        "labels": [
            # valid label
            {"index": 0, "label": "post_training"},
            # valid label with extra ignored keys
            {"index": 1, "label": "agents", "weight": 0.5},
            # off-taxonomy id -> nothing
            {"index": 2, "label": "not_a_type"},
        ]
    }))
    items = [
        Item(title="a", url="https://A.example/", date="d", body="b", source="arxiv"),
        Item(title="b", url="https://b.example", date="d", body="b", source="arxiv"),
        Item(title="c", url="https://c.example", date="d", body="b", source="arxiv"),
    ]
    out = label_mod.label_items(items, workers=1)
    assert out == {
        "https://a.example": {"post_training": 1.0},
        "https://b.example": {"agents": 1.0},
    }
    assert "https://c.example" not in out


def test_flatten_label_single_label():
    assert label_mod._flatten_label({"index": 0, "label": "post_training"}) \
        == {"post_training": 1.0}
    assert label_mod._flatten_label({"label": "not_a_type"}) == {}
    assert label_mod._flatten_label({"label": 5}) == {}
    assert label_mod._flatten_label({}) == {}


def test_label_items_malformed_batch_yields_nothing(monkeypatch):
    monkeypatch.setattr(label_mod, "_chat",
                        lambda payload, model: "not json at all")
    items = [Item(title="a", url="https://a.example", date="d", body="b", source="arxiv")]
    assert label_mod.label_items(items, workers=1) == {}


def test_annotate_brief_labels_is_surgical_and_idempotent(tmp_path):
    root = tmp_path
    brief = root / "podcast_brief.md"
    brief.write_text(
        "# AI News Digest — 2026-09-08\n\n"
        "## Hacker News stories (2)\n\n"
        "- [A](https://a.example) — score 0.90\n"
        "  > Excerpt A.\n"
        "- [B](https://b.example) — score 0.85\n"
        "- [C](https://c.example) — score 0.8 · agents\n",
        encoding="utf-8",
    )
    labels = {
        "https://a.example": {"model_release": 1.0},
        "https://b.example": {"ai_for_science": 1.0},
        "https://c.example": {"agents": 1.0},
    }
    assert label_mod._annotate_brief_labels(root, labels) is True
    text = brief.read_text(encoding="utf-8")
    assert "- [A](https://a.example) — score 0.90 · model_release\n" in text
    assert "- [B](https://b.example) — score 0.85 · ai_for_science\n" in text
    assert "- [C](https://c.example) — score 0.8 · agents\n" in text
    # header + excerpt untouched
    assert "# AI News Digest — 2026-09-08" in text
    assert "  > Excerpt A." in text
    # second run: already labeled -> no change
    assert label_mod._annotate_brief_labels(root, labels) is False


def test_chunk_by_chars_respects_budget():
    items = [Item(title=f"t{i}", url=f"u{i}", date="d", body="x" * 500,
                  source="arxiv") for i in range(10)]
    chunks = label_mod._chunk_by_chars(items, max_chars=1200)
    assert sum(len(c) for c in chunks) == 10
    assert len(chunks) > 1


def test_backfill_labels_writes_episode_labels(tmp_path, monkeypatch):
    root = tmp_path / "history"
    run = root / "05-09-2026"
    run.mkdir(parents=True)
    (run / "episode.json").write_text(json.dumps({"manifest": [
        {"title": "a", "url": "https://a.example", "date": "d", "body": "b",
         "source": "arxiv", "score": 0.8},
        {"title": "b", "url": "https://b.example", "date": "d", "body": "b",
         "source": "arxiv", "score": 0.5},
    ]}))
    monkeypatch.setattr(label_mod.store, "ROOT", root)
    monkeypatch.setattr(
        label_mod, "label_items",
        lambda items, **k: {topics.normalize_url(it.url): {"post_training": 0.9}
                            for it in items})

    touched = label_mod.backfill_labels()
    assert touched == ["05-09-2026"]
    labels = json.loads((run / "labels.json").read_text())
    assert labels["episode_topics"][0]["topic"] == "post_training"
    # already present -> skipped, nothing relabeled
    assert label_mod.backfill_labels() == []


def test_cli_accepts_label_stage_and_steering_flags():
    import pytest as _pt  # noqa: F401
    from pipeline import __main__ as cli
    # --only label is a valid stage; --topics/--steering-alpha parse into run args.
    # We only exercise argument parsing (no LLM/network): a bogus stage raises.
    with pytest.raises(SystemExit):
        cli.main(["run", "--only", "not_a_stage"])
    assert "label" in cli.STAGES
    assert "topics" in cli.USAGE and "steering-alpha" in cli.USAGE


# --- cross-episode memory -------------------------------------------------

def test_memory_derive_summary_parses_and_filters(monkeypatch):
    monkeypatch.setattr(memory_mod, "_chat",
                        lambda payload: {"topics": [
                            {"topic": "post_training", "summary": "Covered RLHF follow-ups."},
                            {"topic": "not_a_topic", "summary": "ignored"},
                            {"topic": "agents", "summary": 5},
                        ]})
    items = [
        RankedItem(title="A", url="https://a.example", date="2026-09-01",
                   body="b", source="arxiv", score=0.9, judge_reason="imp",
                   topics={"post_training": 1.0}),
        RankedItem(title="B", url="https://b.example", date="2026-09-01",
                   body="b", source="hn", score=0.8, judge_reason="imp",
                   topics={"agents": 1.0}),
    ]
    out = memory_mod.derive_summary(items, episode_date="2026-09-08")
    assert out["episode_date"] == "2026-09-08"
    assert out["topics"] == [
        {"topic": "post_training", "summary": "Covered RLHF follow-ups."}]


def test_memory_derive_falls_back_on_llm_failure(monkeypatch):
    monkeypatch.setattr(memory_mod, "_chat",
                        lambda payload: (_ for _ in ()).throw(RuntimeError("boom")))
    items = [
        RankedItem(title="A", url="https://a.example", date="2026-09-01",
                   body="b", source="arxiv", score=0.9,
                   judge_reason="Importance: major new method.",
                   topics={"post_training": 1.0}),
    ]
    out = memory_mod.derive_summary(items, episode_date="2026-09-08")
    assert out["topics"][0]["topic"] == "post_training"
    assert "A" in out["topics"][0]["summary"]
    assert "major new method" in out["topics"][0]["summary"]
    # Fallback is deterministic: same input -> same digest.
    assert memory_mod.derive_summary(items, episode_date="2026-09-08") == out


def test_memory_write_idempotent_and_readable(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_mod.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(memory_mod, "_chat",
                        lambda payload: {"topics": [
                            {"topic": "agents", "summary": "Agent arc."}]})
    items = [RankedItem(title="A", url="u", date="d", body="b", source="hn",
                        score=0.9, judge_reason="r", topics={"agents": 1.0})]
    assert memory_mod.write_memory(items, episode_date="2026-09-08", date="2026-09-08")
    mem = json.loads((tmp_path / memory_mod.MEMORY_FILE).read_text())
    assert mem["topics"][0]["topic"] == "agents"
    assert mem["window_end"] == "2026-09-08"
    # unreadable/missing artifact tolerance
    assert memory_mod.read_episode_date_memory(tmp_path) == mem


def _write_prior_run(root, name, wend, topics, episode_date=None):
    import yaml
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(
        yaml.safe_dump({"window": {"end": wend}}, sort_keys=False),
        encoding="utf-8")
    eps = [(t, f"{name} covers {t}.") for t in topics]
    (d / memory_mod.MEMORY_FILE).write_text(json.dumps(
        {"episode_date": episode_date or name,
         "topics": [{"topic": t, "summary": s} for t, s in eps]}),
        encoding="utf-8")
    return d


def test_memory_context_retention_and_topic_match(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_mod.store, "ROOT", tmp_path / "history")
    root = tmp_path / "history"
    _write_prior_run(root, "25-08-2026", "2026-08-25",
                     ["agents"], episode_date="2026-08-25")
    _write_prior_run(root, "01-09-2026", "2026-09-01",
                     ["post_training", "agents"], episode_date="2026-09-01")
    _write_prior_run(root, "02-09-2026", "2026-09-02",
                     ["model_release"], episode_date="2026-09-02")

    current = [RankedItem(title="C", url="u", date="2026-09-07", body="b",
                          source="hn", score=0.9, judge_reason="r",
                          topics={"agents": 1.0})]
    ctx = memory_mod.context_for(current, mem_windows=2,
                                 episode_date="2026-09-08",
                                 window_span_days=7)
    # Only agents is wanted among the current items' topics. The retention
    # window (2 windows back: window-start 08-18) makes BOTH the 25-08 and
    # 01-09 runs eligible; runs in the CURRENT window (>= 09-02) are excluded,
    # so the 02-09 run is not a memory source for itself.
    assert list(ctx.keys()) == ["agents"]
    entries = ctx["agents"]
    assert len(entries) == 2
    assert any("2026-09-01" in e for e in entries)
    assert any("2026-08-25" in e for e in entries)


def test_generate_writes_memory_artifact(tmp_path, monkeypatch):
    """generate() writes memory.json into the run folder alongside labels."""
    ranked = [
        RankedItem(title="t", url="u", date="d", body="b", source="arxiv",
                   score=0.9, judge_reason="r", topics={"agents": 1.0})
    ]
    monkeypatch.setattr(store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(generate.store, "run_dir", lambda date=None: tmp_path)
    monkeypatch.setattr(memory_mod, "_chat",
                        lambda payload: {"topics": [
                            {"topic": "agents", "summary": "Agent arc."}]})
    ep = generate.generate(ranked, make_audio=False)
    out = json.loads((tmp_path / memory_mod.MEMORY_FILE).read_text())
    assert out["topics"][0]["topic"] == "agents"
    assert len(ep.manifest) == 1


def test_memory_section_and_ordering():
    from podcast_engine.generator import SimplePodcastGenerator
    gen = object.__new__(SimplePodcastGenerator)
    gen.memory_context = {"agents": ["2026-09-01: prior agent arc"],
                          "post_training": []}
    seg = gen._memory_section()
    assert "=== MEMORY: PRIOR EPISODES ===" in seg
    assert "2026-09-01: prior agent arc" in seg
    assert "TOPIC: agents" in seg
    assert "TOPIC: post_training" not in seg  # empty topic skipped

    # Ordering: same-theme items adjacent, unlabeled fall back to legacy.
    class _S:
        def __init__(self, title, url):
            self.title, self.url = title, url
    class _It:
        def __init__(self, url, topics):
            self.url, self.topics = url, topics
    gen.items = [
        _It("https://a.example", {"post_training": 1.0}),
        _It("https://b.example", {"agents": 1.0}),
        _It("https://c.example", {"model_release": 1.0}),
        _It("https://d.example", {}),  # unlabeled -> legacy fallback
    ]
    sources = [_S("Paper A", "https://a.example"),
               _S("HN B", "https://b.example"),
               _S("Paper C", "https://c.example"),
               _S("DiffusionGemma Technical Report", "https://d.example")]
    ordered = gen._order_sources_thematically(sources)
    # Labeled items follow the canonical theme order: post_training, then
    # model_release, then agents. The unlabeled source keeps the legacy order.
    titles = [s.title for s in ordered]
    assert titles.index("Paper A") < titles.index("Paper C")
    assert titles.index("Paper C") < titles.index("HN B")
    assert titles[-1] == "DiffusionGemma Technical Report"


def test_memory_block_chunking_and_part_cap():
    """The cross-episode MEMORY block is its own chunk with a small word cap
    and the PRIOR-COVERAGE-SYNC instruction — never a full topic part."""
    from podcast_engine.content_generator import LongFormContentGenerator
    gen = LongFormContentGenerator(None, None, {
        "per_source_words": 300, "intro_words": 120, "recap_words": 180,
        "memory_words": 90, "continuity_reference_enabled": True})
    combined = (
        "=== INTRO ===\nhi\n=== END INTRO ===\n\n"
        "=== MEMORY: PRIOR EPISODES ===\nTOPIC: agents\n- 2026-09-01: prior arc\n"
        "=== END MEMORY ===\n"
        "=== TOPIC: A ===\nsrc\n=== END TOPIC ===\n"
        "=== RECAP ===\n1. A\n=== END RECAP ===")
    chunks = gen.chunk_content(combined, 0)
    assert len(chunks) == 4
    assert gen._is_memory_chunk(chunks[1])
    assert gen._part_word_cap(1, 4, is_memory=True) == 90
    assert gen._part_word_cap(0, 4) == 120
    assert gen._part_word_cap(2, 4) == 300
    assert gen._part_word_cap(3, 4) == 180
    inst = gen.enhance_prompt_params({"host1_name": "B", "host2_name": "T"},
                                     1, 4, "", is_memory=True)
    assert "PRIOR-COVERAGE SYNC" in inst["instruction"]
    # A topic part (not memory) still gets the continuity guidance.
    topic = gen.enhance_prompt_params({"host1_name": "B", "host2_name": "T"},
                                      2, 4, "")
    assert "CONTINUITY" in topic["instruction"] and "MEMORY" in topic["instruction"]




def _steer_items(n=5):
    return [Item(title=f"t{i}", url=f"https://x{i}.example", date="2026-08-21",
                 body="body", source="arxiv") for i in range(n)]


def _steer_judge(n=5):
    scores = (0.9, 0.8, 0.7, 0.6, 0.2)

    def f(groups, rubric):
        return [(scores[i] if i < len(scores) else 0.5, "r",
                 "ai_for_science" if i == 0 else "post_training")
                for i in range(len(groups))]
    return f


def test_rank_steering_reranks_by_profile(tmp_path, monkeypatch):
    items = _steer_items()
    monkeypatch.setattr(rank, "_judge_batch", _steer_judge())
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)

    out = rank.rank(items, date="07-09-2026",
                    profile={"post_training": 1.0}, alpha=0.5)
    assert [r.url for r in out] == [
        "https://x1.example", "https://x2.example", "https://x3.example",
        "https://x4.example", "https://x0.example"]  # t0 (top importance) demoted
    assert out[0].personal_score == pytest.approx(1.0)
    assert out[-1].personal_score == pytest.approx(0.0)
    assert out[1].final_score > out[-1].final_score


def test_rank_steering_off_is_unchanged(tmp_path, monkeypatch):
    items = _steer_items()
    monkeypatch.setattr(rank, "_judge_batch", _steer_judge())
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)

    out = rank.rank(items, date="07-09-2026")  # no profile -> steering off
    assert [r.url for r in out] == [f"https://x{i}.example" for i in range(5)]
    assert all(r.final_score == r.score for r in out)
    # the judge still returned labels; they just don't move anything
    assert all(r.topics for r in out)


def test_rank_steering_alpha_zero_keeps_importance_order(tmp_path, monkeypatch):
    items = _steer_items()
    monkeypatch.setattr(rank, "_judge_batch", _steer_judge())
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)

    out = rank.rank(items, date="07-09-2026",
                    profile={"post_training": 1.0}, alpha=0.0)
    assert [r.url for r in out] == [f"https://x{i}.example" for i in range(5)]
    assert all(r.final_score == r.score for r in out)


def test_rank_judge_labels_feed_steering_for_all_items(tmp_path, monkeypatch):
    items = _steer_items()
    scores = (0.9, 0.8, 0.7, 0.6, 0.2)

    def judge(groups, rubric):
        out = []
        for i in range(len(groups)):
            label = "post_training" if i == 0 else "ai_for_science"
            if i == 2:  # judge returns no valid label for t2 -> neutral
                label = None
            out.append((scores[i], "r", label))
        return out

    monkeypatch.setattr(rank, "_judge_batch", judge)
    monkeypatch.setattr(rank.store, "run_dir", lambda date=None: tmp_path)

    out = rank.rank(items, date="07-09-2026",
                    profile={"post_training": 1.0}, alpha=0.5)
    # all items carry the judge's label except the deliberately-unlabeled one
    assert {r.url for r in out if r.topics} == {
        "https://x0.example", "https://x1.example",
        "https://x3.example", "https://x4.example"}
    t2 = [r for r in out if r.url == "https://x2.example"][0]
    assert t2.personal_score == pytest.approx(0.5)
    assert t2.final_score == t2.score


# --- selection + config ------------------------------------------------------

def test_select_sources_sorts_by_final_score():
    items = [
        RankedItem(title="a", url="https://a.example", date="d", body="b",
                   source="arxiv", score=0.6, final_score=0.9),
        RankedItem(title="b", url="https://b.example", date="d", body="b",
                   source="arxiv", score=0.9, final_score=0.61),
        RankedItem(title="c", url="https://c.example", date="d", body="b",
                   source="arxiv", score=0.8, final_score=0.8),
    ]
    chosen = generate.select_sources(items, target_n=4)
    assert [c.url for c in chosen] == ["https://a.example", "https://c.example",
                                       "https://b.example"]


def test_config_resolve_accepts_steering():
    cfg = config_mod.resolve(topic_prefs=["post_training"], steering_alpha=0.5)
    assert cfg.topic_prefs == ["post_training"]
    assert cfg.steering_alpha == 0.5


def test_config_resolve_accepts_and_validates_mem_windows():
    cfg = config_mod.resolve(mem_windows=3)
    assert cfg.mem_windows == 3
    assert config_mod.resolve().mem_windows == config_mod.MEM_WINDOWS_DEFAULT
    # to_dict / to_yaml serialize it for the run's stored config.yaml.
    assert config_mod.resolve().to_dict()["podcast"]["mem_windows"] == 2
    assert "mem_windows: 2" in config_mod.resolve().to_yaml()
    with pytest.raises(ValueError):
        config_mod.resolve(mem_windows=0)
    with pytest.raises(ValueError):
        config_mod.resolve(mem_windows=9)


def test_config_rejects_unknown_topic_and_bad_alpha():
    with pytest.raises(ValueError):
        config_mod.resolve(topic_prefs=["bogus_topic"])
    with pytest.raises(ValueError):
        config_mod.resolve(steering_alpha=1.5)


def test_label_pool_removed_steering_alpha_defaults_to_constant():
    # The old top-K label pool is gone — labels ride the judge pass — but the
    # steering alpha still resolves from its constant, so nothing regresses.
    cfg = config_mod.RunConfig(length="short", depth="deep-dive")
    assert cfg.steering_alpha == config_mod.STEERING_ALPHA
    assert not hasattr(cfg, "label_pool_size")


# --- eval harness helpers (no LLM) ------------------------------------------

def test_eval_labels_metrics():
    from pipeline import eval_labels
    assert eval_labels._jaccard({"a": 0.9}, {"a": 0.8, "b": 0.2}) == pytest.approx(0.5)
    assert eval_labels._jaccard({}, {}) == 1.0
    # perfect & imperfect agreement on binary raters
    assert eval_labels.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert eval_labels.cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)
    assert 0.0 <= eval_labels.cohen_kappa([], []) <= 1.0


def test_eval_labels_load_pool(tmp_path):
    from pipeline import eval_labels
    f = tmp_path / "pool.json"
    f.write_text(json.dumps({"papers": [
        {"title": "A", "url": "https://a.example", "abstract": "x"},
        {"title": "B", "arxiv_id": "2501.00001", "abstract": "y"},
    ]}))
    items = eval_labels.load_pool([str(f)])
    assert {it.url for it in items} == {"https://a.example", "https://arxiv.org/abs/2501.00001"}
    assert items[0].body == "x"
    assert eval_labels.load_pool([]) == []
    # raw-list pool (e.g. rank.json) also loads
    raw = tmp_path / "rank.json"
    raw.write_text(json.dumps([
        {"title": "C", "url": "https://c.example", "body": "z", "score": 0.9},
    ]))
    assert "https://c.example" in {it.url for it in eval_labels.load_pool([str(raw)])}


def test_eval_labels_label_id():
    from pipeline import eval_labels
    assert eval_labels._label_id({"post_training": 1.0}) == "post_training"
    assert eval_labels._label_id({"post_training": 1.0, "agents": 0.0}) == "post_training"
    assert eval_labels._label_id({}) is None


def test_eval_labels_label_stats_uses_prevalent_topics():
    from pipeline import eval_labels
    items = [types.SimpleNamespace(url=f"https://x{i}.example", title=f"t{i}", body="b")
             for i in range(2)]
    small = {"https://x0.example": {"post_training": 1.0},
             "https://x1.example": {"agents": 1.0}}
    big = {"https://x0.example": {"post_training": 1.0},
           "https://x1.example": {"agents": 1.0}}
    stats, kappa, match = eval_labels._label_stats(items, small, big)
    # both items agree (post_training, agents) -> kappa 1.0
    assert kappa == 1.0
    assert match == 1.0
    assert stats["post_training"]["prevalence"] == 0.5
    # 'other' was never marked by the reference -> excluded from the mean kappa
    assert stats["other"]["prevalence"] == 0.0


# --- multi-label items (equal-weight, cosine unchanged) -----------------------

def test_clean_labels_normalizes_multi_and_single():
    assert rank._clean_labels(["post_training", "agents"]) == ["post_training", "agents"]
    assert rank._clean_labels(["post_training", "nope", "agents"]) == ["post_training", "agents"]
    assert rank._clean_labels("post_training") == ["post_training"]
    assert rank._clean_labels([]) is None
    assert rank._clean_labels(None) is None
    assert rank._clean_labels(5) is None


def test_rank_to_ranked_multi_label():
    item = Item(title="t", url="u", date="d", body="b", source="arxiv")
    r = rank._to_ranked(item, 0.9, "r", ["post_training", "agents"])
    assert r.topics == {"post_training": 1.0, "agents": 1.0}
    # legacy single-string judge shape still supported
    assert rank._to_ranked(item, 0.9, "r", "post_training").topics == {"post_training": 1.0}
    assert rank._to_ranked(item, 0.9, "r", None).topics == {}
    assert rank._to_ranked(item, 0.9, "r", ["bogus"]).topics == {}


def test_judge_batch_parses_multi_labels(monkeypatch):
    monkeypatch.setattr(rank.llm, "chat", lambda *a, **k: json.dumps({
        "scores": [
            {"index": 0, "score": 0.8, "reason": "r",
             "labels": ["post_training", "agents"]},
            {"index": 1, "score": 0.5, "reason": "s",
             "labels": ["not_a_topic"]},
        ]}))
    items = [Item(title="a", url="u1", date="d", body="b", source="arxiv"),
             Item(title="b", url="u2", date="d", body="b", source="arxiv")]
    out = rank._judge_batch(items, "rubric")
    assert out[0] == (0.8, "r", ["post_training", "agents"])
    # an all-invalid label list is normalized to None (item stays neutral)
    assert out[1] == (0.5, "s", None)


def test_personal_match_multi_label_item():
    vec = {"post_training": 1.0, "agents": 1.0}
    pref = {"post_training": 1.0}
    v = topics.personal_match(vec, pref)
    assert abs(v - 1.0 / math.sqrt(2)) < 1e-9
    # an item matching every profile facet scores full match
    assert topics.personal_match(vec, {"post_training": 1.0, "agents": 1.0}) == pytest.approx(1.0)
    assert topics.personal_match({"agents": 1.0}, pref) == 0.0


def test_flatten_label_multi():
    assert label_mod._flatten_label({"index": 0, "labels": ["post_training", "agents"]}) \
        == {"post_training": 1.0, "agents": 1.0}
    # off-taxonomy ids dropped; a scalar string id is tolerated
    assert label_mod._flatten_label({"index": 0, "labels": ["post_training", "bogus"]}) \
        == {"post_training": 1.0}
    assert label_mod._flatten_label({"labels": "model_release"}) == {"model_release": 1.0}
    # legacy single-label shape still parses
    assert label_mod._flatten_label({"label": "post_training"}) == {"post_training": 1.0}
    assert label_mod._flatten_label({}) == {}


def test_label_items_parses_multi_labels(monkeypatch):
    monkeypatch.setattr(label_mod, "_chat", lambda payload, model: json.dumps({
        "labels": [
            {"index": 0, "labels": ["post_training", "agents"]},
            {"index": 1, "labels": "model_release"},
        ]}))
    items = [Item(title="a", url="https://a.example", date="d", body="b", source="arxiv"),
             Item(title="b", url="https://b.example", date="d", body="b", source="arxiv")]
    out = label_mod.label_items(items, workers=1)
    assert out["https://a.example"] == {"post_training": 1.0, "agents": 1.0}
    assert out["https://b.example"] == {"model_release": 1.0}


def test_generate_brief_multi_labels():
    chosen = [
        RankedItem(title="Paper A", url="https://arxiv.org/abs/2601.00001",
                   date="d", body="abstract", source="arxiv", score=0.9,
                   topics={"post_training": 1.0, "agents": 1.0}),
    ]
    by_source = generate._group_by_source(chosen)
    text = generate._brief_text(chosen, by_source)
    # comma-joined multi-label ids, most salient facet first (dict order)
    line = next(l for l in text.splitlines() if l.startswith("- "))
    m = re.search(r"·\s*([\w-]+(?:,\s*[\w-]+)*)$", line)
    assert m and m.group(1) == "post_training,agents"


def test_annotate_brief_labels_multi_comma(tmp_path):
    root = tmp_path
    brief = root / "podcast_brief.md"
    brief.write_text(
        "# AI News Digest — 2026-09-08\n\n"
        "## Hacker News stories (1)\n\n"
        "- [A](https://a.example) — score 0.90\n",
        encoding="utf-8")
    labels = {"https://a.example": {"agents": 1.0, "post_training": 1.0}}
    assert label_mod._annotate_brief_labels(root, labels) is True
    assert "- [A](https://a.example) — score 0.90 · agents,post_training\n" \
        in brief.read_text(encoding="utf-8")
    # second run is a no-op (already annotated)
    assert label_mod._annotate_brief_labels(root, labels) is False


def test_memory_payload_groups_under_each_label():
    items = [
        RankedItem(title="A", url="https://a.example", date="2026-09-08",
                   body="b", source="arxiv", score=0.9, judge_reason="r",
                   topics={"post_training": 1.0, "agents": 1.0}),
        RankedItem(title="B", url="https://b.example", date="2026-09-08",
                   body="b", source="hn", score=0.8, judge_reason="r", topics={}),
    ]
    payload = memory_mod._payload(items)
    by_topic = {e["topic"]: [i["title"] for i in e["items"]] for e in payload}
    # a multi-label item is summarized under EACH of its topics
    assert by_topic["post_training"] == ["A"]
    assert by_topic["agents"] == ["A"]
    assert by_topic["other"] == ["B"]


# --- runtime taxonomy loading / refresh adoption --------------------------------

def _restore_default_taxonomy():
    topics.reload_taxonomy(pathlib.Path("data/taxonomy.json"))


def test_topics_default_fallback_when_artifact_missing(tmp_path, monkeypatch):
    try:
        topics.reload_taxonomy(tmp_path / "missing.json")
        assert topics.TAXONOMY == topics.DEFAULT_TAXONOMY
    finally:
        _restore_default_taxonomy()


def test_topics_loads_runtime_taxonomy(tmp_path):
    spec = {"taxonomy": [
        {"id": "widgets", "label": "Widgets", "description": "Widget news."},
        {"id": "gadgets", "label": "Gadgets", "definition": "Gadget news."},
    ]}
    p = tmp_path / "taxonomy.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    try:
        taxo = topics.reload_taxonomy(p)
        assert [t["id"] for t in taxo] == ["widgets", "gadgets", "other"]
        assert taxo[0]["label"] == "Widgets"
        # the derive artifact's "definition" key is normalized to "description"
        assert taxo[1]["description"] == "Gadget news."
        # downstream constants track the artifact
        assert topics.TAXONOMY_BY_ID["widgets"]["label"] == "Widgets"
    finally:
        _restore_default_taxonomy()


def test_topics_rejects_degenerate_artifact_and_falls_back(tmp_path):
    p = tmp_path / "taxonomy.json"
    p.write_text(json.dumps({"taxonomy": [{"id": "only_one"}]}), encoding="utf-8")
    try:
        topics.reload_taxonomy(p)
        assert topics.TAXONOMY == topics.DEFAULT_TAXONOMY
    finally:
        _restore_default_taxonomy()


def test_write_runtime_artifact_and_adopt(tmp_path):
    from pipeline import derive_taxonomy as dt
    canon = [
        {"id": "agents", "label": "AI Agents", "definition": "d",
         "covers": ["agentic"]},
        {"id": "model_release", "label": "Model Releases", "definition": "m",
         "covers": ["launch"]},
        {"id": "other", "label": "Other", "definition": "o", "covers": []},
    ]
    target = tmp_path / "taxonomy.json"
    out_dir = tmp_path / "out"
    artifact = dt.write_runtime_artifact(out_dir, canon, "2026-09-08T00:00:00Z",
                                         apply_to=target)
    raw = json.loads(artifact.read_text())
    assert [t["id"] for t in raw["taxonomy"]] == ["agents", "model_release", "other"]
    assert raw["taxonomy"][0]["description"] == "d"
    try:
        # the adopted artifact is what pipeline.topics loads
        assert [t["id"] for t in topics.reload_taxonomy(target)] \
            == ["agents", "model_release", "other"]
    finally:
        _restore_default_taxonomy()
