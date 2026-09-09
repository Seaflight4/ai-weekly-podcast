"""Contract tests for the lean pipeline (collect -> rank -> generate).

No network, no LLM. The judge is monkeypatched; store is isolated to a tmp dir.
"""
import json, pathlib, sys, types
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipeline import Item, RankedItem
from pipeline import rank, store, generate, transcribe
from pipeline import config as config_mod


# --- RankedItem -----------------------------------------------------------

def test_rankeditem_round_trips():
    r = RankedItem(title="t", url="u", date="d", body="b", source="arxiv",
                   score=0.8, judge_reason="imp")
    d = r.__dict__
    r2 = RankedItem(**d)
    assert r2.score == 0.8
    assert r2.judge_reason == "imp"


def test_rankeditem_legacy_load_ignores_unknown_keys():
    # Old caches may carry removed Phase-2 fields (personal_score, final_score,
    # alpha). The tolerant loaders filter to known dataclass fields.
    legacy = {"title": "t", "url": "u", "date": "d", "body": "b",
              "source": "arxiv", "score": 0.8, "judge_reason": "r",
              "personal_score": 0.6, "alpha": 0.7, "final_score": 0.74}
    import dataclasses
    known = {f.name for f in dataclasses.fields(RankedItem)}
    r = RankedItem(**{k: v for k, v in legacy.items() if k in known})
    assert r.score == 0.8
    assert not hasattr(r, "personal_score")
    assert not hasattr(r, "final_score")


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
                   date="d", body="abstract", source="arxiv", score=0.9),
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
    from pipeline import audio
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
    from pipeline import audio
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
    from pipeline import audio

    # Fake SimplePodcastGenerator that avoids the langchain/openai deps.
    class FakeGenerator:
        def __init__(self, papers_dir=None, web_dir=None, **kw):
            self.papers_dir = papers_dir
            self.web_dir = web_dir
            self.tts_model = kw.get("tts_model", "tng")
        def load_brief_and_sources(self, brief_path):
            return "=== INTRO ===\n=== END INTRO ==="
        def generate_transcript(self, combined, on_part=None):
            return "<Person1>hello</Person1>\n<Person2>world</Person2>"
        def generate_audio(self, transcript, output_path, temp_audio_dir=None):
            pathlib.Path(output_path).write_bytes(b"FAKE_MP3")

    # Inject a fake pipeline.podcastfy.generator module so the lazy import
    # inside PodcastfyBackend.generate picks up FakeGenerator without needing
    # the langchain/openai optional deps installed.
    fake_mod = types.ModuleType("pipeline.podcastfy.generator")
    fake_mod.SimplePodcastGenerator = FakeGenerator
    monkeypatch.setitem(sys.modules, "pipeline.podcastfy.generator", fake_mod)

    brief = tmp_path / "podcast_brief.md"
    brief.write_text("# AI News Digest")
    chosen = [RankedItem(title="t", url="u", date="d", body="b", source="arxiv",
                         score=0.9)]

    result = audio.PodcastfyBackend().generate(
        brief=brief, run_dir=tmp_path, chosen=chosen,
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
    from pipeline import audio

    brief = tmp_path / "podcast_brief.md"
    brief.write_text("# AI News Digest\n\n- [A](https://arxiv.org/abs/1)")
    chosen = [
        RankedItem(title="A", url="https://arxiv.org/abs/1", date="d", body="b",
                   source="arxiv", score=0.9),
        RankedItem(title="B", url="https://arxiv.org/abs/2", date="d", body="b",
                   source="arxiv", score=0.8),
    ]
    cfg = {"per_source_words": 330, "depth_factor": 1.6}
    fp1 = audio.transcript_fingerprint(brief, chosen, cfg)
    # Deterministic: same inputs -> same fingerprint (url order is sorted).
    assert fp1 == audio.transcript_fingerprint(brief, list(reversed(chosen)), cfg)
    # Sensitive: brief change, selection change, or config change all invalidate.
    brief.write_text("# AI News Digest v2\n\n- [A](https://arxiv.org/abs/1)")
    assert audio.transcript_fingerprint(brief, chosen, cfg) != fp1
    brief.write_text("# AI News Digest\n\n- [A](https://arxiv.org/abs/1)")
    other = [chosen[0]]
    assert audio.transcript_fingerprint(brief, other, cfg) != fp1
    assert audio.transcript_fingerprint(brief, chosen, {**cfg, "familiar_clause": "x"}) != fp1


def test_podcastfy_backend_reuses_transcript_on_fingerprint_match(tmp_path, monkeypatch):
    """A cached transcript.md is reused only when its stored fingerprint still
    matches the current brief/selection/config (fast TTS-retry path)."""
    from pipeline import audio

    transcript_text = "<Person1>cached content</Person1>"
    (tmp_path / "transcript.md").write_text(transcript_text, encoding="utf-8")

    brief = tmp_path / "podcast_brief.md"
    brief.write_text("# AI News Digest\n\n- [A](https://arxiv.org/abs/1)")
    chosen = [RankedItem(title="A", url="https://arxiv.org/abs/1", date="d",
                         body="b", source="arxiv", score=0.9)]
    cfg = {"per_source_words": 330, "depth_factor": 1.6}

    class FakeGenerator:
        def __init__(self, papers_dir=None, web_dir=None, **kw):
            self.papers_dir = papers_dir
            self.web_dir = web_dir
            self.tts_model = kw.get("tts_model", "tng")
        def load_brief_and_sources(self, brief_path):
            raise AssertionError("LLM path must not run for a matching cache")
        def generate_transcript(self, combined, on_part=None):
            raise AssertionError("LLM path must not run for a matching cache")
        def generate_audio(self, transcript, output_path, temp_audio_dir=None):
            pathlib.Path(output_path).write_bytes(b"FAKE_MP3")

    fake_mod = types.ModuleType("pipeline.podcastfy.generator")
    fake_mod.SimplePodcastGenerator = FakeGenerator
    monkeypatch.setitem(sys.modules, "pipeline.podcastfy.generator", fake_mod)

    # Write the marker matching the current inputs (as a real run would).
    (tmp_path / ".podcastfy-cache").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".podcastfy-cache" / "transcript.fingerprint").write_text(
        audio.transcript_fingerprint(brief, chosen, cfg), encoding="utf-8")

    result = audio.PodcastfyBackend().generate(brief=brief, run_dir=tmp_path,
                                               chosen=chosen, config=cfg)
    assert result.audio_path.exists()
    # Transcript is untouched: the cached one was reused, not regenerated.
    assert result.transcript_path.read_text(encoding="utf-8") == transcript_text


def test_podcastfy_backend_regenerates_when_transcript_stale(tmp_path, monkeypatch):
    """A transcript.md without (or with a stale) fingerprint is REGENERATED —
    e.g. a fresh full run on the same date or an edited brief must not reuse
    audio for the old selection."""
    from pipeline import audio

    (tmp_path / "transcript.md").write_text("<Person1>STALE nine-topic transcript</Person1>")

    brief = tmp_path / "podcast_brief.md"
    brief.write_text("# AI News Digest\n\n- [A](https://arxiv.org/abs/1)")
    chosen = [RankedItem(title="A", url="https://arxiv.org/abs/1", date="d",
                         body="b", source="arxiv", score=0.9)]
    cfg = {"per_source_words": 330, "depth_factor": 1.6}

    calls = {"transcripts": 0}

    class FakeGenerator:
        def __init__(self, papers_dir=None, web_dir=None, **kw):
            self.papers_dir = papers_dir
            self.web_dir = web_dir
            self.tts_model = kw.get("tts_model", "tng")
        def load_brief_and_sources(self, brief_path):
            return "=== INTRO ===\n=== END INTRO ==="
        def generate_transcript(self, combined, on_part=None):
            calls["transcripts"] += 1
            return "<Person1>fresh short transcript</Person1>"

    fake_mod = types.ModuleType("pipeline.podcastfy.generator")
    fake_mod.SimplePodcastGenerator = FakeGenerator
    monkeypatch.setitem(sys.modules, "pipeline.podcastfy.generator", fake_mod)

    result = audio.PodcastfyBackend().generate(brief=brief, run_dir=tmp_path,
                                               chosen=chosen, config=cfg)
    assert calls["transcripts"] == 1
    assert result.transcript_path.read_text(encoding="utf-8") == "<Person1>fresh short transcript</Person1>"
    # And a fingerprint marker is persisted for the next run.
    marker = tmp_path / ".podcastfy-cache" / "transcript.fingerprint"
    assert marker.read_text(encoding="utf-8") == audio.transcript_fingerprint(brief, chosen, cfg)


def test_podcastfy_backend_failure_returns_empty_audio(tmp_path, monkeypatch):
    """On exception, the backend returns an empty audio_path (brief is the
    guaranteed product) and any partial transcript."""
    from pipeline import audio

    class FakeGenerator:
        def __init__(self, papers_dir=None, web_dir=None, **kw):
            pass
        def load_brief_and_sources(self, brief_path):
            raise RuntimeError("boom")

    fake_mod = types.ModuleType("pipeline.podcastfy.generator")
    fake_mod.SimplePodcastGenerator = FakeGenerator
    monkeypatch.setitem(sys.modules, "pipeline.podcastfy.generator", fake_mod)

    brief = tmp_path / "podcast_brief.md"
    brief.write_text("# AI News Digest")
    chosen = [RankedItem(title="t", url="u", date="d", body="b", source="arxiv",
                         score=0.9)]

    result = audio.PodcastfyBackend().generate(
        brief=brief, run_dir=tmp_path, chosen=chosen,
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

    fake_result = generate.audio_mod.AudioResult(
        audio_path=tmp_path / "episode.mp3",
        transcript_path=tmp_path / "transcript.md",
        backend="podcastfy",
    )
    class FakeBackend:
        name = "podcastfy"
        def generate(self, brief, run_dir, chosen, transcript_in=None, config=None):
            return fake_result
    monkeypatch.setattr(generate.audio_mod, "PodcastfyBackend", lambda: FakeBackend())

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

    seen_brief = {}
    class FakeBackend:
        name = "podcastfy"
        def generate(self, brief, run_dir, chosen, transcript_in=None, config=None):
            seen_brief["path"] = brief
            seen_brief["chosen"] = list(chosen)
            return generate.audio_mod.AudioResult(
                audio_path=tmp_path / "episode.mp3",
                transcript_path=tmp_path / "transcript.md",
                backend="podcastfy",
            )
    monkeypatch.setattr(generate.audio_mod, "PodcastfyBackend", lambda: FakeBackend())

    # The run-dir brief must NOT pre-exist (so we can prove we don't write it).
    assert not (tmp_path / "podcast_brief.md").exists()
    ep = generate.generate(ranked, make_audio=True, brief_in=str(brief_path))
    # The audio backend was pointed at the edited brief path (not overwritten).
    assert seen_brief["path"] == tmp_path / "podcast_brief.md"
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
    class FakeBackend:
        name = "podcastfy"
        def generate(self, brief, run_dir, chosen, transcript_in=None, config=None):
            return generate.audio_mod.AudioResult(
                audio_path=tmp_path / "episode.mp3",
                transcript_path=tmp_path / "transcript.md", backend="podcastfy")
    monkeypatch.setattr(generate.audio_mod, "PodcastfyBackend", lambda: FakeBackend())
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

    class FakeBackend:
        name = "podcastfy"
        def generate(self, brief, run_dir, chosen, transcript_in=None, config=None):
            (run_dir / "episode.mp3").write_bytes(b"x")
            (run_dir / "transcript.md").write_text(
                "<Person1>Hello world</Person1>\n<Person2>How are you today</Person2>\n")
            return generate.audio_mod.AudioResult(
                audio_path=run_dir / "episode.mp3",
                transcript_path=run_dir / "transcript.md", backend="podcastfy")
    monkeypatch.setattr(generate.audio_mod, "PodcastfyBackend", lambda: FakeBackend())

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
    targets = {"short": 10.0, "medium": 17.5, "long": 30.0}
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


def test_config_budget_podcastfy_overrides_carries_caps():
    rc = config_mod.RunConfig(length="medium", depth="deep-dive")
    o = rc.podcastfy_overrides()
    assert o["per_source_words"] == 330
    assert o["intro_words"] + o["recap_words"] == 578
    assert o["max_num_chunks"] == 7


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
    from pipeline.podcastfy.content_generator import ContentCleanerMixin
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
    from pipeline.podcastfy.content_generator import LongFormContentGenerator
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

    from pipeline.podcastfy import generator as gen_mod

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
        "roles_person2": "AI researcher",
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
