"""Contract tests for the lean pipeline (collect -> rank -> generate).

No network, no LLM. The judge is monkeypatched; store is isolated to a tmp dir.
"""
import json, pathlib, sys, types
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipeline import Item, RankedItem
from pipeline import rank, store, generate, transcribe


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
    # 15 items >= 0.8 (within floor=10, cap=20) -> all 15 qualify.
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
    assert len(chosen) == 15
    assert all(c.score >= 0.8 for c in chosen)


def test_select_sources_cap_on_strong_week():
    # 25 items >= 0.8, cap=20 -> cut to 20 (top-scored).
    ranked = [
        RankedItem(title=f"t{i}", url=f"u{i}", date="d", body="b", source="arxiv",
                   score=0.95 - i * 0.005)
        for i in range(25)
    ] + [
        RankedItem(title=f"low{i}", url=f"l{i}", date="d", body="b", source="arxiv",
                   score=0.5)
        for i in range(5)
    ]
    chosen = generate.select_sources(ranked)
    assert len(chosen) == 20
    assert chosen[0].score == 0.95
    assert chosen[-1].score == 0.95 - 19 * 0.005


def test_select_sources_floor_on_weak_week():
    # Only 2 items >= 0.8, floor=10 -> pad to 10 with next-highest.
    ranked = [
        RankedItem(title="a", url="u1", date="d", body="b", source="arxiv",
                   score=0.9),
        RankedItem(title="b", url="u2", date="d", body="b", source="arxiv",
                   score=0.82),
    ] + [
        RankedItem(title=f"low{i}", url=f"l{i}", date="d", body="b",
                   source="arxiv", score=0.5 - i * 0.05)
        for i in range(20)
    ]
    chosen = generate.select_sources(ranked)
    assert len(chosen) == 10
    assert chosen[0].score == 0.9
    assert chosen[-1].score < 0.8  # padded from below threshold


def test_select_sources_empty_pool():
    assert generate.select_sources([]) == []


def test_select_sources_small_pool_below_floor():
    # Only 3 items total, all below 0.8; floor=10 but pool=3 -> return all 3.
    ranked = [
        RankedItem(title=f"t{i}", url=f"u{i}", date="d", body="b", source="arxiv",
                   score=0.5)
        for i in range(3)
    ]
    chosen = generate.select_sources(ranked)
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


def test_generate_writes_episode(tmp_path, monkeypatch):
    # 3 items >= 0.8, rest below. floor=10 -> 3 qualify, padded to 10.
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
    assert len(ep.manifest) == 10
    assert ep.manifest[0].score == 0.9
    assert ep.manifest[-1].score < 0.8


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
        def load_brief_and_sources(self, brief_path):
            return "=== INTRO ===\n=== END INTRO ==="
        def generate_transcript(self, combined):
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
    assert result.audio_path.read_bytes() == b"FAKE_MP3"
    assert result.transcript_path == tmp_path / "transcript.md"
    assert "<Person1>hello</Person1>" in result.transcript_path.read_text()
    # Per-run cache dirs were created under the run dir.
    assert (tmp_path / ".podcastfy-cache").is_dir()


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
        def generate(self, brief, run_dir, chosen, transcript_in=None):
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
