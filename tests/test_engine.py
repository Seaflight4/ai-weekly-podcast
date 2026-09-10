"""Tests for the generic podcast_engine package.

No network, no LLM, no TTS. The audio backend is mocked.
"""
import json, pathlib, sys, types, base64

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from podcast_engine import Source, EngineConfig, EpisodeResult
from podcast_engine import engine as engine_mod
from podcast_engine.generator import SimplePodcastGenerator


# --- Source dataclass -------------------------------------------------------

def test_source_defaults():
    s = Source(title="Test")
    assert s.url == ""
    assert s.pdf_url is None
    assert s.kind == "blog"
    assert s.local_path is None


def test_source_arxiv_id_from_pdf_url():
    s = Source(title="P", url="https://arxiv.org/abs/2601.00001",
               pdf_url="https://arxiv.org/pdf/2601.00001", kind="arxiv")
    assert s.arxiv_id == "2601.00001"


def test_source_arxiv_id_none_without_pdf_url():
    s = Source(title="P", url="https://example.com", kind="blog")
    assert s.arxiv_id is None


# --- EngineConfig -----------------------------------------------------------

def test_engine_config_intro_text_includes_topic():
    cfg = EngineConfig(podcast_topic="RAG in enterprise search",
                      podcast_name="Tech Talk")
    intro = cfg._intro_text()
    assert "Tech Talk" in intro
    assert "RAG in enterprise search" in intro


def test_engine_config_overrides_has_podcast_topic():
    cfg = EngineConfig(podcast_topic="quantum computing")
    ov = cfg._overrides()
    assert ov["podcast_topic"] == "quantum computing"
    assert "podcast_name" in ov
    assert "audience_prompt" in ov


def test_engine_config_budget_derives_num_sources():
    cfg = EngineConfig(length="medium", depth="deep-dive")
    ov = cfg._overrides()
    assert ov["max_num_chunks"] >= 4
    assert ov["per_source_words"] > 0


# --- build_combined_content: pdf kind branch --------------------------------

def test_build_combined_content_includes_pdf_source(tmp_path):
    """A kind='pdf' source with a local_path is read via SimplePDFExtractor,
    not downloaded or fetched from the web."""
    # Create a minimal fake PDF.
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"fake pdf content")

    gen = object.__new__(SimplePodcastGenerator)
    gen.papers_dir = str(tmp_path / "papers")
    gen.web_dir = str(tmp_path / "web")
    gen.memory_context = {}
    gen.config_conversation = {
        "per_paper_chars": 10000,
        "per_paper_tail_chars": 5000,
        "per_web_chars": 12000,
    }
    gen.items = []

    # Mock the PDF extractor to avoid pymupdf.
    class FakeExtractor:
        def extract_text(self, path):
            return "Extracted PDF text content for the topic."

    # Patch SimplePDFExtractor in the generator module.
    import podcast_engine.generator as gen_module
    original = gen_module.SimplePDFExtractor
    gen_module.SimplePDFExtractor = FakeExtractor
    try:
        sources = [Source(title="My Doc", url="file://doc.pdf",
                          kind="pdf", local_path=str(pdf_path),
                          excerpt="A summary.")]
        combined = gen.build_combined_content(sources, intro_text="# Test Pod")
    finally:
        gen_module.SimplePDFExtractor = original

    assert "=== TOPIC: My Doc ===" in combined
    assert "SOURCE: pdf" in combined
    assert "Extracted PDF text content" in combined
    assert "=== RECAP ===" in combined


# --- generate_podcast: mocked backend --------------------------------------

def test_generate_podcast_returns_episode_result(tmp_path, monkeypatch):
    """generate_podcast() delegates to PodcastfyBackend and wraps the result."""
    from podcast_engine import audio_backend

    fake_audio = tmp_path / "episode.mp3"
    fake_audio.write_bytes(b"FAKE_MP3")
    fake_transcript = tmp_path / "transcript.md"
    fake_transcript.write_text("<Person1>hello</Person1>", encoding="utf-8")

    def fake_generate(self, **kw):
        return audio_backend.AudioResult(
            audio_path=fake_audio,
            transcript_path=fake_transcript,
            backend="podcastfy",
        )

    monkeypatch.setattr(audio_backend.PodcastfyBackend, "generate", fake_generate)

    sources = [Source(title="T", url="https://example.com", kind="blog")]
    config = EngineConfig(podcast_topic="test topic")
    result = engine_mod.generate_podcast(sources, config, run_dir=tmp_path)

    assert isinstance(result, EpisodeResult)
    assert result.audio_path == fake_audio
    assert result.transcript_path == fake_transcript
    assert result.backend == "podcastfy"
    assert result.transcript_words is not None


# --- generic prompt: no AI-news framing leaks -------------------------------

def test_longform_prompt_has_podcast_topic_not_ai_news():
    from podcast_engine.generator import LONGFORM_PROMPT
    assert "{podcast_topic}" in LONGFORM_PROMPT
    assert "AI research" not in LONGFORM_PROMPT
    assert "AI researchers" not in LONGFORM_PROMPT


def test_default_audience_is_generic():
    """When no audience_prompt is passed, the fallback is generic."""
    gen = object.__new__(SimplePodcastGenerator)
    gen.audience_prompt = None
    gen.podcast_topic = "test"
    gen.config_conversation = {
        "roles_person1": "co-host", "roles_person2": "co-host",
        "host1_name": "A", "host2_name": "B",
        "podcast_name": "Pod", "podcast_tagline": "",
        "output_language": "English",
    }
    gen.familiar_clause = "none"
    # We can't call generate_transcript (needs LLM), but we can check the
    # fallback audience text is generic by inspecting the source.
    import inspect
    source = inspect.getsource(SimplePodcastGenerator.generate_transcript)
    assert "general audience" in source
    assert "AI researchers at a leading" not in source
