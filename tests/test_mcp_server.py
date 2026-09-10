"""Tests for the MCP server: input normalization + job lifecycle.

No network, no LLM, no TTS. The engine's generate_podcast is mocked.
"""
import base64, json, pathlib, sys, types, uuid

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from podcast_engine import EngineConfig, EpisodeResult, Source
from mcp_server import inputs, jobs


# --- input normalization ----------------------------------------------------

def test_normalize_url_source_blog(tmp_path):
    sources = inputs.normalize_sources(
        [{"url": "https://example.com/article"}], tmp_path / "papers")
    assert len(sources) == 1
    assert sources[0].kind == "blog"
    assert sources[0].url == "https://example.com/article"


def test_normalize_arxiv_abs_url_classified_as_arxiv(tmp_path):
    sources = inputs.normalize_sources(
        [{"url": "https://arxiv.org/abs/2601.00001"}], tmp_path / "papers")
    assert sources[0].kind == "arxiv"
    assert sources[0].pdf_url == "https://arxiv.org/pdf/2601.00001"


def test_normalize_arxiv_pdf_url_classified_as_arxiv(tmp_path):
    sources = inputs.normalize_sources(
        [{"url": "https://arxiv.org/pdf/2601.00001"}], tmp_path / "papers")
    assert sources[0].kind == "arxiv"
    assert sources[0].pdf_url == "https://arxiv.org/pdf/2601.00001"


def test_normalize_base64_pdf_source(tmp_path):
    pdf_content = b"%PDF-1.4 fake pdf"
    b64 = base64.b64encode(pdf_content).decode()
    sources = inputs.normalize_sources(
        [{"pdf_base64": b64, "filename": "report.pdf", "title": "Quarterly Report"}],
        tmp_path / "papers")
    assert len(sources) == 1
    assert sources[0].kind == "pdf"
    assert sources[0].title == "Quarterly Report"
    assert sources[0].local_path is not None
    assert pathlib.Path(sources[0].local_path).read_bytes() == pdf_content


def test_normalize_base64_pdf_safe_filename(tmp_path):
    b64 = base64.b64encode(b"pdf").decode()
    sources = inputs.normalize_sources(
        [{"pdf_base64": b64, "filename": "../../etc/passwd"}], tmp_path / "papers")
    path = pathlib.Path(sources[0].local_path)
    # No directory traversal: the path is inside papers_dir (slashes replaced).
    assert path.parent == tmp_path / "papers"
    assert "/" not in path.name
    # The file actually exists at that path.
    assert path.exists()


def test_normalize_mixed_sources(tmp_path):
    b64 = base64.b64encode(b"pdf").decode()
    raw = [
        {"url": "https://arxiv.org/abs/2601.00001"},
        {"url": "https://blog.example.com/post"},
        {"pdf_base64": b64, "filename": "doc.pdf"},
    ]
    sources = inputs.normalize_sources(raw, tmp_path / "papers")
    assert len(sources) == 3
    assert [s.kind for s in sources] == ["arxiv", "blog", "pdf"]


def test_normalize_invalid_source_raises(tmp_path):
    with pytest.raises(ValueError):
        inputs.normalize_sources([{"foo": "bar"}], tmp_path / "papers")


# --- job lifecycle (mocked engine) ------------------------------------------

def test_job_lifecycle_pending_to_completed(tmp_path, monkeypatch):
    """submit() creates a pending job; the worker runs generate_podcast
    (mocked) and updates status to completed."""
    # Mock generate_podcast to return instantly.
    fake_transcript = tmp_path / "transcript.md"
    fake_transcript.write_text("<Person1>hi</Person1>", encoding="utf-8")
    fake_audio = tmp_path / "episode.mp3"
    fake_audio.write_bytes(b"mp3")

    fake_result = EpisodeResult(
        run_dir=tmp_path,
        audio_path=fake_audio,
        transcript_path=fake_transcript,
        backend="podcastfy",
    )
    monkeypatch.setattr(jobs, "generate_podcast", lambda **kw: fake_result)

    # Use a tmp DB path.
    monkeypatch.setattr(jobs, "_DB_PATH", tmp_path / "test_jobs.db")

    sources = [Source(title="T", url="https://example.com", kind="blog")]
    config = EngineConfig(podcast_topic="test")
    job_id = jobs.submit(sources, config)

    # Status should be pending or running initially.
    status = jobs.get(job_id)
    assert status["status"] in ("pending", "running", "completed")

    # Wait for the worker to finish (single submission, fast mock).
    import time
    for _ in range(50):
        status = jobs.get(job_id)
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert status["status"] == "completed"
    assert status["transcript"] == "<Person1>hi</Person1>"
    assert "episode.mp3" in status["audio_path"]


def test_job_lifecycle_failed_on_error(tmp_path, monkeypatch):
    """When generate_podcast raises, the job is marked failed."""
    def boom(**kw):
        raise RuntimeError("TTS service down")

    monkeypatch.setattr(jobs, "generate_podcast", boom)
    monkeypatch.setattr(jobs, "_DB_PATH", tmp_path / "test_jobs.db")

    sources = [Source(title="T", url="https://example.com", kind="blog")]
    config = EngineConfig(podcast_topic="test")
    job_id = jobs.submit(sources, config)

    import time
    for _ in range(50):
        status = jobs.get(job_id)
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert status["status"] == "failed"
    assert "TTS service down" in status["error"]


def test_job_get_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_DB_PATH", tmp_path / "test_jobs.db")
    result = jobs.get("nonexistent-job-id")
    assert result["status"] == "not_found"


# --- server tool + resource registration ------------------------------------

def test_server_registers_generate_podcast_tool():
    from mcp_server.server import server
    import asyncio
    tools = asyncio.run(server.list_tools())
    names = [t.name for t in tools]
    assert "generate_podcast" in names


def test_server_registers_job_resource_template():
    from mcp_server.server import server
    import asyncio
    templates = asyncio.run(server.list_resource_templates())
    uris = [t.uri_template for t in templates]
    assert "podcast://jobs/{job_id}" in uris
