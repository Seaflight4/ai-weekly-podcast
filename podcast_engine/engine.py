"""Generic podcast generation engine.

This module is the domain-agnostic entry point for turning a list of sources
(documents, URLs, PDFs) into a finished podcast episode (transcript + audio).
Both the AI-news pipeline (``pipeline/generate.py``) and the MCP server
(``mcp_server/``) are consumers of this engine.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Callable, Optional

from .sources import Source
from .audio_backend import AudioResult, PodcastfyBackend


# --- budget constants (mirrors pipeline.config for engine standalone use) ----

WPM = 165.0
LENGTH_MINUTES = {"short": 10.0, "medium": 18.0, "long": 30.0}
DEPTH_MINUTES = {"brief": 1.0, "deep-dive": 2.0}
DEPTH_FACTOR = {"brief": 0.8, "deep-dive": 1.6}
INTRO_RECAP_FRACTION = 0.20
INTRO_SHARE = 0.40
MEMORY_WORDS = 90


def _budget(length: str, depth: str) -> dict:
    total_words = round(LENGTH_MINUTES[length] * WPM)
    ir_words = round(total_words * INTRO_RECAP_FRACTION)
    intro_words = round(ir_words * INTRO_SHARE)
    recap_words = ir_words - intro_words
    per_source_words = round(DEPTH_MINUTES[depth] * WPM)
    topic_words = total_words - ir_words
    n = max(4, min(30, round(topic_words / per_source_words)))
    return {
        "total_words": total_words,
        "intro_words": intro_words,
        "recap_words": recap_words,
        "per_source_words": per_source_words,
        "num_sources": n,
    }


# --- config + result ---------------------------------------------------------

@dataclass
class EngineConfig:
    """Configuration for a single podcast generation run.

    All framing is caller-supplied; the engine has no domain defaults.
    """
    podcast_topic: str = "the provided sources"
    podcast_name: str = "Podcast"
    podcast_tagline: str = ""
    host1_name: str = "Brian"
    host2_name: str = "Tina"
    roles_person1: str = "co-host"
    roles_person2: str = "co-host"
    audience: str = (
        "a general audience interested in the topic. Define specialized "
        "terms where helpful and keep explanations concrete and "
        "self-contained."
    )
    familiar_topics: list[str] = field(default_factory=list)
    length: str = "medium"
    depth: str = "deep-dive"

    def _overrides(self) -> dict:
        b = _budget(self.length, self.depth)
        return {
            "depth_factor": DEPTH_FACTOR[self.depth],
            "max_num_chunks": b["num_sources"],
            "per_source_words": b["per_source_words"],
            "intro_words": b["intro_words"],
            "recap_words": b["recap_words"],
            "memory_words": MEMORY_WORDS,
            "audience_prompt": self.audience,
            "familiar_clause": ", ".join(self.familiar_topics) if self.familiar_topics else "none",
            "podcast_topic": self.podcast_topic,
            "podcast_name": self.podcast_name,
            "podcast_tagline": self.podcast_tagline,
            "host1_name": self.host1_name,
            "host2_name": self.host2_name,
            "roles_person1": self.roles_person1,
            "roles_person2": self.roles_person2,
        }

    def _intro_text(self) -> str:
        tagline = f" — {self.podcast_tagline}" if self.podcast_tagline else ""
        return f"# {self.podcast_name}{tagline}\n\nA podcast about {self.podcast_topic}.\n"


@dataclass
class EpisodeResult:
    """Outcome of a podcast generation run."""
    run_dir: pathlib.Path
    audio_path: pathlib.Path
    transcript_path: pathlib.Path
    backend: str = "podcastfy"
    duration_sec: Optional[float] = None
    transcript_words: Optional[int] = None


# --- entry point -------------------------------------------------------------

def generate_podcast(
    sources: list[Source],
    config: EngineConfig,
    run_dir: pathlib.Path | str | None = None,
    memory_context: dict | None = None,
    on_part: Callable[[int, str], None] | None = None,
    transcript_in: pathlib.Path | str | None = None,
) -> EpisodeResult:
    """Generate a podcast episode from a list of provided sources.

    Fetches full content for each source (web pages, arXiv PDFs, or reads
    local PDFs), generates a two-host transcript via the LLM, and synthesizes
    audio via the TNG TTS service.

    Args:
        sources: List of :class:`Source` objects (url, pdf_base64-decoded, etc.)
        config: Engine configuration (topic, audience, length, depth, hosts).
        run_dir: Directory to write outputs into. Created if it doesn't exist.
            Defaults to ``data/engine/<timestamp>/``.
        memory_context: Optional cross-episode memory (empty = no memory).
        on_part: Optional callback ``on_part(idx, text)`` fired after each
            transcript part is generated (for progress reporting).
        transcript_in: Optional path to a cached transcript file; when set,
            skips the LLM and goes straight to TTS.

    Returns:
        :class:`EpisodeResult` with paths to the audio and transcript.
    """
    if run_dir is None:
        import datetime
        ts = datetime.datetime.now().strftime("%d-%m-%Y-%H%M%S")
        run_dir = pathlib.Path("data/engine") / ts
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = pathlib.Path(transcript_in) if transcript_in else None

    backend = PodcastfyBackend()
    result: AudioResult = backend.generate(
        run_dir=run_dir,
        chosen=sources,
        sources=sources,
        intro_text=config._intro_text(),
        config=config._overrides(),
        memory_context=memory_context or {},
        items=sources,
        transcript_in=transcript_path,
    )

    duration_sec = None
    transcript_words = None
    if result.audio_path and result.audio_path.exists():
        duration_sec = _audio_duration_sec(result.audio_path)
    if result.transcript_path and result.transcript_path.exists():
        transcript_words = _transcript_words(
            result.transcript_path.read_text(encoding="utf-8"))

    return EpisodeResult(
        run_dir=run_dir,
        audio_path=result.audio_path,
        transcript_path=result.transcript_path or run_dir / "transcript.md",
        backend=result.backend,
        duration_sec=duration_sec,
        transcript_words=transcript_words,
    )


def _audio_duration_sec(audio_path: pathlib.Path) -> float | None:
    try:
        import subprocess
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(audio_path)],
            stderr=subprocess.DEVNULL, timeout=10)
        return float(out.strip())
    except Exception:
        return None


def _transcript_words(text: str) -> int:
    import re
    return len(re.sub(r"</?Person[12]>", "", text or "").split())
