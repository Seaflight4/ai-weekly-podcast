"""Audio-generation backend.

The ``generate`` stage feeds the week's top items to the vendored podcastfy
stack: full arXiv PDFs + HN/web page text are re-fetched into a per-run cache,
a two-host transcript is generated via an OpenAI-compatible LLM (DeepSeek on
the SkaiNet gateway — same ``SKAINET_API_KEY`` the pipeline already uses), and
voice-cloned audio is synthesized via the TNG qwen3 TTS service.

The backend takes a written ``podcast_brief.md`` + the chosen source items +
the run directory, produces an ``episode.mp3`` and a ground-truth
``transcript.md`` (so the ``transcribe`` stage is a no-op), and returns an
:class:`AudioResult`.
"""
from __future__ import annotations

import logging
import pathlib
import time
from dataclasses import dataclass

from . import RankedItem

logger = logging.getLogger(__name__)


# --- result -----------------------------------------------------------------

@dataclass
class AudioResult:
    """Outcome of an audio-generation run.

    Attributes:
        audio_path: Path to the produced ``episode.mp3``, or empty when the
            backend failed (the brief is still the guaranteed product).
        transcript_path: Path to the ground-truth ``transcript.md`` produced
            by the backend. ``None`` only when generation failed before the
            transcript was written — the ``transcribe`` stage then runs Whisper
            on the audio.
        backend: Name of the backend that produced this result.
    """
    audio_path: pathlib.Path
    transcript_path: pathlib.Path | None
    backend: str


# --- Podcastfy backend (vendored DeepSeek + TNG qwen3 TTS) -------------------

class PodcastfyBackend:
    """Vendored podcastfy audio backend.

    Re-fetches full arXiv PDFs (10k head + 3k tail) and HN/web page text (22k)
    into a per-run cache, generates a two-host transcript via an
    OpenAI-compatible LLM (DeepSeek on the SkaiNet gateway — same
    ``SKAINET_API_KEY`` the pipeline already uses), and synthesizes
    voice-cloned audio via the TNG qwen3 TTS service.

    Produces a ground-truth ``transcript.md`` alongside the audio, so the
    ``transcribe`` stage is a no-op for this backend.
    """

    name = "podcastfy"

    def generate(
        self,
        brief: pathlib.Path,
        run_dir: pathlib.Path,
        chosen: list[RankedItem],
        transcript_in: pathlib.Path | None = None,
    ) -> AudioResult:
        from .podcastfy.generator import SimplePodcastGenerator

        audio_out = run_dir / "episode.mp3"
        transcript_out = run_dir / "transcript.md"
        # Per-run cache so re-runs don't collide and don't pollute the repo root.
        cache_dir = run_dir / ".podcastfy-cache"
        papers_dir = cache_dir / "papers"
        web_dir = cache_dir / "web"
        temp_audio_dir = cache_dir / "audio-tmp"
        cache_dir.mkdir(parents=True, exist_ok=True)

        gen = SimplePodcastGenerator(
            papers_dir=str(papers_dir),
            web_dir=str(web_dir),
        )
        try:
            # Source fetching only happens when we need the LLM (i.e. no cached
            # transcript). When reusing a transcript we skip the brief fetch +
            # LLM call entirely, so retrying TTS after a transient outage is fast.
            t_trans = 0.0
            if transcript_in is not None:
                src = pathlib.Path(transcript_in)
                if not src.exists():
                    raise FileNotFoundError(
                        f"--transcript-in not found: {src}")
                transcript = src.read_text(encoding="utf-8")
                transcript_out = src
                print(f"[podcastfy] reusing cached transcript {src.name} "
                      f"({len(transcript)} chars); skipping LLM")
            elif transcript_out.exists():
                transcript = transcript_out.read_text(encoding="utf-8")
                print(f"[podcastfy] reusing existing {transcript_out.name} "
                      f"({len(transcript)} chars); skipping LLM")
            else:
                t0 = time.perf_counter()
                print("[podcastfy] loading brief and fetching sources...")
                combined = gen.load_brief_and_sources(str(brief))
                t_fetch = time.perf_counter() - t0
                print(f"[podcastfy] fetched sources in {t_fetch:.1f}s")

                t0 = time.perf_counter()
                print("[podcastfy] generating transcript...")
                transcript = gen.generate_transcript(combined)
                t_trans = time.perf_counter() - t0
                transcript_out.write_text(transcript, encoding="utf-8")
                print(f"[podcastfy] transcript -> {transcript_out.name} "
                      f"({len(transcript)} chars) in {t_trans:.1f}s")

            t0 = time.perf_counter()
            print("[podcastfy] checking TTS service health...")
            from .podcastfy.tts.providers.tng import TNGTTS
            import os as _os
            TNGTTS.wait_until_healthy(_os.environ.get("SKAINET_API_KEY", ""))
            t_health = time.perf_counter() - t0
            print(f"[podcastfy] TTS healthy ({t_health:.1f}s)")

            t0 = time.perf_counter()
            print("[podcastfy] generating audio...")
            gen.generate_audio(
                transcript,
                str(audio_out),
                temp_audio_dir=str(temp_audio_dir),
            )
            t_audio = time.perf_counter() - t0
            print(f"[podcastfy] audio -> {audio_out.name} in {t_audio:.1f}s")

            total = t_trans + t_audio
            print(f"[podcastfy] total: {total:.1f}s "
                  f"(transcript {t_trans:.1f}s + audio {t_audio:.1f}s)")
        except Exception as e:
            logger.exception("podcastfy audio generation failed")
            print(f"      [warn] podcastfy audio generation failed: {e}")
            print(f"      [warn] {audio_out.name} not produced; "
                  f"the brief is still available")
            return AudioResult(
                audio_path=pathlib.Path(""),
                transcript_path=transcript_out if transcript_out.exists() else None,
                backend=self.name,
            )

        return AudioResult(
            audio_path=audio_out,
            transcript_path=transcript_out,
            backend=self.name,
        )
