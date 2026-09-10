"""Audio-generation backend.

The ``generate`` stage feeds the week's top items to the vendored podcastfy
stack: full arXiv PDFs + HN/web page text are re-fetched into a per-run cache,
a two-host transcript is generated via an OpenAI-compatible LLM (DeepSeek on
the SkaiNet gateway — same ``SKAINET_API_KEY`` the pipeline already uses), and
voice-cloned audio is synthesized via the TNG qwen3 TTS service.

The backend takes the chosen source items + the run directory, produces an
``episode.mp3`` and a ground-truth ``transcript.md`` (so the ``transcribe``
stage is a no-op), and returns an :class:`AudioResult`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline import RankedItem

logger = logging.getLogger(__name__)

# Approximate clean spoken pace measured from past episodes (~130-150 wpm, so
# ~0.40-0.45 s per spoken word). A garbled/looping TTS part is typically many
# multiples of this — the part-audio sanity check uses it to catch such parts
# before they are merged into the episode as meaningless noise.
SECONDS_PER_WORD = 0.45


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


def transcript_fingerprint(chosen: list,
                           config: dict | None,
                           memory_context: dict | None = None,
                           sources: list | None = None) -> str:
    """Fingerprint of everything that determines the transcript's content.

    Covers the source URLs/titles, the generator config (word budgets, depth,
    audience, familiar topics). A cached transcript may only be reused when
    this value is unchanged; otherwise the LLM would silently re-synthesize
    audio for a different selection/config.

    Cross-episode memory (``memory_context``) also changes what the transcript
    says (a continuation thread), so it is part of the fingerprint too — a
    changed memory never reuses a stale cached transcript.
    """
    brief_text = ""
    if sources is not None:
        brief_text = "\n".join(
            f"{getattr(s, 'title', '')}|{getattr(s, 'url', '')}"
            for s in sorted(sources, key=lambda s: (getattr(s, "url", "") or "").strip().lower())
        )
    urls = sorted((getattr(c, "url", "") or "").strip().lower() for c in chosen)
    cfg = json.dumps(config or {}, sort_keys=True, default=str)
    mem = json.dumps(memory_context or {}, sort_keys=True)
    payload = f"{brief_text}\x00{urls}\x00{cfg}\x00{mem}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fingerprint_matches(marker: pathlib.Path, candidate: str) -> bool:
    try:
        return marker.read_text(encoding="utf-8").strip() == candidate
    except OSError:
        return False


def _spoken_words(text: str) -> int:
    """Word count of a part's dialogue, ignoring ``<PersonN>`` tags."""
    return len(re.sub(r"</?Person[12]>", "", text or "").split())


def part_audio_ok(part_text: str, part_path: pathlib.Path) -> bool:
    """True if a synthesized part's audio duration is plausible for its words.

    TTS occasionally returns a garbled/looping part whose audio is far longer
    than its spoken text can account for (and is meaningless noise). Check the
    duration against the word-derived expectation with a generous ceiling, so
    the caller can re-synthesize (and eventually drop) the part instead of
    merging noise into the episode. Returns False on unreadable audio too.
    """
    from pydub import AudioSegment
    try:
        dur = AudioSegment.from_file(str(part_path), format="mp3").duration_seconds
    except Exception:
        return False
    expected = max(1, _spoken_words(part_text)) * SECONDS_PER_WORD
    max_ok = max(expected * 2.5, expected + 90.0)
    return dur <= max_ok


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
        run_dir: pathlib.Path,
        chosen: list | None = None,
        transcript_in: pathlib.Path | None = None,
        config: dict | None = None,
        memory_context: dict | None = None,
        items: list | None = None,
        sources: list | None = None,
        intro_text: str = "",
    ) -> AudioResult:
        from .generator import SimplePodcastGenerator

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
            memory_context=memory_context or {},
            items=items or [],
            **(config or {}),
        )
        # Marker recording which brief/selection/config/memory produced a cached
        # transcript; reuse is only allowed when it still matches (so a full
        # re-run of the same date or an edited brief never re-synthesizes
        # audio from a stale transcript).
        marker_path = cache_dir / "transcript.fingerprint"
        fingerprint = transcript_fingerprint(chosen or [], config,
                                             memory_context=memory_context,
                                             sources=sources)
        try:
            # Source fetching only happens when we need the LLM (no cached
            # transcript for these inputs). When reusing a matching transcript
            # we skip the brief fetch + LLM call entirely, so retrying TTS
            # after a transient outage is fast.
            t_trans = 0.0
            t_audio = 0.0
            use_pipeline = False
            if transcript_in is not None:
                src = pathlib.Path(transcript_in)
                if not src.exists():
                    raise FileNotFoundError(
                        f"--transcript-in not found: {src}")
                transcript = src.read_text(encoding="utf-8")
                transcript_out = src
                print(f"[podcastfy] reusing cached transcript {src.name} "
                      f"({len(transcript)} chars); skipping LLM")
            elif transcript_out.exists() and _fingerprint_matches(marker_path, fingerprint):
                transcript = transcript_out.read_text(encoding="utf-8")
                print(f"[podcastfy] reusing existing {transcript_out.name} "
                      f"({len(transcript)} chars); skipping LLM "
                      f"(brief/selection/config unchanged)")
            else:
                use_pipeline = True
                t0 = time.perf_counter()
                print("[podcastfy] fetching provided sources...")
                combined = gen.build_combined_content(sources,
                                                      intro_text=intro_text)
                t_fetch = time.perf_counter() - t0
                print(f"[podcastfy] fetched sources in {t_fetch:.1f}s")

                # Build TTS early (health check + voice loading) BEFORE
                # transcript so audio synthesis can overlap with LLM calls.
                from .text_to_speech import TextToSpeech
                from .tts.providers.tng import TNGTTS
                import os as _os
                from concurrent.futures import ThreadPoolExecutor

                t0 = time.perf_counter()
                TNGTTS.wait_until_healthy(_os.environ.get("SKAINET_API_KEY", ""))
                print(f"[podcastfy] TTS healthy ({time.perf_counter()-t0:.1f}s)")

                part_audio_dir = cache_dir / "audio-parts"
                part_audio_dir.mkdir(parents=True, exist_ok=True)
                tts = TextToSpeech(
                    model=gen.tts_model,
                    conversation_config={"text_to_speech": {
                        "temp_audio_dir": str(temp_audio_dir),
                    }},
                )

                # Pipeline: transcript runs in the main thread; each completed
                # part is submitted to a 1-worker TTS pool. Each part TTS call
                # internally uses max_workers=4 for piece-level concurrency, so
                # peak load is 1 LLM + 4 TTS HTTP (different endpoints, no
                # shared rate limit). TTS per-part (~10s) is faster than
                # transcript per-part (~16s), so TTS keeps up — total is
                # transcript-bound.
                part_futures: dict = {}
                tts_pool = ThreadPoolExecutor(max_workers=1)

                def on_part(idx: int, part_text: str):
                    part_path = str(part_audio_dir / f"part_{idx:03d}.mp3")
                    def _tts_part():
                        # TTS occasionally synthesizes a garbled/looping part
                        # whose audio is far longer (and noisier) than its words
                        # allow. Verify each part's duration against its spoken
                        # length and re-synthesize once; drop the part (never
                        # merge noise) if it is still out of proportion.
                        for attempt in (1, 2):
                            try:
                                tts.convert_to_speech(part_text, part_path)
                            except Exception as e:
                                logger.warning(
                                    "[tts] part %d TTS error (attempt %d): %s",
                                    idx, attempt, e)
                                return None
                            if part_audio_ok(part_text, pathlib.Path(part_path)):
                                return part_path
                            logger.warning(
                                "[tts] part %d audio out-of-band for its word "
                                "count (attempt %d) — re-synthesizing", idx, attempt)
                        logger.error(
                            "[tts] part %d audio still out-of-band after retry "
                            "— dropping garbled part from episode", idx)
                        return None
                    part_futures[idx] = tts_pool.submit(_tts_part)

                t0 = time.perf_counter()
                print("[podcastfy] generating transcript + audio (pipeline)...")
                transcript = gen.generate_transcript(combined, on_part=on_part)
                t_trans = time.perf_counter() - t0
                transcript_out.write_text(transcript, encoding="utf-8")
                marker_path.write_text(fingerprint, encoding="utf-8")
                print(f"[podcastfy] transcript -> {transcript_out.name} "
                      f"({len(transcript)} chars) in {t_trans:.1f}s")

                # Wait for all part TTS jobs, then merge part mp3s in order.
                t0 = time.perf_counter()
                tts_pool.shutdown(wait=True)

                from pydub import AudioSegment
                combined_audio = AudioSegment.empty()
                n_parts = len(part_futures)
                for idx in sorted(part_futures.keys()):
                    try:
                        part_path = part_futures[idx].result()
                        if part_path is None:
                            logger.warning(
                                "Skipping part %d (no usable audio after retries)", idx)
                            continue
                        p = pathlib.Path(part_path)
                        if p.exists() and p.stat().st_size > 0:
                            combined_audio += AudioSegment.from_file(
                                str(p), format="mp3")
                        else:
                            logger.warning("Skipping empty part audio: %s", part_path)
                    except Exception as e:
                        logger.warning("Part %d TTS failed: %s (skipping)", idx, e)
                combined_audio.export(str(audio_out), format="mp3")
                t_audio = time.perf_counter() - t0
                print(f"[podcastfy] audio -> {audio_out.name} in {t_audio:.1f}s "
                      f"(merged {n_parts} parts)")

            if not use_pipeline:
                # Serial TTS (cached transcript: no pipeline overlap needed).
                t0 = time.perf_counter()
                print("[podcastfy] checking TTS service health...")
                from .tts.providers.tng import TNGTTS
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
