from . import RankedItem, Episode
from . import store
from . import audio as audio_mod
from . import config as config_mod
import pathlib, datetime, re, subprocess


def generate(ranked: list[RankedItem], make_audio: bool = True,
             date: str | None = None,
             transcript_in: str | None = None,
             brief_in: str | None = None,
             config: config_mod.RunConfig | None = None) -> Episode:
    """Feed the week's top items to the podcastfy audio backend and produce audio.

    By default the source set is the top-N by score above a quality floor,
    where N is derived from the run config's podcast length / per-topic depth
    (``config.RunConfig``), then written to ``podcast_brief.md``. The brief is
    a flat, source-grouped digest aimed at the configured audience.

    ``brief_in`` overrides the brief with an existing markdown file (e.g. one
    the user edited in the service UI to drop items). When set, selection is
    skipped and the episode manifest is reconstructed by matching the brief's
    URLs against ``ranked``; ``episode.json`` records ``selection_source:
    "customized"`` so the UI can show that the user curated the set.

    ``transcript_in`` reuses a cached transcript file and skips the LLM step,
    going straight to TTS — useful for retrying audio after a transient TTS
    outage without paying the multi-minute LLM cost again.
    """
    run = store.run_dir(date)
    brief = run / "podcast_brief.md"
    if config is None:
        config = config_mod.RunConfig()

    if brief_in is not None:
        src_brief = pathlib.Path(brief_in)
        if not src_brief.exists():
            raise SystemExit(f"--brief-in: {src_brief} not found")
        chosen = _chosen_from_brief(ranked, src_brief.read_text(encoding="utf-8"))
        selection_source = "customized"
        # Canonicalize the edited brief into the run dir so the run always
        # carries the brief that produced this audio (and the UI reads it
        # from one place). When the service passes --brief-in pointing at the
        # run-dir brief it already wrote, this is a same-content overwrite.
        brief.write_text(src_brief.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"      using edited brief {src_brief.name} ({len(chosen)} items)")
    else:
        chosen = select_sources(ranked, target_n=config.num_sources())
        selection_source = "auto"
        start, end = config.resolve_window()
        brief.write_text(_brief_text(chosen, _group_by_source(chosen),
                                     audience_desc=config.audience_prompt(),
                                     episode_date=end.isoformat()), encoding="utf-8")

    by_source = _group_by_source(chosen)

    # Forward monitor: surface low gate_score among aired items. A recurring
    # low-score-but-aired item signals the small-model gate is misaligned with
    # the big-LLM judge and the prefilter floor should be relaxed.
    scored = [c.gate_score for c in chosen if c.gate_score]
    if scored:
        print(f"      monitor: chosen gate_score min={min(scored):.2f} "
              f"median={sorted(scored)[len(scored)//2]:.2f} max={max(scored):.2f} "
              f"({len(scored)}/{len(chosen)} scored)")

    transcript_source = "whisper"
    transcript_path: pathlib.Path | None = None
    backend_name: str | None = None
    if make_audio:
        try:
            result = audio_mod.PodcastfyBackend().generate(
                brief=brief, run_dir=run, chosen=chosen,
                transcript_in=pathlib.Path(transcript_in) if transcript_in else None,
                config=config.podcastfy_overrides(),
            )
            audio = result.audio_path
            backend_name = result.backend
            if result.transcript_path is not None:
                transcript_path = result.transcript_path
                transcript_source = "generated"
        except Exception as e:
            print(f"      [warn] audio generation failed: {e}")
            print(f"      [warn] episode.mp3 not produced; the brief is still available")
            audio = pathlib.Path("")
    else:
        print("      skipping audio (--no-audio)")
        audio = pathlib.Path("")

    ep = Episode(
        audio_path=str(audio) if audio else "",
        manifest=chosen,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config=config.to_dict(),
    )

    # Record measured duration + spoken-word count (display telemetry only —
    # length is enforced upstream by the word budget, not corrected here).
    duration_sec = None
    transcript_words = None
    if make_audio and audio and audio.exists():
        duration_sec = _audio_duration_sec(audio)
    if transcript_path is not None and transcript_path.exists():
        transcript_words = _transcript_words(
            transcript_path.read_text(encoding="utf-8"))
    if duration_sec:
        mm, ss = divmod(int(duration_sec), 60)
        print(f"      episode duration ~{mm}m {ss:02d}s · "
              f"{transcript_words or '?'} transcript words")

    # Persist the resolved config alongside the run for reproducibility and so
    # ``--only generate`` re-runs can be replayed with the same knobs.
    (run / "config.yaml").write_text(config.to_yaml(), encoding="utf-8")
    store.write("episode.json", {
        "audio_path": str(audio),
        "created_at": ep.created_at,
        "manifest": [r.__dict__ for r in ep.manifest],
        "transcript_source": transcript_source,
        "backend": backend_name,
        "selection_source": selection_source,
        "config": ep.config,
        "duration_sec": duration_sec,
        "transcript_words": transcript_words,
    }, date=date)
    print(f"      wrote {brief.name} ({len(chosen)} items)")
    return ep


def _chosen_from_brief(ranked: list[RankedItem], brief_text: str) -> list[RankedItem]:
    """Reconstruct the chosen items from an edited brief by URL-matching.

    The brief format is fixed (see ``_brief_text``): each source is a markdown
    bullet ``- [Title](url)``. We extract the URLs in order and keep the
    matching ``RankedItem`` from the pool, preserving the brief's order so the
    manifest reflects what the user curated. URLs absent from the pool (e.g.
    a hand-pasted extra) are skipped silently.
    """
    import re
    by_url = {(r.url or "").strip().rstrip("/").lower(): r for r in ranked}
    chosen: list[RankedItem] = []
    for m in re.finditer(r"^- \[[^\]]+\]\(([^)]+)\)", brief_text, re.MULTILINE):
        url = m.group(1).strip().rstrip("/").lower()
        r = by_url.get(url)
        if r is not None:
            chosen.append(r)
    return chosen


# --- source selection: top-N by score above a quality floor -----------------

def select_sources(ranked: list[RankedItem],
                   target_n: int | None = None) -> list[RankedItem]:
    """Pick the source set for the audio backend: top-N by score above a floor.

    ``target_n`` is the desired source count, derived from the run config's
    podcast length / per-topic depth (e.g. 9 for medium + deep-dive). We take
    the top ``target_n`` items by score that clear ``MIN_SCORE_FLOOR`` so a
    high target never airs weak items; if fewer than ``target_n`` clear the
    floor we air fewer (no padding with junk). ``target_n`` defaults to the
    derived count of the default config when omitted.
    """
    sorted_items = sorted(ranked, key=lambda r: r.score, reverse=True)
    if not sorted_items:
        return []
    if target_n is None:
        target_n = config_mod.RunConfig().num_sources()
    target_n = max(MIN_SOURCES, min(MAX_SOURCES, target_n))
    pass_floor = [r for r in sorted_items if r.score >= MIN_SCORE_FLOOR]
    chosen = pass_floor[:target_n]
    if chosen:
        print(f"      select: top {target_n} by score above floor {MIN_SCORE_FLOOR} "
              f"-> {len(chosen)} items (score {chosen[0].score:.3f}..{chosen[-1].score:.3f})")
    else:
        print(f"      select: nothing scored >= {MIN_SCORE_FLOOR}; airing nothing")
    return chosen


# --- brief: a flat, source-grouped digest fed to the backend as a source ----

def _brief_text(chosen: list[RankedItem],
                by_source: dict[str, list[RankedItem]],
                audience_desc: str = "AI researchers",
                episode_date: str | None = None) -> str:
    day = episode_date or datetime.date.today().isoformat()
    lines = [
        f"# AI News Digest — {day}",
        "",
        f"A pipeline-generated, source-grouped digest of this week's AI work. "
        f"Audience: {audience_desc}.",
        "",
    ]
    for source in ("arxiv", "hn"):
        items = by_source.get(source, [])
        if not items:
            continue
        label = "arXiv papers" if source == "arxiv" else "Hacker News stories"
        lines += [f"## {label} ({len(items)})", ""]
        for m in items:
            pdf = _paper_pdf_url(m)
            link = f"- [{m.title}]({m.url})"
            if pdf and pdf != m.url:
                link += f" · [PDF]({pdf})"
            link += f" — score {m.score:.2f}"
            lines.append(link)
            if m.body:
                lines.append(f"  > {m.body[:500]}")
        lines.append("")
    return "\n".join(lines)


def _group_by_source(items: list[RankedItem]) -> dict[str, list[RankedItem]]:
    out: dict[str, list[RankedItem]] = {}
    for it in items:
        out.setdefault(it.source, []).append(it)
    return out


def _paper_pdf_url(item: RankedItem) -> str | None:
    ARXIV_ABS = "https://arxiv.org/abs/"
    ARXIV_PDF = "https://arxiv.org/pdf/"
    if item.url.startswith(ARXIV_ABS):
        return item.url.replace(ARXIV_ABS, ARXIV_PDF, 1)
    return None


# Source-set selection: top-N by score above a quality floor. The floor keeps
# a high source count from airing weak items; N is derived from the config.
MIN_SCORE_FLOOR = 0.5
MIN_SOURCES = 4
MAX_SOURCES = 30


def _audio_duration_sec(path: pathlib.Path) -> float | None:
    """Episode mp3 duration in seconds (display telemetry). Uses ffprobe
    (ffmpeg is a hard dependency for pydub); returns None if unavailable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            return round(float(out.stdout.strip()), 1)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _transcript_words(text: str) -> int:
    """Spoken-word count of a tagged transcript (``<PersonN>`` stripped)."""
    return len(re.sub(r"</?Person\d+>", "", text).split())
