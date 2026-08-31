from . import RankedItem, Episode
from . import store
from . import audio as audio_mod
import pathlib, datetime


def generate(ranked: list[RankedItem], make_audio: bool = True,
             date: str | None = None,
             transcript_in: str | None = None) -> Episode:
    """Feed the week's top items to the podcastfy audio backend and produce audio.

    The source set is selected by a fixed score threshold (>= 0.8) with a
    fixed floor/cap of 10–20 items. The brief is a flat, source-grouped digest
    aimed at an AI-researcher audience; the vendored podcastfy stack uses its
    own tuned defaults for transcript + TTS styling.

    ``transcript_in`` reuses a cached transcript file and skips the LLM step,
    going straight to TTS — useful for retrying audio after a transient TTS
    outage without paying the multi-minute LLM cost again.
    """
    run = store.run_dir(date)
    brief = run / "podcast_brief.md"

    chosen = select_sources(ranked)
    by_source = _group_by_source(chosen)

    brief.write_text(_brief_text(chosen, by_source), encoding="utf-8")

    transcript_source = "whisper"
    transcript_path: pathlib.Path | None = None
    backend_name: str | None = None
    if make_audio:
        try:
            result = audio_mod.PodcastfyBackend().generate(
                brief=brief, run_dir=run, chosen=chosen,
                transcript_in=pathlib.Path(transcript_in) if transcript_in else None,
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
    )
    store.write("episode.json", {
        "audio_path": str(audio),
        "created_at": ep.created_at,
        "manifest": [r.__dict__ for r in ep.manifest],
        "transcript_source": transcript_source,
        "backend": backend_name,
    }, date=date)
    print(f"      wrote {brief.name} ({len(chosen)} items)")
    return ep


# --- source selection: threshold + floor/cap -------------------------------

def select_sources(ranked: list[RankedItem]) -> list[RankedItem]:
    """Pick the source set for the audio backend by score threshold + floor/cap.

    All items with score >= THRESHOLD (0.8) are "must include". The selection
    then honours a fixed floor and cap (10 and 20):

    - weak week  (few items >= 0.8): pad with the next-highest-scored items
      down to the floor.
    - strong week (many items >= 0.8): cut at the cap, keeping the top-scored.
    - typical week: all items >= 0.8 qualify (within floor/cap).
    """
    sorted_items = sorted(ranked, key=lambda r: r.score, reverse=True)
    if not sorted_items:
        return []

    above = [r for r in sorted_items if r.score >= THRESHOLD]
    n = len(above)

    if n >= CAP:
        chosen = sorted_items[:CAP]
        reason = f"cap (>=0.8: {n}, cut to {CAP})"
    elif n < FLOOR:
        chosen = sorted_items[:FLOOR]
        reason = f"floor (>=0.8: {n}, pad to {FLOOR})"
    else:
        chosen = above
        reason = f"threshold ({n} items >=0.8)"

    print(f"      select: {reason} -> {len(chosen)} items "
          f"(score {chosen[0].score:.3f}..{chosen[-1].score:.3f})")
    return chosen


# --- brief: a flat, source-grouped digest fed to the backend as a source ----

def _brief_text(chosen: list[RankedItem],
                by_source: dict[str, list[RankedItem]]) -> str:
    lines = [
        f"# AI News Digest — {datetime.date.today().isoformat()}",
        "",
        f"A pipeline-generated, source-grouped digest of this week's AI work, "
        f"aimed at AI researchers.",
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


# Source-set selection: fixed score threshold + fixed floor/cap.
THRESHOLD = 0.80
FLOOR = 10
CAP = 20
