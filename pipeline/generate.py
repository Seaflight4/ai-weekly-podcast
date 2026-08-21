from . import RankedItem, Topic, EpisodePlan, Episode
from . import store
import asyncio, pathlib, datetime

NOTEBOOK = "AI News Digest"

ARXIV_ABS = "https://arxiv.org/abs/"
ARXIV_PDF = "https://arxiv.org/pdf/"

# NotebookLM's server-side ingester cannot fetch these (JS-rendered, auth-walled,
# or bot-hostile), so skip the URL source and fall back to text ingestion.
UNFETCHABLE_HOSTS = {
    "twitter.com",
    "x.com",
    "twitterusercontent.com",
    "t.co",
}

def generate(plan: EpisodePlan, topics: list[Topic], make_audio: bool = True) -> Episode:
    """Render the narrative plan as a topic-grouped digestible brief, then
    optionally produce audio via NotebookLM fed a rich instruction scaffold.

    `plan` is the narrative outline; `topics` carries the members so the brief
    can link each item for deeper reading.
    """
    run = store.run_dir()
    audio = run / "episode.mp3"
    brief = run / "podcast_brief.md"

    by_id = {t.id: t for t in topics}
    lines = _brief_lines(plan, by_id)
    brief.write_text("\n".join(lines), encoding="utf-8")

    members = [m for seg in plan.segments for m in by_id.get(seg.topic_id, Topic(id=seg.topic_id)).members]

    if make_audio:
        try:
            asyncio.run(_generate_audio(plan, by_id, members, brief, audio))
        except Exception as e:
            print(f"      [warn] audio generation failed: {e}")
            print(f"      [warn] {audio.name} not produced; drop the brief into NotebookLM manually")
            audio = pathlib.Path("")
    else:
        print("      skipping audio (generate(make_audio=False) or --no-audio)")

    ep = Episode(
        audio_path=str(audio) if audio else "",
        manifest=members,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    store.write("episode.json", {
        "audio_path": str(audio),
        "created_at": ep.created_at,
        "manifest": [r.__dict__ for r in ep.manifest],
    })
    print(f"      wrote {brief.name} ({len(plan.segments)} topics)")
    return ep

# --- brief rendering ------------------------------------------------------

def _brief_lines(plan: EpisodePlan, by_id: dict[str, Topic]) -> list[str]:
    total = sum(s.minutes for s in plan.segments)
    lines = [
        f"# AI News Digest — {plan.date}",
        "",
        f"A pipeline-generated, topic-grouped overview of this week's AI work. "
        f"~{total:.0f} min listen.",
        "",
        f"**Motif this week**: *{plan.motif or '(none)'}*",
        "",
        f"> Hook: {plan.hook or '(none)'}",
        "",
    ]
    for i, seg in enumerate(plan.segments, 1):
        t = by_id.get(seg.topic_id)
        lines += _section_lines(i, seg, t)
    lines += ["## Outro", "", plan.outro or "(none)", ""]
    # Links appendix: every member URL once
    lines += ["## Links", ""]
    seen: set[str] = set()
    for seg in plan.segments:
        t = by_id.get(seg.topic_id)
        if not t:
            continue
        for m in t.members:
            if m.url in seen:
                continue
            seen.add(m.url)
            pdf = _paper_pdf_url(m)
            label = m.title
            link = f"- [{label}]({m.url})"
            if pdf and pdf != m.url:
                link += f" · [PDF]({pdf})"
            lines.append(link)
    return lines

def _section_lines(i: int, seg, t: Topic | None) -> list[str]:
    out = [f"## {i}. {t.title if t else seg.topic_id}  ({seg.minutes:.1f} min)", ""]
    if t and t.why:
        out.append(f"**Why it matters**: {t.why}")
        out.append("")
    if seg.opening:
        out.append(f"> {seg.opening}")
        out.append("")
    if t and t.members:
        out.append("**Items in this topic:**")
        out.append("")
        for m in t.members:
            pdf = _paper_pdf_url(m)
            link = f"- [{m.title}]({m.url})"
            if pdf and pdf != m.url:
                link += f" · [PDF]({pdf})"
            link += f" — score {m.score:.2f}"
            out.append(link)
        out.append("")
    if seg.signposts:
        out.append("**Signposts:** " + "; ".join(seg.signposts))
        out.append("")
    if seg.transition_out:
        out.append(f"_{seg.transition_out}_")
        out.append("")
    return out

def _paper_pdf_url(item: RankedItem) -> str | None:
    if item.url.startswith(ARXIV_ABS):
        return item.url.replace(ARXIV_ABS, ARXIV_PDF, 1)
    return None

# --- NotebookLM audio -----------------------------------------------------

def _instructions(plan: EpisodePlan, by_id: dict[str, Topic]) -> str:
    """Build a rich instruction scaffold from the plan (replaces the old
    one-liner). NotebookLM still writes the actual spoken script; this steers
    motif, pacing, signposts, and topic allocation."""
    seg_lines = []
    for i, seg in enumerate(plan.segments, 1):
        t = by_id.get(seg.topic_id)
        title = t.title if t else seg.topic_id
        sources = ", ".join(sorted({m.source for m in t.members})) if t else "?"
        seg_lines.append(
            f"{i}. Topic \"{title}\" (~{seg.minutes:.0f} min; sources: {sources}). "
            f"Open with: {seg.opening or '(setup)'}. "
            f"Signposts: {'; '.join(seg.signposts) or '(none)'}. "
            f"Transition: {seg.transition_out or '(bridge to next)'}."
        )
    return (
        f"A lively two-host AI news podcast for AI researchers. "
        f"Episode motif to return to in transitions: \"{plan.motif or '(decide naturally)'}\". "
        f"Hook to open: {plan.hook or '(open with the biggest story)'}. "
        f"Cover these topics in this order (do not exceed the minutes):\n"
        + "\n".join(seg_lines)
        + f"\n\nFor each topic: weave the paper(s) and the HN coverage together — they "
        f"are the same story; cross-reference them, do not read items as a bullet list. "
        f"Pacing: break stats-heavy segments with a host question; no single topic runs "
        f"more than ~4 min straight. Close with: {plan.outro or '(point to the links)'}."
    )

async def _generate_audio(plan: EpisodePlan, by_id: dict[str, Topic],
                          members: list[RankedItem], brief: pathlib.Path, audio: pathlib.Path):
    from notebooklm import NotebookLMClient, AudioFormat, AudioLength
    from notebooklm.artifacts import with_rate_limit_retry
    from notebooklm.exceptions import RateLimitError, SourceError, RPCError

    async with NotebookLMClient.from_storage(profile="personal") as client:
        notebooks = await client.notebooks.list()
        nb = next((n for n in notebooks if n.title == NOTEBOOK), None)
        if nb is None:
            nb = await client.notebooks.create(NOTEBOOK)

        await _clear_sources(client, nb.id)

        for i, item in enumerate(members):
            url = _paper_pdf_url(item) or item.url
            if _host(url) in UNFETCHABLE_HOSTS:
                print(f"      [source {i+1}/{len(members)}] skipping unfetchable host {url}")
                await _add_text_source(client, nb.id, item)
                continue
            try:
                await with_rate_limit_retry(
                    lambda url=url: client.sources.add_url(nb.id, url, wait=True),
                    max_retries=3,
                )
                print(f"      [source {i+1}/{len(members)}] added {url}")
            except RateLimitError as e:
                print(f"      [source {i+1}/{len(members)}] rate-limited adding {url}, skipped: {e}")
            except (SourceError, RPCError) as e:
                print(f"      [source {i+1}/{len(members)}] url-add failed for {url}: {e}")
                await _add_text_source(client, nb.id, item)
            await asyncio.sleep(1)

        try:
            await with_rate_limit_retry(
                lambda: client.sources.add_file(nb.id, str(brief), wait=True),
                max_retries=3,
            )
            print(f"      [source] added {brief.name}")
        except RateLimitError:
            print(f"      [source] rate-limited adding {brief.name}, skipped")

        sources = await client.sources.list(nb.id)
        print(f"      [sources] {len(sources)} sources now in notebook:")
        for s in sources:
            print(f"        - {s.title or s.url}")

        instructions = _instructions(plan, by_id)
        print(f"      [instructions] {len(instructions)} chars (was a one-liner before)")
        status = await client.artifacts.generate_audio(
            nb.id,
            instructions=instructions,
            audio_format=AudioFormat.DEEP_DIVE,
            audio_length=AudioLength.DEFAULT,
        )
        await client.artifacts.wait_for_completion(nb.id, status.task_id, timeout=1200)
        await client.artifacts.download_audio(nb.id, str(audio))
        print(f"      audio -> {audio.name} (via notebooklm-py)")

def _host(url: str) -> str:
    from urllib.parse import urlparse
    return (urlparse(url).netloc or "").lower().removeprefix("www.")

async def _clear_sources(client, notebook_id: str) -> None:
    sources = await client.sources.list(notebook_id)
    for s in sources:
        await client.sources.delete(notebook_id, s.id)
        print(f"      [source] removed {s.title or s.url}")

async def _add_text_source(client, notebook_id: str, item: RankedItem) -> None:
    body = (item.body or "").strip()
    if not body:
        print(f"        no body for {item.url}; skipping source")
        return
    try:
        await client.sources.add_text(
            notebook_id, title=item.title, content=body, wait=True,
        )
        print(f"        added text source for {item.title}")
    except (SourceError, RPCError) as e:
        print(f"        text-fallback also failed for {item.title}: {e}")
