from . import RankedItem, Episode
from . import store, profile as profile_mod
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

# Map profile.format -> notebooklm.AudioFormat
_FORMAT_MAP = {
    "deep_dive": "DEEP_DIVE",
    "brief": "BRIEF",
    "critique": "CRITIQUE",
    "debate": "DEBATE",
}
# Map profile.target_length -> notebooklm.AudioLength
_LENGTH_MAP = {
    "short": "SHORT",
    "default": "DEFAULT",
    "long": "LONG",
}


def generate(ranked: list[RankedItem], make_audio: bool = True,
             profile: profile_mod.Profile | None = None) -> Episode:
    """Feed the top-N ranked items to NotebookLM and produce audio.

    The narrative style comes from `profile.md` (loaded if `profile` is None):
    `format`/`target_length` reach NotebookLM's audio knobs; the rest template
    into the `instructions` string that steers it. No per-run planner LLM call.
    """
    if profile is None:
        profile = profile_mod.load_profile()

    run = store.run_dir()
    audio = run / "episode.mp3"
    brief = run / "podcast_brief.md"

    chosen = sorted(ranked, key=lambda r: r.final_score, reverse=True)[:profile.top_n]
    by_source = _group_by_source(chosen)

    brief.write_text(_brief_text(chosen, by_source, profile), encoding="utf-8")

    if make_audio:
        try:
            asyncio.run(_generate_audio(chosen, brief, audio, profile))
        except Exception as e:
            print(f"      [warn] audio generation failed: {e}")
            print(f"      [warn] {audio.name} not produced; drop the brief into NotebookLM manually")
            audio = pathlib.Path("")
    else:
        print("      skipping audio (--no-audio)")

    ep = Episode(
        audio_path=str(audio) if audio else "",
        manifest=chosen,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    store.write("episode.json", {
        "audio_path": str(audio),
        "created_at": ep.created_at,
        "manifest": [r.__dict__ for r in ep.manifest],
    })
    print(f"      wrote {brief.name} ({len(chosen)} items)")
    return ep


# --- brief: a flat, source-grouped digest fed to NotebookLM as a source ----

def _brief_text(chosen: list[RankedItem],
                by_source: dict[str, list[RankedItem]],
                profile: profile_mod.Profile) -> str:
    lines = [
        f"# AI News Digest — {datetime.date.today().isoformat()}",
        "",
        f"A pipeline-generated, source-grouped digest of this week's AI work. "
        f"Audio: {profile.format} format, {profile.target_length} length, "
        f"tone: {profile.tone}, audience: {profile.knowledge_level}.",
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
            link += f" — score {m.final_score:.2f}"
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
    if item.url.startswith(ARXIV_ABS):
        return item.url.replace(ARXIV_ABS, ARXIV_PDF, 1)
    return None


# --- NotebookLM instructions: templated from the profile -------------------

def _instructions(profile: profile_mod.Profile,
                   chosen: list[RankedItem]) -> str:
    """Build the instruction scaffold from the profile + this week's items.

    NotebookLM still writes the spoken script; this steers format, tone,
    audience level, and opening/closing style without padding. The style
    knobs come from `profile.md`, not a frozen prompt.
    """
    intro_hint = {
        "theme-first": "Open by naming the week's dominant theme, then lead into the first story.",
        "biggest-story": "Open with the single biggest story of the week.",
        "bullet": "Open with a short bullet list of the 2-3 biggest items, then dive in.",
    }[profile.intro_style]
    outro_hint = {
        "links": "Close by pointing listeners to the source links for deeper reading.",
        "recap": "Close with a one-sentence recap of the top 2 items.",
        "teaser": "Close with a teaser for next week's likely themes.",
    }[profile.outro_style]
    transition_hint = {
        "next": "Use simple 'Next:' transitions between topics.",
        "bridge": "Use one bridge phrase between topics where it helps the flow.",
        "motif": "Return to the week's motif in transitions between topics.",
    }[profile.transition_style]
    tone_hint = {
        "dense": "Dense and information-rich: no filler, no restating the obvious, no template phrases.",
        "conversational": "Conversational but efficient: a natural back-and-forth, no filler.",
        "casual": "Casual and relaxed: two hosts chatting, keep it light but accurate.",
    }[profile.tone]
    level_hint = {
        "undergrad": "Assume a CS undergraduate audience; explain jargon briefly.",
        "researcher": "Assume the audience are AI researchers; skip basics.",
        "expert": "Assume deep expertise; go straight to the substance.",
    }[profile.knowledge_level]

    titles = [f'{i+1}. "{m.title}"' for i, m in enumerate(chosen)]
    return (
        f"A two-host AI news podcast. {tone_hint} {level_hint}\n"
        f"Cover these items in this order (do not omit any):\n"
        + "\n".join(titles)
        + f"\nFor each item: explain what happened, how it works, and why it matters. "
        f"Cross-reference an arXiv paper and a HN story about the same event when both "
        f"are present — they are the same story, do not read items as a bullet list. "
        f"Pacing: no single item runs more than ~3 min straight.\n"
        f"Open: {intro_hint}\n"
        f"Transitions: {transition_hint}\n"
        f"Close: {outro_hint}"
    )


# --- NotebookLM audio -----------------------------------------------------

async def _generate_audio(chosen: list[RankedItem], brief: pathlib.Path,
                          audio: pathlib.Path, profile: profile_mod.Profile):
    from notebooklm import NotebookLMClient, AudioFormat, AudioLength
    from notebooklm.artifacts import with_rate_limit_retry
    from notebooklm.exceptions import RateLimitError, SourceError, RPCError

    audio_format = AudioFormat[_FORMAT_MAP[profile.format]]
    audio_length = AudioLength[_LENGTH_MAP[profile.target_length]]

    async with NotebookLMClient.from_storage(profile="personal") as client:
        notebooks = await client.notebooks.list()
        nb = next((n for n in notebooks if n.title == NOTEBOOK), None)
        if nb is None:
            nb = await client.notebooks.create(NOTEBOOK)

        await _clear_sources(client, nb.id)

        for i, item in enumerate(chosen):
            url = _paper_pdf_url(item) or item.url
            if _host(url) in UNFETCHABLE_HOSTS:
                print(f"      [source {i+1}/{len(chosen)}] skipping unfetchable host {url}")
                await _add_text_source(client, nb.id, item)
                continue
            try:
                await with_rate_limit_retry(
                    lambda url=url: client.sources.add_url(nb.id, url, wait=True),
                    max_retries=3,
                )
                print(f"      [source {i+1}/{len(chosen)}] added {url}")
            except RateLimitError as e:
                print(f"      [source {i+1}/{len(chosen)}] rate-limited adding {url}, skipped: {e}")
            except (SourceError, RPCError) as e:
                print(f"      [source {i+1}/{len(chosen)}] url-add failed for {url}: {e}")
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
        print(f"      [sources] {len(sources)} sources now in notebook")
        for s in sources:
            print(f"        - {s.title or s.url}")

        instructions = _instructions(profile, chosen)
        print(f"      [instructions] {len(instructions)} chars")
        status = await client.artifacts.generate_audio(
            nb.id,
            instructions=instructions,
            audio_format=audio_format,
            audio_length=audio_length,
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
