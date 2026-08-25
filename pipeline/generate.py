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

# Source-set selection: fixed score threshold + pace-controlled floor/cap.
# All items with final_score >= THRESHOLD are "must include". Pace sets the
# floor (pad with lower-scored items on a weak week) and cap (cut on a strong
# week). This replaces the previous score-tier elbow algorithm, which did not
# generalise across the two distribution shapes the judge produces.
THRESHOLD = 0.80
PACE_LIMITS = {
    "deep_dive": (5, 10),
    "brief":     (10, 20),
}


def generate(ranked: list[RankedItem], make_audio: bool = True,
             profile: profile_mod.Profile | None = None,
             date: str | None = None) -> Episode:
    """Feed the week's top items to NotebookLM and produce audio.

    The source set is selected by a fixed score threshold (>= 0.8) with
    pace-controlled floor/cap. The narrative style comes from `profile.md`:
    `format`/`target_length` reach NotebookLM's audio knobs; `knowledge_level`
    and `pace` template into the `instructions` string.
    """
    if profile is None:
        profile = profile_mod.load_profile()

    run = store.run_dir(date)
    audio = run / "episode.mp3"
    brief = run / "podcast_brief.md"

    chosen = select_sources(ranked, profile.pace)
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
    }, date=date)
    print(f"      wrote {brief.name} ({len(chosen)} items)")
    return ep


# --- source selection: threshold + pace floor/cap ------------------------

def select_sources(ranked: list[RankedItem], pace: str) -> list[RankedItem]:
    """Pick the source set for NotebookLM by score threshold + floor/cap.

    All items with final_score >= THRESHOLD (0.8) are "must include". Pace
    then sets a floor and cap:
      - deep_dive: floor=5,  cap=10  (fewer items, each in depth)
      - brief:     floor=10, cap=20  (more items, each briefly)

    On a weak week (few items >= 0.8): pad with the next-highest-scored
    items down to the floor. On a strong week (many items >= 0.8): cut at
    the cap, keeping the top-scored. Otherwise: all items >= 0.8 qualify.
    """
    floor, cap = PACE_LIMITS.get(pace, PACE_LIMITS["deep_dive"])
    sorted_items = sorted(ranked, key=lambda r: r.final_score, reverse=True)
    if not sorted_items:
        return []

    above = [r for r in sorted_items if r.final_score >= THRESHOLD]
    n = len(above)

    if n >= cap:
        chosen = sorted_items[:cap]
        reason = f"cap (>=0.8: {n}, cut to {cap})"
    elif n < floor:
        chosen = sorted_items[:floor]
        reason = f"floor (>=0.8: {n}, pad to {floor})"
    else:
        chosen = above
        reason = f"threshold ({n} items >=0.8)"

    print(f"      select: pace={pace}, {reason} -> {len(chosen)} items "
          f"(score {chosen[0].final_score:.3f}..{chosen[-1].final_score:.3f})")
    return chosen


# --- brief: a flat, source-grouped digest fed to NotebookLM as a source ----

def _brief_text(chosen: list[RankedItem],
                by_source: dict[str, list[RankedItem]],
                profile: profile_mod.Profile) -> str:
    lines = [
        f"# AI News Digest — {datetime.date.today().isoformat()}",
        "",
        f"A pipeline-generated, source-grouped digest of this week's AI work. "
        f"Audio: {profile.format} format, {profile.target_length} length, "
        f"pace: {profile.pace}, audience: {profile.knowledge_level}.",
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

    NotebookLM still writes the spoken script; this steers audience level,
    pace, and opening/closing style. The knobs come from `profile.md`, not a
    frozen prompt. `pace` is a soft instruction tested against the hard
    `AudioFormat` knob — `pace=brief` asks for breadth while `format=deep_dive`
    requests the Deep Dive format, to see which wins.
    """
    level_hint = {
        "researcher": (
            "Assume the audience are AI researchers. Don't explain basics — no "
            "need to define what an attention mechanism is, only explain the new "
            "architecture; no need to define RLHF, only explain the new result. "
            "Skip prerequisites, go straight to the substance."
        ),
        "undergrad": (
            "Assume a CS undergraduate audience. Briefly explain jargon the first "
            "time it appears — define the attention mechanism in one sentence "
            "before discussing the new architecture; explain what RLHF is before "
            "quoting the result."
        ),
    }[profile.knowledge_level]

    pace_hint = {
        "brief": (
            "Cover more topics this episode — each explained briefly. For each "
            "topic: what it is, why it matters, one concrete detail. Do not go "
            "deep on any single topic."
        ),
        "deep_dive": (
            "Cover fewer topics this episode, each in depth. For each: explain "
            "how it works, walk through the method, and discuss implications. "
            "Depth over breadth."
        ),
    }[profile.pace]

    titles = [f'- "{m.title}"' for m in chosen]
    return (
        f"A two-host AI news podcast. {level_hint}\n"
        f"Pace: {pace_hint}\n"
        f"Cover these items (do not omit any). Group related items "
        f"thematically where possible:\n"
        + "\n".join(titles)
        + f"\nFor each item: explain what happened, how it works, and why it "
        f"matters. Cross-reference an arXiv paper and a HN story about the same "
        f"event when both are present — they are the same story, do not read "
        f"items as a bullet list.\n"
        f"Open with an overview: summarize each topic in one sentence, grouped "
        f"by common themes where they exist. Example shape: \"In today's "
        f"episode, we'll cover two major model releases — X which achieves Y, "
        f"and Z which uses a new architecture. We'll also discuss industry news "
        f"including W. Let's begin.\"\n"
        f"Use clear transitions between topics. Either a brief (~2 second) "
        f"pause of silence, or an explicit transitional sentence such as: "
        f"\"Now let's jump to the next topic, which is ...\". Do not blend "
        f"topics together without a break.\n"
        f"Close with a summary of the topics covered in this episode."
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
