from . import RankedItem, Episode
import asyncio, json, pathlib, datetime

DATA = pathlib.Path("data")
NOTEBOOK = "AI News Digest"
INTRO = """# AI News Digest — {date}

A pipeline-generated brief of this week's most interesting AI work.

- **Deep dives**: the arXiv papers worth studying closely.
- **Quick briefs**: the Hacker News stories clients will be asking about.

"""

ARXIV_ABS = "https://arxiv.org/abs/"
ARXIV_PDF = "https://arxiv.org/pdf/"

def generate(items: list[RankedItem], make_audio: bool = True) -> Episode:
    audio = DATA / "episode.mp3"
    brief = DATA / "podcast_brief.md"
    today = datetime.date.today().isoformat()

    deep = [r for r in items if r.kind == "deep"]
    brief_items = [r for r in items if r.kind != "deep"]

    lines = [INTRO.format(date=today)]

    _write_section(lines, "Deep dives", deep)
    _write_section(lines, "Quick briefs", brief_items)

    brief.write_text("\n".join(lines), encoding="utf-8")

    if make_audio:
        try:
            asyncio.run(_generate_audio(items, brief, audio))
        except Exception as e:
            # NotebookLM creds may be absent (no stored auth) — the brief is
            # still valid for the manual Audio Overview step.
            print(f"      [warn] audio generation failed: {e}")
            print(f"      [warn] {audio.name} not produced; drop the brief into NotebookLM manually")
            audio = pathlib.Path("")  # mark as not produced
    else:
        print("      skipping audio (generate(make_audio=False) or --no-audio)")

    ep = Episode(
        audio_path=str(audio) if audio else "",
        manifest=items,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    _write("episode.json", {
        "audio_path": ep.audio_path,
        "created_at": ep.created_at,
        "manifest": [r.__dict__ for r in ep.manifest],
    })
    print(f"      wrote {brief.name} ({len(items)} sections)")
    return ep

def _write_section(lines: list[str], title: str, items: list[RankedItem]) -> None:
    if not items:
        lines.append(f"## {title}")
        lines.append("_(none this week)_")
        lines.append("")
        return
    lines.append(f"## {title}")
    lines.append("")
    for i, item in enumerate(items, 1):
        lines.append(f"### {i}. {item.title}")
        lines.append(f"- Source: {item.source} | URL: {item.url}")
        lines.append(f"- Score: {item.score:.2f} — {item.judge_reason}")
        lines.append("")
        lines.append(item.body)
        lines.append("")

def _paper_pdf_url(item: RankedItem) -> str | None:
    """Rewrite an arXiv abstract URL to its PDF URL, else None."""
    if item.url.startswith(ARXIV_ABS):
        return item.url.replace(ARXIV_ABS, ARXIV_PDF, 1)
    return None

async def _generate_audio(items: list[RankedItem], brief: pathlib.Path, audio: pathlib.Path):
    from notebooklm import NotebookLMClient, AudioFormat, AudioLength
    from notebooklm.artifacts import with_rate_limit_retry
    from notebooklm.exceptions import RateLimitError

    async with NotebookLMClient.from_storage() as client:
        notebooks = await client.notebooks.list()
        nb = next((n for n in notebooks if n.title == NOTEBOOK), None)
        if nb is None:
            nb = await client.notebooks.create(NOTEBOOK)

        existing = await client.sources.list(nb.id)
        existing_urls = {s.url for s in existing if s.url}
        existing_titles = {s.title for s in existing if s.title}

        for i, item in enumerate(items):
            url = _paper_pdf_url(item)
            if url is None:
                print(f"      [source {i+1}/{len(items)}] {item.url}: not an arXiv URL, skipped")
                continue
            if url in existing_urls:
                print(f"      [source {i+1}/{len(items)}] already present, skipped {url}")
                continue
            try:
                await with_rate_limit_retry(
                    lambda url=url: client.sources.add_url(nb.id, url, wait=True),
                    max_retries=3,
                )
                print(f"      [source {i+1}/{len(items)}] added {url}")
            except RateLimitError:
                print(f"      [source {i+1}/{len(items)}] rate-limited adding {url}, skipped")
            await asyncio.sleep(1)

        if brief.name not in existing_titles:
            try:
                await with_rate_limit_retry(
                    lambda: client.sources.add_file(nb.id, str(brief), wait=True),
                    max_retries=3,
                )
                print(f"      [source] added {brief.name}")
            except RateLimitError:
                print(f"      [source] rate-limited adding {brief.name}, skipped")
        else:
            print(f"      [source] already present, skipped {brief.name}")

        status = await client.artifacts.generate_audio(
            nb.id,
            instructions="a lively two-host AI news podcast",
            audio_format=AudioFormat.BRIEF,
            audio_length=AudioLength.SHORT,
        )
        await client.artifacts.wait_for_completion(nb.id, status.task_id, timeout=1200)
        await client.artifacts.download_audio(nb.id, str(audio))
        print(f"      audio -> {audio.name} (via notebooklm-py)")

def _write(name, payload):
    DATA.mkdir(exist_ok=True)
    (DATA / name).write_text(json.dumps(payload, indent=2))
