from . import RankedItem, Episode
import asyncio, json, pathlib, datetime

DATA = pathlib.Path("data")
NOTEBOOK = "AI News Digest"
INTRO = """# AI News Digest — {date}

A pipeline-generated brief of this week's most interesting AI work.
Each section is one paper you may want to discuss on your podcast.

"""

def generate(items: list[RankedItem]) -> Episode:
    audio = DATA / "episode.mp3"
    brief = DATA / "podcast_brief.md"
    today = datetime.date.today().isoformat()

    lines = [INTRO.format(date=today)]
    for i, item in enumerate(items, 1):
        lines.append(f"## {i}. {item.title}")
        lines.append(f"- Source: {item.source} | URL: {item.url}")
        lines.append(f"- Score: {item.score:.2f} — {item.judge_reason}")
        lines.append("")
        lines.append(item.body)
        lines.append("")
    brief.write_text("\n".join(lines), encoding="utf-8")

    asyncio.run(_generate_audio(brief, audio))

    ep = Episode(
        audio_path=str(audio),
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

async def _generate_audio(brief: pathlib.Path, audio: pathlib.Path):
    from notebooklm import NotebookLMClient, AudioFormat, AudioLength
    async with NotebookLMClient.from_storage() as client:
        notebooks = await client.notebooks.list()
        nb = next((n for n in notebooks if n.title == NOTEBOOK), None)
        if nb is None:
            nb = await client.notebooks.create(NOTEBOOK)
        await client.sources.add_file(nb.id, str(brief), wait=True)
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
