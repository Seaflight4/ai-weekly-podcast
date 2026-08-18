from . import RankedItem, Episode
import json, pathlib, datetime

DATA = pathlib.Path("data")
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

def _write(name, payload):
    DATA.mkdir(exist_ok=True)
    (DATA / name).write_text(json.dumps(payload, indent=2))
