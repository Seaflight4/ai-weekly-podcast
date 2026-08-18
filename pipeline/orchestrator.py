from . import Item, RankedItem, StoryGroup, collect, distil, rank, generate
import json, os, pathlib

DATA = pathlib.Path("data")

def _load_env():
    env = pathlib.Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_env()

def _load(name: str, cls) -> list:
    path = DATA / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"cannot run stage: {name} needs {path.name}, which doesn't exist")
    raw = json.loads(path.read_text())
    return [cls(**i) for i in raw]

def _load_story_groups() -> list:
    path = DATA / "distil.json"
    if not path.exists():
        raise SystemExit("cannot run stage: distil needs data/distil.json, which doesn't exist")
    raw = json.loads(path.read_text())
    return [
        StoryGroup(
            title=g["title"],
            content=g["content"],
            urls=g["urls"],
            sources=set(g["sources"]),
            first_date=g["first_date"],
            consensus=g["consensus"],
            rep_url=g["rep_url"],
        )
        for g in raw
    ]

def run(only: str | None = None):
    if only is None:
        print("[1/4] collecting...")
        items = collect.collect()
        print(f"      {len(items)} items")

        print("[2/4] distilling...")
        groups = distil.distil(items)
        print(f"      {len(groups)} story groups (deduped)")

        print("[3/4] ranking...")
        ranked = rank.rank(groups)
        print(f"      top {len(ranked)}")

        print("[4/4] generating...")
        episode = generate.generate(ranked)
        print(f"      audio -> {episode.audio_path}")
        print(f"      manifest has {len(episode.manifest)} items")
        return

    if only == "collect":
        print("[1/4] collecting...")
        collect.collect()
        return

    if only == "distil":
        items = _load("collect", Item)
        print(f"[2/4] distilling {len(items)} cached items...")
        distil.distil(items)
        return

    if only == "rank":
        if (DATA / "distil.json").exists():
            groups = _load_story_groups()
            print(f"[3/4] ranking {len(groups)} cached story groups...")
        else:
            items = _load("collect", Item)
            print(f"[3/4] ranking {len(items)} cached items (no distil.json)...")
            groups = items
        rank.rank(groups)
        return

    if only == "generate":
        ranked = _load("rank", RankedItem)
        print(f"[4/4] generating from {len(ranked)} cached items...")
        episode = generate.generate(ranked)
        print(f"      audio -> {episode.audio_path}")
        print(f"      manifest has {len(episode.manifest)} items")
        return

    raise SystemExit(f"unknown stage: {only!r} (expected collect, distil, rank, or generate)")

if __name__ == "__main__":
    run()
