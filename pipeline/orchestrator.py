from . import Item, RankedItem, collect, rank, generate
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

def run(only: str | None = None, no_audio: bool = False):
    if only is None:
        print("[1/3] collecting...")
        items = collect.collect()
        print(f"      {len(items)} items")

        print("[2/3] ranking...")
        ranked = rank.rank(items)
        print(f"      top {len(ranked)}")

        print("[3/3] generating...")
        episode = generate.generate(ranked)
        print(f"      audio -> {episode.audio_path}")
        print(f"      manifest has {len(episode.manifest)} items")
        return

    if only == "collect":
        print("[1/3] collecting...")
        collect.collect()
        return

    if only == "rank":
        items = _load("collect", Item)
        print(f"[2/3] ranking {len(items)} cached items...")
        rank.rank(items)
        return

    if only == "generate":
        ranked = _load("rank", RankedItem)
        make_audio = not no_audio
        print(f"[3/3] generating from {len(ranked)} cached items (audio={'yes' if make_audio else 'no'})...")
        episode = generate.generate(ranked, make_audio=make_audio)
        print(f"      audio -> {episode.audio_path}")
        print(f"      manifest has {len(episode.manifest)} items")
        return

    raise SystemExit(f"unknown stage: {only!r} (expected collect, rank, or generate)")

if __name__ == "__main__":
    run()
