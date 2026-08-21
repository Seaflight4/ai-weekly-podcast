from . import Item, RankedItem, collect, rank, generate
from . import store
import os, pathlib, time

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
    raw = store.read(f"{name}.json")
    return [cls(**i) for i in raw]

def _timed(label: str, fn, *args, **kwargs):
    t0 = time.monotonic()
    result = fn(*args, **kwargs)
    dt = time.monotonic() - t0
    print(f"      [{label} took {dt:.1f}s]")
    return result

def run(only: str | None = None, no_audio: bool = False):
    if only is None:
        print("[1/3] collecting...")
        items = _timed("collect", collect.collect)
        print(f"      {len(items)} items")

        print("[2/3] ranking...")
        ranked = _timed("rank", rank.rank, items)
        print(f"      top {len(ranked)}")

        print("[3/3] generating...")
        episode = _timed("generate", generate.generate, ranked)
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
