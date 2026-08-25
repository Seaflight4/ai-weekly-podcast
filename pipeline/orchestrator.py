from . import Item, RankedItem, collect, rank, generate
from . import store, profile as profile_mod
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

def _load(name: str, cls, date: str | None = None) -> list:
    raw = store.read(f"{name}.json", date=date)
    return [cls(**i) for i in raw]

def _timed(label: str, fn, *args, **kwargs):
    t0 = time.monotonic()
    result = fn(*args, **kwargs)
    dt = time.monotonic() - t0
    print(f"      [{label} took {dt:.1f}s]")
    return result

STAGES = ("collect", "rank", "generate")

def run(only: str | None = None, no_audio: bool = False,
        from_cache: str | None = None, date: str | None = None):
    if only is None:
        print("[1/3] collecting...")
        items = _timed("collect", collect.collect, date)
        print(f"      {len(items)} items")

        print("[2/3] ranking...")
        ranked = _timed("rank", rank.rank, items, date=date)
        print(f"      scored {len(ranked)} items")

        print("[3/3] generating...")
        profile = profile_mod.load_profile()
        episode = _timed("generate", generate.generate, ranked,
                         make_audio=not no_audio, profile=profile, date=date)
        print(f"      audio -> {episode.audio_path}")
        print(f"      manifest has {len(episode.manifest)} items")
        return

    if only == "collect":
        print("[1/3] collecting...")
        collect.collect(date)
        return

    if only == "rank":
        if from_cache:
            print(f"[2/3] ranking from cache {from_cache}...")
            ranked = _timed("rank_from_cache", rank.rank_from_cache, from_cache, date)
            print(f"      scored {len(ranked)} items")
            return
        items = _load("collect", Item, date=date)
        print(f"[2/3] ranking {len(items)} cached items...")
        ranked = _timed("rank", rank.rank, items, date=date)
        print(f"      scored {len(ranked)} items")
        return

    if only == "generate":
        ranked = _load("rank", RankedItem, date=date)
        make_audio = not no_audio
        profile = profile_mod.load_profile()
        print(f"[3/3] generating from {len(ranked)} ranked items "
              f"(audio={'yes' if make_audio else 'no'}, pace={profile.pace})...")
        episode = _timed("generate", generate.generate, ranked,
                         make_audio=make_audio, profile=profile, date=date)
        print(f"      audio -> {episode.audio_path}")
        print(f"      manifest has {len(episode.manifest)} items")
        return

    raise SystemExit(f"unknown stage: {only!r} (expected {', '.join(STAGES)})")

if __name__ == "__main__":
    run()
