from . import Item, RankedItem, Topic, EpisodePlan, collect, rank, cluster, plan, generate
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

def _load_topics() -> list[Topic]:
    raw = store.read("cluster.json")
    out: list[Topic] = []
    for t in raw:
        members = [RankedItem(**m) for m in t.pop("members", [])]
        out.append(Topic(**t, members=members))
    return out

def _load_plan() -> EpisodePlan:
    from . import Segment
    raw = store.read("episode_plan.json")
    segs = [Segment(**s) for s in raw.pop("segments", [])]
    return EpisodePlan(**raw, segments=segs)

def _timed(label: str, fn, *args, **kwargs):
    t0 = time.monotonic()
    result = fn(*args, **kwargs)
    dt = time.monotonic() - t0
    print(f"      [{label} took {dt:.1f}s]")
    return result

STAGES = ("collect", "rank", "cluster", "plan", "generate")

def run(only: str | None = None, no_audio: bool = False, ab: bool = False):
    if only is None:
        print("[1/5] collecting...")
        items = _timed("collect", collect.collect)
        print(f"      {len(items)} items")

        print("[2/5] ranking...")
        ranked = _timed("rank", rank.rank, items)
        print(f"      scored {len(ranked)} items")

        print("[3/5] clustering...")
        topics = _timed("cluster", cluster.cluster, ranked)
        print(f"      {len(topics)} topics")

        print("[4/5] planning...")
        ep_plan = _timed("plan", plan.plan, topics)
        print(f"      {len(ep_plan.segments)} segments")

        print("[5/5] generating...")
        episode = _timed("generate", generate.generate, ep_plan, topics,
                         make_audio=not no_audio)
        print(f"      audio -> {episode.audio_path}")
        print(f"      manifest has {len(episode.manifest)} items")
        return

    if only == "collect":
        print("[1/5] collecting...")
        collect.collect()
        return

    if only == "rank":
        items = _load("collect", Item)
        if ab:
            print(f"[2/5] A/B ranking {len(items)} cached items (both rubric arms)...")
            rank.rank_ab(items)
        else:
            print(f"[2/5] ranking {len(items)} cached items...")
            ranked = _timed("rank", rank.rank, items)
            print(f"      scored {len(ranked)} items")
        return

    if only == "cluster":
        ranked = _load("rank", RankedItem)
        print(f"[3/5] clustering {len(ranked)} cached items...")
        topics = _timed("cluster", cluster.cluster, ranked)
        print(f"      {len(topics)} topics")
        return

    if only == "plan":
        topics = _load_topics()
        print(f"[4/5] planning from {len(topics)} cached topics...")
        ep_plan = _timed("plan", plan.plan, topics)
        print(f"      {len(ep_plan.segments)} segments")
        return

    if only == "generate":
        ep_plan = _load_plan()
        topics = _load_topics()
        make_audio = not no_audio
        print(f"[5/5] generating from {len(ep_plan.segments)} segments "
              f"(audio={'yes' if make_audio else 'no'})...")
        episode = _timed("generate", generate.generate, ep_plan, topics, make_audio=make_audio)
        print(f"      audio -> {episode.audio_path}")
        print(f"      manifest has {len(episode.manifest)} items")
        return

    raise SystemExit(f"unknown stage: {only!r} (expected {', '.join(STAGES)})")

if __name__ == "__main__":
    run()
