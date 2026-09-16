from . import Item, RankedItem, collect, rank, generate, transcribe
from . import store
from . import config as config_mod
from . import label as label_mod
from . import DATA_ROOT
import dataclasses, os, pathlib, time

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

def _load_podcast_config() -> dict | None:
    """The persistent per-deployment settings (``data/podcast_config.yaml``),
    or None when absent/unreadable.

    The pipeline reads this so a plain CLI ``python -m pipeline run`` honours
    the same saved settings the web UI uses (audience, familiar topics,
    steering, length/depth, window span). The service remains the only writer.
    """
    path = DATA_ROOT / "podcast_config.yaml"
    if not path.exists():
        return None
    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None

def _load(name: str, cls, date: str | None = None) -> list:
    """Load a stage's JSON into a list of dataclass instances.

    Tolerant of extra/legacy keys: only fields declared on the dataclass are
    passed to its constructor, so older run files (e.g. a stale ``kind`` field
    from a prior schema) don't break re-runs.
    """
    raw = store.read(f"{name}.json", date=date)
    if dataclasses.is_dataclass(cls):
        known = {f.name for f in dataclasses.fields(cls)}
        return [cls(**{k: v for k, v in i.items() if k in known}) for i in raw]
    return [cls(**i) for i in raw]

def _timed(label: str, fn, *args, **kwargs):
    t0 = time.monotonic()
    result = fn(*args, **kwargs)
    dt = time.monotonic() - t0
    print(f"      [{label} took {dt:.1f}s]")
    return result

STAGES = ("collect", "rank", "generate", "transcribe", "label")

def run(only: str | None = None, no_audio: bool = False,
        from_cache: str | None = None, date: str | None = None,
        brief_in: str | None = None,
        config_path: str | None = None,
        window_start: str | None = None, window_end: str | None = None,
        audience_level: str | None = None,
        familiar_topics: list[str] | None = None,
        topic_prefs: list[str] | None = None,
        steering_alpha: float | None = None,
        length: str | None = None, depth: str | None = None,
        mem_windows: int | None = None,
        force_label: bool = False):
    """Resolve the run config (defaults < config file < CLI overrides), then
    dispatch to the requested stage(s). ``date`` is the window end / anchor
    (back-compat with ``--date``)."""
    podcast_cfg = _load_podcast_config() if only is None else None
    try:
        cfg = config_mod.resolve(
            config_path, date=date, window_start=window_start, window_end=window_end,
            audience_level=audience_level, familiar_topics=familiar_topics,
            topic_prefs=topic_prefs, steering_alpha=steering_alpha,
            length=length, depth=depth, mem_windows=mem_windows,
            podcast_config=podcast_cfg,
        )
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    profile = {t: 1.0 for t in cfg.topic_prefs} or None
    # Stage re-runs with an explicit window may target that window's folder
    # (back-compat: --date / --window-end name the folder the stage reads).
    if cfg is not None and cfg.window_end is not None and date is None and only is not None:
        date = cfg.window_end
    # A fresh full run always gets a unique time-stamped run id (unless --date
    # pins the folder), so every CLI-generated episode lands in a
    # DD-MM-YYYY-HHMMSS folder — the same scheme the web UI uses — and thus
    # gets the same "DD-MM-YYYY HH:MM" title.
    if only is None and date is None:
        _, end_date = cfg.resolve_window()
        date = store.make_run_id(end_date.isoformat())
    if only is None:
        print("[1/3] collecting...")
        items = _timed("collect", collect.collect, cfg.window_end,
                       window_start=cfg.window_start, anchor=date)
        print(f"      {len(items)} items")

        print("[2/3] ranking...")
        ranked = _timed("rank", rank.rank, items, date=date,
                        profile=profile, alpha=cfg.steering_alpha)
        print(f"      scored {len(ranked)} items")

        print("[3/3] generating...")
        episode = _timed("generate", generate.generate, ranked,
                         make_audio=not no_audio, date=date,
                         brief_in=brief_in,
                         config=cfg)
        print(f"      audio -> {episode.audio_path}")
        print(f"      manifest has {len(episode.manifest)} items")
        return

    if only == "collect":
        if date is None:
            _, end_date = cfg.resolve_window()
            date = store.make_run_id(end_date.isoformat())
        print("[1/3] collecting...")
        collect.collect(cfg.window_end, window_start=cfg.window_start, anchor=date)
        return

    if only == "rank":
        if from_cache:
            print(f"[2/3] ranking from cache {from_cache}...")
            ranked = _timed("rank_from_cache", rank.rank_from_cache, from_cache, date)
            print(f"      scored {len(ranked)} items")
            return
        items = _load("collect", Item, date=date)
        print(f"[2/3] ranking {len(items)} cached items...")
        ranked = _timed("rank", rank.rank, items, date=date,
                        profile=profile, alpha=cfg.steering_alpha)
        print(f"      scored {len(ranked)} items")
        return

    if only == "generate":
        ranked = _load("rank", RankedItem, date=date)
        make_audio = not no_audio
        print(f"[3/3] generating from {len(ranked)} ranked items "
              f"(audio={'yes' if make_audio else 'no'})...")
        episode = _timed("generate", generate.generate, ranked,
                         make_audio=make_audio, date=date,
                         brief_in=brief_in,
                         config=cfg)
        print(f"      audio -> {episode.audio_path}")
        print(f"      manifest has {len(episode.manifest)} items")
        return

    if only == "transcribe":
        print("[transcribe] transcribing episode...")
        out = _timed("transcribe", transcribe.transcribe, date=date)
        print(f"      transcript -> {out}")
        return

    if only == "label":
        print("[label] backfilling episode topic labels from manifests...")
        touched = label_mod.backfill_labels(date=date, force=force_label)
        print(f"      labeled {len(touched)} run(s)")
        return

    raise SystemExit(f"unknown stage: {only!r} (expected {', '.join(STAGES)})")

if __name__ == "__main__":
    run()
