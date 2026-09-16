import sys
from .orchestrator import run

USAGE = ("usage: python -m pipeline run [--only collect|rank|generate|transcribe|label] "
         "[--no-audio] [--brief-in <path>] "
         "[--from-cache <rank.json>] [--date YYYY-MM-DD] "
         "[--force-label] "
         "[--config <config.yaml>] [--window-start D] [--window-end D] "
         "[--audience LEVEL] [--familiar S] [--topics S] [--steering-alpha F] "
         "[--length X] [--depth Y] [--mem-windows N]")
STAGES = ("collect", "rank", "generate", "transcribe", "label")

def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(USAGE)
    if argv[0] != "run":
        raise SystemExit(f"{USAGE}\n(error: unknown subcommand {argv[0]!r}; expected `run`)")
    args = argv[1:]
    only = None
    no_audio = False
    force_label = False
    brief_in = None
    from_cache = None
    date = None
    config_path = None
    window_start = None
    window_end = None
    audience_level = None
    familiar_topics = None
    topic_prefs = None
    steering_alpha = None
    length = None
    depth = None
    mem_windows = None
    rest = []
    while args:
        a = args.pop(0)
        if a == "--no-audio":
            no_audio = True
        elif a == "--force-label":
            force_label = True
        elif a == "--brief-in":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--brief-in` needs a path)")
            brief_in = args.pop(0)
        elif a == "--from-cache":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--from-cache` needs a path)")
            from_cache = args.pop(0)
        elif a == "--date":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--date` needs a YYYY-MM-DD date)")
            date = args.pop(0)
        elif a == "--only":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--only` needs a stage)")
            only = args.pop(0)
            if only not in STAGES:
                stage_list = "|".join(STAGES)
                raise SystemExit(f"unknown stage: {only!r} (expected {stage_list})")
        elif a == "--config":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--config` needs a path)")
            config_path = args.pop(0)
        elif a == "--window-start":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--window-start` needs a YYYY-MM-DD date)")
            window_start = args.pop(0)
        elif a == "--window-end":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--window-end` needs a YYYY-MM-DD date)")
            window_end = args.pop(0)
        elif a == "--audience":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--audience` needs a level)")
            audience_level = args.pop(0)
        elif a == "--familiar":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--familiar` needs a comma-separated list)")
            familiar_topics = [t.strip() for t in args.pop(0).split(",") if t.strip()]
        elif a == "--topics":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--topics` needs a comma-separated list)")
            topic_prefs = [t.strip() for t in args.pop(0).split(",") if t.strip()]
        elif a == "--steering-alpha":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--steering-alpha` needs 0..1)")
            steering_alpha = float(args.pop(0))
        elif a == "--length":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--length` needs short|medium|long)")
            length = args.pop(0)
        elif a == "--depth":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--depth` needs brief|deep-dive)")
            depth = args.pop(0)
        elif a == "--mem-windows":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--mem-windows` needs an int 1-4)")
            mem_windows = int(args.pop(0))
        else:
            rest.append(a)
    if rest:
        raise SystemExit(f"{USAGE}\n(error: unexpected argument {rest[0]!r})")
    try:
        run(only=only, no_audio=no_audio,
            from_cache=from_cache, date=date,
            brief_in=brief_in,
            config_path=config_path, window_start=window_start, window_end=window_end,
            audience_level=audience_level, familiar_topics=familiar_topics,
            topic_prefs=topic_prefs, steering_alpha=steering_alpha,
            length=length, depth=depth, mem_windows=mem_windows,
            force_label=force_label)
    except SystemExit:
        raise
    except Exception:
        # Parse/usage errors above already raise SystemExit with a message;
        # anything reaching here is a stage failure. Emit a loud, self-contained
        # error block so the service's job log (and thus the panel) shows the
        # real reason instead of a bare exit code.
        import traceback
        traceback.print_exc()
        err = traceback.format_exc().strip().splitlines()[-1]
        print("\nERROR: pipeline run failed\n" + err)
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])
