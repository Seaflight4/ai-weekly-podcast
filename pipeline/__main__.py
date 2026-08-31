import sys
from .orchestrator import run

USAGE = ("usage: python -m pipeline run [--only collect|rank|generate|transcribe] "
         "[--no-audio] [--transcript-in <path>] [--from-cache <rank.json>] "
         "[--date YYYY-MM-DD]")
STAGES = ("collect", "rank", "generate", "transcribe")

def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(USAGE)
    if argv[0] != "run":
        raise SystemExit(f"{USAGE}\n(error: unknown subcommand {argv[0]!r}; expected `run`)")
    args = argv[1:]
    only = None
    no_audio = False
    transcript_in = None
    from_cache = None
    date = None
    rest = []
    while args:
        a = args.pop(0)
        if a == "--no-audio":
            no_audio = True
        elif a == "--transcript-in":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--transcript-in` needs a path)")
            transcript_in = args.pop(0)
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
        else:
            rest.append(a)
    if rest:
        raise SystemExit(f"{USAGE}\n(error: unexpected argument {rest[0]!r})")
    run(only=only, no_audio=no_audio,
        from_cache=from_cache, date=date, transcript_in=transcript_in)

if __name__ == "__main__":
    main(sys.argv[1:])
