import sys
from .orchestrator import run

USAGE = ("usage: python -m pipeline run [--only collect|rank|generate] "
         "[--no-audio] [--from-cache <rank.json>]\n"
         "       python -m pipeline mark <url> kept|skipped")
STAGES = ("collect", "rank", "generate")

def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(USAGE)
    if argv[0] == "mark":
        _mark(argv[1:])
        return
    if argv[0] != "run":
        raise SystemExit(f"{USAGE}\n(error: unknown subcommand {argv[0]!r}; expected `run` or `mark`)")
    args = argv[1:]
    only = None
    no_audio = False
    from_cache = None
    rest = []
    while args:
        a = args.pop(0)
        if a == "--no-audio":
            no_audio = True
        elif a == "--from-cache":
            if not args:
                raise SystemExit(f"{USAGE}\n(error: `--from-cache` needs a path)")
            from_cache = args.pop(0)
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
    run(only=only, no_audio=no_audio, from_cache=from_cache)

def _mark(args: list[str]) -> None:
    from . import feedback
    if len(args) != 2 or args[1] not in ("kept", "skipped"):
        raise SystemExit(f"{USAGE}\n(error: `mark` needs <url> kept|skipped)")
    url, verdict = args
    path = feedback.mark(url, kept=(verdict == "kept"))
    print(f"marked {url} as {verdict} -> {path}")

if __name__ == "__main__":
    main(sys.argv[1:])
