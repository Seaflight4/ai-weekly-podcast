import sys
from .orchestrator import run

USAGE = ("usage: python -m pipeline run [--only collect|rank|cluster|plan|generate] "
         "[--no-audio] [--ab]")
STAGES = ("collect", "rank", "cluster", "plan", "generate")

def main(argv: list[str]) -> None:
    if not argv or argv[0] != "run":
        raise SystemExit(f"{USAGE}\n(error: the `run` subcommand is required)")
    args = argv[1:]
    only = None
    no_audio = False
    ab = False
    rest = []
    while args:
        a = args.pop(0)
        if a == "--no-audio":
            no_audio = True
        elif a == "--ab":
            ab = True
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
    run(only=only, no_audio=no_audio, ab=ab)

if __name__ == "__main__":
    main(sys.argv[1:])
