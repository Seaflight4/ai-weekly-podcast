import sys
from .orchestrator import run

USAGE = "usage: python -m pipeline run [--only collect|rank|generate]"
STAGES = ("collect", "rank", "generate")

def main(argv: list[str]) -> None:
    if not argv or argv[0] != "run":
        raise SystemExit(f"{USAGE}\n(error: the `run` subcommand is required)")
    args = argv[1:]
    only = None
    if args:
        if args[0] != "--only":
            raise SystemExit(f"{USAGE}\n(error: unexpected argument {args[0]!r})")
        if len(args) < 2:
            raise SystemExit(f"{USAGE}\n(error: `--only` needs a stage)")
        if len(args) > 2:
            raise SystemExit(f"{USAGE}\n(error: unexpected argument {args[2]!r})")
        only = args[1]
        if only not in STAGES:
            stage_list = "|".join(STAGES)
            raise SystemExit(f"unknown stage: {only!r} (expected {stage_list})")
    run(only=only)

if __name__ == "__main__":
    main(sys.argv[1:])
