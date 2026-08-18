import sys
from .orchestrator import run

def main(argv: list[str]) -> None:
    only = None
    if len(argv) > 1 and argv[1] == "--only":
        if len(argv) < 3:
            raise SystemExit("usage: python -m pipeline [--only collect|rank|generate]")
        only = argv[2]
    run(only=only)

if __name__ == "__main__":
    main(sys.argv[1:])
