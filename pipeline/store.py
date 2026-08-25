import datetime, json, pathlib

ROOT = pathlib.Path("data")
DATE_FORMAT = "%d-%m-%Y"


def _normalize_date(date: str) -> str:
    """Accept ISO (YYYY-MM-DD) or DD-MM-YYYY; return DD-MM-YYYY (folder format)."""
    for fmt in ("%Y-%m-%d", DATE_FORMAT):
        try:
            return datetime.datetime.strptime(date, fmt).strftime(DATE_FORMAT)
        except ValueError:
            continue
    raise SystemExit(f"bad date {date!r} — expected YYYY-MM-DD or DD-MM-YYYY")


def run_dir(date: str | None = None) -> pathlib.Path:
    """The output folder for a run, named 'DD-MM-YYYY' (defaults to today).
    Accepts ISO (YYYY-MM-DD) or DD-MM-YYYY."""
    if date is None:
        date = datetime.date.today().strftime(DATE_FORMAT)
    else:
        date = _normalize_date(date)
    d = ROOT / date
    d.mkdir(parents=True, exist_ok=True)
    return d


def latest_dir() -> pathlib.Path:
    """The most recently created run folder, for single-stage re-runs.

    Parses each dir name as DD-MM-YYYY and sorts by date (lexicographic sort
    is wrong across year boundaries). Skips dirs that don't match the format.
    """
    if not ROOT.exists():
        raise SystemExit(f"cannot run stage: no run folders exist under {ROOT}/")
    dated: list[tuple[datetime.date, pathlib.Path]] = []
    for p in ROOT.iterdir():
        if not p.is_dir():
            continue
        try:
            d = datetime.datetime.strptime(p.name, DATE_FORMAT).date()
        except ValueError:
            continue  # not a run folder
        dated.append((d, p))
    if not dated:
        raise SystemExit(f"cannot run stage: no run folders exist under {ROOT}/")
    dated.sort(key=lambda t: t[0])
    return dated[-1][1]


def write(name: str, payload, date: str | None = None) -> pathlib.Path:
    d = run_dir(date)
    path = d / name
    path.write_text(json.dumps(payload, indent=2))
    return path


def read(name: str, date: str | None = None):
    d = run_dir(date) if date else latest_dir()
    path = d / name
    if not path.exists():
        raise SystemExit(f"cannot run stage: {name} needs {path}, which doesn't exist")
    return json.loads(path.read_text())
