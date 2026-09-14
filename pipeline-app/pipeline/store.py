import datetime, json, os, pathlib

# Same tree the service reads (service/episodes.py). Overridable per
# subprocess via PIPELINE_DATA_ROOT, which the service always sets.
ROOT = pathlib.Path(os.environ.get("PIPELINE_DATA_ROOT", "data/history"))
DATE_FORMAT = "%d-%m-%Y"
# Run folders are time-stamped so a second episode generated the same day gets
# its own folder instead of overwriting the first. Legacy date-only folders
# (DATE_FORMAT) remain readable.
RUN_ID_FORMAT = "%d-%m-%Y-%H%M%S"


def _normalize_date(date: str) -> str:
    """Accept ISO (YYYY-MM-DD), DD-MM-YYYY, or a run id DD-MM-YYYY-HHMMSS;
    return the folder name (ISO -> DD-MM-YYYY; other forms unchanged)."""
    for fmt in ("%Y-%m-%d", RUN_ID_FORMAT, DATE_FORMAT):
        try:
            parsed = datetime.datetime.strptime(date, fmt)
        except ValueError:
            continue
        return parsed.strftime(DATE_FORMAT if fmt == "%Y-%m-%d" else fmt)
    raise SystemExit(f"bad date {date!r} — expected "
                     f"YYYY-MM-DD, DD-MM-YYYY, or DD-MM-YYYY-HHMMSS")


def _parse_run_id(name: str) -> datetime.datetime | None:
    """Parse a run folder name (DD-MM-YYYY or DD-MM-YYYY-HHMMSS) to a datetime,
    or None if it isn't a run folder."""
    for fmt in (RUN_ID_FORMAT, DATE_FORMAT):
        try:
            return datetime.datetime.strptime(name, fmt)
        except ValueError:
            continue
    return None


def run_dir(date: str | None = None) -> pathlib.Path:
    """The output folder for a run, named 'DD-MM-YYYY' or the run id
    'DD-MM-YYYY-HHMMSS' (defaults to a plain today folder). Accepts ISO
    (YYYY-MM-DD), DD-MM-YYYY, or a run id."""
    if date is None:
        date = datetime.date.today().strftime(DATE_FORMAT)
    else:
        date = _normalize_date(date)
    d = ROOT / date
    d.mkdir(parents=True, exist_ok=True)
    return d


def latest_dir() -> pathlib.Path:
    """The most recently created run folder, for single-stage re-runs.

    Parses each dir name as DD-MM-YYYY or DD-MM-YYYY-HHMMSS and sorts by
    datetime (lexicographic sort is wrong across year boundaries). Skips dirs
    that don't match either format.
    """
    if not ROOT.exists():
        raise SystemExit(f"cannot run stage: no run folders exist under {ROOT}/")
    dated: list[tuple[datetime.datetime, pathlib.Path]] = []
    for p in ROOT.iterdir():
        if not p.is_dir():
            continue
        dt = _parse_run_id(p.name)
        if dt is None:
            continue  # not a run folder
        dated.append((dt, p))
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
