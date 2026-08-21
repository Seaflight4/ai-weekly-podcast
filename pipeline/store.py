import datetime, json, pathlib

ROOT = pathlib.Path("data")
DATE_FORMAT = "%d-%m-%Y"

def run_dir(date: str | None = None) -> pathlib.Path:
    """The output folder for a run, named 'DD-MM-YYYY' (defaults to today)."""
    if date is None:
        date = datetime.date.today().strftime(DATE_FORMAT)
    d = ROOT / date
    d.mkdir(parents=True, exist_ok=True)
    return d

def latest_dir() -> pathlib.Path:
    """The most recently created run folder, for single-stage re-runs."""
    dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])
    if not dirs:
        raise SystemExit(f"cannot run stage: no run folders exist under {ROOT}/")
    return dirs[-1]

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