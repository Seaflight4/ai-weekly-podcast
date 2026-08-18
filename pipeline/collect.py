from . import Item
import json, pathlib, urllib.request, datetime

DATA = pathlib.Path("data")
HF_API = "https://huggingface.co/api/papers"

def collect(date: str | None = None) -> list[Item]:
    if date is None:
        date = datetime.date.today().isoformat()
    url = f"{HF_API}?date={date}&limit=20"
    print(f"      fetching {url}")
    raw = _get(url)
    items = [_to_item(p, date) for p in raw]
    _write("collect.json", [i.__dict__ for i in items])
    return items

def _get(url: str) -> list[dict]:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def _to_item(paper: dict, query_date: str) -> Item:
    return Item(
        title=paper["title"],
        url=f"https://arxiv.org/abs/{paper['id']}",
        date=paper.get("publishedAt", query_date),
        body=paper["summary"],
        source="hf-papers",
    )

def _write(name, payload):
    DATA.mkdir(exist_ok=True)
    (DATA / name).write_text(json.dumps(payload, indent=2))