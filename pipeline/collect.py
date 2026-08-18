from . import Item
import json, pathlib, urllib.request, urllib.parse, datetime, re, html as H, xml.etree.ElementTree as ET

DATA = pathlib.Path("data")
HF_API = "https://huggingface.co/api/papers"
HN_API = "https://hn.algolia.com/api/v1/search"
BATCH_INDEX = "https://www.deeplearning.ai/the-batch/"
BATCH_ISSUE = "https://www.deeplearning.ai/the-batch/issue-{n}"
IMPORT_AI_FEED = "https://importai.substack.com/feed"
BENSBITES_FEED = "https://www.bensbites.com/feed"
LAST_WEEK_IN_AI_FEED = "https://lastweekin.ai/feed"

def collect(
    date: str | None = None,
    sources: list[str] | None = None,
) -> list[Item]:
    if date is None:
        date = datetime.date.today().isoformat()
    if sources is None:
        sources = ["hf-papers", "rss:import-ai", "rss:bensbites", "batch", "rss:lwiai"]

    items: list[Item] = []
    for name in sources:
        if name not in _SOURCES:
            known = ", ".join(_SOURCES)
            raise SystemExit(f"collect: unknown source {name!r} (known: {known})")
        fetched = _SOURCES[name](date)
        print(f"      {name}: {len(fetched)} items")
        items.extend(fetched)

    if not items:
        raise SystemExit("collect: no items from any source — refusing to write an empty file")

    _write("collect.json", [i.__dict__ for i in items])
    return items

# --- source: Hugging Face Daily Papers (JSON array) ---
def _hf_papers(date: str) -> list[Item]:
    url = f"{HF_API}?date={date}&limit=5"
    print(f"      fetching {url}")
    raw = _get_json(url)
    return [
        Item(
            title=p["title"],
            url=f"https://arxiv.org/abs/{p['id']}",
            date=p.get("publishedAt", date),
            body=p["summary"],
            source="hf-papers",
        )
        for p in raw
    ]

# --- source: Hacker News (Algolia JSON) ---
def _hn(date: str) -> list[Item]:
    day = datetime.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    cutoff = int((day - datetime.timedelta(days=7)).timestamp())
    query = {"query": "AI", "tags": "story", "hitsPerPage": 30,
             "numericFilters": f"created_at_i>{cutoff}"}
    url = f"{HN_API}?{urllib.parse.urlencode(query)}"
    print(f"      fetching {url}")
    raw = _get_json(url)
    return [
        Item(
            title=h["title"],
            url=h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}",
            date=h["created_at"],
            body=h.get("story_text") or "",
            source="hn",
        )
        for h in raw.get("hits", [])
    ]

# --- source: The Batch (HTML scrape of latest weekly issue) ---
def _batch(date: str) -> list[Item]:
    day = datetime.date.fromisoformat(date)
    print(f"      fetching {BATCH_INDEX}")
    index = _get(BATCH_INDEX)
    nums = [int(n) for n in re.findall(r"/the-batch/issue-(\d+)", index)]
    if not nums:
        return []
    latest = max(nums)
    print(f"      fetching {BATCH_ISSUE.format(n=latest)}")
    issue_html = _get(BATCH_ISSUE.format(n=latest))
    issue_date = _find_issue_date(issue_html)
    if issue_date is None or not _is_recent(issue_date, day):
        print(f"      issue {latest} dated {issue_date} outside 7-day window")
        return []
    return [
        Item(
            title=a["title"],
            url=a["url"] or BATCH_ISSUE.format(n=latest),
            date=issue_date,
            body=a["intro"],
            source="batch",
        )
        for a in _extract_articles(issue_html)
    ]

# --- sources: Substack newsletters (RSS / XML family) ---
def _substack_rss(feed_url: str, source_tag: str, date: str) -> list[Item]:
    day = datetime.date.fromisoformat(date)
    print(f"      fetching {feed_url}")
    root = ET.fromstring(_get(feed_url))
    items = []
    for entry in root.iter("item"):
        link = entry.findtext("link") or entry.findtext("guid")
        if not link:
            continue
        pub = _to_iso(entry.findtext("pubDate") or "")
        if pub and not _is_recent(pub, day):
            continue
        body = entry.findtext("{http://purl.org/rss/1.0/modules/content/}encoded")
        if body is None:
            body = entry.findtext("description") or ""
        body = H.unescape(re.sub(r"<[^>]+>", " ", body))
        body = re.sub(r"\s+", " ", body).strip()
        items.append(Item(
            title=(entry.findtext("title") or "").strip(),
            url=link,
            date=pub,
            body=body,
            source=source_tag,
        ))
    return items

def _to_iso(rfc822: str) -> str:
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z",
                "%a, %d %b %Y %H:%M:%S %z",
                "%d %b %Y %H:%M:%S %Z"):
        try:
            return datetime.datetime.strptime(rfc822, fmt).date().isoformat()
        except ValueError:
            continue
    return ""

def _find_issue_date(html_text: str) -> str | None:
    m = re.search(r"[A-Z][a-z]{2} \d{1,2}, \d{4}", html_text)
    if not m:
        return None
    return datetime.datetime.strptime(m.group(0), "%b %d, %Y").date().isoformat()

def _extract_articles(html_text: str) -> list[dict]:
    starts = [
        (m.start(), re.sub(r"<[^>]+>", "", html_text[m.end():html_text.find("</h1>", m.end())]).strip())
        for m in re.finditer(r"<h1[^>]*>", html_text)
    ]
    article_idx = [i for i, (_, t) in enumerate(starts) if t.lower() != "news"]
    out = []
    for k, i in enumerate(article_idx):
        pos, title = starts[i]
        end = starts[article_idx[k + 1]][0] if k + 1 < len(article_idx) else len(html_text)
        seg = html_text[pos:end]
        pm = re.search(r"<p>(.*?)</p>", seg, re.S)
        intro = re.sub(r"<[^>]+>", " ", pm.group(1)).strip() if pm else ""
        intro = H.unescape(re.sub(r"\s+", " ", intro))
        links = re.findall(r'href="(https?://[^"]+)"', seg)
        foreign = [
            l for l in links
            if all(x not in l for x in ("deeplearning.ai", "bit.ly", "charonhub",
                                        "facebook", "twitter", "youtube", "linkedin"))
        ]
        out.append({"title": title, "url": foreign[0] if foreign else None, "intro": intro})
    return out

def _is_recent(pub: str, day: datetime.date) -> bool:
    try:
        dt = datetime.date.fromisoformat(pub)
        return day - datetime.timedelta(days=7) <= dt <= day
    except ValueError:
        return False

def _get(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ai-weekly-podcast-poc/0.1"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")

def _get_json(url: str):  # list (HF) or dict (HN) — caller knows which
    return json.loads(_get(url))

def _write(name, payload):
    DATA.mkdir(exist_ok=True)
    (DATA / name).write_text(json.dumps(payload, indent=2))

_SOURCES = {
    "hf-papers": _hf_papers,
    "hn": _hn,
    "batch": _batch,
    "rss:import-ai": lambda date: _substack_rss(IMPORT_AI_FEED, "rss:import-ai", date),
    "rss:bensbites": lambda date: _substack_rss(BENSBITES_FEED, "rss:bensbites", date),
    "rss:lwiai": lambda date: _substack_rss(LAST_WEEK_IN_AI_FEED, "rss:lwiai", date),
}
