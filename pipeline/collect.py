from . import Item
from . import store
from . import llm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, urllib.request, urllib.parse, datetime, re
import xml.etree.ElementTree as ET
import trafilatura

HN_API = "https://hn.algolia.com/api/v1/search"
ARXIV_API = "https://export.arxiv.org/api/query"

RELEVANCE_GATE_BATCH = 100   # HN titles per LLM relevance call
ARXIV_GATE_BATCH = 20         # arXiv title+abstract per relevance call (~9k tokens/chunk)
MAX_FETCH_BYTES = 1_000_000   # cap a single HN target page (~1 MB raw HTML)
BODY_FETCH_WORKERS = 8        # parallel trafilatura body fetches
RELEVANCE_MODEL = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

RELEVANCE_PROMPT = """You are screening Hacker News stories for a weekly AI podcast
for AI researchers at a software consulting firm.

You are given a JSON array of candidate stories, each with "index", "title", and
"url". Decide which are worth including in a podcast summarizing the week's most
important AI research and product news.

Include a story if it is relevant to AI researchers in a company: model releases,
research papers or results, tools/libraries for AI engineering, datasets and
benchmarks, noteworthy AI industry/company news. Exclude general tech, personal
essays about productivity, UI/programming-language churn, releases of unrelated
software, and pure entertainment.

Return ONLY a JSON object with a single key "relevant", an array of the integer
indices of the stories that pass. If none pass, return {"relevant": []}.
No prose before or after. No markdown fences.
"""

ARXIV_GATE_PROMPT = """You are screening arXiv papers for a weekly AI podcast for AI
researchers at a software consulting firm.

You are given a JSON array of papers, each with an integer "index", a "title",
and a "body" (the abstract). Decide which papers are worth carrying into a
ranking stage that scores the week's most important AI research.

Keep a paper if it is plausibly interesting to practitioners: client outcomes
(agents and agentic RL, evals & reliability, inference cost and latency,
security, data tooling), a concrete method or framework we could borrow, a
genuinely novel result, or a rigor/evidence win we can reason about. Drop only
obvious noise: narrow-focus incremental results on a niche benchmark, toy
settings, pure theory with no application path, or work well outside AI (pure
math, physics, non-AI linguistics).

Be recall-leaning — when in doubt, KEEP. This gate only trims the obvious
junk; a fine-grained scoring judge downstream decides what actually airs.
Prefer to over-keep rather than silently drop a borderline-but-important paper.

Return ONLY a JSON object with a single key "relevant", an array of the integer
indices of the papers that pass. If none pass, return {"relevant": []}.
No prose before or after. No markdown fences.
"""

def collect(date: str | None = None) -> list[Item]:
    """Fetch the week's AI news from HN (points>100 + LLM gate + body fetch) and arXiv (cs.AI)."""
    if date is None:
        date = datetime.date.today().isoformat()

    items: list[Item] = []
    for name in ("hn", "arxiv"):
        fetched = _SOURCES[name](date)
        print(f"      {name}: {len(fetched)} raw items")
        items.extend(fetched)

    items = _dedup_hn_arxiv(items)

    # HN-specific post-processing: relevance gate first (cheap), body fetch last (slow)
    hn_items = [i for i in items if i.source == "hn"]
    if hn_items:
        print(f"      hn: {len(hn_items)} survive arXiv dedup, gating relevance...")
        kept = _hn_relevant(hn_items)
        print(f"      hn: {len(kept)} relevant, fetching bodies in parallel (up to {BODY_FETCH_WORKERS} workers)...")
        t0 = datetime.datetime.now(datetime.timezone.utc)
        with ThreadPoolExecutor(max_workers=BODY_FETCH_WORKERS) as ex:
            futures = {ex.submit(_fetch_body, it.url): it for it in kept if not it.body}
            done = 0
            for fut in as_completed(futures):
                it = futures[fut]
                try:
                    it.body = fut.result() or ""
                except Exception:
                    it.body = ""
                done += 1
                if done % 10 == 0 or done == len(futures):
                    print(f"      hn: {done}/{len(kept)} bodies fetched")
        dt = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
        print(f"      hn: body fetch took {dt:.1f}s")
        items = [i for i in items if i.source != "hn"] + kept

    # arXiv relevance gate: trim the ~987/wk stream to the plausibly-relevant
    # survivors before the judge scores any of them (~5x cheaper rank).
    arxiv_items = [i for i in items if i.source == "arxiv"]
    if arxiv_items:
        print(f"      arxiv: gating {len(arxiv_items)} papers by relevance...")
        items = [i for i in items if i.source != "arxiv"] + _arxiv_relevant(arxiv_items)

    if not items:
        raise SystemExit("collect: no items from any source — refusing to write an empty file")

    store.write("collect.json", [i.__dict__ for i in items])
    return items

# --- source: Hacker News (Algolia JSON): points>100 candidates, title+URL only ---
def _hn(date: str) -> list[Item]:
    day = datetime.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    cutoff = int((day - datetime.timedelta(days=7)).timestamp())
    hits: list[dict] = []
    page = 0
    per_page = 100
    nb_pages = None
    while True:
        query = {"tags": "story", "hitsPerPage": per_page, "page": page,
                 "numericFilters": f"points>100,created_at_i>{cutoff}"}
        url = f"{HN_API}?{urllib.parse.urlencode(query)}"
        print(f"      fetching {url}")
        raw = _get_json(url)
        got = raw.get("hits", [])
        hits.extend(got)
        nb_pages = raw.get("nbPages", nb_pages)
        if not got or nb_pages is None or page + 1 >= nb_pages:
            break
        page += 1

    # dedupe by objectID (multi-page paging can repeat). A repost of the same
    # URL gets a new objectID, so also dedupe by (normalized) URL keeping the
    # highest-points posting. First-seen order is preserved for both kinds.
    seen: set[str] = set()
    slots: dict[str, dict] = {}     # slot key -> winning hit
    order: list[str] = []
    url_slot: dict[str, str] = {}   # normalized URL -> its slot key
    for h in hits:
        oid = h["objectID"]
        if oid in seen:
            continue
        seen.add(oid)
        url = (h.get("url") or "").strip().rstrip("/").lower()
        if url:
            slot = url_slot.get(url)
            if slot is None:
                slot = "url:" + url
                url_slot[url] = slot
                order.append(slot)
            if h.get("points", 0) > slots.get(slot, {}).get("points", 0):
                slots[slot] = h
        else:
            slot = "oid:" + oid
            order.append(slot)
            slots[slot] = h

    out: list[Item] = []
    for slot in order:
        h = slots[slot]
        out.append(Item(
            title=h["title"],
            url=h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}",
            date=h["created_at"],
            body="",  # points used only to select; never emitted
            source="hn",
        ))
    return out

# --- source: arXiv (official Atom API): cat:cs.AI, 7-day window, latest version ---
def _arxiv(date: str) -> list[Item]:
    day = datetime.date.fromisoformat(date)
    cutoff = day - datetime.timedelta(days=7)
    entries: list[Item] = []
    seen_ids: set[str] = set()
    start = 0
    max_results = 100
    while True:
        params = {"search_query": "cat:cs.AI", "sortBy": "submittedDate",
                  "sortOrder": "descending", "start": start, "max_results": max_results}
        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
        print(f"      fetching {url}")
        xml = _get(url)
        feed = ET.fromstring(xml)
        page = feed.findall("{http://www.w3.org/2005/Atom}entry")
        if not page:
            break
        for e in page:
            published = (e.findtext("{http://www.w3.org/2005/Atom}published") or "").strip()
            pub_date = published[:10]
            if _is_recent(pub_date, day):
                base_id = _arxiv_id(e.findtext("{http://www.w3.org/2005/Atom}id") or "")
                if base_id and base_id not in seen_ids and not _withdrawn(e):
                    seen_ids.add(base_id)
                    entries.append(Item(
                        title=_strip_tags(e.findtext("{http://www.w3.org/2005/Atom}title") or ""),
                        url=f"https://arxiv.org/abs/{base_id}",
                        date=pub_date,
                        body=_strip_tags(e.findtext("{http://www.w3.org/2005/Atom}summary") or ""),
                        source="arxiv",
                    ))
        total = feed.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
        if start + len(page) >= int(total or 0):
            break
        oldest_on_page = page[-1].findtext("{http://www.w3.org/2005/Atom}published") or ""
        if oldest_on_page[:10] and oldest_on_page[:10] < cutoff.isoformat():
            break  # fully past the window; no more to page
        start += len(page)
        if start > 2000:
            raise SystemExit("collect: arXiv paging exceeded 2000 entries — window too wide?")
    return entries

def _withdrawn(entry: ET.Element) -> bool:
    """True if the entry's arxiv:comment or attributes mark it withdrawn."""
    NS = "{http://arxiv.org/schemas/atom}"
    comment = (entry.findtext(f"{NS}comment") or "").lower()
    atype = (entry.attrib.get(f"{NS}announce_type") or "")
    return "withdraw" in comment or "withdrawn" in atype

def _arxiv_id(url_or_id: str) -> str | None:
    """Extract a bare arXiv ID from an abs/pdf URL or an Atom id like .../abs/2608.18076v1."""
    if not url_or_id:
        return None
    # arXiv IDs: YYMM.NNNNN or YYYY.NNNNN (4- or 5-digit second part). Must be
    # delimited so a longer trailing number doesn't partially match.
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?$", url_or_id.strip().rstrip("/"))
    if not m:
        m = re.search(r"(?<![\d/])(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", url_or_id.strip().rstrip("/"))
    return m.group(1) if m else None

# --- cross-source dedup: drop HN items pointing at a paper we already collected ---
def _dedup_hn_arxiv(items: list[Item]) -> list[Item]:
    arxiv_ids = {_arxiv_id(i.url) for i in items if i.source == "arxiv"}
    if not arxiv_ids:
        return items
    kept: list[Item] = []
    dropped = 0
    for it in items:
        if it.source == "hn" and _arxiv_id(it.url) in arxiv_ids:
            dropped += 1
            continue
        kept.append(it)
    if dropped:
        print(f"      dedup: dropped {dropped} HN item(s) pointing at a collected arXiv paper")
    return kept

# --- batched LLM relevance gate (shared by HN titles and arXiv abstracts) ---
def _gate(items: list[Item], prompt: str, model: str, batch_size: int,
          label: str, include_body: bool = False) -> list[Item]:
    """Batched recall-leaning relevance gate. Returns the kept subset.

    Same input -> same verdict (temperature=0) so re-runs are reproducible. On a
    malformed response, keep the whole chunk rather than drop it (recall over
    precision): a bad batch must never silently zero out stories.
    """
    if not items:
        return []
    kept: list[Item] = []
    n_batches = (len(items) + batch_size - 1) // batch_size
    gate_t0 = datetime.datetime.now(datetime.timezone.utc)
    for bi, lo in enumerate(range(0, len(items), batch_size), 1):
        chunk = items[lo:lo + batch_size]
        payload = [
            {"index": i, "title": it.title, "url": it.url,
             **({"body": it.body[:2000]} if include_body else {})}
            for i, it in enumerate(chunk)
        ]
        print(f"      {label} relevance gate: batch {bi}/{n_batches} ({len(chunk)} items)...")
        t0 = datetime.datetime.now(datetime.timezone.utc)
        raw = _chat(payload, prompt, model)
        dt = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
        try:
            rel = llm.parse_json(raw).get("relevant") or []
        except ValueError:
            print(f"      {label} relevance gate: bad response, keeping whole chunk ({len(chunk)})")
            rel = list(range(len(chunk)))
        kept_idx: list[int] = []
        for x in rel:
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(chunk):
                kept_idx.append(i)
        kept_idx = sorted(set(kept_idx))
        kept.extend(chunk[i] for i in kept_idx)
        print(f"      {label} relevance gate: batch {bi}/{n_batches} done in {dt:.1f}s, kept {len(kept_idx)}/{len(chunk)}")
    gate_dt = (datetime.datetime.now(datetime.timezone.utc) - gate_t0).total_seconds()
    print(f"      {label} relevance gate: {len(items)} -> {len(kept)} across {n_batches} batches, took {gate_dt:.1f}s total")
    return kept

def _hn_relevant(items: list[Item]) -> list[Item]:
    return _gate(items, RELEVANCE_PROMPT, RELEVANCE_MODEL, RELEVANCE_GATE_BATCH, "hn")

def _arxiv_relevant(items: list[Item]) -> list[Item]:
    kept = _gate(items, ARXIV_GATE_PROMPT, RELEVANCE_MODEL, ARXIV_GATE_BATCH, "arxiv", include_body=True)
    if not kept:
        raise SystemExit("collect: arXiv gate dropped every paper — refusing to write an empty file")
    return kept

def _chat(payload, prompt: str, model: str) -> str:
    resp = llm.get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0,  # greedy: same input -> same gate verdict (reproducible)
    )
    return resp.choices[0].message.content

# --- fetch an arbitrary HN target page and extract the main text ---
def _fetch_body(url: str) -> str:
    try:
        html = _get(url, max_bytes=MAX_FETCH_BYTES)
    except Exception:
        return ""
    try:
        text = trafilatura.extract(html, url=url, include_links=False, include_images=False,
                                   favor_recall=False)
    except Exception:
        text = None
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:8000]

def _strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()

def _is_recent(pub: str, day: datetime.date) -> bool:
    try:
        dt = datetime.date.fromisoformat(pub)
        return day - datetime.timedelta(days=7) <= dt <= day
    except ValueError:
        return False

def _get(url: str, max_bytes: int | None = None) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ai-weekly-podcast-poc/0.1"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read(max_bytes) if max_bytes else r.read()
    return data.decode("utf-8", "replace")

def _get_json(url: str):  # HN returns a dict — caller knows which
    return json.loads(_get(url))

_SOURCES = {
    "hn": _hn,
    "arxiv": _arxiv,
}
