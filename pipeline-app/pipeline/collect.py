from . import Item
from . import store
from . import llm
from . import arxiv_oai
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable, Callable
import json, urllib.request, urllib.error, urllib.parse, datetime, re, time
import xml.etree.ElementTree as ET
import trafilatura

HN_API = "https://hn.algolia.com/api/v1/search"
ARXIV_API = "https://export.arxiv.org/api/query"

RELEVANCE_GATE_BATCH = 100   # HN titles per LLM relevance call
ARXIV_GATE_BATCH = 20         # arXiv title+abstract per relevance call (~9k tokens/chunk)
ARXIV_TITLE_GATE_BATCH = 100  # arXiv titles only per coarse relevance call (cheap trim before Qwen)
MAX_FETCH_BYTES = 1_000_000   # cap a single HN target page (~1 MB raw HTML)
BODY_FETCH_WORKERS = 8        # parallel trafilatura body fetches
GATE_WORKERS = 4              # concurrent relevance-gate batches (small model)
ARXIV_TITLE_GATE_WORKERS = 8  # arXiv title gate is the collect bottleneck; overlap fetch+judge with more workers
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

Return ONLY a JSON object with two keys:
- "relevant": an array of the integer indices of the stories that pass.
- "scores": an array of objects, one per story in the input, each with
  "index" (the integer index) and "score" (a float 0.0 to 1.0, how relevant
  this story is to AI researchers — used to prefilter the expensive ranker
  downstream). Score EVERY index in the batch, whether or not it is in
  "relevant".
If none pass, return {"relevant": [], "scores": [{"index": 0, "score": 0.0}, ...]}.
No prose before or after. No markdown fences.
"""

ARXIV_GATE_PROMPT = """You are screening arXiv papers for a weekly AI podcast for AI
researchers at a software consulting firm. The audience advises clients and
builds systems; they want to know what CHANGES how we build, buy, or advise
on AI THIS WEEK.

You are given a JSON array of papers, each with an integer "index", a "title",
and a "body" (the abstract). Score each paper 0.0 to 1.0 on importance THIS
WEEK using a SELECTIVE scale, then mark in "relevant" the papers worth
carrying into a downstream ranking stage.

Score calibration:
- 0.9+ : a must-know, week-defining result — a major model/capability change,
  an incident, or a paper whose method we would borrow immediately.
- 0.7-0.89 : useful context; a solid but non-urgent paper.
- 0.4-0.69 : niche or incremental.
- below 0.4 : skip.

Be DISCRIMINATING. Most papers are NOT important this week. Only a small
minority should score 0.7 or higher. Never give a paper 0.8+ just because it
is relevant or competent — reserve high scores for papers that change
decisions or announce a real capability step.

ALWAYS HIGH-SCORE (include — do not under-credit): papers announcing a
capability threshold or a step change in what AI can do autonomously, even if
the title sounds niche:
- AI designing, verifying, or deploying real hardware/silicon (tapeouts,
  accelerators, AI-written verified RTL).
- Frontier/superintelligence capability or benchmark work (e.g. ASI-bench,
  progress toward AGI/ASI).
- Agent step-level credit assignment, reward learning, or auditing/eval
  methodology with concrete results.
- Provable/verified reasoning (kernel-checked proofs) that enables safe
  autonomous work.

USUALLY NICHE — score BELOW 0.6 unless there is a genuinely striking result
(keep only the standout, e.g. a major benchmark jump):
- vision-language-action / embodied robotics increments,
- unlearning, jailbreak, or refusal micro-attacks,
- EHR/clinical applications, mobile/edge apps, text-to-SQL,
- speculative-decoding / prefill / micro latency optimizations,
- enterprise/cloud tooling, detector/attribution papers,
- low-level tool-use engineering.

Return ONLY a JSON object with two keys:
- "relevant": an array of the integer indices of the papers that pass.
- "scores": an array of objects, one per paper in the input, each with
  "index" (the integer index) and "score" (a float 0.0 to 1.0, how relevant
  this paper is to practitioners — used to prefilter the expensive ranker
  downstream). Score EVERY index in the batch, whether or not it is in
  "relevant".
If none pass, return {"relevant": [], "scores": [{"index": 0, "score": 0.0}, ...]}.
No prose before or after. No markdown fences.
"""

ARXIV_TITLE_GATE_PROMPT = """You are screening arXiv paper TITLES for a weekly AI podcast
for AI researchers at a software consulting firm. This is a CHEAP COARSE FILTER
that trims the raw cs.AI stream before an expensive ranker scores the survivors.

You are given a JSON array of papers, each with an integer "index", a "title",
and a "url". Decide which titles are plausibly worth carrying into a downstream
ranking stage. Keep anything that could matter to practitioners THIS WEEK;
this stage favors RECALL — the downstream ranker does the fine discrimination.

Score each title 0.0 to 1.0 on how likely the paper is important to AI
researchers, then mark in "relevant" the titles worth keeping.

ALWAYS KEEP / HIGH-SCORE (do not drop on a niche-sounding title):
- AI designing, verifying, or deploying real hardware/silicon (tapeouts,
  accelerators, AI-written verified RTL).
- Frontier/superintelligence capability or benchmark work (e.g. ASI-bench,
  progress toward AGI/ASI).
- Agent step-level credit assignment, reward learning, or auditing/eval
  methodology.
- Provable/verified reasoning (kernel-checked proofs) that enables safe
  autonomous work.

USUALLY DROP / LOW-SCORE unless the title promises a striking result:
- vision-language-action / embodied robotics increments,
- unlearning, jailbreak, or refusal micro-attacks,
- EHR/clinical applications, mobile/edge apps, text-to-SQL,
- speculative-decoding / prefill / micro latency optimizations,
- enterprise/cloud tooling, detector/attribution papers,
- low-level tool-use engineering.

A title is usually enough to spot these topics; when in doubt, KEEP (recall).

Return ONLY a JSON object with two keys:
- "relevant": an array of the integer indices of the titles that pass.
- "scores": an array of objects, one per title in the input, each with
  "index" (the integer index) and "score" (a float 0.0 to 1.0, likelihood of
  importance — used to prefilter the expensive ranker downstream). Score EVERY
  index in the batch, whether or not it is in "relevant".
If none pass, return {"relevant": [], "scores": [{"index": 0, "score": 0.0}, ...]}.
No prose before or after. No markdown fences.
"""

def _collect_hn(date: str, start: datetime.date) -> list[Item]:
    """HN branch: fetch -> relevance gate (title-only, cheap) -> body fetch (slow, parallel).

    An HN item whose body fetch comes back empty is DROPPED here — a bodyless
    story never reaches the rank stage (HN is the only source whose body can be
    empty; arXiv always carries its abstract).
    """
    fetched = _hn(date, start)
    print(f"      hn: {len(fetched)} raw items")
    hn_items = [i for i in fetched if i.source == "hn"]
    if not hn_items:
        return []
    print(f"      hn: gating {len(hn_items)} stories by relevance...")
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
    kept = _drop_empty_bodies("hn", kept)
    return kept


def _collect_arxiv(date: str, start: datetime.date) -> list[Item]:
    """arXiv branch: OAI metadata mirror first, throttled query API as fallback.

    The preferred path reads the locally-mirrored ``cs:cs:AI`` metadata for the
    window (``pipeline/arxiv_oai.py``) — arXiv's sanctioned bulk-metadata sync,
    not subject to the query API's 429 throttling. The query API is used only
    when the mirror fails outright or holds no records for the window.
    """
    try:
        end = datetime.date.fromisoformat(date)
        arxiv_oai.ensure_coverage(
            start - datetime.timedelta(days=arxiv_oai.COVERAGE_BUFFER_DAYS))
        records = arxiv_oai.records_for_window(start, end)
    except Exception as e:  # noqa: BLE001 — mirror is best-effort, never fatal
        print(f"      arxiv: OAI mirror unavailable ({e}); "
              f"falling back to query API")
        return _collect_arxiv_queryapi(date, start)
    if not records:
        print("      arxiv: OAI mirror has no cs.AI records for this window; "
              "falling back to query API")
        return _collect_arxiv_queryapi(date, start)

    items = [
        Item(title=r["title"], url=f"https://arxiv.org/abs/{r['id']}",
             date=r["first_submitted"], body=r["abstract"], source="arxiv")
        for r in records if r.get("abstract")
    ]
    print(f"      arxiv: {len(items)} raw items from OAI mirror")
    kept = _arxiv_relevant_titles(iter(items))
    print(f"      arxiv: {len(kept)}/{len(items)} survived the coarse title "
          f"gate (OAI mirror)")
    return kept


def _collect_arxiv_queryapi(date: str, start: datetime.date) -> list[Item]:
    """arXiv fallback: streamed query-API fetch + coarse title-only gate.

    Pages are fetched one at a time (arXiv throttles to ~1 req/3s + 5s sleep)
    and each accumulated gate batch is judged as soon as it is full, so the LLM
    gate overlaps the inter-page sleeps instead of waiting for the full fetch.
    Used when the OAI mirror (``_collect_arxiv``) cannot serve the window.
    """
    n_raw = 0
    def items():
        nonlocal n_raw
        for page in _arxiv_pages(date, start):
            n_raw += len(page)
            yield from page
    kept = _arxiv_relevant_titles(items())
    print(f"      arxiv: {n_raw} raw items, {len(kept)} survived the coarse title gate")
    return kept


def collect(date: str | None = None, window_start: str | None = None,
            anchor: str | None = None) -> list[Item]:
    """Fetch the week's AI news from every registered source concurrently.

    ``SOURCES`` maps a source id to its collector (``fn(date, start) -> list[Item]``);
    branches run in parallel (inputs are disjoint, gates are shared small-model
    LLM calls, outputs only need a single join): HN goes fetch -> title gate ->
    body fetch; arXiv goes fetch -> coarse title gate. Cross-source dedup is a
    data-driven rule list (``DEDUP_RULES``), so adding a source is
    register-a-collector + a rule.

    Fault-tolerant: a branch that raises (e.g. arXiv throttling with HTTP 429)
    is dropped with a warning and the run continues with the surviving sources —
    the episode then omits that source's content. ``collect`` only aborts when
    NO source yields any item (the empty-file guard).

    ``date`` is the window end (ISO date, defaults to today) and also the
    storage anchor unless ``anchor`` is given — a full-run passes its unique
    run id here so collect.json lands in the same folder as rank/episode.
    ``window_start`` (ISO date) overrides the default 7-day-back cutoff.
    """
    if date is None:
        date = datetime.date.today().isoformat()
    end = datetime.date.fromisoformat(date)
    start = datetime.date.fromisoformat(window_start) if window_start else end - datetime.timedelta(days=7)

    print(f"      collect: running {len(SOURCES)} source branches in parallel...")
    results: dict[str, list] = {}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=len(SOURCES)) as ex:
        futures = {name: ex.submit(fn, date, start) for name, fn in SOURCES.items()}
        for name in SOURCES:
            try:
                results[name] = futures[name].result()
            except Exception as e:  # noqa: BLE001 — never let one source kill the run
                failed.append(name)
                print(f"      collect: source '{name}' failed: {e}")
                print(f"      collect: continuing WITHOUT '{name}' — the episode "
                      f"will lack {name} content")
    if failed:
        ok = ", ".join(n for n in SOURCES if n not in failed)
        print(f"      collect: source(s) failed ({', '.join(failed)}); "
              f"proceeding with: {ok or 'none'}")

    items = []
    for name in SOURCES:          # registry order, so output order is stable
        if name in results:
            items += results[name]

    # Cross-source dedup at the join (after all branches finish), via the
    # data-driven DEDUP_RULES table.
    items = _dedup(items, DEDUP_RULES)

    if not items:
        raise SystemExit("collect: no items from any source — refusing to write an empty file")

    store.write("collect.json", [i.__dict__ for i in items], date=anchor or date)
    return items

# --- source: Hacker News (Algolia JSON): points>100 candidates, title+URL only ---
def _hn(date: str, start: datetime.date) -> list[Item]:
    """HN branch: full-inclusive ``[start, day]`` window on ``created_at``.

    The upper bound is midnight UTC of the day AFTER the end date, so the whole
    end day is included (matching arXiv's ``submittedDate:[... TO ...2359]``).
    A one-day window therefore returns that day's stories instead of an empty
    range.
    """
    cutoff = int(datetime.datetime.combine(start, datetime.time.min, tzinfo=datetime.timezone.utc).timestamp())
    day = datetime.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    end_next = int(day.timestamp()) + 86400  # midnight UTC of the day after end (exclusive upper)
    hits: list[dict] = []
    page = 0
    per_page = 100
    nb_pages = None
    while True:
        query = {"tags": "story", "hitsPerPage": per_page, "page": page,
                 "numericFilters": f"points>100,created_at_i>={cutoff},created_at_i<{end_next}"}
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
            hn_points=h.get("points", 0) or 0,  # the popularity signal for rank
        ))
    return out

def _drop_empty_bodies(label: str, items: list[Item]) -> list[Item]:
    """Drop items whose content fetch came back empty (HN only).

    arXiv always carries its abstract, so it never hits this path. An item with
    no body would reach rank with nothing for the judge to read and would air as
    an empty topic — so it is dropped here, never fed to the rank stage.
    """
    before = len(items)
    kept = [it for it in items if (it.body or "").strip()]
    dropped = before - len(kept)
    if dropped:
        print(f"      {label}: dropped {dropped}/{before} empty-body item(s) "
              f"(no fetched content); they never reach rank")
    return kept

# --- source: arXiv (official Atom API): cat:cs.AI, window, latest version ---
def _arxiv_pages(date: str, start: datetime.date):
    """Yield arXiv items page-by-page (a list per page), streaming.

    A generator so the caller can start gating page N while page N+1 is still
    being fetched: the 5s inter-page politeness sleep is overlapped with the
    relevance gate instead of blocking collection.
    """
    day = datetime.date.fromisoformat(date)
    cutoff = start
    # Query the specific date range directly so we don't page through months of
    # newer entries when collecting a past week. arXiv's submittedDate filter
    # uses [from TO to] (inclusive, format YYYYMMDDHHMM).
    date_range = f"submittedDate:[{cutoff.strftime('%Y%m%d0000')} TO {day.strftime('%Y%m%d2359')}]"
    seen_ids: set[str] = set()
    start = 0
    max_results = 100
    while True:
        params = {"search_query": f"cat:cs.AI AND {date_range}",
                   "sortBy": "submittedDate", "sortOrder": "descending",
                   "start": start, "max_results": max_results}
        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
        print(f"      fetching {url}")
        xml = _get_arxiv(url)
        feed = ET.fromstring(xml)
        page = feed.findall("{http://www.w3.org/2005/Atom}entry")
        if not page:
            break
        page_items: list[Item] = []
        for e in page:
            published = (e.findtext("{http://www.w3.org/2005/Atom}published") or "").strip()
            pub_date = published[:10]
            if _is_recent(pub_date, cutoff, day):
                base_id = _arxiv_id(e.findtext("{http://www.w3.org/2005/Atom}id") or "")
                if base_id and base_id not in seen_ids and not _withdrawn(e):
                    seen_ids.add(base_id)
                    page_items.append(Item(
                        title=_strip_tags(e.findtext("{http://www.w3.org/2005/Atom}title") or ""),
                        url=f"https://arxiv.org/abs/{base_id}",
                        date=pub_date,
                        body=_strip_tags(e.findtext("{http://www.w3.org/2005/Atom}summary") or ""),
                        source="arxiv",
                    ))
        yield page_items
        total = feed.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
        if start + len(page) >= int(total or 0):
            break
        start += len(page)
        if start > 2000:
            raise SystemExit("collect: arXiv paging exceeded 2000 entries — window too wide?")
        time.sleep(5)  # be polite to the arXiv API; avoids 429s

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

def _canonical_key(item: Item) -> str | None:
    """A stable cross-reference identity for a collected item's own URL."""
    u = (item.url or "").strip().rstrip("/")
    if item.source == "arxiv" and _arxiv_id(u):
        return _arxiv_id(u)
    return u.lower()

# --- cross-source dedup: data-driven rule list -------------------------------
# Each rule is (winner_source, dropper_source, dropper_key): when a dropper
# item's key (computed by ``dropper_key`` from its url/body) matches the URL
# identity of any winner item, drop the dropper (the winner is the collected
# arXiv paper, which carries the full abstract).
DEDUP_RULES = [
    ("arxiv", "hn", lambda it: _arxiv_id(it.url)),    # HN twin of a collected paper
]

def _dedup(items: list[Item], rules: list[tuple]) -> list[Item]:
    """Apply ``rules`` in order; each drops dropper items that collide with a
    winner item. Deterministic, O(n) per rule via a key-set hash lookup.

    HN is the pipeline's link-only source: a dropped HN twin carries no unique
    content, but its ``hn_points`` (community attention) is folded onto the
    surviving winner so the rank judge can weigh that a paper/model was
    prominently discussed on HN this week (see ``rank.UNIFIED_RUBRIC``).
    """
    kept = list(items)
    for winner_src, dropper_src, dropper_key in rules:
        winner_by_key: dict[str, Item] = {}
        for i in kept:
            if i.source == winner_src:
                k = _canonical_key(i)
                if k:
                    winner_by_key.setdefault(k, i)  # first-seen wins (stable)
        if not winner_by_key:
            continue
        dropped = 0
        boosted = 0
        out: list[Item] = []
        for it in kept:
            if it.source == dropper_src:
                k = dropper_key(it)
                if k in winner_by_key:
                    dropped += 1
                    if it.hn_points:
                        winner = winner_by_key[k]
                        if it.hn_points > winner.hn_points:
                            winner.hn_points = it.hn_points  # carry attention
                            boosted += 1
                    continue
            out.append(it)
        if dropped:
            extra = f" (carried {boosted} HN upvote signal)" if boosted else ""
            print(f"      dedup: dropped {dropped} {dropper_src} item(s) matching a "
                  f"collected {winner_src} item{extra}")
        kept = out
    return kept


# --- batched LLM relevance gate (shared by HN titles and arXiv abstracts) ---
def _gate(items: list[Item], prompt: str, model: str, batch_size: int,
          label: str, include_body: bool = False,
          workers: int = GATE_WORKERS) -> list[Item]:
    """Batched recall-leaning relevance gate. Returns the kept subset, each
    item carrying its small-model ``gate_score``.

    Batches run in parallel (``workers``); order is restored by batch
    index. Same input -> same verdict (temperature=0) regardless of completion
    order, so parallelism does not change the kept set. On a malformed
    response, keep the whole chunk rather than drop it (recall over precision):
    a bad batch must never silently zero out stories.
    """
    if not items:
        return []
    chunks = [items[lo:lo + batch_size] for lo in range(0, len(items), batch_size)]
    n_batches = len(chunks)
    gate_t0 = datetime.datetime.now(datetime.timezone.utc)
    print(f"      {label} relevance gate: {len(items)} items in {n_batches} batches "
          f"({workers} workers)...")
    results: dict[int, tuple[list[int], dict[int, float]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_gate_batch, chunk, prompt, model, bi, n_batches,
                            label, include_body): bi
                   for bi, chunk in enumerate(chunks)}
        for fut in as_completed(futures):
            bi = futures[fut]
            results[bi] = fut.result()
    kept: list[Item] = []
    for bi in sorted(results):
        kept_idx, scores = results[bi]
        chunk = chunks[bi]
        for local_i in kept_idx:
            it = chunk[local_i]
            it.gate_score = scores.get(local_i, 0.0)
            kept.append(it)
    gate_dt = (datetime.datetime.now(datetime.timezone.utc) - gate_t0).total_seconds()
    print(f"      {label} relevance gate: {len(items)} -> {len(kept)} across "
          f"{n_batches} batches, took {gate_dt:.1f}s total")
    return kept


def _gate_batch(chunk: list[Item], prompt: str, model: str, bi: int,
                n_batches: int, label: str,
                include_body: bool = False) -> tuple[list[int], dict[int, float]]:
    """One relevance-gate LLM call. Returns (kept_local_indices, scores_map).

    kept_local_indices is sorted. scores_map covers every item the model
    scored (kept or dropped). On a malformed response the whole chunk is kept
    (recall) and scores default to 0.0.
    """
    payload = [
        {"index": i, "title": it.title, "url": it.url,
         **({"body": it.body[:2000]} if include_body else {})}
        for i, it in enumerate(chunk)
    ]
    t0 = datetime.datetime.now(datetime.timezone.utc)
    obj = None
    last_raw = ""
    for _ in range(3):
        raw = _chat(payload, prompt, model)
        last_raw = raw
        try:
            obj = llm.parse_json(raw)
            if "relevant" in obj:
                break
        except ValueError:
            pass
    dt = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    where = f"batch {bi+1}" if n_batches is None else f"batch {bi+1}/{n_batches}"
    if obj is None or "relevant" not in obj:
        print(f"      {label} relevance gate: {where} bad response after retries, "
              f"keeping whole chunk ({len(chunk)})")
        return list(range(len(chunk))), {}
    rel = obj.get("relevant") or []
    scores = {}
    for e in (obj.get("scores") or []):
        try:
            scores[int(e["index"])] = float(e["score"])
        except (KeyError, TypeError, ValueError):
            continue
    kept_idx: list[int] = []
    for x in rel:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(chunk):
            kept_idx.append(i)
    kept_idx = sorted(set(kept_idx))
    print(f"      {label} relevance gate: {where} done in {dt:.1f}s, "
          f"kept {len(kept_idx)}/{len(chunk)}")
    return kept_idx, scores

def _hn_relevant(items: list[Item]) -> list[Item]:
    return _gate(items, RELEVANCE_PROMPT, RELEVANCE_MODEL, RELEVANCE_GATE_BATCH, "hn")

def _arxiv_relevant(items: list[Item]) -> list[Item]:
    kept = _gate(items, ARXIV_GATE_PROMPT, RELEVANCE_MODEL, ARXIV_GATE_BATCH, "arxiv", include_body=True)
    if not kept:
        raise SystemExit("collect: arXiv gate dropped every paper — refusing to write an empty file")
    return kept

def _arxiv_relevant_titles(items: list[Item]) -> list[Item]:
    """Cheap coarse arXiv filter: title-only Mistral gate (recall-leaning),
    streamed — pages arrive one at a time and batches are judged as they fill
    (``_gate_stream``), so gating overlaps the inter-page fetch sleeps.

    Unlike ``_arxiv_relevant`` this does NOT raise when everything is dropped:
    it feeds the survivors into Qwen and the top-level empty-pool guard decides
    whether to abort. A lost arXiv branch must not kill a healthy HN branch
    running in parallel.
    """
    kept = _gate_stream(iter(items), ARXIV_TITLE_GATE_PROMPT, RELEVANCE_MODEL,
                        ARXIV_TITLE_GATE_BATCH, "arxiv-title", include_body=False,
                        workers=ARXIV_TITLE_GATE_WORKERS)
    if not kept:
        print("      arxiv-title relevance gate: dropped every paper (keeping none — ranker will see no arXiv items)")
    return kept

def _gate_stream(items: Iterable[Item], prompt: str, model: str, batch_size: int,
                 label: str, include_body: bool = False,
                 workers: int = GATE_WORKERS) -> list[Item]:
    """Streaming relevance gate: judge items as they arrive.

    ``items`` is an iterator (e.g. an arXiv page generator that sleeps between
    fetches). Chunks of ``batch_size`` are submitted to the worker pool the
    moment they fill, so the LLM calls overlap the producer's slow fetches.

    Batch composition and kept-order match ``_gate`` on the same ordered item
    sequence, and (temperature=0) a re-run over a fully-fetched list yields the
    identical kept set. On a malformed response the whole chunk is kept
    (recall), same as ``_gate``.
    """
    gate_t0 = datetime.datetime.now(datetime.timezone.utc)
    print(f"      {label} relevance gate: streaming '{model}' ({workers} workers) as pages arrive...")
    jobs: dict[int, tuple[list[Item], object]] = {}
    buf: list[Item] = []
    seq = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for it in items:
            buf.append(it)
            if len(buf) >= batch_size:
                chunk, buf = buf, []
                jobs[seq] = (chunk, ex.submit(_gate_batch, chunk, prompt, model,
                                              seq, None, label, include_body))
                seq += 1
        if buf:
            jobs[seq] = (buf, ex.submit(_gate_batch, buf, prompt, model,
                                        seq, None, label, include_body))
            seq += 1
        n_total = sum(len(c) for c, _ in jobs.values())
        n_batches = len(jobs)
        kept: list[Item] = []
        for s in sorted(jobs):
            chunk, future = jobs[s]
            kept_idx, scores = future.result()
            for local_i in kept_idx:
                it = chunk[local_i]
                it.gate_score = scores.get(local_i, 0.0)
                kept.append(it)
    gate_dt = (datetime.datetime.now(datetime.timezone.utc) - gate_t0).total_seconds()
    print(f"      {label} relevance gate: {n_total} -> {len(kept)} across "
          f"{n_batches} batches, took {gate_dt:.1f}s total")
    return kept

def _chat(payload, prompt: str, model: str) -> str:
    resp = llm.get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0,  # greedy: same input -> same gate verdict (reproducible)
        response_format={"type": "json_object"},
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

def _is_recent(pub: str, start: datetime.date, end: datetime.date) -> bool:
    try:
        dt = datetime.date.fromisoformat(pub)
        return start <= dt <= end
    except ValueError:
        return False

def _get(url: str, max_bytes: int | None = None) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ai-weekly-podcast-poc/0.1"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read(max_bytes) if max_bytes else r.read()
    return data.decode("utf-8", "replace")

def _get_arxiv(url: str, max_bytes: int | None = None, retries: int = 3,
               base_wait: float = 12.0, cap: float = 45.0) -> str:
    """arXiv page GET with exponential backoff on transient failures.

    arXiv throttles aggressively and answers with HTTP 429 (often "Unknown
    Error") for a stretch after too many requests from an IP; the documented
    recovery is to wait and retry. The budget is deliberately short (~40s
    total): if a throttle persists past that, the collect stage degrades and
    keeps the other sources instead of stalling the whole run for minutes.

    Only used for the arXiv page fetch — HN body fetches keep the fast
    fail-empty path in ``_fetch_body``.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return _get(url, max_bytes=max_bytes)
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code not in (429, 500, 502, 503, 504):
                raise
            last = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError) as e:
            last_exc = e
            last = "network error"
        if attempt == retries - 1:
            break
        wait = min(base_wait * (2 ** attempt), cap)
        print(f"      arXiv {last} on page — backing off {wait:.0f}s then retrying "
              f"({attempt + 1}/{retries})")
        time.sleep(wait)
    print(f"      arXiv {last} — giving up after {retries} retries ({url})")
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"arxiv fetch failed: {last} ({url})")

def _get_json(url: str):  # HN returns a dict — caller knows which
    return json.loads(_get(url))

# --- source registry: adding a source = a collector fn + a label + dedup rules
# Registry order is the output order of the join (and the brief's section order).
SOURCES: dict[str, Callable[[str, datetime.date], list[Item]]] = {
    "hn": _collect_hn,
    "arxiv": _collect_arxiv,
}
