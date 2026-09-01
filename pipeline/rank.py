from . import Item, RankedItem
from . import store
from . import llm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

BATCH_SIZE = 25            # stories per batch-judge request (empirically fastest)
MAX_JUDGE_TRIES = 3       # retries per batch before giving up on malformed JSON
JUDGE_WORKERS = 8         # concurrent batch-judge calls to the SkAInet backend

UNIFIED_RUBRIC = """You are a senior practitioner at a software consulting firm choosing which
items from this week's AI news and research deserve airtime on the internal AI
podcast. The audience is AI researchers who advise clients and build systems.

You are given a JSON array of items, each with an integer "index", a "title",
a "url", a "source" ("arxiv" or "hn"), and a "body" (an abstract for arxiv, the
extracted page text for hn — which may be empty if the fetch failed). Score
each 0.0 to 1.0 on **how much a busy researcher needs to know this week**,
using ONE source-agnostic scale.

Weight:
- Importance this week: does this change how we build, buy, or advise on AI
  systems? A major model release, capability change, incident, or a paper
  whose method we could borrow counts heavily.
- Concreteness: real capabilities, numbers, benchmarks, prices, named events —
  not hype.
- Takeaway: can a listener leave with one fact or one recommendation?

Calibrate against these named reference points so arxiv and hn scores are
directly comparable:
- A typical must-study paper ≈ 0.85.
- A typical major model release / capability change / incident ≈ 0.85
  (a release is AT LEAST as important as a must-study paper — do not discount
  it for being a blog post or announcement; that is its purpose).
- A typical niche or incremental paper ≈ 0.45.
- A typical low-signal HN post ≈ 0.3.

Worked per-source anchors (the two sources are judged together, but the pool
is often lopsided — arXiv may far outnumber HN or vice versa. Do NOT let the
majority source inflate its scores; rate each item against the anchors below,
never against the other items in the batch):
- arXiv: a groundbreaking method paper adopted industry-wide within weeks = 0.9;
  a routine ablation or engineering trick = 0.4; a weak/tangential abstract = 0.3.
- HN: a frontier model release, major incident, or pricing shock announced via
  blog = 0.9; a well-argued engineering writeup with real numbers = 0.7;
  a cool demo with no method transfer = 0.35; a gossip/pro-tip post = 0.3.

Thin-evidence rule: if an item NAMES a major event but has a thin or empty
body (e.g. a launch post whose fetch failed), score it on the EVENT's
importance, not the text length. An empty body never earns credit on its own,
but never penalize a clearly important event for a failed fetch.

Anchor bands: 0.9+ = must-know this week; 0.7-0.89 = useful context; 0.4-0.69
= niche or soft; below 0.4 = skip.

Return ONLY a JSON object with a single key "scores", an array of objects,
one per item, each with keys "index" (the item's integer index), "score" (a
float), and "reason" (a one-sentence string). Each index must appear exactly
once. No prose before or after. No markdown fences.
"""


def rank(items: list[Item], date: str | None = None,
         top_k: int | None = None, score_floor: float | None = None) -> list[RankedItem]:
    """Score every item and return the full ranked pool (sorted desc).

    Pure importance ranking — there is no personalization pass.

    Optional prefilter (``top_k`` / ``score_floor``) shrinks the big-LLM input
    using the small-model ``gate_score`` set by the collect stage, per source
    (HN and arXiv gate scores come from different prompts and are not on a
    common scale, so the cut is applied within each source). ``top_k`` is a
    total budget split proportionally across sources; ``score_floor`` drops
    items below the floor first. Both None = score everything (legacy).
    """
    items = _dedup_by_url(items)
    print(f"      rank: {len(items)} items")
    if top_k is not None or score_floor is not None:
        before = len(items)
        items = _prefilter_by_gate(items, top_k=top_k, score_floor=score_floor)
        print(f"      rank: prefilter {before} -> {len(items)} by gate_score "
              f"(top_k={top_k}, floor={score_floor})")

    ranked = _rank_pool(items, UNIFIED_RUBRIC)
    ranked.sort(key=lambda r: r.score, reverse=True)
    store.write("rank.json", [r.__dict__ for r in ranked], date=date)
    return ranked


def _prefilter_by_gate(items: list[Item], top_k: int | None,
                       score_floor: float | None) -> list[Item]:
    """Per-source gate_score cut. ``top_k`` is a total budget split across
    sources in proportion to their share of the pool (min 10 each). Returns
    items surviving the cut, in gate_score-descending order within source.
    """
    if not items:
        return items
    by_src: dict[str, list[Item]] = {}
    for it in items:
        by_src.setdefault(it.source, []).append(it)
    kept: list[Item] = []
    total = len(items)
    for src, group in by_src.items():
        group.sort(key=lambda i: i.gate_score, reverse=True)
        if score_floor is not None:
            group = [i for i in group if i.gate_score >= score_floor]
        if top_k is not None:
            quota = max(10, int(round(top_k * len(group) / total)))
            group = group[:quota]
        kept.extend(group)
    return kept


def rank_from_cache(cache_path: str, date: str | None = None) -> list[RankedItem]:
    """Load a saved rank.json, sort by score, and re-write.

    Tolerant of extra/legacy keys: only fields declared on ``RankedItem`` are
    passed to its constructor, so older caches (e.g. ones carrying the removed
    ``personal_score``/``final_score`` fields) don't break re-runs.

    Staleness guard: warns if the cache predates rank.py's mtime, since a
    rubric change invalidates the cached importance scores.
    """
    import pathlib, json, dataclasses
    cache = pathlib.Path(cache_path)
    if not cache.exists():
        raise SystemExit(f"--from-cache: {cache} not found")
    cache_mtime = cache.stat().st_mtime
    rank_mtime = pathlib.Path(__file__).resolve().stat().st_mtime
    if cache_mtime < rank_mtime:
        print(f"      WARN: cache {cache.name} predates rank.py — importance "
              f"scores may be stale; re-run `--only rank` without --from-cache "
              f"to refresh.")

    raw = json.loads(cache.read_text())
    known = {f.name for f in dataclasses.fields(RankedItem)}
    ranked = [RankedItem(**{k: v for k, v in r.items() if k in known}) for r in raw]
    ranked.sort(key=lambda r: r.score, reverse=True)
    print(f"      rank_from_cache: loaded {len(ranked)} items from {cache.name}")
    store.write("rank.json", [r.__dict__ for r in ranked], date=date)
    return ranked


def _rank_pool(items: list[Item], rubric: str) -> list[RankedItem]:
    ranked: list[RankedItem] = []
    chunks = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    results: dict[int, list] = {}
    if chunks:
        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
            futures = {ex.submit(_judge_batch, chunk, rubric): ci
                       for ci, chunk in enumerate(chunks)}
            print(f"      judging {len(chunks)} batches (up to {JUDGE_WORKERS} in parallel)...")
            done = 0
            for fut in as_completed(futures):
                ci = futures[fut]
                t_b = time.time()
                results[ci] = fut.result()
                dt = time.time() - t_b
                done += 1
                print(f"      batch {ci + 1}/{len(chunks)} done in {dt:.1f}s ({done}/{len(chunks)} complete)")
    for ci in sorted(results):
        for (score, reason), item in zip(results[ci], chunks[ci]):
            ranked.append(_to_ranked(item, score, reason))
    return ranked


def _dedup_by_url(items: list[Item]) -> list[Item]:
    """Drop duplicate-URL items, keeping the first occurrence."""
    seen: set[str] = set()
    out: list[Item] = []
    for item in items:
        key = (item.url or "").strip().rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _story_to_dict(group: Item, idx: int) -> dict:
    return {"index": idx, "title": group.title,
            "url": group.url, "source": group.source, "body": group.body}


def _judge_batch(groups: list[Item], rubric: str) -> list[tuple[float, str]]:
    """Judge a batch of stories in one LLM request. Returns (score, reason) per story."""
    import json
    payload = [_story_to_dict(g, i) for i, g in enumerate(groups)]
    user_msg = json.dumps(payload)
    last_err: ValueError | None = None
    for attempt in range(MAX_JUDGE_TRIES):
        raw = llm.chat(user_msg, rubric)
        try:
            entries = llm.parse_json(raw).get("scores")
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"model returned no 'scores' list.\nraw response:\n{raw}")
            by_index: dict[int, tuple[float, str]] = {}
            for e in entries:
                try:
                    by_index[int(e["index"])] = (float(e["score"]), e["reason"])
                except (KeyError, TypeError, ValueError):
                    continue
            out = []
            missing = []
            for i in range(len(groups)):
                if i not in by_index:
                    missing.append(i)
                else:
                    out.append(by_index[i])
            if missing:
                raise ValueError(
                    f"model omitted indices {missing} from batch.\nraw response:\n{raw}"
                )
            return out
        except ValueError as e:
            last_err = e
    raise last_err


def _to_ranked(item: Item, score: float, reason: str) -> RankedItem:
    return RankedItem(
        **item.__dict__,
        score=score, judge_reason=reason,
    )
