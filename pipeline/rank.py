from . import Item, RankedItem
from . import store
from . import profile as profile_mod
from . import feedback as feedback_mod
from . import llm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, time

BATCH_SIZE = 25            # stories per batch-judge request (empirically fastest)
MAX_JUDGE_TRIES = 3       # retries per batch before giving up on malformed JSON
JUDGE_WORKERS = 4         # concurrent batch-judge calls to the SkAInet backend

# Phase 2 personalization blend weight for importance vs personal score.
# ALPHA=1.0 = pure importance (Phase 1 behaviour); ALPHA=0.0 = pure personal.
ALPHA = float(os.environ.get("ALPHA", "0.7"))
if not (0.0 <= ALPHA <= 1.0):
    raise SystemExit(f"ALPHA={ALPHA!r} — expected a float in [0.0, 1.0]")

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

PERSONAL_PROMPT = """You are scoring this week's AI news items on how well they
match an individual researcher's stated interests.

The researcher's profile:
- Topics they care about (promote): {topics}
- Topics they want less of (demote): {anti_topics}
- What they're working on (verbatim):
{body}

Recent feedback (last few runs):
- Items they kept (found useful): {kept}
- Items they skipped (found not useful): {skipped}

You are given a JSON array of items, each with an integer "index", a "title",
a "url", a "source" ("arxiv" or "hn"), and a "body". Score each 0.0 to 1.0 on
**how well it matches this specific researcher's interests**.

Scoring guide:
- 0.9+ = core interest this quarter; directly about a listed topic.
- 0.7-0.89 = adjacent; touches a listed topic or a method they could borrow.
- 0.4-0.69 = tangential; same broad field but not their focus.
- below 0.4 = off-target; matches an anti-topic or unrelated to their work.

Anti-topics actively demote: an item about an anti-topic should score at most
0.4, even if well-executed. Use the kept/skipped feedback as a weak prior only
(items similar to kept ones rise slightly; similar to skipped ones fall
slightly), never as a veto.

Return ONLY a JSON object with a single key "scores", an array of objects,
each with keys "index" (the item's integer index), "score" (a float), and
"reason" (a one-sentence string explaining the match or non-match). Each
index must appear exactly once. No prose before or after. No markdown fences.
"""


def rank(items: list[Item]) -> list[RankedItem]:
    """Score every item and return the full ranked pool (sorted desc).

    Phase 2: if a profile.md exists and ALPHA < 1.0, a personal-match pass
    scores every item and `final_score = ALPHA*score + (1-ALPHA)*personal_score`.
    Otherwise `final_score = score` (Phase 1 behaviour).
    """
    items = _dedup_by_url(items)
    print(f"      rank: {len(items)} items")

    ranked = _rank_pool(items, UNIFIED_RUBRIC)

    _apply_personal(ranked)

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    store.write("rank.json", [r.__dict__ for r in ranked])
    return ranked


def rank_from_cache(cache_path: str) -> list[RankedItem]:
    """Load a saved rank.json (with importance scores), skip the importance
    pass, run only the personal pass + blend. Writes the blended result to
    the current run dir's rank.json. (Fast iteration mode.)

    Staleness guard: warns if the cache predates rank.py's mtime, since a
    rubric change invalidates the cached importance scores.
    """
    import pathlib
    cache = pathlib.Path(cache_path)
    if not cache.exists():
        raise SystemExit(f"--from-cache: {cache} not found")
    cache_mtime = cache.stat().st_mtime
    rank_mtime = pathlib.Path(__file__).resolve().stat().st_mtime
    if cache_mtime < rank_mtime:
        print(f"      WARN: cache {cache.name} predates rank.py — importance "
              f"scores may be stale; re-run `--only rank` without --from-cache "
              f"to refresh.")

    import json
    raw = json.loads(cache.read_text())
    ranked = [RankedItem(**r) for r in raw]
    print(f"      rank_from_cache: loaded {len(ranked)} items from {cache.name}")
    _apply_personal(ranked)
    ranked.sort(key=lambda r: r.final_score, reverse=True)
    store.write("rank.json", [r.__dict__ for r in ranked])
    return ranked


def _apply_personal(ranked: list[RankedItem]) -> None:
    """Run the personal-match pass on `ranked` in place, then blend.

    No-op (every item keeps final_score == score) when:
      - profile.md is absent (no personal signal), or
      - ALPHA == 1.0 (pure importance, Phase 1 behaviour).
    Otherwise scores every item and writes `personal_score`, `personal_reason`,
    `alpha`, `final_score`.
    """
    prof = profile_mod.load_profile()
    if not prof.topics and not prof.body:
        print("      rank: no profile.md found — skipping personal pass (Phase 1 mode)")
        for r in ranked:
            r.final_score = r.score
        return
    if ALPHA >= 1.0:
        print(f"      rank: ALPHA={ALPHA} — skipping personal pass (pure importance)")
        for r in ranked:
            r.final_score = r.score
        return

    log = feedback_mod.load_feedback()
    kept_titles = _titles_for_urls(ranked, log.kept_urls)
    skipped_titles = _titles_for_urls(ranked, log.skipped_urls)
    print(f"      rank: personal pass (ALPHA={ALPHA}, {len(ranked)} items, "
          f"{len(kept_titles)} kept, {len(skipped_titles)} skipped)...")
    _personal_pass(ranked, prof, kept_titles, skipped_titles)
    for r in ranked:
        r.alpha = ALPHA
        if r.personal_score > 0.0 or r.personal_reason:
            r.final_score = ALPHA * r.score + (1.0 - ALPHA) * r.personal_score
        else:
            r.final_score = r.score


def _titles_for_urls(ranked: list[RankedItem], urls: set[str]) -> list[str]:
    by_url = {r.url: r.title for r in ranked}
    return [by_url[u] for u in urls if u in by_url]


def _personal_pass(ranked: list[RankedItem], prof,
                   kept_titles: list[str], skipped_titles: list[str]) -> None:
    prompt = PERSONAL_PROMPT.format(
        topics=", ".join(prof.topics) or "(none)",
        anti_topics=", ".join(prof.anti_topics) or "(none)",
        body=prof.body or "(none)",
        kept="; ".join(kept_titles) or "(none yet)",
        skipped="; ".join(skipped_titles) or "(none yet)",
    )
    chunks = [ranked[i:i + BATCH_SIZE] for i in range(0, len(ranked), BATCH_SIZE)]
    results: dict[int, list] = {}
    if chunks:
        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
            futures = {ex.submit(_judge_batch, chunk, prompt): ci
                       for ci, chunk in enumerate(chunks)}
            done = 0
            for fut in as_completed(futures):
                ci = futures[fut]
                results[ci] = fut.result()
                done += 1
                print(f"      personal batch {done}/{len(chunks)} done")
    for ci in sorted(results):
        for (score, reason), item in zip(results[ci], chunks[ci]):
            item.personal_score = score
            item.personal_reason = reason


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
        personal_score=0.0, personal_reason="",
        alpha=1.0, final_score=score,
    )
