from . import Item, RankedItem
from . import store
from . import profile as profile_mod
from . import feedback as feedback_mod
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import json, os, re, time

SKAINET_BASE_URL = "https://chat.model.tngtech.com/v1/"
DEEP_N = 5               # (legacy) arXiv deep dives — kept for the separated arm
BRIEF_N = 5              # (legacy) HN quick briefs — kept for the separated arm
TOP_N = DEEP_N + BRIEF_N
BATCH_SIZE = 25            # stories per batch-judge request (empirically fastest: ~3x vs 1)
MAX_JUDGE_TRIES = 3        # retries per batch before giving up on malformed JSON
JUDGE_WORKERS = 4          # concurrent batch-judge calls to the SkAInet backend
SKAINET_DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3.8-27B")

# Phase 2 personalization: blend weight for importance vs personal score.
# ALPHA=1.0 = pure importance (Phase 1 behaviour); ALPHA=0.0 = pure personal.
ALPHA = float(os.environ.get("ALPHA", "0.7"))
if not (0.0 <= ALPHA <= 1.0):
    raise SystemExit(f"ALPHA={ALPHA!r} — expected a float in [0.0, 1.0]")

# Which rubric arm to run. "unified" = one source-agnostic rubric over the whole
# pool; "separated" = the legacy two-track (PAPER_RUBRIC for arxiv, NEWS_RUBRIC
# for hn). The A/B test (issue 02) compares both on the same collect.json.
RUBRIC_MODE = os.environ.get("RUBRIC_MODE", "unified").strip().lower()
if RUBRIC_MODE not in ("unified", "separated"):
    raise SystemExit(f"RUBRIC_MODE={RUBRIC_MODE!r} — expected 'unified' or 'separated'")

# Size of the candidate pool the cluster stage embeds. Capped at top-K by score,
# or at the score-distribution knee if that is a real dropoff (issue 03).
CLUSTER_POOL_MAX = 50
MIN_POOL = 25                      # never knee-cut below this — a wobble isn't a cliff

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

PAPER_RUBRIC = """You are a senior researcher at a software consulting firm choosing
which arXiv papers deserve a deep-dive segment on this week's internal AI podcast.

You are given a JSON array of papers, each with an integer "index", a "title",
"url", and "body" (the abstract). Score each 0.0 to 1.0 on how valuable it is
for a consulting practitioner to study closely. Weight heavily:

- Client outcomes: could this plausibly change how we build, buy, or advise on
  AI systems — agents and agentic RL, evals & reliability, inference cost and
  latency, security, data tooling — for real engagements?
- Borrowable method: is it a concrete method, framework, or evaluation insight
  we could apply, or an incremental result on a niche benchmark?
- Rigor & evidence: clean ablations, reproducible numbers, honest failure
  analysis. Prefer studies with results you can reason about.
- Novelty this week: a meaningful step forward, not a rehash.

Anchor scores to these bands: 0.9+ = must-study; 0.7-0.89 = worth a skim before
client work; 0.4-0.69 = minor or niche; below 0.4 = skip.

Penalize narrow domains with no path to client work, toy-setting evals, and
well-written-but-hollow work. Score the substance, never the packaging.

Return ONLY a JSON object with a single key "scores", an array of objects, one
per paper, each with keys "index" (the story's integer index), "score" (a
float), and "reason" (a one-sentence string). Each index must appear exactly
once. No prose before or after. No markdown fences.
"""

NEWS_RUBRIC = """You are a tech lead at a software consulting firm choosing which
Hacker News stories (model releases, product launches, incidents, analysis posts)
deserve a quick brief on this week's internal AI podcast.

You are given a JSON array of stories, each with an integer "index", a "title",
"url", and "body". Score each 0.0 to 1.0 on how much this week's practitioners
should know. Weight heavily:

- Client signal: does it change a model, tool, pricing, or security reality
  clients will ask about — frontier/open model releases, capability or price
  changes, agent tooling, vulnerabilities, vendor moves?
- Concreteness: real capabilities, numbers, benchmarks, prices, named
  incidents — not hype. An empty body earns no credit, but don't punish the
  topic for a failed fetch; judge on the evidence present.
- Takeaway: can a listener leave with one fact or one recommendation?
- Community momentum is a weak prior only: high engagement may reflect real
  importance, never use points as evidence of quality.

Anchor scores to these bands: 0.9+ = clients will ask this week; 0.7-0.89 =
useful context for our own stack choices; 0.4-0.69 = niche or soft; below 0.4 = skip.

Do NOT discount a story because it is a blog post or product announcement rather
than a paper — that is this track's purpose.

Return ONLY a JSON object with a single key "scores", an array of objects, one
per story, each with keys "index" (the story's integer index), "score" (a
float), and "reason" (a one-sentence string). Each index must appear exactly
once. No prose before or after. No markdown fences.
"""

# Phase 2: personal-match scoring. The prompt is FIXED (the no-prompt-chasing
# rule). The listener tunes the profile, not the prompt. The profile body +
# topics/anti_topics + recent feedback are injected as context.
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

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=SKAINET_BASE_URL,
            api_key=os.environ["SKAINET_API_KEY"],
            default_headers={"x-user-agent": "tng/practice-judge"},
        )
    return _client

def rank(items: list[Item]) -> list[RankedItem]:
    """Score every item and return the full ranked pool (sorted desc).

    No top-N slice — the cluster/plan stages select from this pool. Two rubric
    arms are supported via RUBRIC_MODE:
      - "unified":    one source-agnostic rubric over the whole merged pool.
      - "separated":  the legacy two-track (PAPER_RUBRIC for arxiv, NEWS_RUBRIC
                      for hn), each judged in its own pool.

    Phase 2: if a profile.md exists and ALPHA < 1.0, a personal-match pass
    scores every item and `final_score = ALPHA*score + (1-ALPHA)*personal_score`.
    Otherwise `final_score = score` (Phase 1 behaviour).
    """
    items = _dedup_by_url(items)
    print(f"      rank: RUBRIC_MODE={RUBRIC_MODE}, {len(items)} items")

    if RUBRIC_MODE == "unified":
        ranked = _rank_pool(items, UNIFIED_RUBRIC, kind="deep")
    else:
        papers = [i for i in items if i.source == "arxiv"]
        news = [i for i in items if i.source == "hn"]
        print(f"      judging {len(papers)} arXiv papers (deep dives, keep {DEEP_N})...")
        deep = _rank_pool(papers, PAPER_RUBRIC, kind="deep") if papers else []
        print(f"      judging {len(news)} HN stories (quick briefs, keep {BRIEF_N})...")
        brief = _rank_pool(news, NEWS_RUBRIC, kind="brief") if news else []
        ranked = deep + brief

    # Phase 2: personal-match pass + linear blend.
    _apply_personal(ranked)

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    store.write("rank.json", [r.__dict__ for r in ranked])
    return ranked


def rank_from_cache(cache_path: str) -> list[RankedItem]:
    """Load a saved rank.json (with importance scores), skip the importance
    pass, run only the personal pass + blend. Writes the blended result to
    the current run dir's rank.json. (Ticket 06 — fast iteration mode.)

    Staleness guard: warns if the cache predates rank.py's mtime, since a
    rubric change invalidates the cached importance scores.
    """
    import pathlib, time as _time
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
    ranked = [RankedItem(**r) for r in raw]
    print(f"      rank_from_cache: loaded {len(ranked)} items from {cache.name}")
    _apply_personal(ranked)
    ranked.sort(key=lambda r: r.final_score, reverse=True)
    store.write("rank.json", [r.__dict__ for r in ranked])
    return ranked


def rank_ab_personal(cache_path: str, top_n: int = 30) -> dict[str, list[RankedItem]]:
    """Run plain (ALPHA=1.0) vs. personal (configured ALPHA) arms on the same
    cached rank.json, write both + a comparison report. (Ticket 08 — the
    primary tuning digest.)

    Saves `rank_plain.json` / `rank_personal.json` + `rank_report.md`. Does
    NOT overwrite `rank.json` (the active arm stays in charge there), mirroring
    the existing `--ab` rubric A/B.
    """
    import pathlib
    from . import inspect
    cache = pathlib.Path(cache_path)
    if not cache.exists():
        raise SystemExit(f"--ab-personal: {cache} not found")
    raw = json.loads(cache.read_text())
    base = [RankedItem(**r) for r in raw]
    print(f"      [A/B personal] loaded {len(base)} items from {cache.name}")

    # Plain arm: ALPHA=1.0, no personal pass. final_score == score.
    plain = [RankedItem(**r.__dict__) for r in base]
    for r in plain:
        r.alpha = 1.0
        r.personal_score = 0.0
        r.personal_reason = ""
        r.final_score = r.score
    plain.sort(key=lambda r: r.final_score, reverse=True)
    store.write("rank_plain.json", [r.__dict__ for r in plain])

    # Personal arm: run the personal pass + blend at the configured ALPHA.
    personal = [RankedItem(**r.__dict__) for r in base]
    _apply_personal(personal)
    personal.sort(key=lambda r: r.final_score, reverse=True)
    store.write("rank_personal.json", [r.__dict__ for r in personal])

    inspect.personal_ab_report(
        [r.__dict__ for r in plain],
        [r.__dict__ for r in personal],
        top_n=top_n,
    )
    return {"plain": plain, "personal": personal}


def _apply_personal(ranked: list[RankedItem]) -> None:
    """Run the personal-match pass on `ranked` in place, then blend.

    No-op (every item keeps final_score == score) when:
      - profile.md is absent (no personal signal), or
      - ALPHA == 1.0 (pure importance, Phase 1 behaviour).
    Otherwise scores every item (or a subset, per `scope_fn`) and writes
    `personal_score`, `personal_reason`, `alpha`, `final_score`.
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
    # Titles for kept/skipped context (urls alone are useless to the judge).
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
            # out-of-scope item (ticket 07): final_score falls back to importance
            r.final_score = r.score


def _titles_for_urls(ranked: list[RankedItem], urls: set[str]) -> list[str]:
    """Look up titles for a set of URLs in the ranked pool."""
    by_url = {r.url: r.title for r in ranked}
    return [by_url[u] for u in urls if u in by_url]


def _personal_pass(ranked: list[RankedItem], prof, kept_titles: list[str],
                   skipped_titles: list[str]) -> None:
    """Score every item in `ranked` against the profile. Writes
    `personal_score` and `personal_reason` in place."""
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

def rank_ab(items: list[Item]) -> dict[str, list[RankedItem]]:
    """Run both rubric arms on the same collect.json and write a comparison
    report (issue 02). Saves `rank_unified.json` / `rank_separated.json` and
    `rank_report.md`. Does NOT overwrite `rank.json` (the active arm stays in
    charge there).
    """
    from . import inspect
    global RUBRIC_MODE
    items = _dedup_by_url(items)
    results: dict[str, list[RankedItem]] = {}

    for mode, rubric, kind in (("unified", UNIFIED_RUBRIC, "deep"),
                              ("separated", None, None)):
        RUBRIC_MODE = mode
        print(f"      [A/B] {mode} arm: judging {len(items)} items...")
        if mode == "unified":
            ranked = _rank_pool(items, rubric, kind="deep")
        else:
            papers = [i for i in items if i.source == "arxiv"]
            news = [i for i in items if i.source == "hn"]
            deep = _rank_pool(papers, PAPER_RUBRIC, kind="deep") if papers else []
            brief = _rank_pool(news, NEWS_RUBRIC, kind="brief") if news else []
            ranked = deep + brief
        ranked.sort(key=lambda r: r.score, reverse=True)
        store.write(f"rank_{mode}.json", [r.__dict__ for r in ranked])
        results[mode] = ranked
        print(f"      [A/B] {mode}: top score {ranked[0].score:.2f}, "
              f"top-50 ratio {_ratio_top(results[mode])}")

    RUBRIC_MODE = os.environ.get("RUBRIC_MODE", "unified").strip().lower()
    inspect.rubric_ab_report(
        [r.__dict__ for r in results["unified"]],
        [r.__dict__ for r in results["separated"]],
    )
    return results

def _ratio_top(ranked: list[RankedItem], n: int = 50) -> str:
    top = ranked[:n]
    a = sum(1 for r in top if r.source == "arxiv")
    h = sum(1 for r in top if r.source == "hn")
    return f"{a}:{h}"

def _rank_pool(items: list[Item], rubric: str, kind: str) -> list[RankedItem]:
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
            ranked.append(_to_ranked(item, score, reason, kind))
    return ranked

def select_pool(ranked: list[RankedItem], max_n: int = CLUSTER_POOL_MAX,
                min_n: int = MIN_POOL) -> tuple[list[RankedItem], int]:
    """Pick the candidate pool for clustering: top-K by score, capped at the
    score-distribution knee if that is smaller (but never below `min_n`).

    Returns (pool, knee_index) where knee_index is the 0-based index into the
    sorted pool at which the score curve bends (or len(pool)-1 if no clear knee).
    The pool is ranked[:cut] where cut = min(max_n, knee+1) only when the knee
    is a *real* dropoff (>= 1.5x the median local step) and above the floor;
    otherwise we keep the full top-max_n.
    """
    ranked = sorted(ranked, key=lambda r: r.score, reverse=True)
    if len(ranked) <= max_n:
        return ranked, len(ranked) - 1
    knee = _knee(ranked[:max_n + 10])  # look a little past the cap to find a bend
    if knee is not None and knee + 1 >= min_n and _is_real_drop(ranked, knee):
        cut = min(knee + 1, max_n)
        return ranked[:cut], knee
    return ranked[:max_n], max_n - 1

def _is_real_drop(ranked: list[RankedItem], knee: int) -> bool:
    """True if the score drop at the knee is a real cliff, not a wobble.

    The drop at the knee must be >= 1.5x the median step size over the curve,
    so a 0.02 wobble in a 0.82 plateau doesn't trigger an early cut.
    """
    ys = [r.score for r in ranked[:max(knee + 4, len(ranked))]]
    steps = [ys[i] - ys[i + 1] for i in range(len(ys) - 1)]
    if not steps:
        return False
    steps_sorted = sorted(steps)
    median_step = steps_sorted[len(steps_sorted) // 2] or 1e-9
    drop_at_knee = ys[knee] - ys[knee + 1] if knee + 1 < len(ys) else 0.0
    return drop_at_knee >= 1.5 * median_step

def _knee(ranked: list[RankedItem]) -> int | None:
    """L-method knee: the point of max curvature on the sorted score curve.

    Models the curve as two linear segments and finds the split k that
    minimizes total RMSE of the two fits. Returns the 0-based index of the
    knee, or None if the curve is too flat / short to find one.
    """
    n = len(ranked)
    if n < 4:
        return None
    ys = [r.score for r in ranked]
    xs = list(range(n))
    # guard: if the curve is nearly flat, there is no meaningful knee
    if max(ys) - min(ys) < 0.05:
        return None

    def _rmse(x, y):
        if len(x) < 2:
            return 0.0
        mx = sum(x) / len(x)
        my = sum(y) / len(y)
        denom = sum((xi - mx) ** 2 for xi in x) or 1e-9
        slope = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / denom
        intercept = my - slope * mx
        return (sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y)) / len(x)) ** 0.5

    best_k, best_cost = None, float("inf")
    for k in range(2, n - 2):
        c1 = _rmse(xs[:k + 1], ys[:k + 1])
        c2 = _rmse(xs[k:], ys[k:])
        # weight each segment by its share of points (L-method)
        cost = (k * c1 + (n - k) * c2) / n
        if cost < best_cost:
            best_cost, best_k = cost, k
    return best_k

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
    payload = [_story_to_dict(g, i) for i, g in enumerate(groups)]
    user_msg = json.dumps(payload)
    last_err: ValueError | None = None
    for attempt in range(MAX_JUDGE_TRIES):
        raw = _chat(user_msg, rubric)
        try:
            parsed = _parse_json(raw)
            entries = parsed.get("scores")
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

def _chat(user_msg: str, rubric: str) -> str:
    resp = _get_client().chat.completions.create(
        model=SKAINET_DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": rubric},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content

def _to_ranked(item: Item, score: float, reason: str, kind: str) -> RankedItem:
    return RankedItem(
        **item.__dict__,
        score=score, judge_reason=reason, kind=kind,
        personal_score=0.0, personal_reason="",
        alpha=1.0, final_score=score,
    )

def _parse_json(raw: str) -> dict:
    s = raw.strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    start = s.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in model response:\n{raw}")
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                cand = s[start : i + 1]
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"found braces but no valid JSON object in model response:\n{raw}")
