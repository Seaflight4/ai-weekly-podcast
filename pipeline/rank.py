from . import Item, RankedItem
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import json, os, pathlib, time

DATA = pathlib.Path("data")
DEEP_N = 5               # arXiv papers to deep-dive
BRIEF_N = 5              # HN stories to quick-brief
TOP_N = DEEP_N + BRIEF_N
BATCH_SIZE = 25            # stories per batch-judge request (empirically fastest: ~3x vs 1)
MAX_JUDGE_TRIES = 3        # retries per batch before giving up on malformed JSON
JUDGE_WORKERS = 4          # concurrent batch-judge calls to the SkAInet backend
SKAINET_BASE_URL = "https://chat.model.tngtech.com/v1/"
SKAINET_DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-ai/DeepSeek-V4-Flash-0731")

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
    papers = [i for i in items if i.source == "arxiv"]
    news = [i for i in items if i.source == "hn"]

    print(f"      judging {len(papers)} arXiv papers (deep dives, keep {DEEP_N})...")
    deep = _rank_pool(papers, PAPER_RUBRIC, DEEP_N, kind="deep")

    print(f"      judging {len(news)} HN stories (quick briefs, keep {BRIEF_N})...")
    brief = _rank_pool(news, NEWS_RUBRIC, BRIEF_N, kind="brief")

    ranked = deep + brief
    _write("rank.json", [r.__dict__ for r in ranked])
    return ranked

def _rank_pool(items: list[Item], rubric: str, n: int, kind: str) -> list[RankedItem]:
    items = _dedup_by_url(items)
    ranked: list[RankedItem] = []
    chunks = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    results: dict[int, list] = {}
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
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:n]

def _dedup_by_url(items: list[Item]) -> list[Item]:
    """Drop duplicate-URL items, keeping the first occurrence.

    Belt-and-suspenders: collect() already dedups by URL (highest points), but
    this guarantees podcast slots never contain the same URL twice even if a
    later stage reintroduces a duplicate.
    """
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
            "url": group.url, "body": group.body}

def _judge_batch(groups: list[Item], rubric: str) -> list[tuple[float, str]]:
    """Judge a batch of stories in one LLM request. Returns (score, reason) per story.

    Retries up to MAX_JUDGE_TRIES if the model returns malformed JSON or omits an
    index (reasoning models occasionally do), since one bad batch shouldn't kill
    a long ranking run.
    """
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
    return RankedItem(**item.__dict__, score=score, judge_reason=reason, kind=kind)

def _parse_json(raw: str) -> dict:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    # Reasoning models often prefix a `thinking` prose block before the JSON.
    # Find every balanced JSON `{...}` span and return the first one that
    # actually parses as a dict (the full object), tolerating nested braces
    # and trailing prose.
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

def _write(name, payload):
    DATA.mkdir(exist_ok=True)
    (DATA / name).write_text(json.dumps(payload, indent=2))
