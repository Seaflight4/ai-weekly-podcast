from . import Item, RankedItem
from openai import OpenAI
import json, os, pathlib

DATA = pathlib.Path("data")
TOP_N = 10
BATCH_SIZE = 25            # stories per batch-judge request (empirically fastest: ~3x vs 1)
MAX_JUDGE_TRIES = 3        # retries per batch before giving up on malformed JSON
SKAINET_BASE_URL = "https://chat.model.tngtech.com/v1/"
SKAINET_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

RUBRIC = """You are judging AI news stories for a weekly podcast aimed at
AI researchers at a software consulting firm.

You are given a JSON array of stories, each with an integer "index", and a
"title", "url", and "body". Score each story from 0.0 to 1.0 on how well it
would translate into a podcast segment the listener would find interesting.
Consider:

- Is the claim concrete, not vague hype?
- Would a listener come away with something they could use in client work?
- Is the source primary (paper, official blog) or derivative?

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
    ranked: list[RankedItem] = []
    for lo in range(0, len(items), BATCH_SIZE):
        chunk = items[lo:lo + BATCH_SIZE]
        print(f"      judging {lo+1}-{lo+len(chunk)}/{len(items)} (batch {BATCH_SIZE})")
        result = _judge_batch(chunk)
        for (score, reason), group in zip(result, chunk):
            ranked.append(_to_ranked(group, score, reason))
    ranked.sort(key=lambda r: r.score, reverse=True)
    top = ranked[:TOP_N]
    _write("rank.json", [r.__dict__ for r in top])
    return top

def _story_to_dict(group: Item, idx: int) -> dict:
    return {"index": idx, "title": group.title,
            "url": group.url, "body": group.body}

def _judge_batch(groups: list[Item]) -> list[tuple[float, str]]:
    """Judge a batch of stories in one LLM request. Returns (score, reason) per story.

    Retries up to MAX_JUDGE_TRIES if the model returns malformed JSON or omits an
    index (reasoning models occasionally do), since one bad batch shouldn't kill
    a long ranking run.
    """
    payload = [_story_to_dict(g, i) for i, g in enumerate(groups)]
    user_msg = json.dumps(payload)
    last_err: ValueError | None = None
    for attempt in range(MAX_JUDGE_TRIES):
        raw = _chat(user_msg)
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

def _chat(user_msg: str) -> str:
    resp = _get_client().chat.completions.create(
        model=SKAINET_DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content

def _to_ranked(group: Item, score: float, reason: str) -> RankedItem:
    return RankedItem(**group.__dict__, score=score, judge_reason=reason)

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
