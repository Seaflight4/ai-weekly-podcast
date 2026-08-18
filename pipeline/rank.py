from . import Item, RankedItem
from openai import OpenAI
import json, os, pathlib

DATA = pathlib.Path("data")
TOP_N = 5
SKAINET_BASE_URL = "https://chat.model.tngtech.com/v1/"
SKAINET_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

RUBRIC = """You are judging AI news items for a weekly podcast aimed at
AI researchers at a software consulting firm.

Score each item from 0.0 to 1.0 on how well it would translate into a
podcast segment the listener would find interesting. Consider:

- Is the claim concrete, not vague hype?
- Would a listener come away with something they could use in client work?
- Is the source primary (paper, official blog) or derivative?

Return ONLY a JSON object with two keys: a float "score" and a one-sentence string "reason". No prose before or after. No markdown fences.
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
    ranked = []
    for i, item in enumerate(items):
        print(f"      [{i+1}/{len(items)}] {item.title[:60]}")
        score, reason = _judge(item)
        ranked.append(RankedItem(**item.__dict__, score=score, judge_reason=reason))
    ranked.sort(key=lambda r: r.score, reverse=True)
    top = ranked[:TOP_N]
    _write("rank.json", [r.__dict__ for r in top])
    return top

def _judge(item: Item) -> tuple[float, str]:
    user_msg = f"Title: {item.title}\nSource: {item.source}\nURL: {item.url}\n\nAbstract:\n{item.body}"
    resp = _get_client().chat.completions.create(
        model=SKAINET_DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = resp.choices[0].message.content
    parsed = _parse_json(raw)
    try:
        return float(parsed["score"]), parsed["reason"]
    except KeyError:
        raise ValueError(
            f"model returned JSON without 'score'/'reason' keys.\n"
            f"raw response:\n{raw}\n"
            f"parsed object: {parsed}"
        )

def _parse_json(raw: str) -> dict:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    last = s.rfind("}")
    if last == -1:
        raise ValueError(f"no JSON object found in model response:\n{raw}")
    cursor = 0
    while cursor <= last:
        start = s.find("{", cursor)
        if start == -1 or start > last:
            break
        try:
            return json.loads(s[start : last + 1])
        except json.JSONDecodeError:
            cursor = start + 1
    raise ValueError(f"found braces but no valid JSON object in model response:\n{raw}")

def _write(name, payload):
    DATA.mkdir(exist_ok=True)
    (DATA / name).write_text(json.dumps(payload, indent=2))
