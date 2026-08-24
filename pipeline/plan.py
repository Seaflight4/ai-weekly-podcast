"""Narrative planner: budget fill + episode outline.

Two responsibilities (issue 06):

  1. Budget fill (deterministic): per-topic `est_minutes` from the primary
     member's body length at ~150 wpm, +~0.75 min per extra member. Greedy-fill
     topics in aggregate-score order to 28-30 min; clamp the total to
     [24, 34] min. This replaces the old fixed top-10 with a week-adaptive,
     budget-aware episode length.

  2. Narrative plan (one structured LLM call): a JSON outline — hook, ordered
     segments with allocated minutes + opening + 2 signposts + transition +
     speakers, an episode-wide motif, and an outro. This is the "narrative
     planner" of the delivery engine; `generate` renders it into the brief
     and feeds NotebookLM a rich instruction scaffold.

Writes `episode_plan.json` + `budget_report.md` + `plan_report.md`.
"""
from __future__ import annotations

from . import Topic, EpisodePlan, Segment, EditorialGroup, store
from . import inspect
from .rank import _get_client, _parse_json
import datetime, json, os

WPM = 150                     # speaking rate for minute estimation
MIN_PER_EXTRA_MEMBER = 0.75   # each non-primary member adds this much airtime
INTRO_MIN = 1.5
OUTRO_MIN = 1.0
TRANSITION_MIN = 0.25
TARGET_MIN = 30
MIN_MIN = 24
MAX_MIN = 34
PLAN_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3.8-27B")

EDITORIAL_ROLES = ("release", "method", "incident", "evaluation",
                    "industry", "analysis", "standalone")

PLAN_PROMPT = """You are the narrative planner for a weekly AI news podcast for AI researchers

You do two jobs in one response: (1) group the topics into 3-5 editorial
meta-groups, and (2) produce the narrative outline for the episode.

You are given a JSON object with "date", "target_minutes", "topics" — an
array of this week's topics in importance order, each with "id", "title", "why",
"minutes" (the budget-allocated minutes for that topic), and "members" (each
with "title", "source", "url", "score").

=== JOB 1: editorial_groups ===
Group the topics into 3-5 editorial meta-groups that give the episode a
narrative shape beyond a flat list.

Each group has:
- "label": a human-readable group name that frames the week.
- "role": exactly one of: release, method, incident, evaluation, industry,
  analysis, standalone.
- "topic_ids": the topic ids in this group (each topic in exactly one group;
  no overlap).
- "opener": one short sentence leading into the first topic. Keep it minimal.

Role definitions:
- release: a new model, tool, or asset that shipped.
- method: a concrete technique the listener could borrow.
- incident: a failure, vulnerability, or attack.
- evaluation: a benchmark, capability measurement, or reproduction/debunking.
- industry: a business move, acquisition, pricing change, hardware, or financing.
- analysis: a trend synthesis, opinion piece, or practitioner writeup.
- standalone: doesn't fit any of the above.

Rules for groups:
- Each topic appears in exactly one group.
- Use 3-5 groups (not 1, not 10).
- Prefer fewer, larger groups over many singletons.

=== JOB 2: narrative outline ===
Produce a podcast outline that gives a listener the shape of the week concisely.

Design choices:
- "hook": one or two sentences tying the week's biggest theme to the first
  topic. Concrete, not generic.
- "motif": a single short phrase the hosts can return to in transitions.
  Must genuinely fit this week's themes.
- "segments": one per topic, ordered by editorial group (all topics in the
  first group, then the second, etc.). For each:
  - "opening": go straight to the topic. Do NOT repeat the group's opener —
    the listener already heard it. One sentence: what this topic is.
  - "signposts": 0-2 listener cues, ONLY when a key number or non-obvious
    takeaway genuinely aids understanding. Most segments need zero. NEVER use
    template phrases like "The key number here is..." or "What to take away
    is..." — if a signpost is worth including, state the number or takeaway
    directly.
  - "transition_out": one short line bridging to the next topic (or the outro
    for the last). A simple "Next:" or one bridge phrase is enough. Do NOT
    write a paragraph.
  - "speakers": two short host names, e.g. ["A","B"].
- "outro": one sentence pointing listeners to the links for deeper reading.

Keep the total minutes across segments within the target. Do NOT exceed the
"minutes" budget per topic.

Return ONLY a JSON object with keys "editorial_groups" (array of objects
each with "label", "role", "topic_ids", "opener"), "hook", "motif",
"segments" (array, one per topic in order, each with "opening", "signposts"
(array, 0-2 strings), "transition_out", "speakers" (array of 2 strings)),
and "outro". No prose before or after. No markdown fences.
"""

def plan(topics: list[Topic], date: str | None = None) -> EpisodePlan:
    """Fill the budget, run editorial grouping + narrative planner in a single
    LLM call, write episode_plan.json + reports."""
    if date is None:
        date = datetime.date.today().isoformat()

    seg_minutes = _budget_fill(topics)
    chosen = topics[:len(seg_minutes)]
    ep_topics = [{
        "id": t.id, "title": t.title, "why": t.why, "minutes": seg_minutes[i],
        "members": [{"title": m.title, "source": m.source, "url": m.url,
                     "score": m.score} for m in t.members],
    } for i, t in enumerate(chosen)]

    raw = _chat(json.dumps({"date": date, "target_minutes": TARGET_MIN,
                           "topics": ep_topics}),
                PLAN_PROMPT, PLAN_MODEL)
    try:
        obj = _parse_json(raw)
    except ValueError:
        print(f"      plan: bad response, falling back to minimal outline:\n{raw}")
        obj = {"editorial_groups": [], "hook": "", "motif": "", "segments": [], "outro": ""}

    groups = _parse_groups(obj.get("editorial_groups", []), ep_topics)
    print(f"      plan: {len(groups)} editorial groups")

    # Build a topic_id -> group label map so each segment knows its group.
    group_label_by_topic: dict[str, str] = {}
    for g in groups:
        for tid in g.topic_ids:
            group_label_by_topic[tid] = g.label

    segments = []
    for i, seg in enumerate(obj.get("segments", [])):
        if i >= len(chosen):
            break
        tid = chosen[i].id
        segments.append(Segment(
            topic_id=tid, minutes=seg_minutes[i],
            opening=seg.get("opening", ""), signposts=seg.get("signposts", [])[:2],
            transition_out=seg.get("transition_out", ""),
            speakers=seg.get("speakers", [])[:2],
            editorial_group=group_label_by_topic.get(tid, ""),
        ))

    ep = EpisodePlan(
        date=date, target_minutes=TARGET_MIN,
        hook=obj.get("hook", ""), motif=obj.get("motif", ""),
        segments=segments, outro=obj.get("outro", ""),
        editorial_groups=groups,
    )
    store.write("episode_plan.json", _plan_dict(ep))

    # reports
    inspect.budget_report(
        [m.__dict__ for t in topics for m in t.members],
        _plan_dict(ep),
    )
    inspect.plan_report(_plan_dict(ep))
    return ep


def _parse_groups(raw_groups: list, ep_topics: list[dict]) -> list[EditorialGroup]:
    """Parse editorial groups from the merged LLM response. Falls back to a
    single standalone group if parsing fails or no groups are returned."""
    groups: list[EditorialGroup] = []
    for g in raw_groups:
        role = g.get("role", "standalone")
        if role not in EDITORIAL_ROLES:
            role = "standalone"
        groups.append(EditorialGroup(
            label=g.get("label", ""),
            role=role,
            topic_ids=g.get("topic_ids", []),
            opener=g.get("opener", ""),
        ))
    if not groups:
        groups = [EditorialGroup(
            label="This week's topics", role="standalone",
            topic_ids=[t["id"] for t in ep_topics],
            opener="",
        )]
    return groups

def _budget_fill(topics: list[Topic]) -> list[float]:
    """Greedy-fill topics in aggregate-score order to TARGET_MIN.

    Returns per-topic minutes for the chosen topics (in the order they were
    chosen). The fill stops when adding the next topic would exceed MAX_MIN.
    Always includes at least one topic.
    """
    order = sorted(topics, key=lambda t: t.aggregate_score, reverse=True)
    minutes: list[float] = []
    cum = INTRO_MIN + OUTRO_MIN
    for t in order:
        tm = _topic_minutes(t)
        if cum + tm + TRANSITION_MIN > MAX_MIN and minutes:
            break
        minutes.append(tm)
        cum += tm + TRANSITION_MIN
    if not minutes and order:
        minutes.append(_topic_minutes(order[0]))
    # clamp the running total into [MIN_MIN, MAX_MIN]
    total = INTRO_MIN + OUTRO_MIN + sum(minutes) + TRANSITION_MIN * max(len(minutes) - 1, 0)
    if total < MIN_MIN and len(order) > len(minutes):
        # try to pull in more topics to reach the floor
        for t in order[len(minutes):]:
            tm = _topic_minutes(t)
            minutes.append(tm)
            total = INTRO_MIN + OUTRO_MIN + sum(minutes) + TRANSITION_MIN * max(len(minutes) - 1, 0)
            if total >= MIN_MIN:
                break
    return minutes

def _topic_minutes(t: Topic) -> float:
    """Estimate minutes for one topic from the primary member's body length."""
    primary = next((m for m in t.members if m.url == t.primary_url), None)
    primary = primary or (t.members[0] if t.members else None)
    words = len((primary.body or "").split()) if primary else 0
    base = max(words / WPM, 1.5)
    extra = max(len(t.members) - 1, 0) * MIN_PER_EXTRA_MEMBER
    return round(base + extra, 1)

def _chat(user_msg: str, prompt: str, model: str) -> str:
    resp = _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content

def _plan_dict(ep: EpisodePlan) -> dict:
    return {
        "date": ep.date, "target_minutes": ep.target_minutes,
        "hook": ep.hook, "motif": ep.motif, "outro": ep.outro,
        "segments": [{
            "topic_id": s.topic_id, "minutes": s.minutes, "opening": s.opening,
            "signposts": s.signposts, "transition_out": s.transition_out,
            "speakers": s.speakers, "editorial_group": s.editorial_group,
        } for s in ep.segments],
        "editorial_groups": [{
            "label": g.label, "role": g.role,
            "topic_ids": g.topic_ids, "opener": g.opener,
        } for g in ep.editorial_groups],
    }
