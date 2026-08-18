from . import Item, StoryGroup
from .rank import _get_client, _parse_json, SKAINET_DEFAULT_MODEL as _MODEL
import json, pathlib, datetime

DATA = pathlib.Path("data")
MIN_CONSENSUS = 1  # any group qualifies; only same-story near-dups merge

DEDUP_PROMPT = """You are clustering AI news items for a weekly podcast.

You are given a flat list of items. Several items may describe the SAME underlying
story (for example, the same paper appearing both as a Hugging Face entry and
inside a newsletter's multi-story issue; or the same announcement covered by two
different newsletters).

Group items into clusters by these rules, PRECISELY:

- Group two items ONLY when they describe the same underlying story (same paper,
  same announcement, same product launch).
- An item with no near-duplicate forms its own single-item cluster.
- A multi-story newsletter item may need to be recognized as one item whose body
  mentions several stories; match it against its best-fitting peer(s) rather than
  splitting it.

Return ONLY a JSON object with a single key "clusters", an array of cluster
objects. There must be a cluster for EVERY item — no item may be dropped. Each
cluster object has these keys:
- "title": a short canonical title for the story
- "content": the single most complete body text among the cluster's items
- "indices": the integer array of this item's index within the "items" array (input order)
- "consensus": the number of DISTINCT sources covering this cluster, as an integer
  (count how many different "source" values appear across the cluster's items)

No prose before or after. No markdown fences.
"""

def distil(items: list[Item]) -> list[StoryGroup]:
    payload = [
        {"index": i, "title": it.title, "source": it.source,
         "date": it.date, "url": it.url, "body": it.body[:4000]}
        for i, it in enumerate(items)
    ]
    user_msg = json.dumps(payload, indent=1)

    raw = _chat("cluster_dedup", user_msg)
    parsed = _parse_json(raw)
    clusters = parsed.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise ValueError(
            f"model returned no cluster list.\nraw response:\n{raw}"
        )

    groups: list[StoryGroup] = []
    seen: set[int] = set()
    for cl in clusters:
        idxs = [int(i) for i in (cl.get("indices") or [])]
        members = []
        for i in idxs:
            if 0 <= i < len(items) and i not in seen:
                seen.add(i)
                members.append(items[i])
        if not members:
            continue
        members.sort(key=lambda it: it.date)
        sources = {it.source for it in members}
        rep = max(members, key=lambda it: len(it.body))
        groups.append(StoryGroup(
            title=(cl.get("title") or rep.title).strip(),
            content=cl.get("content") or rep.body,
            urls=[it.url for it in members],
            sources=sources,
            first_date=members[0].date,
            consensus=len(sources),
            rep_url=rep.url,
        ))

    missing = [it for i, it in enumerate(items) if i not in seen]
    if missing:
        # fail loud: every item must land in exactly one group
        raise ValueError(
            f"{len(missing)} item(s) were not assigned to any cluster.\n"
            f"raw response:\n{raw}"
        )

    _write("distil.json", [_g_dict(g) for g in groups])
    return groups

def _chat(role: str, user_msg: str) -> str:
    resp = _get_client().chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": DEDUP_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content

def _g_dict(g: StoryGroup) -> dict:
    return {
        "title": g.title,
        "content": g.content,
        "urls": g.urls,
        "sources": sorted(g.sources),
        "first_date": g.first_date,
        "consensus": g.consensus,
        "rep_url": g.rep_url,
    }

def _write(name, payload):
    DATA.mkdir(exist_ok=True)
    (DATA / name).write_text(json.dumps(payload, indent=2))
