"""Topic labeling of run items, plus episode aggregation and history backfill.

For NEW runs, topic labels are assigned by the rank-stage judge in the SAME
LLM pass that scores importance (score + reason + label over the full body,
see ``rank.UNIFIED_RUBRIC``) — this module no longer runs a labeler during
ranking.

What this module is used for now:
- ``label_items`` / ``backfill_labels``: label ELIGIBLE already-aired episodes
  (from their manifests) with the cheap Mistral-Small labeler so the history
  topic filter works for episodes aired before judging carried labels.
- ``write_episode_labels`` / ``episode_topics_from_items``: aggregate chosen
  items' topics (from rank) into the episode ``labels.json`` vector.

Inputs are always the items' already-fetched bodies (in-memory or
``collect.json``/``rank.json``) — labeling never re-fetches content.
"""
from __future__ import annotations

import datetime
import json
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import llm
from . import store
from . import topics
from .collect import RELEVANCE_MODEL as LABEL_MODEL

# One small-model request per this many input chars (titles + body head), so
# requests stay fast and never blow the context window.
LABEL_BATCH_CHARS = 9000
# Keep only the head of each body for labeling — titles + lead paragraph carry
# the topic; trimming keeps each request small and fast.
BODY_CHARS = 2000
LABEL_WORKERS = 8
MAX_TRIES = 3


def item_payload(item, local_idx: int) -> dict:
    """One item as it goes to the labeler (title/url + trimmed body)."""
    return {
        "index": local_idx,
        "title": getattr(item, "title", ""),
        "url": getattr(item, "url", ""),
        "body": (getattr(item, "body", "") or "")[:BODY_CHARS],
    }


def _chat(payload: list[dict], model: str) -> str:
    resp = llm.get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": topics.label_prompt()},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0,  # greedy: same input -> same labels (reproducible)
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def _chunk_by_chars(items: list, max_chars: int = LABEL_BATCH_CHARS) -> list[list]:
    """Chunk items into requests of ~``max_chars`` input (title + body head).

    Fixed-count batching ignores that HN bodies are ~8k chars vs ~600-char
    abstracts; char-budgeted batching keeps every request comparable.
    """
    chunks: list[list] = []
    cur: list = []
    size = 0
    for it in items:
        approx = len(getattr(it, "title", "") or "") + min(
            len(getattr(it, "body", "") or ""), BODY_CHARS)
        if cur and size + approx > max_chars:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(it)
        size += approx
    if cur:
        chunks.append(cur)
    return chunks


def _label_chunk(chunk: list, model: str) -> dict[str, dict[str, float]]:
    """One labeler request. Returns ``{normalized_url: {topic: weight}}``.

    The response uses the multi-label shape (a ``labels`` array per item, see
    ``topics.label_prompt``); it is flattened into a salience-weighted vector
    over the globally-unique ids. On a malformed response the whole chunk
    yields no labels (items stay neutral for steering) rather than failing the
    run.
    """
    payload = [item_payload(it, i) for i, it in enumerate(chunk)]
    obj = None
    for _ in range(MAX_TRIES):
        raw = _chat(payload, model)
        try:
            obj = llm.parse_json(raw)
            if "labels" in obj:
                break
        except ValueError:
            continue
    if obj is None or "labels" not in obj:
        return {}

    out: dict[str, dict[str, float]] = {}
    for e in obj.get("labels", []):
        try:
            idx = int(e["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= idx < len(chunk)):
            continue
        cleaned = _flatten_label(e)
        if cleaned:
            out[topics.normalize_url(getattr(chunk[idx], "url", ""))] = cleaned
    return out


def _flatten_label(entry: dict) -> dict[str, float]:
    """Flatten one multi-label entry into a salience-weighted vector.

    Each assigned label id is weighted by its position (most salient first):
    primary = 1.0, then decaying per ``topics.SALIENCE_WEIGHTS``. Accepts the
    ``labels`` array shape the labeler prompt returns, and the legacy single
    ``label`` id for backward-compat caches. Off-taxonomy ids are dropped.
    """
    raw = entry.get("labels")
    if raw is None:
        raw = entry.get("label")
    return topics.salience_weighted(raw)


def label_items(items: list,
                model: str = LABEL_MODEL,
                workers: int = LABEL_WORKERS,
                max_chars: int = LABEL_BATCH_CHARS) -> dict[str, dict[str, float]]:
    """Label every item in ``items`` with a small model, in parallel batches.

    Returns ``{normalized_url: {topic: weight}}`` (sparse; only non-zero,
    taxonomy-valid topics). Items label nothing on a malformed batch.
    """
    if not items:
        return {}
    chunks = _chunk_by_chars(items, max_chars)
    results: dict[int, dict] = {}
    if chunks:
        print(f"      label: {len(items)} items in {len(chunks)} batches "
              f"({workers} workers, model {model})...")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_label_chunk, c, model): ci
                       for ci, c in enumerate(chunks)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
    out: dict[str, dict[str, float]] = {}
    for ci in sorted(results):
        out.update(results[ci])
    print(f"      label: {len(out)}/{len(items)} items labeled ({model})")
    return out


# --- episode-level labels ----------------------------------------------------

def episode_topics_from_items(chosen) -> dict[str, float]:
    """Aggregate chosen items' ``.topics`` into a normalized episode vector,
    weighted by each item's importance score."""
    vecs, weights = [], []
    for r in chosen:
        t = getattr(r, "topics", None)
        if t:
            vecs.append(t)
            weights.append(getattr(r, "score", 1.0))
    return topics.episode_vector(vecs, weights)


def write_episode_labels(chosen, date=None) -> bool:
    """Write ``labels.json`` for a run from its chosen items' topics.

    Returns True when written. Legacy manifests whose items carry no topics
    produce no vector and leave any existing ``labels.json`` (e.g. from a
    backfill) untouched.
    """
    vec = episode_topics_from_items(chosen)
    if not any(w > 0 for w in vec.values()):
        return False
    store.write("labels.json", {
        "source": "manifest",
        "model": LABEL_MODEL,
        "taxonomy_version": len(topics.TAXONOMY),
        "episode_topics": topics.top_topics(vec, k=len(topics.TAXONOMY)),
        "topics": vec,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, date=date)
    return True


# --- backfill: label existing runs (for history filtering) ------------------

def _all_run_roots() -> list[pathlib.Path]:
    if not store.ROOT.exists():
        return []
    return [p for p in store.ROOT.iterdir()
            if p.is_dir() and store._parse_run_id(p.name) is not None]


# Matches a brief item bullet and extracts its URL (same shape as the items
# ``generate._brief_text`` emits: "- [Title](url) · [PDF](pdf) — score X.XX").
_BULLET_URL_RE = re.compile(r"^-\s+\[[^\]]+\]\(([^)]+)\)")


def _annotate_brief_labels(root: pathlib.Path,
                           labels: dict[str, dict[str, float]]) -> bool:
    """Append ``· <topic_id>`` to each labeled item bullet of an existing
    episode's ``podcast_brief.md``.

    Surgical: header, section titles and excerpt blockquotes are left intact;
    bullets already carrying a label token are skipped. Returns True when the
    file changed.
    """
    brief_path = root / "podcast_brief.md"
    if not brief_path.exists():
        return False
    text = brief_path.read_text(encoding="utf-8")
    out: list[str] = []
    changed = False
    for line in text.splitlines():
        m = _BULLET_URL_RE.match(line)
        if m:
            labs = labels.get(topics.normalize_url(m.group(1))) or {}
            if labs and not re.search(r"·\s*[\w, -]+\s*$", line.rstrip()):
                line = line.rstrip() + f" · {','.join(labs)}"
                changed = True
        out.append(line)
    if changed:
        brief_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def backfill_labels(date=None, force: bool = False) -> list[str]:
    """Label the chosen items of existing runs, write per-run episode
    ``labels.json`` (used by the history topic filter for pre-steering
    episodes), and annotate each run's ``podcast_brief.md`` with the per-item
    topics (``— score 0.88 · topic_a,topic_b``). Skips runs that already have
    ``labels.json`` unless forced. Returns the run folder names touched.
    """
    from . import RankedItem
    touched: list[str] = []
    roots = [store.run_dir(date)] if date else _all_run_roots()
    for root in roots:
        labels_path = root / "labels.json"
        if not force and labels_path.exists():
            continue
        ep_path = root / "episode.json"
        if not ep_path.exists():
            continue
        try:
            ep = json.loads(ep_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        manifest = ep.get("manifest") or []
        if not manifest:
            continue
        items = [
            RankedItem(title=m.get("title", ""), url=m.get("url", ""),
                       date=m.get("date", ""), body=m.get("body", ""),
                       source=m.get("source", ""), gate_score=m.get("gate_score", 0.0),
                       hn_points=m.get("hn_points", 0),
                       score=m.get("score", 0.0), judge_reason=m.get("judge_reason", ""))
            for m in manifest
        ]
        labels = label_items(items)
        for it in items:
            it.topics = labels.get(topics.normalize_url(it.url)) or {}
        brief_changed = _annotate_brief_labels(root, labels)
        if write_episode_labels(items, date=root.name):
            touched.append(root.name)
            print(f"      label: backfilled {root.name} "
                  f"({sum(1 for it in items if it.topics)}/{len(items)} items labeled, "
                  f"brief{' annotated' if brief_changed else ' unchanged'})")
    return touched
