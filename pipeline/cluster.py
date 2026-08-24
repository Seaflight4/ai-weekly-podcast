"""Cluster the ranked pool into topics.

Two-phase, matching the design in `.scratch/podcast-quality/`:

  Phase A (MiniLM, deterministic, cheap): embed the top-K candidate pool
  (`rank.select_pool`), build candidate clusters by cosine similarity merges
  (greedy single-linkage). Nails *topic* grouping; misses *event*-level grouping
  where two items describe the same release with no shared words.

  Phase B (LLM, per-cluster label + cross-cluster merge): one LLM call per
  candidate cluster produces a 2-5 word title, a "why this matters" sentence,
  and a primary-member pick; one final LLM call sees all labelled clusters and
  merges same-event clusters, producing the final `Topic[]` with mixed-source
  members. This is where paper+HN cross-referencing actually happens.

Writes `cluster.json`, `cluster_report.md`, `merge_report.md`.
"""
from __future__ import annotations

from . import RankedItem, Topic, store
from . import rank as rank_mod
from . import inspect
from .rank import _get_client, _parse_json
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, os

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Re-tuned 2026-08-24 for BGE-small (was 0.55 for MiniLM). BGE embeddings have a
# much tighter cosine distribution (median pairwise ~0.67 vs MiniLM's ~0.25),
# so the threshold must be much higher to avoid collapse. 0.82 was chosen by
# calibration on the 21-08-2026 top-50 pool: produces 4 clean multi-member
# clusters (test-time scaling, browser/GUI agents, reasoning compute,
# agentic RL harnesses) without spurious merges. Override via env for A/B.
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.82"))
MERGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3.8-27B")
LABEL_WORKERS = int(os.environ.get("LABEL_WORKERS", "4"))

LABEL_PROMPT = """You are labelling a cluster of this week's AI news items for a podcast topic.

You are given a JSON object with "id" and "members" (each member has "title",
"source" ("arxiv" or "hn"), and "score"). Produce:
- "title": a 2-5 word topic title (no trailing punctuation).
- "why": one sentence on why this matters to AI researchers this week.
- "primary_url": the url of the member that best anchors the topic (the most
  concrete / primary source; prefer an arxiv paper when one is present, else the
  highest-score hn post).

Return ONLY a JSON object with keys "title", "why", "primary_url". No prose
before or after. No markdown fences.
"""

MERGE_PROMPT = """You are merging podcast topic clusters that describe the same underlying event.

You are given a JSON array of clusters, each with "id", "title", "why", and
"members" (each member has "title", "source", "url", "score"). Two clusters
should be MERGED when they describe the same underlying story, event, or
release — even if their words differ (e.g. a model release post and a paper
analyzing that model). Do NOT merge clusters that merely share a broad theme
(e.g. "RLHF" appears in many unrelated papers) — only merge when they are about
the SAME specific event/thing this week.

Return ONLY a JSON object with a single key "merges", an array of objects each
{"into": "<cluster id to keep>", "from": "<cluster id to merge away>"}. Each
"from" id may appear once. If no merges, return {"merges": []}. No prose before
or after. No markdown fences.
"""

_model = None

def _embed_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"      cluster: loading {EMBED_MODEL} (first run downloads ~90MB)...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def cluster(ranked: list[RankedItem]) -> list[Topic]:
    """Build topics from the ranked pool: MiniLM candidate groups -> LLM
    label + cross-cluster merge. Writes cluster.json + reports."""
    pool, knee = rank_mod.select_pool(ranked)
    print(f"      cluster: candidate pool = top {len(pool)} (knee @ {knee})")

    candidates = _minilm_clusters(pool)
    print(f"      cluster: {len(candidates)} candidate groups from MiniLM")

    labelled = _label_clusters(candidates)
    print(f"      cluster: labelled {len(labelled)} clusters")

    final = _merge_clusters(labelled)
    print(f"      cluster: {len(final)} final topics after merge")

    store.write("cluster.json", [_topic_dict(t) for t in final])
    inspect.cluster_report([_topic_dict(t) for t in final])
    inspect.merge_report([_topic_dict(t) for t in labelled], [_topic_dict(t) for t in final])
    return final

# --- Phase A: MiniLM candidate grouping -----------------------------------

def _minilm_clusters(pool: list[RankedItem]) -> list[Topic]:
    """Embed the pool, greedy single-linkage merge by cosine similarity.

    Each item starts as its own cluster; walk items in score order and merge
    into the first existing cluster whose max cosine similarity to any member
    is >= SIM_THRESHOLD. Singletons stay as one-item topics.
    """
    texts = [f"{r.title}. {r.body[:500]}" for r in pool]
    vecs = _embed_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    import numpy as np
    vecs = np.asarray(vecs, dtype="float32")

    clusters: list[Topic] = []
    cluster_vecs: list[list] = []   # parallel: list of member vectors per cluster
    for i, r in enumerate(pool):
        v = vecs[i]
        best_c, best_sim = None, -1.0
        for ci, members_v in enumerate(cluster_vecs):
            sims = [float(v @ mv) for mv in members_v]
            s = max(sims)
            if s > best_sim:
                best_sim, best_c = s, ci
        if best_c is not None and best_sim >= SIM_THRESHOLD:
            clusters[best_c].members.append(r)
            cluster_vecs[best_c].append(v)
        else:
            clusters.append(Topic(id=f"t{len(clusters) + 1}", members=[r],
                                  aggregate_score=r.score, primary_url=r.url))
            cluster_vecs.append([v])
    # recompute aggregate score = max member score
    for t in clusters:
        t.aggregate_score = max((m.score for m in t.members), default=0.0)
    clusters.sort(key=lambda t: t.aggregate_score, reverse=True)
    # renumber after sort
    for i, t in enumerate(clusters, 1):
        t.id = f"t{i}"
    return clusters

# --- Phase B: LLM per-cluster label ---------------------------------------

def _label_clusters(candidates: list[Topic]) -> list[Topic]:
    """One LLM call per candidate cluster: title, why, primary_url.

    Calls are independent and run in parallel (up to LABEL_WORKERS at a time).
    """
    if not candidates:
        return candidates

    def _label_one(t: Topic) -> None:
        payload = {
            "id": t.id,
            "members": [{"title": m.title, "source": m.source,
                         "url": m.url, "score": m.score} for m in t.members],
        }
        raw = _chat(json.dumps(payload), LABEL_PROMPT, MERGE_MODEL)
        try:
            obj = _parse_json(raw)
            t.title = obj.get("title", "").strip()
            t.why = obj.get("why", "").strip()
            primary = obj.get("primary_url", "").strip()
            t.primary_url = primary if primary else (t.members[0].url if t.members else "")
        except ValueError:
            t.title = (t.members[0].title if t.members else t.id)
            t.why = ""
            t.primary_url = t.members[0].url if t.members else ""

    with ThreadPoolExecutor(max_workers=LABEL_WORKERS) as ex:
        futures = {ex.submit(_label_one, t): t for t in candidates}
        done = 0
        for fut in as_completed(futures):
            fut.result()
            done += 1
            print(f"      cluster: labelled {done}/{len(candidates)}")
    return candidates

# --- Phase C: cross-cluster LLM merge ------------------------------------

def _merge_clusters(labelled: list[Topic]) -> list[Topic]:
    """One LLM call across all labelled clusters: merge same-event clusters."""
    payload = [{
        "id": t.id, "title": t.title, "why": t.why,
        "members": [{"title": m.title, "source": m.source, "url": m.url,
                     "score": m.score} for m in t.members],
    } for t in labelled]
    raw = _chat(json.dumps(payload), MERGE_PROMPT, MERGE_MODEL)
    merges: dict[str, str] = {}   # from_id -> into_id
    try:
        for m in _parse_json(raw).get("merges", []):
            merges[m["from"]] = m["into"]
    except (ValueError, KeyError, TypeError):
        print(f"      cluster: bad merge response, skipping merge pass:\n{raw}")

    by_id = {t.id: t for t in labelled}
    final: list[Topic] = []
    consumed: set[str] = set()      # source ids already merged away
    placed: set[str] = set()        # target ids already appended to final
    for t in labelled:
        if t.id in consumed:
            continue
        # follow merge chains: a -> b -> c
        cur = t.id
        seen_chain: set[str] = set()
        while cur in merges and merges[cur] not in seen_chain:
            seen_chain.add(cur)
            cur = merges[cur]
        target = by_id[cur]
        if target is not t:
            target.members.extend(t.members)
            consumed.add(t.id)
        if target.id not in placed:
            final.append(target)
            placed.add(target.id)
    for t in final:
        t.aggregate_score = max((m.score for m in t.members), default=0.0)
        # re-set primary if missing
        if not t.primary_url and t.members:
            t.primary_url = t.members[0].url
    final.sort(key=lambda t: t.aggregate_score, reverse=True)
    for i, t in enumerate(final, 1):
        t.id = f"t{i}"
    return final

# --- chat + serialization -------------------------------------------------

def _chat(user_msg: str, prompt: str, model: str) -> str:
    resp = _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content

def _topic_dict(t: Topic) -> dict:
    return {
        "id": t.id, "title": t.title, "why": t.why,
        "primary_url": t.primary_url, "aggregate_score": t.aggregate_score,
        "members": [m.__dict__ for m in t.members],
    }
