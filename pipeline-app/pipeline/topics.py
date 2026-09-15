"""Multi-label topic taxonomy + shared helpers for item/episode labels.

The taxonomy is derived from the news itself (``pipeline.derive_taxonomy.py``:
free-form label a 4-week corpus sample -> cluster -> verify) and curated by
hand. Every item gets ONE OR MORE topic ids ordered most salient first, so a
label set is a weighted ``{id: weight}`` vector over every facet that applies
(a paper can be both ``post_training`` and ``agents``) with the primary label
weighted highest (see ``SALIENCE_WEIGHTS``). Nothing downstream hard-codes a
single label: cosine ``personal_match``, the ranking blend, episode
aggregation, the brief and history filtering all operate on these
salience-weighted vectors unchanged.

The taxonomy can be refreshed every few months by re-running the derivation
pipeline (see ``pipeline.derive_taxonomy.py``) and dropping its runtime
artifact at ``TAXONOMY_PATH`` (default ``data/taxonomy.json``). If present it
is loaded at import; otherwise the curated ``DEFAULT_TAXONOMY`` is used.
"""
from __future__ import annotations

import json
import math
import os
import pathlib

from . import DATA_ROOT

DEFAULT_TAXONOMY: tuple[dict, ...] = (
    {"id": "agents", "label": "AI Agents",
     "description": "Autonomous or semi-autonomous AI systems that plan, act, and interact with environments."},
    {"id": "ai_for_science", "label": "AI for Science",
     "description": "Applications of AI methods to scientific discovery, simulation, and research domains."},
    {"id": "safety_alignment", "label": "Safety & Alignment",
     "description": "Research and practice focused on making AI systems safe, aligned, and controllable."},
    {"id": "benchmarks", "label": "Benchmarks & Evaluation",
     "description": "Standardized tests, leaderboards, and evaluation methodologies for AI models."},
    {"id": "inference_infrastructure", "label": "Inference & Infrastructure",
     "description": "Hardware, serving stacks, and infrastructure for running AI models at scale."},
    {"id": "multimodal", "label": "Multimodal",
     "description": "Models and techniques spanning text, vision, audio, and cross-modal understanding."},
    {"id": "robotics", "label": "Robotics",
     "description": "AI-driven robotic systems, manipulation, navigation, and embodied intelligence."},
    {"id": "business_economics", "label": "Business & AI Economics",
     "description": "Commercial, market, and economic aspects of the AI industry."},
    {"id": "post_training", "label": "Post-training",
     "description": "Fine-tuning, RLHF/RL, alignment, and other post-training methods applied after base pretraining."},
    {"id": "pretraining", "label": "Pretraining & Scaling",
     "description": "Base-model pretraining, scaling laws, and training data."},
    {"id": "policy", "label": "Policy & Regulation",
     "description": "Government, regulatory, and institutional policy affecting AI development and deployment."},
    {"id": "model_release", "label": "Model Releases",
     "description": "Announcements and launches of new AI models or major model updates."},
    {"id": "interpretability", "label": "Interpretability",
     "description": "Methods and research for understanding, explaining, and probing internal model representations."},
    {"id": "incident", "label": "Incidents & Failures",
     "description": "Notable AI system failures, outages, or operational incidents."},
    {"id": "research_theory", "label": "Research & Theory",
     "description": "Fundamental AI research, theoretical results, and representation learning."},
    {"id": "architecture_world_models", "label": "Architecture & World Models",
     "description": "Novel model architectures and world-model approaches for AI systems."},
    {"id": "retrieval_rag", "label": "Retrieval & RAG",
     "description": "Retrieval-augmented generation and information retrieval techniques for AI systems."},
    {"id": "other", "label": "Other",
     "description": "Relevant but fits no named topic."},
)

# Path of the runtime-loaded taxonomy artifact (see derive_taxonomy.py). When
# it exists and parses, it overrides the curated default so a refreshed
# taxonomy can be adopted without code edits. Overridable via TAXONOMY_PATH.
TAXONOMY_PATH = pathlib.Path(
    os.environ.get("TAXONOMY_PATH", str(DATA_ROOT / "taxonomy.json")))


def _load_taxonomy() -> tuple[dict, ...]:
    """The active taxonomy: ``data/taxonomy.json`` when present, else the
    built-in curated set. Tolerant of a missing/malformed artifact (the
    pipeline never breaks on a bad refresh) and normalizes ``definition`` (the
    derivation artifact's key) to ``description``."""
    spec = None
    try:
        raw = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
        spec = raw.get("taxonomy") or raw.get("canonicals") or raw.get("ids")
    except (OSError, ValueError, AttributeError):
        spec = None
    if not isinstance(spec, list) or not spec:
        return DEFAULT_TAXONOMY
    out: list[dict] = []
    for t in spec:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")
        label = t.get("label") or t.get("id")
        desc = t.get("description") or t.get("definition") or ""
        if isinstance(tid, str) and tid.strip():
            out.append({"id": tid.strip(), "label": str(label or tid),
                        "description": str(desc or "")})
    if not out:
        return DEFAULT_TAXONOMY
    if "other" not in {t["id"] for t in out}:
        out.append({"id": "other", "label": "Other",
                    "description": "Relevant but fits no named topic."})
    # Fail closed to the curated set if the artifact has fewer than two real
    # (non-"other") buckets — too degenerate to steer on.
    if len({t["id"] for t in out if t["id"] != "other"}) < 2:
        return DEFAULT_TAXONOMY
    return tuple(out)


TAXONOMY = _load_taxonomy()
TAXONOMY_IDS = tuple(t["id"] for t in TAXONOMY)
TAXONOMY_BY_ID = {t["id"]: t for t in TAXONOMY}


def reload_taxonomy(path: str | os.PathLike | None = None) -> tuple[dict, ...]:
    """Re-read the runtime taxonomy (used by tests / a hot refresh). Returns
    the newly active ``TAXONOMY`` (mutating the module globals)."""
    global TAXONOMY, TAXONOMY_IDS, TAXONOMY_BY_ID, TAXONOMY_PATH
    if path is not None:
        TAXONOMY_PATH = pathlib.Path(path)
    TAXONOMY = _load_taxonomy()
    TAXONOMY_IDS = tuple(t["id"] for t in TAXONOMY)
    TAXONOMY_BY_ID = {t["id"]: t for t in TAXONOMY}
    return TAXONOMY

# personal_match for an item when no user profile is configured (steering off):
# the blend is neutral, so final == importance.
NEUTRAL_PERSONAL = 0.5

# Label salience weights. The judge returns an item's labels ordered most
# salient first; each label's weight decays by position (primary = 1.0). The
# rubric caps labels at 3, so positions beyond this array drop out of the
# vector (no weight -> no steering contribution).
SALIENCE_WEIGHTS = (1.0, 0.6, 0.35)


def salience_weighted(labels) -> dict[str, float]:
    """Map an ordered label list (most salient first) to a weighted vector.

    ``labels`` may be a list of taxonomy ids, a single id string, or None.
    On-taxonomy ids get ``SALIENCE_WEIGHTS[pos]`` by position; off-taxonomy
    ids are dropped and duplicates keep their first (highest) weight.
    None/empty -> {}.
    """
    if isinstance(labels, str):
        labels = [labels]
    if not isinstance(labels, list):
        return {}
    out: dict[str, float] = {}
    for i, lab in enumerate(labels):
        if (not isinstance(lab, str) or lab not in TAXONOMY_BY_ID
                or lab in out or i >= len(SALIENCE_WEIGHTS)):
            continue
        out[lab] = SALIENCE_WEIGHTS[i]
    return out


def label_prompt() -> str:
    """System prompt shared by the production labeler and the eval reference
    model. Lists the multi-label taxonomy and asks for the exact JSON shape
    the labeler parses."""
    lines = [
        "You label AI news items and research papers for a weekly AI podcast.",
        "",
        "You are given a JSON array of items, each with an integer \"index\", a "
        "\"title\", a \"url\", and a \"body\" (an abstract for arxiv, extracted "
        "page text for hn — it may be long; the head usually suffices).",
        "",
        "For EACH item pick 1 to 3 topic ids from this taxonomy that genuinely "
        "apply (every facet the item actually belongs to):",
    ]
    for t in TAXONOMY:
        lines.append(f"- {t['id']}: {t['description']}")
    lines += [
        "",
        "Rules:",
        "- Give the item ALL ids that meaningfully describe it (a paper can be "
          "both its technical area and another facet, e.g. an agent-training "
          "paper is post_training AND agents; a model launch with a pricing "
          "story is model_release AND business_economics).",
        "- 1 id is usually right; use 2-3 only when the item genuinely spans "
          "multiple buckets. Prefer the most salient facet first.",
        "- A research paper gets its technical area (e.g. post_training, "
          "benchmarks) — NOT model_release.",
        "- An event story (launch, incident, deal) gets the event topic "
          "(model_release, incident, business_economics).",
        "- If nothing fits, use [\"other\"].",
        "",
        "Return ONLY a JSON object with a single key \"labels\": an array of "
        "objects, one per item in input order, each with \"index\" (the item's "
        "integer index) and \"labels\" (an ARRAY of 1-3 string ids). Every "
        "index must appear exactly once. No prose before or after. No markdown "
        "fences.",
    ]
    return "\n".join(lines)


def validate_topic_map(topics) -> list[str]:
    """Return the invalid topic ids in ``topics`` (empty list = valid)."""
    if not isinstance(topics, dict):
        return ["<not a map>"]
    return [k for k in topics if k not in TAXONOMY_BY_ID]


def normalize_url(url: str) -> str:
    """Stable cross-use identity for item/URL keys (mirrors collect._canonical_key)."""
    return ((url or "").strip().rstrip("/").lower())


def personal_match(item_topics: dict[str, float],
                   profile: dict[str, float]) -> float:
    """Cosine similarity of an item's topic vector and the user's profile.

    Labels are salience-weighted vectors (primary facet 1.0, later labels
    decayed — see ``SALIENCE_WEIGHTS``), so the cosine naturally measures how
    much of an item's most-salient content overlaps the profile.

    Returns ``NEUTRAL_PERSONAL`` (0.5) when no profile is configured so the
    blend is a no-op; 0.0 when the item has no labels or no overlap. Empty
    profile => neutral (not zero), so important items keep their rank.
    """
    if not profile:
        return NEUTRAL_PERSONAL
    if not item_topics:
        return 0.0
    num = sum(item_topics.get(k, 0.0) * v for k, v in profile.items())
    d1 = math.sqrt(sum(v * v for v in item_topics.values()))
    d2 = math.sqrt(sum(v * v for v in profile.values()))
    if d1 == 0.0 or d2 == 0.0:
        return 0.0
    return num / (d1 * d2)


def final_score(importance: float, personal: float, alpha: float) -> float:
    """Interpolated additive blend: (1-α)·importance + α·personal."""
    return (1.0 - alpha) * importance + alpha * personal


def episode_vector(item_vectors: list[dict[str, float] | None],
                   weights: list[float] | None = None) -> dict[str, float]:
    """Aggregate per-item topic vectors into one episode vector over the full
    taxonomy.

    Averages the item vectors (optionally weighted by ``weights``, e.g. each
    item's importance score) across all taxonomy ids and normalizes to sum 1 —
    topics the items never touched end at 0, which is what makes episodes
    comparable for filtering and steering.
    """
    weights = weights or [1.0] * len(item_vectors)
    total_w = sum(weights) or 1.0
    agg = {tid: 0.0 for tid in TAXONOMY_IDS}
    for vec, w in zip(item_vectors, weights):
        if not vec:
            continue
        for k, v in vec.items():
            if k in agg:
                agg[k] += w * max(0.0, min(1.0, v))
    norm = sum(agg.values()) or 1.0
    return {k: v / norm for k, v in agg.items()}


def top_topics(vector: dict[str, float], k: int = 5) -> list[dict]:
    """Top-K topics (id + rounded weight) descending by weight, weight > 0.

    Used by the UI/API for chips and the history topic filter.
    """
    items = sorted(((tid, w) for tid, w in vector.items() if w > 0),
                   key=lambda t: (-t[1], t[0]))
    return [{"topic": tid, "weight": round(w, 3)} for tid, w in items[:k]]
