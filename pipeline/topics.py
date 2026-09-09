"""Three-axis topic taxonomy + shared helpers for item/episode labels.

Labels live on three orthogonal axes so item/episode vectors are comparable
and steerable by interest:

- ``PRIMARY``  — the headline *kind* of news (single choice per item).
- ``TECHNICAL`` — the *technical area* the item is actually about (multi-topic,
  weighted). This is the main steering signal.
- ``APPLICATION`` — the *domain* it serves (multi-topic, weighted).

Every id is globally unique across axes, so the rest of the pipeline treats a
label set as one flat ``{id: weight}`` vector: cosine ``personal_match``, the
ranking blend, episode aggregation and history filtering all work unchanged.

Axes are split deliberately: at eval time we can measure per-axis agreement
(small model vs a big reference), and the small production labeler only has to
choose within a homogeneous set per axis instead of over one mixed list — the
main cause of its earlier mislabels (research papers tagged ``model_release``).
"""
from __future__ import annotations

import math

# Primary axis: single, most-salient headline type. Carried at reduced weight
# (PRIMARY_WEIGHT) in the flat vector so a generic type like "research_findings"
# never swamps the technical facets for steering or episode chips; it stays
# first-class for filtering.
PRIMARY_WEIGHT = 0.5

PRIMARY = [
    {"id": "model_release", "label": "Model releases",
     "description": "A new or updated model is announced/released (incl. capability, pricing or availability notes)."},
    {"id": "research_findings", "label": "Research findings",
     "description": "A research paper/result presenting a method or empirical finding."},
    {"id": "evaluation", "label": "Evaluation",
     "description": "A benchmark, eval, leaderboard or measurement study."},
    {"id": "safety_risk", "label": "Safety & risk",
     "description": "Analysis/report of safety, alignment or risk (not a live incident)."},
    {"id": "incident", "label": "Incident",
     "description": "A concrete incident: breach, escape, outage, real-world failure."},
    {"id": "infrastructure", "label": "Infrastructure",
     "description": "Hardware, chips, data centers, training/serving infra news."},
    {"id": "open_source", "label": "Open source",
     "description": "Open-weights/model hub/licensing/community platform news."},
    {"id": "business", "label": "Business",
     "description": "M&A, funding, pricing, partnerships, company/market news."},
    {"id": "policy", "label": "Policy",
     "description": "Regulation, law, standards, governance, international coordination."},
    {"id": "other", "label": "Other",
     "description": "Relevant but fits no primary type."},
]

TECHNICAL = [
    {"id": "pretraining", "label": "Pretraining",
     "description": "Pretraining runs, scaling laws, data quality/curation, compute, architecture."},
    {"id": "post_training", "label": "Post-training",
     "description": "Fine-tuning, RLHF/RLVR, SFT, DPO, synthetic data, alignment tuning."},
    {"id": "inference_efficiency", "label": "Inference & efficiency",
     "description": "Inference speed/cost, quantization, speculative decoding, serving, kernels."},
    {"id": "agents_tool_use", "label": "Agents & tool use",
     "description": "Agentic systems, tool/computer use, long-horizon tasks, agent scaffolding."},
    {"id": "multimodality", "label": "Multimodal (vision/audio/video)",
     "description": "Image/video/audio/speech models, vision-language, generation, perception."},
    {"id": "robotics_embodied", "label": "Robotics & embodied",
     "description": "Robotics, embodied agents, manipulation, world models for control."},
    {"id": "ai_for_science", "label": "AI for science",
     "description": "AI used for scientific discovery, formal verification, proofs, research automation."},
    {"id": "safety_alignment", "label": "Safety & alignment research",
     "description": "Research on alignment, interpretability, misuse, jailbreaks, system governance."},
    {"id": "benchmarks_evals", "label": "Benchmarks & evals",
     "description": "New benchmarks, evaluation methodology, red-teaming, measurement."},
]

APPLICATION = [
    {"id": "coding", "label": "Coding", "description": "Software engineering, code generation, dev tooling."},
    {"id": "enterprise", "label": "Enterprise", "description": "Business/workplace software, agents-for-work, integration."},
    {"id": "healthcare", "label": "Healthcare", "description": "Clinical, biomedical, health applications."},
    {"id": "finance", "label": "Finance", "description": "Financial, trading, fintech applications."},
    {"id": "education", "label": "Education", "description": "Learning, tutoring, education applications."},
    {"id": "government", "label": "Government & public", "description": "Public sector, defense, civic applications."},
    {"id": "science", "label": "Science", "description": "Scientific domains (bio, chem, physics...) as application."},
    {"id": "creative_media", "label": "Creative & media", "description": "Art, music, video, content creation, games."},
    {"id": "consumer", "label": "Consumer", "description": "Consumer products, assistants, apps for general users."},
]

AXES: dict[str, list[dict]] = {
    "primary": PRIMARY,
    "technical": TECHNICAL,
    "application": APPLICATION,
}

# Flattened view: {id: {id, label, description, axis}} + ordered tuples.
TAXONOMY: tuple[dict, ...] = tuple(
    {**t, "axis": axis}
    for axis, entries in AXES.items()
    for t in entries
)
TAXONOMY_IDS = tuple(t["id"] for t in TAXONOMY)
TAXONOMY_BY_ID = {t["id"]: t for t in TAXONOMY}
AXIS_OF = {t["id"]: t["axis"] for t in TAXONOMY}
AXIS_IDS = {axis: tuple(t["id"] for t in entries)
            for axis, entries in AXES.items()}

# personal_match for an item when no user profile is configured (steering off):
# the blend is neutral, so final == importance.
NEUTRAL_PERSONAL = 0.5


def label_prompt() -> str:
    """System prompt shared by the production labeler and the eval reference
    model. Lists each axis separately so the model picks within one facet at a
    time, and asks for the exact JSON shape the labeler parses."""
    lines = [
        "You label AI news items and research papers for a weekly AI podcast.",
        "",
        "You are given a JSON array of items, each with an integer \"index\", a "
        "\"title\", a \"url\", and a \"body\" (an abstract for arxiv, extracted "
        "page text for hn — it may be long; the head usually suffices).",
        "",
        "For EACH item, pick one PRIMARY type and assign weights to the "
        "relevant TECHNICAL and APPLICATION topics. Use ONLY these ids.",
        "",
        "PRIMARY (pick EXACTLY ONE — the most salient headline type):",
    ]
    for t in PRIMARY:
        lines.append(f"- {t['id']}: {t['description']}")
    lines += ["", "TECHNICAL (0.0-1.0 weights; only include what the item is "
                    "about, weight>0, may be empty):"]
    for t in TECHNICAL:
        lines.append(f"- {t['id']}: {t['description']}")
    lines += ["", "APPLICATION (0.0-1.0 weights; the domain it serves, "
                    "weight>0, may be empty):"]
    for t in APPLICATION:
        lines.append(f"- {t['id']}: {t['description']}")
    lines += [
        "",
        "Rules:",
        "- primary must be exactly one id from the PRIMARY list.",
        "- technical/application weights say how much of the item is about that "
          "topic (1.0 = almost entirely); omit zero-weight topics (sparse).",
        "- A research paper is research_findings (or evaluation for benchmark/"
          "measurement work) — NOT model_release.",
        "- If nothing fits, use \"other\" for primary.",
        "",
        "Return ONLY a JSON object with a single key \"labels\": an array of "
        "objects, one per item in input order, each with \"index\" (the item's "
        "integer index), \"primary\" (a string id), \"technical\" (a JSON "
        "object mapping ids to weights) and \"application\" (a JSON object "
        "mapping ids to weights). Every index must appear exactly once. No "
        "prose before or after. No markdown fences.",
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
