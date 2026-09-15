"""Derive the topic taxonomy bottom-up from a news corpus.

This is the re-runnable label-generation function: every few months, when
production content drifts, sample a fresh corpus and re-derive the taxonomy
set (see the ``derive`` function — also exposed as the ``derive_taxonomy``
CLI). Per-item multi-labeling at rating time rides whatever taxonomy is
active (see ``pipeline.topics`` / the rank judge).

Pipeline:
  1. Load a corpus (~1000 sampled items, see pipeline/fetch_corpus.py).
  2. Big-model free-form 1-3 labels per item (the "assign facets" pass), most
     salient first, so multi-facet items feed every facet's cluster.
  3. Consolidate raw labels (with frequencies) into a FINER canonical set
     (data-driven K, ~25-30 buckets): split technical threads while keeping
     event-type buckets coarse (the "pick/merge" pass).
  4. Verify on a held-out split with multi-label metrics: coverage gap, exact
     label-set match and top-label match small-vs-big, mean per-topic Cohen's
     kappa over prevalent topics, a light ablation, and corner-case mappings.

Usage:
    python -m pipeline.fetch_corpus --out data/eval/corpus__x
    python -m pipeline.derive_taxonomy --pool data/eval/corpus__x/corpus.json \
        --out data/eval/taxonomy__derived__x [--apply data/taxonomy.json]

Output (under --out): raw_labels.json (item -> free label list), frequency.json,
canonical_labels.json (the derived set + mapping), taxonomy.runtime.json (the
artifact ``pipeline.topics`` loads — adopted with ``--apply``), report.json
with the go/no-go metrics, and corner/ablation details. Every LLM pass is
cached on disk, so re-running with tweaks doesn't re-pay model calls.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import random
import types
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import llm
from . import label as label_mod
from . import topics  # noqa: F401  (keeps id conventions consistent upstream)
from .eval_labels import cohen_kappa
from . import DATA_ROOT

HOLD_OUT_FRAC = 0.15
SEED = 42
MIN_COUNT_MAP = 3       # raw labels below this count may be left unmapped
ABLATE_N = 4            # ablation depth (top labels by prevalence)
ABLATE_SAMPLE = 60      # max items per ablated label to re-label
GAP_MAX = 0.05          # acceptance: hold-out coverage gap (fraction -> "other")
SET_MATCH_MIN = 0.50    # acceptance: multi-label exact-set match, small vs big
KAPPA_MIN = 0.60        # acceptance: mean per-topic kappa (prevalent), small vs big

CORNER_CASES = [
    {"title": "Claude Fable 5.1 and Claude Mythos 5.1",
     "url": "https://example.com/claude",
     "body": "Anthropic releases Claude Fable 5.1 and Mythos 5.1, its most "
             "advanced coding and knowledge models, with a 25% price cut."},
    {"title": "Formalizing Fermat's Last Theorem",
     "url": "https://example.com/flt",
     "body": "Claude produces the first complete computer-checked proof of "
             "Fermat's Last Theorem in Lean over 11 days."},
    {"title": "Nvidia to acquire Hugging Face",
     "url": "https://example.com/hf",
     "body": "Nvidia agrees to buy the open-source AI platform Hugging Face "
             "for $12.9 billion."},
    {"title": "OpenAI begins rolling out GPT-6 Astra",
     "url": "https://example.com/astra",
     "body": "OpenAI begins rolling out GPT-6 Astra, its new frontier model, "
             "with phased access."},
]


def _load_env() -> None:
    env = pathlib.Path(".env")
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def load_corpus(paths: list[str]) -> list:
    out: list = []
    seen: set[str] = set()
    for p in paths:
        raw = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        entries = raw.get("items") if isinstance(raw, dict) else raw
        for e in entries:
            key = (e.get("url") or f"title:{e.get('title', '')}") \
                .strip().rstrip("/").lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(types.SimpleNamespace(
                title=e.get("title", ""), url=e.get("url", ""),
                body=e.get("body") or e.get("abstract") or "",
                source=e.get("source", "")))
    return out


CONSOLIDATE_TIMEOUT = 180.0  # per-request bound for the big cluster/name calls
CLUSTER_TOP_N = 120          # labels shown to the clustering pass (covers count>=3)



def _chat(payload: list, system: str, model: str, json_mode: bool = True) -> str:
    kwargs = {"temperature": 0.0, "timeout": CONSOLIDATE_TIMEOUT}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = llm.get_client().chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": json.dumps(payload)}],
        **kwargs,
    )
    return (resp.choices[0].message.content or "")


FREE_SYSTEM = (
    "You are building the topic taxonomy for a weekly AI-news podcast for AI "
    "researchers. You are given a JSON array of news items, each with an "
    "integer \"index\", a \"title\", a \"url\", and a \"body\" (may be long; "
    "the head suffices).\n"
    "For EACH item return 1 to 3 RECURRING, broadly-applicable AI-news topic "
    "facets that genuinely apply — every meaningful bucket a researcher would "
    "filter on or steer toward, most salient facet first. A typical refined "
    "pool has ~25-35 such facets; split coarse buckets into finer ones where "
    "the field really distinguishes them — e.g. \"post-training (rlhf)\", "
    "\"post-training (distillation)\", \"evals / benchmarks\", \"agents\", "
    "\"agent tooling / orchestration\", \"models — release\", \"inference / "
    "GPU serving\", \"multimodal\", \"robotics\", \"AI for science\", "
    "\"policy / regulation\", \"incident\", \"business / deals\", "
    "\"interpretability\", \"safety / alignment\", \"world models\", "
    "\"retrieval / RAG\". A research paper gets its technical facet(s); an "
    "event story (launch, incident, deal) gets the event facet. NOT a paper "
    "title and NOT a specific product/model name, unless the release/event "
    "itself is the story. Reuse the same noun phrase for the same kind of "
    "item so raw labels stay consistent across items. 1 id is usually right; "
    "use 2-3 only when the item genuinely spans buckets (e.g. an "
    "agent-training paper is \"post-training (rlhf)\" AND \"agents\"). If "
    "nothing fits, use [\"other\"].\n"
    "Return ONLY JSON: {\"labels\":[{\"index\":0,\"labels\":[\"...\", \"...\"]}, ...]}. "
    "Every index exactly once. No prose, no markdown fences."
)


def _extract(payload: dict, chunk: list) -> dict[str, list[str]]:
    """From a parsed ``{labels:[{index, ...}]}`` response, pull ``{url: [labels]}``.

    Accepts the multi-facet ``labels`` array shape (most salient first) and the
    legacy single ``label`` string for backward-compat caches. Empty entries are
    skipped, so an item the model couldn't label simply stays absent.
    """
    out: dict[str, list[str]] = {}
    for e in payload.get("labels", []):
        try:
            idx = int(e["index"])
        except (TypeError, ValueError, KeyError):
            continue
        if not (0 <= idx < len(chunk)):
            continue
        raw = e.get("labels")
        if raw is None:
            raw = e.get("label")
        if isinstance(raw, str):
            raw = [raw]
        vals = [str(v).strip() for v in (raw or [])
                if isinstance(v, str) and str(v).strip()]
        if vals:
            out[topics.normalize_url(getattr(chunk[idx], "url", ""))] = vals
    return out


def _label_chunk(chunk: list, model: str, system: str) -> dict[str, list[str]]:
    payload = [label_mod.item_payload(it, i) for i, it in enumerate(chunk)]
    for _ in range(3):
        raw = _chat(payload, system, model)
        try:
            obj = llm.parse_json(raw)
            if "labels" in obj:
                return _extract(obj, chunk)
        except ValueError:
            continue
    return {}


def label_all(items: list, model: str, system: str,
              cache_path: pathlib.Path, workers: int = 8) -> dict[str, list[str]]:
    """Label ``items`` with 1-3 facets each -> ``{url: [labels]}``.

    Cached to disk (url -> label list); re-runs skip already-labeled items.
    Legacy single-label cache entries (url -> "raw label") are normalized to
    singleton lists so old caches keep working through a refresh.
    """
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}
    for url, v in list(cache.items()):  # normalize legacy single-label entries
        if isinstance(v, str):
            cache[url] = [v]
    missing = [it for it in items if topics.normalize_url(it.url) not in cache]
    if missing:
        chunks = label_mod._chunk_by_chars(missing)
        print(f"  label: {len(missing)} in {len(chunks)} batches ({model})...", flush=True)
        results: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_label_chunk, c, model, system): ci
                       for ci, c in enumerate(chunks)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        for ci in sorted(results):
            cache.update(results[ci])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print(f"  label: {len(cache)} items cached", flush=True)
    return cache


def consolidate(counts: dict[str, int], model: str) -> list[dict]:
    """Two-step consolidation: cluster the observed raw labels, then name each
    cluster. Returns canonicals [{id,label,definition,covers:[raw_labels]}]."""

    top = dict(sorted(counts.items(), key=lambda t: -t[1])[:CLUSTER_TOP_N])
    table = "\n".join(f"- {k} ({v})" for k, v in top.items())

    cluster_system = (
        "Cluster the raw topic labels below (each with its frequency in a sample "
        "of AI news) into thematic groups. Requirements:\n"
        "- Group labels that mean the same kind of AI-news topic.\n"
        "- Aim for 25-30 groups at a FINER, state-of-the-field granularity: "
          "split technical threads researchers actually distinguish (e.g. "
          "post-training/rlhf vs evals/benchmarks vs agents vs agent tooling), "
          "but do not split one concept into near-identical tiny groups and do "
          "not merge unrelated topics.\n"
        "- Event-type buckets (release, incident, business/deals, policy) stay "
          "coarse; technical facets may be finer.\n"
        "- Each group's \"raw_labels\" must be EXACT strings copied verbatim "
          "from the list below (do not reword them). Rare/noisy labels may be "
          "left ungrouped.\n"
        "Return ONLY JSON: {\"groups\":[{\"raw_labels\":[\"...\"]}, ...]}. "
        "No prose."
    )
    groups = None
    last_raw = ""
    for _attempt in range(5):
        raw = _chat([{"task": "cluster"}], cluster_system + "\n\n" + table, model,
                    json_mode=False)
        last_raw = raw
        try:
            obj = llm.parse_json(raw)
            gs = obj.get("groups")
            if isinstance(gs, dict):            # tolerate {"groups": {...}}
                gs = list(gs.values())
            if isinstance(gs, list) and gs:
                norm: list[dict] = []
                for g in gs:
                    if isinstance(g, dict) and g.get("raw_labels"):
                        norm.append(g)
                    elif isinstance(g, list) and g:
                        norm.append({"raw_labels": list(g)})
                if norm:
                    groups = norm
                    break
        except Exception:
            continue
    if not groups:
        raise SystemExit(f"clustering failed to produce groups.\nraw was:\n{last_raw[:2000]}")

    # Name each cluster (pass the labels + aggregate count so definitions fit).
    name_block = ""
    for i, g in enumerate(groups):
        labs = g["raw_labels"]
        n = sum(counts.get(_normalize(l), 0) for l in labs)
        name_block += f"{i}: {', '.join(labs)} (total {n})\n"
    name_system = (
        "Name these clusters of AI-news topics (one canonical per cluster, same "
        "order). Each canonical: \"id\" (unique snake_case), \"label\" (short "
        "human title), \"definition\" (one line). Use dry, filterable names;\n"
        "merge nothing further. Here are the clusters:\n" + name_block +
        "\nReturn ONLY JSON: {\"canonicals\":[{\"id\":\"...\",\"label\":\"...\","
        "\"definition\":\"...\"}, ...]}. No prose."
    )
    canonicals = None
    last_raw = ""
    for _attempt in range(5):
        raw = _chat([{"task": "name"}], name_system, model, json_mode=False)
        last_raw = raw
        try:
            obj = llm.parse_json(raw)
            cs = obj.get("canonicals")
            if isinstance(cs, list) and len(cs) == len(groups):
                if len({c.get("id") for c in cs}) == len(cs) and \
                   all(c.get("id") for c in cs):
                    canonicals = cs
                    break
        except Exception:
            continue
    if canonicals is None:
        raise SystemExit(f"naming failed to produce canonicals.\nraw was:\n{last_raw[:2000]}")

    out = []
    for c, g in zip(canonicals, groups):
        out.append({**c, "covers": list(g["raw_labels"])})
    return out


def closed_system(canonicals: list[dict]) -> str:
    lines = ["You label AI-news items with 1 to 3 topic ids from this taxonomy "
             "(every facet that actually applies, most salient first):"]
    for c in canonicals:
        lines.append(f"- {c['id']}: {c.get('definition') or c.get('label')}")
    lines += [
        "",
        "Rules: a research paper gets its technical facet(s) (e.g. post_training, "
        "benchmarks) — NOT model_release; an event story (launch, incident, "
        "deal) gets the event topic. 1 id is usually right; use 2-3 only when "
        "the item genuinely spans buckets; each returned id must be one of the "
        "ids above. If nothing fits, use [\"other\"].",
        "",
        "Return ONLY JSON: {\"labels\":[{\"index\":0,\"labels\":[\"id\", ...]}, ...]}. "
        "Every index exactly once. No prose.",
    ]
    return "\n".join(lines)


def _normalize(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def map_raw(label: str, canon_by_label: dict[str, str],
            canonicals_list: list[dict]) -> str:
    """Map one free-form raw label to a canonical id (``"other"`` when it maps
    to nothing). ``canon_by_label`` maps normalized raw label -> canonical id;
    ``canonicals_list`` provides the id-level fallback so an id used verbatim
    (e.g. "other") still maps to itself."""
    k = _normalize(label)
    if k in canon_by_label:
        return canon_by_label[k]
    for lab, cid in canon_by_label.items():  # forgiveness for reworded labels
        if k in lab or lab in k:
            return cid
    for c in canonicals_list:
        if k == _normalize(c.get("id", "")):
            return c["id"]
    return "other"


def map_many(labels: list, canon_by_label: dict[str, str],
             canonicals_list: list[dict]) -> list[str]:
    """Map an item's free-form label list to its canonical ids (deduped, order
    preserved). ``"other"`` results and duplicates are dropped, so an item that
    maps to nothing degrades to an empty list."""
    seen: set[str] = set()
    out: list[str] = []
    for lab in labels:
        cid = map_raw(lab, canon_by_label, canonicals_list)
        if cid == "other" or cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    return out


def write_runtime_artifact(out_dir: str | pathlib.Path,
                           canonicals: list[dict],
                           generated_at: str,
                           apply_to: str | pathlib.Path | None = None) -> pathlib.Path:
    """Write the runtime-loadable taxonomy artifact (``id``/``label``/
    ``description``/``covers`` per canonical) — the shape ``pipeline.topics``
    loads at import (see topics.TAXONOMY_PATH). With ``apply_to`` the refresh
    is adopted there (default data/taxonomy.json) so the pipeline switches
    without code edits. Returns the artifact path.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = {
        "generated_at": generated_at,
        "source_report": f"{out_dir.name}/report.json",
        "taxonomy": [
            {"id": c.get("id"), "label": c.get("label") or c.get("id"),
             "description": c.get("definition", ""),
             "covers": list(c.get("covers") or [])}
            for c in canonicals
        ],
    }
    artifact = out_dir / "taxonomy.runtime.json"
    artifact.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    if apply_to:
        target = pathlib.Path(apply_to)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(artifact.read_text(encoding="utf-8"), encoding="utf-8")
    return artifact


def _real_labels(d: dict, url: str) -> set[str]:
    """Non-'other' label ids for an item from a url-keyed cache.

    The cached maps (``label_all`` output) are keyed by the *normalized* url,
    so lookups must normalize too — a raw-``.get`` silently misses items whose
    cached key differs in case / trailing slash, inflating the coverage gap
    and skewing the verification metrics.
    """
    return {x for x in (d.get(topics.normalize_url(url)) or []) if x and x != "other"}


def _top_label(d: dict, url: str) -> str:
    """Most salient non-'other' label for an item (url-normalized lookup)."""
    for x in (d.get(topics.normalize_url(url)) or []):
        if x and x != "other":
            return x
    return "other"


def derive(corpus_paths: list[str],
           out_dir: str | pathlib.Path,
           big: str | None = None,
           small: str | None = None,
           consolidate_model: str | None = None,
           canonicals: str | None = None,
           apply_to: str | pathlib.Path | None = None,
           hold_out_frac: float = HOLD_OUT_FRAC,
           seed: int = SEED) -> dict:
    """Run the full taxonomy-derivation pipeline and write all its artifacts.

    Returns the report dict (same shape as ``report.json``). Deterministic for
    a given corpus + cached LLM labels. This is the re-runnable, every-few-
    months entry point for refreshing a drifting taxonomy:

        derive(["data/eval/corpus__x/corpus.json"],
               "data/eval/taxonomy__refresh_01",
               apply_to="data/taxonomy.json")

    The runtime artifact (``id``/``label``/``description`` per canonical
    bucket plus ``covers``) is always written to ``<out_dir>/taxonomy.runtime.
    json``; with ``apply_to`` set it is also copied there, so the pipeline
    (``pipeline.topics``) adopts the refresh without code edits. ``big`` /
    ``small`` / ``consolidate_model`` default to the stage's pinned models.
    ``canonicals`` (path to ``{"canonicals": [...]}``) skips free-labeling and
    clustering and only re-verifies that curated set.
    """
    big = big or llm.JUDGE_MODEL
    small = small or label_mod.LABEL_MODEL
    consolidate_model = consolidate_model or llm.JUDGE_MODEL

    items = load_corpus(corpus_paths)
    if len(items) < 200:
        raise SystemExit(f"corpus too small: {len(items)} items")
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_hold = max(1, int(len(shuffled) * hold_out_frac))
    dev, hold = shuffled[:-n_hold], shuffled[-n_hold:]
    print(f"  corpus={len(items)} dev={len(dev)} hold_out={len(hold)}", flush=True)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = lambda m: m.replace("/", "__").replace(".", "_")  # noqa: E731

    # 2. Free-form 1-3 labels over the dev split (big model) — or load a
    #    curated canonical set (with raw cache/frequency from this out dir) for
    #    cheap re-verification without re-deriving/clustering.
    if canonicals:
        data = json.loads(pathlib.Path(canonicals).read_text())
        canonicals_list = data["canonicals"]
        rawp = out / f"raw__{slug(big)}.json"
        raw = json.loads(rawp.read_text()) if rawp.exists() else {}
        freqp = out / "frequency.json"
        counts = json.loads(freqp.read_text()) if freqp.exists() else {}
        if not counts:
            for labels in raw.values():
                vals = labels if isinstance(labels, list) else [labels]
                for lab in vals:
                    k = _normalize(lab)
                    if k:
                        counts[k] = counts.get(k, 0) + 1
        print(f"  loaded {len(canonicals_list)} canonicals from {canonicals} "
              f"(raw={len(raw)} items)", flush=True)
    else:
        raw = label_all(dev, big, FREE_SYSTEM,
                        out / f"raw__{slug(big)}.json")
        counts: dict[str, int] = {}
        for labels in raw.values():
            for lab in labels:
                k = _normalize(lab)
                if k:
                    counts[k] = counts.get(k, 0) + 1
        (out / "frequency.json").write_text(
            json.dumps({k: counts[k] for k in sorted(counts, key=lambda t: -counts[t])},
                       indent=2), encoding="utf-8")

    # 3. Canonical set (data-driven K or curated) + force 'other' fallback.
    if not canonicals:
        canonicals_list = consolidate(counts, consolidate_model)
    ids = [c["id"] for c in canonicals_list]
    if "other" not in ids:
        canonicals_list = canonicals_list + [{
            "id": "other", "label": "Other",
            "definition": "Relevant but fits no named topic.", "covers": []}]
        ids.append("other")

    canon_by_label: dict[str, str] = {}
    for c in canonicals_list:
        for x in c.get("covers", []):
            canon_by_label.setdefault(_normalize(x), c["id"])

    mapped = {url: map_many(labels, canon_by_label, canonicals_list)
              for url, labels in raw.items()}
    prev: dict[str, int] = {}
    for ids_c in mapped.values():
        for cid in ids_c:
            prev[cid] = prev.get(cid, 0) + 1
    covered = sum(1 for ids_c in mapped.values() if ids_c)
    coverage = covered / max(1, len(mapped))
    unmapped = sorted({
        _normalize(l) for labels in raw.values() for l in labels
        if map_raw(l, canon_by_label, canonicals_list) == "other"
        and counts.get(_normalize(l), 0) >= MIN_COUNT_MAP})
    top_raw = sorted(counts.items(), key=lambda t: -t[1])[:30]
    (out / "canonical_labels.json").write_text(json.dumps({
        "canonicals": canonicals_list,
        "derived": {"n_items": len(mapped), "coverage": round(coverage, 4),
                    "prevalence": {k: round(v / len(mapped), 4)
                                   for k, v in sorted(prev.items(), key=lambda t: -t[1])},
                    "unmapped_above_floor": unmapped,
                    "top_raw_labels": dict(top_raw)},
    }, indent=2), encoding="utf-8")

    # 4. Verify on the hold-out with the derived (closed) set, using MULTI-LABEL
    #    metrics: exact label-set match and top-label match small-vs-big, plus
    #    per-topic Cohen's kappa over the derived ids (prevalent-weighted mean).
    setver = abs(hash(",".join(ids))) % 10 ** 9
    big_closed = label_all(hold, big, closed_system(canonicals_list),
                           out / f"closed__{slug(big)}__set{setver}.json")
    small_closed = label_all(hold, small, closed_system(canonicals_list),
                             out / f"closed__{slug(small)}__set{setver}.json")

    def real(d: dict[str, list[str]], url: str) -> set[str]:
        return _real_labels(d, url)

    def top_label(d: dict[str, list[str]], url: str) -> str:
        return _top_label(d, url)

    n = len(hold)
    gaps = [it for it in hold if not real(big_closed, it.url)]
    gap_rate = len(gaps) / n

    set_match = sum(1 for it in hold
                    if real(small_closed, it.url) == real(big_closed, it.url)) / n
    top_match = sum(1 for it in hold
                    if top_label(small_closed, it.url) == top_label(big_closed, it.url)) / n
    per_label: dict[str, dict] = {}
    kappas: list[float] = []
    for cid in ids:
        a = [1 if cid in real(small_closed, it.url) else 0 for it in hold]
        b = [1 if cid in real(big_closed, it.url) else 0 for it in hold]
        prev_c = sum(b) / n
        per_label[cid] = {"kappa": round(cohen_kappa(a, b), 3),
                          "prevalence": round(prev_c, 3)}
        if prev_c > 0:
            kappas.append(cohen_kappa(a, b))
    mean_kappa = (sum(kappas) / len(kappas)) if kappas else 0.0

    # Ablation: top labels by prevalence on the dev mapping; how much of a
    # label's items are UNABLE to express any facet once the label is removed?
    top_ids = sorted(prev, key=lambda k: -prev[k])[:ABLATE_N]
    ablation: dict[str, dict] = {}
    for cid in top_ids:
        if cid == "other":
            continue
        narrowed = [c for c in canonicals_list if c["id"] != cid]
        sample_items = [it for it in hold if cid in real(big_closed, it.url)][:ABLATE_SAMPLE]
        if len(sample_items) < 5:
            continue
        relabeled = label_all(sample_items, big, closed_system(narrowed),
                              out / f"ablate__{cid}__{slug(big)}__set{setver}.json")
        # With multi-label, an item "degrades" when removing the label leaves it
        # with NO remaining facet (only "other") — keeping another facet means
        # the label wasn't load-bearing for that item.
        degrade = sum(1 for it in sample_items if not real(relabeled, it.url))
        ablation[cid] = {"n": len(sample_items),
                         "degrade_to_other": round(degrade / len(sample_items), 3)}

    # Corner cases (must map cleanly).
    corner_items = [types.SimpleNamespace(**c) for c in CORNER_CASES]
    corner_sys = closed_system(canonicals_list)
    corner_out = {}
    for i, it in enumerate(corner_items):
        res = label_all([it], big, corner_sys,
                        out / f"corner{i}__{slug(big)}__set{setver}.json")
        corner_out[it.title] = sorted(real(res, it.url)) or ["other"]

    report = {
        "run_id": out.name,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {"pool": [str(p) for p in corpus_paths], "big": big, "small": small,
                   "n_corpus": len(items), "n_dev": len(dev), "n_hold": n},
        "canonical_ids": ids,
        "coverage_gap": round(gap_rate, 4),
        "set_match_small_vs_big": round(set_match, 4),
        "top_label_match_small_vs_big": round(top_match, 4),
        "mean_kappa_small_vs_big": round(mean_kappa, 4),
        "per_label": per_label,
        "ablation": ablation,
        "corner_cases": corner_out,
        "decision": {
            "coverage_gap_ok": gap_rate <= GAP_MAX,
            "set_match_ok": set_match >= SET_MATCH_MIN,
            "kappa_ok": mean_kappa >= KAPPA_MIN,
            "accept": (gap_rate <= GAP_MAX and set_match >= SET_MATCH_MIN
                       and mean_kappa >= KAPPA_MIN),
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 5. Runtime taxonomy artifact: the canonical ids/labels/descriptions in
    # the shape ``pipeline.topics`` loads (see topics.TAXONOMY_PATH). With
    # ``apply_to`` the refresh is adopted immediately (no code edit).
    artifact = write_runtime_artifact(out, canonicals_list, report["generated_at"],
                                      apply_to=apply_to)
    if apply_to:
        print(f"  adopted runtime taxonomy -> {apply_to} (K={len(ids)})", flush=True)

    print(f"  K={len(ids)} canonicals; dev coverage={coverage:.2%}; "
          f"hold-out gap={gap_rate:.1%} (want <= {GAP_MAX:.0%}); "
          f"small-vs-big set-match={set_match:.0%} (want >= {SET_MATCH_MIN:.0%}); "
          f"mean kappa={mean_kappa:.2f} (want >= {KAPPA_MIN:.2f})")
    print("  prevalence:", {k: round(v, 2) for k, v in
                            sorted(prev.items(), key=lambda t: -t[1])[:12]})
    print("  corner cases:", json.dumps(corner_out), flush=True)
    print(f"  wrote {out / 'report.json'}", flush=True)
    return report


def main() -> None:
    _load_env()
    ap = argparse.ArgumentParser(
        description="Derive (and optionally adopt) the topic taxonomy")
    ap.add_argument("--pool", nargs="+", required=True, help="corpus JSON file(s)")
    ap.add_argument("--out", default=str(DATA_ROOT / "eval/taxonomy__derived__latest"))
    ap.add_argument("--big", default=llm.JUDGE_MODEL,
                    help="model for the free-label pass")
    ap.add_argument("--small", default=label_mod.LABEL_MODEL,
                    help="small model used for the closed-set verification")
    ap.add_argument("--consolidate-model", default=llm.JUDGE_MODEL,
                    help="model used for cluster/name (Qwen handles these large "
                         "JSON calls; the DeepSeek flash model times out on them)")
    ap.add_argument("--canonicals", help="JSON with {\"canonicals\":[...]}: skip "
                    "free-labeling/clustering and verify this fixed set instead "
                    "(reuses raw cache + frequency.json from --out if present)")
    ap.add_argument("--apply", metavar="TARGET", default=None,
                    help="adopt the derived taxonomy by copying taxonomy."
                         "runtime.json here (default data/taxonomy.json)")
    args = ap.parse_args()
    derive(args.pool, args.out, big=args.big, small=args.small,
           consolidate_model=args.consolidate_model,
           canonicals=args.canonicals, apply_to=args.apply)


if __name__ == "__main__":
    main()
