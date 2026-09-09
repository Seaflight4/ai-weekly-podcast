"""Derive a single-label topic taxonomy bottom-up from a news corpus.

Pipeline:
  1. Load a corpus (~1000 sampled items, see pipeline/fetch_corpus.py).
  2. Big-model free-form single label per item (the "assign one label" pass).
  3. Consolidate raw labels (with frequencies) into canonical labels, data-driven
     K, splitting generic buckets (the "pick/merge" pass).
  4. Verify on a held-out split: coverage gap, small-vs-big agreement on the
     derived set, prevalence/balance, a light ablation, and corner-case mappings.

Usage:
    python -m pipeline.fetch_corpus --out data/eval/corpus__x
    python -m pipeline.derive_taxonomy --pool data/eval/corpus__x/corpus.json \
        --out data/eval/taxonomy__derived__x

Output (under --out): raw_labels.json (item -> free label), frequency.json,
canonical_labels.json (the derived set + mapping), report.json with the go/no-go
metrics, and corner/ablation details. Every LLM pass is cached on disk, so
re-running with tweaks doesn't re-pay model calls.
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

HOLD_OUT_FRAC = 0.15
SEED = 42
MIN_COUNT_MAP = 3       # raw labels below this count may be left unmapped
ABLATE_N = 4            # ablation depth (top labels by prevalence)
ABLATE_SAMPLE = 60      # max items per ablated label to re-label
GAP_MAX = 0.05          # acceptance: hold-out coverage gap
AGREE_MIN = 0.60        # acceptance: small-vs-big top-1 agreement

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
    "For EACH item return EXACTLY ONE coarse topic label — one of the RECURRING, "
    "broadly-applicable AI-news buckets a researcher would filter on (a typical "
    "set has ~15-25 such buckets, e.g. \"model release\", \"business\", "
    "\"policy\", \"post-training\", \"agents\", \"safety/alignment\", "
    "\"benchmarks\", \"inference\", \"multimodal\", \"robotics\", \"AI for "
    "science\", \"incident\"). NOT a paper title and NOT a specific "
    "product/model name. Reuse the same noun phrase for the same kind of item. "
    "If none fits, use \"other\".\n"
    "Return ONLY JSON: {\"labels\":[{\"index\":0,\"label\":\"...\"}, ...]}. "
    "Every index exactly once. No prose, no markdown fences."
)


def _extract(payload: dict, chunk: list) -> dict[str, str]:
    """From a parsed {labels:[{index, ...}]} response, pull {url: text}."""
    out: dict[str, str] = {}
    for e in payload.get("labels", []):
        try:
            idx = int(e["index"])
        except (TypeError, ValueError, KeyError):
            continue
        if 0 <= idx < len(chunk):
            val = str(e.get("label", "") or "").strip()
            if val:
                out[topics.normalize_url(getattr(chunk[idx], "url", ""))] = val
    return out


def _label_chunk(chunk: list, model: str, system: str) -> dict[str, str]:
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
              cache_path: pathlib.Path, workers: int = 8) -> dict[str, str]:
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}
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
        "- Aim for 14-20 groups of roughly state-of-the-field granularity. Do "
          "not split a concept into many tiny groups, do not merge unrelated "
          "topics.\n"
        "- Split any group that would cover more than ~40% of the sample into "
          "more specific groups.\n"
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
    lines = ["You label AI-news items with EXACTLY ONE topic from this taxonomy:"]
    for c in canonicals:
        lines.append(f"- {c['id']}: {c.get('definition') or c.get('label')}")
    lines += [
        "",
        "Rules: a research paper gets its technical area (e.g. post-training, "
        "benchmarks) — NOT model_release; an event story (launch, incident, "
        "deal) gets the event topic. Pick the single most important label; if "
        "nothing fits, use \"other\".",
        "",
        "Return ONLY JSON: {\"labels\":[{\"index\":0,\"label\":\"id\"}, ...]}. "
        "Every index exactly once. No prose.",
    ]
    return "\n".join(lines)


def _normalize(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def main() -> None:
    _load_env()
    ap = argparse.ArgumentParser(description="Derive a single-label taxonomy")
    ap.add_argument("--pool", nargs="+", required=True, help="corpus JSON file(s)")
    ap.add_argument("--out", default="data/eval/taxonomy__derived__latest")
    ap.add_argument("--big", default=llm.JUDGE_MODEL)
    ap.add_argument("--small", default=label_mod.LABEL_MODEL)
    ap.add_argument("--consolidate-model", default=llm.JUDGE_MODEL,
                    help="model used for cluster/name (Qwen handles these large "
                         "JSON calls; the DeepSeek flash model times out on them)")
    ap.add_argument("--canonicals", help="JSON with {\"canonicals\":[...]}: skip "
                    "free-labeling/clustering and verify this fixed set instead "
                    "(reuses raw cache + frequency.json from --out if present)")
    args = ap.parse_args()

    items = load_corpus(args.pool)
    if len(items) < 200:
        raise SystemExit(f"corpus too small: {len(items)} items")
    rng = random.Random(SEED)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_hold = max(1, int(len(shuffled) * HOLD_OUT_FRAC))
    dev, hold = shuffled[:-n_hold], shuffled[-n_hold:]
    print(f"  corpus={len(items)} dev={len(dev)} hold_out={len(hold)}", flush=True)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = lambda m: m.replace("/", "__").replace(".", "_")  # noqa: E731

    # 2. Free-form single label over the dev split (big model) — or load a
    #    curated canonical set (with raw cache/frequency from this out dir) for
    #    cheap re-verification without re-deriving/clustering.
    if args.canonicals:
        data = json.loads(pathlib.Path(args.canonicals).read_text())
        canonicals = data["canonicals"]
        rawp = out / f"raw__{slug(args.big)}.json"
        raw = json.loads(rawp.read_text()) if rawp.exists() else {}
        freqp = out / "frequency.json"
        counts = json.loads(freqp.read_text()) if freqp.exists() else {}
        if not counts:
            for lab in raw.values():
                k = _normalize(lab)
                if k:
                    counts[k] = counts.get(k, 0) + 1
        print(f"  loaded {len(canonicals)} canonicals from {args.canonicals} "
              f"(raw={len(raw)} labels)", flush=True)
    else:
        raw = label_all(dev, args.big, FREE_SYSTEM,
                        out / f"raw__{slug(args.big)}.json")
        counts: dict[str, int] = {}
        for lab in raw.values():
            k = _normalize(lab)
            if k:
                counts[k] = counts.get(k, 0) + 1
        (out / "frequency.json").write_text(
            json.dumps({k: counts[k] for k in sorted(counts, key=lambda t: -counts[t])},
                       indent=2), encoding="utf-8")

    # 3. Canonical set (data-driven K or curated) + force 'other' fallback.
    if not args.canonicals:
        canonicals = consolidate(counts, args.consolidate_model)
    ids = [c["id"] for c in canonicals]
    if "other" not in ids:
        canonicals = canonicals + [{
            "id": "other", "label": "Other",
            "definition": "Relevant but fits no named topic.", "covers": []}]
        ids.append("other")

    canon_by_label: dict[str, str] = {}
    for c in canonicals:
        for x in c.get("covers", []):
            canon_by_label.setdefault(_normalize(x), c["id"])

    def map_raw(label: str) -> str:
        k = _normalize(label)
        if k in canon_by_label:
            return canon_by_label[k]
        for lab, cid in canon_by_label.items():  # forgiveness for reworded labels
            if k in lab or lab in k:
                return cid
        for c in canonicals:
            if k == _normalize(c.get("id", "")):
                return c["id"]
        return "other"

    mapped = {url: map_raw(lab) for url, lab in raw.items()}
    prev: dict[str, int] = {}
    for cid in mapped.values():
        prev[cid] = prev.get(cid, 0) + 1
    covered = sum(v for k, v in prev.items() if k != "other")
    coverage = covered / max(1, len(mapped))
    unmapped = sorted({_normalize(l) for l in raw.values() if map_raw(l) == "other"
                       and counts.get(_normalize(l), 0) >= MIN_COUNT_MAP})
    top_raw = sorted(counts.items(), key=lambda t: -t[1])[:30]
    (out / "canonical_labels.json").write_text(json.dumps({
        "canonicals": canonicals,
        "derived": {"n_items": len(mapped), "coverage": round(coverage, 4),
                    "prevalence": {k: round(v / len(mapped), 4)
                                   for k, v in sorted(prev.items(), key=lambda t: -t[1])},
                    "unmapped_above_floor": unmapped,
                    "top_raw_labels": dict(top_raw)},
    }, indent=2), encoding="utf-8")

    # 4. Verify on the hold-out with the derived (closed) set.
    setver = abs(hash(",".join(ids))) % 10 ** 9
    big_closed = label_all(hold, args.big, closed_system(canonicals),
                           out / f"closed__{slug(args.big)}__set{setver}.json")
    small_closed = label_all(hold, args.small, closed_system(canonicals),
                             out / f"closed__{slug(args.small)}__set{setver}.json")

    def hit(d: dict[str, str], url: str) -> str:
        return (d.get(url) or "other").strip() or "other"

    n = len(hold)
    gaps = [it for it in hold if hit(big_closed, it.url) == "other"]
    gap_rate = len(gaps) / n

    agree = sum(1 for it in hold if hit(small_closed, it.url) == hit(big_closed, it.url))
    top1 = agree / n
    per_label: dict[str, dict] = {}
    for cid in ids:
        a = [1 if hit(small_closed, it.url) == cid else 0 for it in hold]
        b = [1 if hit(big_closed, it.url) == cid else 0 for it in hold]
        per_label[cid] = {"kappa": round(cohen_kappa(a, b), 3),
                          "prevalence": round(sum(b) / n, 3)}

    # Ablation: top labels by prevalence on the dev mapping; how much of a
    # label's items degrade to 'other' once the label is removed?
    top_ids = sorted(prev, key=lambda k: -prev[k])[:ABLATE_N]
    ablation: dict[str, dict] = {}
    for cid in top_ids:
        if cid == "other":
            continue
        narrowed = [c for c in canonicals if c["id"] != cid]
        sample_items = [it for it in hold if hit(big_closed, it.url) == cid][:ABLATE_SAMPLE]
        if len(sample_items) < 5:
            continue
        relabeled = label_all(sample_items, args.big, closed_system(narrowed),
                              out / f"ablate__{cid}__{slug(args.big)}__set{setver}.json")
        degrade = sum(1 for it in sample_items
                      if (relabeled.get(it.url) or "other").strip() == "other")
        ablation[cid] = {"n": len(sample_items),
                         "degrade_to_other": round(degrade / len(sample_items), 3)}

    # Corner cases (must map cleanly).
    corner_items = [types.SimpleNamespace(**c) for c in CORNER_CASES]
    corner_sys = closed_system(canonicals)
    corner_out = {}
    for i, it in enumerate(corner_items):
        res = label_all([it], args.big, corner_sys,
                        out / f"corner{i}__{slug(args.big)}__set{setver}.json")
        corner_out[it.title] = res.get(it.url) or "other"

    report = {
        "run_id": out.name,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {"pool": list(args.pool), "big": args.big, "small": args.small,
                   "n_corpus": len(items), "n_dev": len(dev), "n_hold": n},
        "canonical_ids": ids,
        "coverage_gap": round(gap_rate, 4),
        "agreement_top1_small_vs_big": round(top1, 4),
        "per_label": per_label,
        "ablation": ablation,
        "corner_cases": corner_out,
        "decision": {
            "coverage_gap_ok": gap_rate <= GAP_MAX,
            "agreement_ok": top1 >= AGREE_MIN,
            "accept": gap_rate <= GAP_MAX and top1 >= AGREE_MIN,
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"  K={len(ids)} canonicals; dev coverage={coverage:.2%}; "
          f"hold-out gap={gap_rate:.1%} (want <= {GAP_MAX:.0%}); "
          f"small-vs-big top-1={top1:.0%} (want >= {AGREE_MIN:.0%})")
    print("  prevalence:", {k: round(v, 2) for k, v in
                            sorted(prev.items(), key=lambda t: -t[1])[:12]})
    print("  corner cases:", json.dumps(corner_out), flush=True)
    print(f"  wrote {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
