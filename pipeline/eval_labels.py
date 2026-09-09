"""Offline eval: the cheap production labeler vs a big reference model.

Labels the same items with two models (identical taxonomy prompt from
``pipeline.topics``) and reports how well the small model agrees with the big
one, so you can decide whether Mistral-Small's labels are good enough for
steering (acceptance: mean per-topic Cohen's kappa ~>= 0.6).

Run manually (no service involvement):

    python -m pipeline.eval_labels \
        --pool data/eval/batches/*.json \
        --small mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
        --big Qwen/Qwen3.8-27B \
        --out data/eval/labels__small__vs__big

Writes ``data/eval/<name>/eval_report.json`` with per-item label-set agreement
(Jaccard/IoU), per-topic Cohen's kappa (big = reference), weight agreement, and
qualitative disagreement samples. Per-model results are cached to disk under
``<out>/`` so re-running with new metrics doesn't re-call the LLMs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import types

from . import label as label_mod
from . import topics


def _to_item(entry: dict) -> types.SimpleNamespace | None:
    title = (entry.get("title") or "").strip()
    if not title:
        return None
    url = (entry.get("url")
           or (f"https://arxiv.org/abs/{entry['arxiv_id']}" if entry.get("arxiv_id") else "")
           or f"title:{title}")
    body = (entry.get("body") or entry.get("abstract")
            or entry.get("summary") or "")
    return types.SimpleNamespace(title=title, url=url, body=body)


def load_pool(paths: list[str]) -> list:
    items: list = []
    seen: set[str] = set()
    for p in paths:
        raw = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            entries: list = raw
        else:
            entries = raw.get("papers") or raw.get("items") or []
        for e in entries:
            it = _to_item(e)
            if it is None:
                continue
            key = topics.normalize_url(it.url)
            if key in seen:
                continue
            seen.add(key)
            items.append(it)
    return items


def _labels_or_cache(items, model: str, cache_path: pathlib.Path) -> dict:
    """Return url->topics, labeling items not already in the disk cache."""
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}
    missing = [it for it in items
               if topics.normalize_url(it.url) not in cache]
    if missing:
        fresh = label_mod.label_items(missing, model=model)
        cache.update(fresh)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache


def _slug(model: str) -> str:
    return model.replace("/", "__").replace(".", "_").replace(":", "_")


def _jaccard(a: dict, b: dict) -> float:
    sa, sb = set(a), set(b)
    if not (sa | sb):
        return 1.0
    return len(sa & sb) / len(sa | sb)


def cohen_kappa(a: list, b: list) -> float:
    """Binary Cohen's kappa between two raters (lists of 0/1)."""
    n = len(a)
    if n == 0:
        return 0.0
    sa = sum(a)
    sb = sum(b)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = (sa / n) * (sb / n) + (1 - sa / n) * (1 - sb / n)
    if pa >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pa) / (1 - pa)


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    d1 = math.sqrt(sum((a - mx) ** 2 for a in x))
    d2 = math.sqrt(sum((b - my) ** 2 for b in y))
    if d1 == 0 or d2 == 0:
        return 0.0
    return num / (d1 * d2)


def _load_env() -> None:
    """Read a local .env (API key) so the script works like `python -m pipeline run`."""
    env = pathlib.Path(".env")
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _axis_stats(items, small, big, axis: str) -> tuple[dict, float, float]:
    """Per-topic and axis-level agreement within one axis.

    Returns (topic_stats, used_kappa, used_match_frac). ``used_kappa`` averages
    only topics the reference model actually marked (prevalence > 0) — rare
    facets otherwise drag down an imbalance-biased kappa.
    """
    topic_stats: dict = {}
    for tid in topics.AXIS_IDS[axis]:
        a = [1 if tid in small.get(topics.normalize_url(it.url), {}) else 0
             for it in items]
        b = [1 if tid in big.get(topics.normalize_url(it.url), {}) else 0
             for it in items]
        prevalence = sum(b) / len(items)
        k = cohen_kappa(a, b) if prevalence > 0 else 0.0
        m = sum(1 for x, y in zip(a, b) if x == y) / len(items)
        topic_stats[tid] = {"match": round(m, 3), "kappa": round(k, 3),
                            "prevalence": round(prevalence, 3)}
    used = [v["kappa"] for v in topic_stats.values() if v["prevalence"] > 0]
    used_match = [v["match"] for v in topic_stats.values() if v["prevalence"] > 0]
    mean_kappa = (sum(used) / len(used)) if used else 0.0
    mean_match = (sum(used_match) / len(used_match)) if used_match else 0.0
    return topic_stats, round(mean_kappa, 3), round(mean_match, 3)


def _primary_id(vector: dict) -> str | None:
    """The single primary id in a flat vector (primary is 1-hot, weight>0)."""
    for tid in topics.AXIS_IDS["primary"]:
        if vector.get(tid, 0.0) > 0:
            return tid
    return None


def main(argv: list[str] | None = None) -> None:
    _load_env()
    ap = argparse.ArgumentParser(description="Compare small vs big topic labeler")
    ap.add_argument("--pool", nargs="+", required=True,
                    help="JSON file(s) of items (papers/items lists or raw list)")
    ap.add_argument("--small", default=label_mod.LABEL_MODEL,
                    help="production labeler model id")
    ap.add_argument("--big", default="Qwen/Qwen3.8-27B",
                    help="reference model id")
    ap.add_argument("--out", required=True, help="output dir (e.g. data/eval/<name>)")
    args = ap.parse_args(argv)

    items = load_pool(args.pool)
    if not items:
        raise SystemExit("no items found in the given pools")

    out = pathlib.Path(args.out)
    small = _labels_or_cache(items, args.small, out / f"{_slug(args.small)}.cache.json")
    big = _labels_or_cache(items, args.big, out / f"{_slug(args.big)}.cache.json")

    # Per-item IoU over the non-zero topic sets + primary exact-match.
    ious = []
    primary_hits = 0
    samples_bad, samples_good = [], []
    for it in items:
        sa = small.get(topics.normalize_url(it.url), {})
        sb = big.get(topics.normalize_url(it.url), {})
        iou = _jaccard(sa, sb)
        ious.append(iou)
        if _primary_id(sa) == _primary_id(sb):
            primary_hits += 1
        row = {"title": it.title[:120], "small": dict(sa), "big": dict(sb)}
        (samples_bad if iou < 0.5 else samples_good).append((iou, row))
    samples_bad.sort(key=lambda t: t[0])
    samples_good.sort(key=lambda t: -t[0])

    # Per-axis agreement (primary is the gate-adjacent single-choice axis;
    # technical is the steering axis we gate adoption on).
    per_axis_metrics = {}
    gate = {
        "primary_exact_match": round(primary_hits / len(items), 3),
    }
    for axis in ("primary", "technical", "application"):
        stats, mean_kappa, mean_match = _axis_stats(items, small, big, axis)
        per_axis_metrics[axis] = {"mean_kappa": mean_kappa,
                                  "mean_match": mean_match,
                                  "topics": stats}
        gate[f"{axis}_kappa"] = mean_kappa
        gate[f"{axis}_match_frac"] = mean_match

    # Weight agreement on topic/item pairs either labeler marked (>0).
    xs, ys = [], []
    for it in items:
        sa = small.get(topics.normalize_url(it.url), {})
        sb = big.get(topics.normalize_url(it.url), {})
        for tid in set(sa) | set(sb):
            xs.append(sa.get(tid, 0.0))
            ys.append(sb.get(tid, 0.0))
    corr = _pearson(xs, ys) if xs else 0.0
    mean_abs_diff = (sum(abs(a - b) for a, b in zip(xs, ys)) / len(xs)) if xs else 0.0

    report = {
        "run_id": out.name,
        "config": {
            "small": args.small,
            "big": args.big,
            "prompt": "topics.label_prompt (builtin, 3-axis)",
            "pools": list(args.pool),
        },
        "n_items": len(items),
        "gate_metrics": gate,
        "aggregated_metrics": {
            "label_set_iou": {"mean": round(sum(ious) / len(ious), 3)},
            "weight_corr": round(corr, 3),
            "weight_mean_abs_diff": round(mean_abs_diff, 3),
        },
        "per_axis_metrics": per_axis_metrics,
        "samples": {
            "worst_5": [{"iou": round(i, 3), **r} for i, r in samples_bad[:5]],
            "best_5": [{"iou": round(i, 3), **r} for i, r in samples_good[:5]],
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"      eval_labels: {len(items)} items, {args.small} vs {args.big}")
    g = report["gate_metrics"]
    print(f"      gate: technical_kappa={g['technical_kappa']} "
          f"primary_exact_match={g['primary_exact_match']} "
          f"application_kappa={g['application_kappa']} "
          f"IoU={report['aggregated_metrics']['label_set_iou']['mean']}")
    print(f"      wrote {out / 'eval_report.json'}")


if __name__ == "__main__":
    main()
