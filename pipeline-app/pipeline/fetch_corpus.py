"""Fetch a raw news corpus (past 4 weeks: arXiv cs.AI + Hacker News), then
stratified-sample ~1000 items for taxonomy derivation.

Deliberately bypasses the pipeline's relevance gates — we want RAW news (what
actually exists out there), not what the pipeline already deemed relevant, so
the derived taxonomy reflects the whole stream.

Usage:
    python -m pipeline.fetch_corpus --out data/eval/corpus__<date>
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import collect

WINDOW_WEEKS = 4
SAMPLE_N = 1000
HN_CAP = 300          # arXiv would otherwise drown the sample
SEED = 42
BODY_CHARS = 2000     # head of HN body only — enough for topic labeling
BODY_WORKERS = 8


def _windows() -> list[tuple[datetime.date, datetime.date]]:
    today = datetime.date.today()
    return [(today - datetime.timedelta(days=7 * (i + 1)) + datetime.timedelta(days=1),
             today - datetime.timedelta(days=7 * i))
            for i in range(WINDOW_WEEKS)]


def _norm_url(item) -> str:
    if getattr(item, "source", "") == "arxiv":
        aid = collect._arxiv_id(item.url)
        if aid:
            return "arxiv:" + aid
    return "url:" + (item.url or "").strip().rstrip("/").lower()


def fetch() -> tuple[list, list]:
    """Return (arxiv_items, hn_items) across the window, deduped."""
    arxiv_seen: set[str] = set()
    hn_seen: set[str] = set()
    arxiv_items: list = []
    hn_items: list = []
    for start, end in _windows():
        print(f"  fetch window {start} .. {end}")
        for page in collect._arxiv_pages(end.isoformat(), start):
            for it in page:
                k = _norm_url(it)
                if k not in arxiv_seen:
                    arxiv_seen.add(k)
                    arxiv_items.append(it)
        for it in collect._hn(end.isoformat(), start):
            k = _norm_url(it)
            if k not in hn_seen:
                hn_seen.add(k)
                hn_items.append(it)
    print(f"  fetched raw: {len(arxiv_items)} arxiv, {len(hn_items)} hn")
    return arxiv_items, hn_items


def _fetch_body(url: str) -> str:
    try:
        return collect._fetch_body(url)[:BODY_CHARS]
    except Exception:
        return ""


def sample(arxiv_items: list, hn_items: list) -> list[dict]:
    """Deterministic stratified sample: all HN (capped) + random arXiv to N."""
    rng = random.Random(SEED)
    hn = list(hn_items)
    rng.shuffle(hn)
    hn = hn[:HN_CAP]
    n_arxiv = SAMPLE_N - len(hn)
    arxiv = list(arxiv_items)
    rng.shuffle(arxiv)
    arxiv = arxiv[:max(0, n_arxiv)]
    return arxiv, hn


def to_corpus(items: list, source: str) -> list[dict]:
    out = []
    for it in items:
        out.append({
            "title": it.title,
            "url": it.url,
            "body": (getattr(it, "body", "") or "")[:BODY_CHARS],
            "source": source,
            "date": getattr(it, "date", ""),
            "hn_points": int(getattr(it, "hn_points", 0) or 0),
        })
    return out


def _fetch_hn_bodies(hn_items: list) -> None:
    if not hn_items:
        return
    with ThreadPoolExecutor(max_workers=BODY_WORKERS) as ex:
        futures = {ex.submit(_fetch_body, it.url): it for it in hn_items}
        done = 0
        for fut in as_completed(futures):
            it = futures[fut]
            try:
                it.body = fut.result() or ""
            except Exception:
                it.body = ""
            done += 1
        print(f"  fetched {done} HN bodies")
    hn_items[:] = [it for it in hn_items if (it.body or "").strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval/corpus__latest")
    args = ap.parse_args()

    arxiv, hn = fetch()
    arxiv_sample, hn_sample = sample(arxiv, hn)
    print(f"  sampling -> {len(arxiv_sample)} arxiv + {len(hn_sample)} hn")
    _fetch_hn_bodies(hn_sample)

    corpus = to_corpus(arxiv_sample, "arxiv") + to_corpus(hn_sample, "hn")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus.json").write_text(json.dumps({
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "windows": [[str(a), str(b)] for a, b in _windows()],
        "seed": SEED,
        "items": corpus,
    }, indent=2), encoding="utf-8")
    print(f"  wrote {out / 'corpus.json'} ({len(corpus)} items)")


if __name__ == "__main__":
    main()
