"""Fetch a raw AI news corpus (past 4 weeks: arXiv cs.AI + Hacker News), then
stratified-sample ~1000 items for taxonomy derivation.

arXiv comes in raw from the OAI metadata mirror (cs.AI already); HN is put
through the pipeline's AI-relevance gate so general-news noise (non-AI
stories that happen to be popular) does not shape the derived label pool —
the whole corpus drives the taxonomy, so it should reflect AI-relevant news.

Usage:
    python -m pipeline.fetch_corpus --out data/eval/corpus__<date>
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import collect
from . import arxiv_oai
from . import Item
from . import DATA_ROOT

WINDOW_WEEKS = 4
SAMPLE_N = 1000
HN_CAP = 500          # upper bound on AI-relevant HN stories kept
HN_FRAC = 0.40        # target HN share of the corpus (arXiv fills the rest)
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
    """Return (arxiv_items, hn_items) across the window, deduped.

    arXiv reads the OAI mirror first (the bulk metadata sync — not subject to
    the query API's 429 throttling), falling back to the throttled query API
    only when the mirror is unavailable or holds no records for a window.
    """
    arxiv_seen: set[str] = set()
    hn_seen: set[str] = set()
    arxiv_items: list = []
    hn_items: list = []
    for start, end in _windows():
        print(f"  fetch window {start} .. {end}")
        mirror = _arxiv_from_mirror(start, end)
        if mirror:
            print(f"  arxiv: {len(mirror)} raw items from OAI mirror")
            for it in mirror:
                k = _norm_url(it)
                if k not in arxiv_seen:
                    arxiv_seen.add(k)
                    arxiv_items.append(it)
        else:
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


def _arxiv_from_mirror(start: datetime.date, end: datetime.date) -> list:
    """Raw, ungated cs.AI items for a window from the OAI mirror.

    ``[]`` when the mirror is unavailable or holds no records for the window,
    so the caller falls back to the throttled query API.
    """
    try:
        arxiv_oai.ensure_coverage(
            start - datetime.timedelta(days=arxiv_oai.COVERAGE_BUFFER_DAYS))
        records = arxiv_oai.records_for_window(start, end)
    except Exception:
        return []
    return [
        Item(title=r["title"], url=f"https://arxiv.org/abs/{r['id']}",
             date=(r.get("first_submitted") or "")[:10],
             body=r.get("abstract") or "", source="arxiv")
        for r in records if r.get("abstract")
    ]


def _fetch_body(url: str) -> str:
    try:
        return collect._fetch_body(url)[:BODY_CHARS]
    except Exception:
        return ""


def sample(arxiv_items: list, hn_items: list) -> tuple[list, list]:
    """Deterministic stratified sample targeting an ``HN_FRAC`` HN share.

    AI-relevant HN stories are the scarcer stream (post-gate), so the corpus
    total is sized so they land at ~``HN_FRAC`` of it (up to ``SAMPLE_N``) and
    arXiv fills the remainder — a 40/60 HN:arXiv corpus by default.
    """
    rng = random.Random(SEED)
    hn = list(hn_items)
    rng.shuffle(hn)
    hn = hn[:HN_CAP]
    n_total = min(SAMPLE_N, int(round(len(hn) / HN_FRAC)))
    n_arxiv = max(0, n_total - len(hn))
    arxiv = list(arxiv_items)
    rng.shuffle(arxiv)
    arxiv = arxiv[:n_arxiv]
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


def main() -> None:
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_ROOT / "eval/corpus__latest"))
    args = ap.parse_args()

    arxiv, hn = fetch()
    # AI-filter the FULL raw HN set before sampling so general-news noise
    # (bread recipes, unrelated tech) never shapes the derived taxonomy, and
    # so the 40/60 sample split survives the gate. arXiv is already cs.AI-only.
    hn_gated = collect._hn_relevant(hn)
    print(f"  hn relevance gate -> {len(hn_gated)} AI-relevant of {len(hn)}")
    arxiv_sample, hn_sample = sample(arxiv, hn_gated)
    print(f"  sampling -> {len(arxiv_sample)} arxiv + {len(hn_sample)} hn "
          f"({len(hn_sample) / max(1, len(arxiv_sample) + len(hn_sample)):.0%} HN)")
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
