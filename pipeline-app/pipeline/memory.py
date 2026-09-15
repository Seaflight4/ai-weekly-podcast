"""Cross-episode memory: per-episode topic summaries + continuity plumbing.

Design (see the project README / plan):

- Each episode writes ``memory.json`` into its run folder: ``topics[].summary``
  — a compact, dated, per-topic recap of what the episode aired (what was
  covered, what it established, open threads). Built by ONE small LLM call
  over the aired manifest (grouped by the judge's taxonomy label), with a
  deterministic bullet-digest fallback so the pipeline never breaks.
- The NEXT episode's ``generate`` stage builds a ``memory_context`` from prior
  runs whose window-end falls within ``mem_windows × window_days`` of the
  current window-end, matched to the current items' topic labels, and hands it
  to the audio backend. The backend injects a ``=== MEMORY ===`` block into
  the transcript-generation input, and the part instructions let a part
  reference a listed prior item only when there is a direct continuation
  (the same product/model/paper/thread getting a new development or successor
  release; never a same-topic-but-different-story bridge, never a fabricated
  "last episode").

Deleting an episode deletes its folder and therefore its memory; the filesystem
remains the single source of truth (see ``pipeline/store.py``).
"""
from __future__ import annotations

import datetime
import json
import pathlib

from . import llm
from . import store
from . import topics

# The model that composes each episode's topic summaries (one short call per
# episode; temperature 0 for reproducible output).
MEMORY_MODEL = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
# Hard cap on one topic summary (chars). Keeps the injected MEMORY block small
# enough that it never meaningfully grows a part's context.
MAX_SUMMARY_CHARS = 900
# Max prior summaries injected per topic (keep the block bounded); the most
# recent episodes' summaries are kept (see context_for).
MAX_PER_TOPIC = 2

MEMORY_FILE = "memory.json"


# --- summary derivation -------------------------------------------------------

def derive_summary(items, episode_date: str | None = None) -> dict:
    """One per-topic summary for an episode from its aired ``items``.

    Uses a small LLM call (a single, non-realtime request per episode). On any
    failure — malformed JSON, LLM error — falls back to a deterministic,
    judge_reason-based digest so generation never blocks on memory.

    ``items`` is any iterable of objects with ``title``, ``url``, ``topics``
    (dict of taxonomy id -> weight), and ``judge_reason``. Returns
    ``{"episode_date": ..., "topics": [{"topic", "summary"}, ...]}``.
    """
    payload = _payload(items)
    if not payload:
        return {"episode_date": episode_date, "topics": []}
    try:
        obj = _chat(payload) or {}
    except Exception:
        obj = {}
    topics_out = obj.get("topics") if isinstance(obj, dict) else None
    if not isinstance(topics_out, list):
        topics_out = _fallback_digest(payload)
    out: list[dict] = []
    for t in topics_out:
        if not isinstance(t, dict):
            continue
        tid = t.get("topic")
        summary = t.get("summary")
        if not (isinstance(tid, str) and tid in topics.TAXONOMY_BY_ID
                and isinstance(summary, str) and summary.strip()):
            continue
        out.append({"topic": tid, "summary": summary[:MAX_SUMMARY_CHARS]})
    if not out:
        out = _fallback_digest(payload)
    return {"episode_date": episode_date, "topics": out}


def _payload(items) -> list[dict]:
    """The aired items, grouped by their taxonomy labels, in the compact
    shape the summary prompt consumes (titles give the next episode a concrete
    subject to ground its "same story?" judgment on). An item with several
    labels appears under EACH of its topics, so every topic summary sees the
    items that actually belong to it."""
    grouped: dict[str, list[dict]] = {}
    for it in items:
        t = getattr(it, "topics", None) or {}
        tids = [k for k in t if t[k] > 0 and k in topics.TAXONOMY_BY_ID]
        if not tids:
            tids = ["other"]
        date = (getattr(it, "date", "") or "")[:10]
        # Ordered topic ids (dict insertion order preserved from rank, so the
        # first id is the judge's most salient facet). Lets the summary prompt
        # know where a multi-labeled item's full treatment belongs.
        entry = {
            "title": getattr(it, "title", ""),
            "url": getattr(it, "url", ""),
            "date": date,
            "judge_reason": getattr(it, "judge_reason", ""),
            "labels": list(tids),
        }
        for tid in tids:
            grouped.setdefault(tid, []).append(entry)
    return [{"topic": tid, "items": grouped[tid]} for tid in topics.TAXONOMY_IDS
            if tid in grouped]


_PROMPT_TEMPLATE = """You summarized a podcast episode for a weekly AI news podcast so that the NEXT
episode can naturally reference genuine continuations — without inventing any.

INPUT: a JSON array of this episode's aired topics. Each entry is one taxonomy
topic with the items aired under it: [{{"topic":"agents","items":[{{"title", "url",
"date", "judge_reason", "labels"}}]}}].

Items may belong to several topics and then appear under EACH of those topics'
item lists; each entry's "labels" lists its topic ids, most salient first. Write
an item's FULL treatment only under its FIRST (most salient) topic; under its
other topics give just the facet-relevant detail that topic's record needs. A
story is never retold in full more than once.

Write ONE summary per topic. Each summary is 2-4 sentences, factual, naming the
concrete actors/models/stories covered (cite the item titles by name and date,
e.g. a paper or a release). It must capture:
1. WHAT was covered in this episode on this topic (the most reportable items).
2. WHAT was established / concluded (key facts, numbers, claims the audience
   should remember).
3. OPEN THREADS — unresolved questions, expected follow-ups, or tensions the
   next episode might resolve or advance.

Keep each summary under {max_chars} characters. A future episode will treat
this text as the complete record of prior coverage on the topic; make it
self-contained and concrete, never vague.

Return ONLY a JSON object: {{"topics":[{{"topic":"<taxonomy id>","summary":"<text>"}}]}}.
Emit one entry for every topic in the INPUT. No prose, no markdown fences."""


def _chat(payload: list[dict]) -> dict | None:
    system = _PROMPT_TEMPLATE.format(max_chars=MAX_SUMMARY_CHARS)
    raw = llm.chat(json.dumps(payload, ensure_ascii=False), system,
                   model=MEMORY_MODEL, temperature=0)
    return llm.parse_json(raw)


def _fallback_digest(payload: list[dict]) -> list[dict]:
    """Deterministic per-topic digest from the items' judge_reason (used when
    the summary LLM call fails) — keeps an episode's memory usable, never
    blocks generation."""
    import re as _re
    out = []
    for entry in payload:
        tid = entry["topic"]
        points = []
        for it in entry.get("items", [])[:5]:
            reason = _re.sub(r"\s+", " ", (it.get("judge_reason") or "")).strip()
            title = it.get("title", "")
            date = it.get("date", "") or ""
            points.append(f"{title} ({date}): {reason}" if reason
                          else f"{title} ({date})")
        summary = "Covered: " + ("; ".join(points) if points else "no items.")
        out.append({"topic": tid, "summary": summary[:MAX_SUMMARY_CHARS]})
    return out


# --- persistence -------------------------------------------------------------

def write_memory(items, episode_date: str | None = None, date: str | None = None,
                 force: bool = False) -> bool:
    """Derive and write ``memory.json`` for a run. Returns True when written.

    Idempotent: re-running generate for the same run just rewrites the same
    artifact (derive is deterministic at temperature 0, and the fallback is
    deterministic), so ``--only generate`` re-renders are safe.
    """
    root = store.run_dir(date)
    path = root / MEMORY_FILE
    payload = derive_summary(items, episode_date=episode_date)
    payload["window_end"] = episode_date
    payload["generated_at"] = (
        datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def read_episode_date_memory(root: pathlib.Path) -> dict | None:
    """Load a prior run's memory.json (or None) — tolerant of missing/corrupt
    artifacts, so any run folder can be a memory source without crashing."""
    p = root / MEMORY_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _window_end(root: pathlib.Path) -> datetime.date | None:
    """The run's collection-window end, from its stored config.yaml."""
    cfg = root / "config.yaml"
    if not cfg.exists():
        return None
    try:
        import yaml
        raw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        end = (raw.get("window") or {}).get("end")
        if end:
            return datetime.date.fromisoformat(str(end))
    except (OSError, ValueError, TypeError):
        pass
    return None


# --- retention + context -----------------------------------------------------

def _norm(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def current_window_end(items) -> str | None:
    """The current run's window end, taken from the (first) item's date when no
    explicit date is available — enough for the retention cutoff."""
    for it in items:
        d = (getattr(it, "date", "") or "")[:10]
        if d:
            return d
    return None


def context_for(items, mem_windows: int, episode_date: str | None = None,
                window_span_days: int | None = None) -> dict:
    """Prior-episode memory that the current items may build on.

    Returns ``{topic: [ "episode_date: summary", ... ]}`` for topics present
    among the current items AND in an eligible prior run. Eligibility: the
    prior run's window-end is at least one window before the current run's
    window-end (so same-window duplicate episodes never cross-pollenate) and
    within the retention horizon (``mem_windows × window_span``).

    ``window_span_days`` is the current episode's window length (drives the
    retention + same-window cutoff); when None it is approximated from the
    current items' dates (fallback for callers that don't know the window).

    Pure + deterministic: reads only run folders + config.yaml files. Exporting
    it from ``memory.py`` keeps the generate stage and the audio backend thin.
    """
    end = episode_date or current_window_end(items)
    if not end:
        return {}
    try:
        cur_end = datetime.date.fromisoformat(str(end)[:10])
    except ValueError:
        return {}
    if not store.ROOT.exists():
        return {}

    span_days = window_span_days
    if span_days is None:
        span_guess = None
        for it in items:
            d = (getattr(it, "date", "") or "")[:10]
            if not d:
                continue
            try:
                it_d = datetime.date.fromisoformat(d)
                if it_d <= cur_end:
                    span = cur_end - it_d
                    span_guess = max(span_guess or span, span)
            except ValueError:
                continue
        span_days = span_guess.days if span_guess is not None else (
            precedence_default_window())
    span_days = max(1, span_days)
    # The current run's own window is never a memory source for itself; its
    # start (window_start) is one full span back and boundary-exclusive.
    min_age = cur_end - datetime.timedelta(days=span_days - 1)
    # Retention: keep runs whose window-end is at least this far back, i.e.
    # mem_windows completed windows (current + mem_windows-1 previous).
    cutoff = cur_end - datetime.timedelta(days=span_days * mem_windows)

    # Topics the current episode could continue.
    wanted: set[str] = set()
    for it in items:
        t = getattr(it, "topics", None) or {}
        for k in t:
            if t[k] > 0 and k in topics.TAXONOMY_BY_ID:
                wanted.add(k)

    prior: dict[str, list[str]] = {}
    eligible: list[tuple[datetime.date, pathlib.Path]] = []
    for p in store.ROOT.iterdir():
        if not p.is_dir():
            continue
        wend = _window_end(p)
        if wend is None:
            continue
        # Only strictly-older windows are eligible (no same-window echo), and
        # only within the retention horizon.
        if not (wend < min_age):
            continue
        if wend < cutoff:
            continue
        eligible.append((wend, p))
    # Chronological by window-end (folder-name sort is day-major and not date-
    # ordered across months), most recent last.
    for _wend, p in sorted(eligible, key=lambda r: (r[0], r[1].name)):
        mem = read_episode_date_memory(p)
        if not mem:
            continue
        ep_date = mem.get("episode_date") or p.name
        for t in mem.get("topics", []):
            tid = t.get("topic")
            summary = t.get("summary")
            if tid in wanted and isinstance(summary, str) and summary.strip():
                prior.setdefault(tid, []).append(f"{ep_date}: {summary}")
    # Keep the most RECENT episodes' summaries per topic: entries are appended
    # in chronological window-end order, so the tail holds the newest MAX_PER_TOPIC.
    return {tid: entries[-MAX_PER_TOPIC:] for tid, entries in prior.items()}


def precedence_default_window() -> int:
    """Fallback window span (days) when the current items carry no usable
    dates — mirrors the pipeline's DEFAULT_WINDOW_DAYS."""
    from .config import DEFAULT_WINDOW_DAYS
    return int(DEFAULT_WINDOW_DAYS)
