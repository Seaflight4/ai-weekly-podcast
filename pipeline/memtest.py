"""Cross-episode memory inspection harness.

Fabricates prior-episode memories plus a current episode that genuinely
continues two of them (and carries one topic whose prior coverage is unrelated),
then drives a transcript-only generation through the same code path production
uses (``pipeline.memory.context_for`` -> ``SimplePodcastGenerator._memory_section``
-> ``generate_transcript``) and scans the output for cross-episode references.

This is the fast alternative to generating several real episodes: no collect, no
rank judge, no arXiv/web fetching, no TTS. The only real cost is the single
transcript LLM call, which is unavoidable because references only exist in model
output. Fabricated data guarantees genuine continuations exist (unlike recent
real data, which may have none), plus a control topic to confirm the model does
not invent a "last episode".

Run from the repo root (needs SKAINET_API_KEY; loaded from .env like the rest
of the stack):

    python -m pipeline.memtest [--scratch DIR] [--mem-windows N] [--no-llm]

``--no-llm`` assembles the fabricated memory context + input blocks and stops
before the LLM call (deterministic dry run). Everything is written under the
scratch dir; ``data/history`` is never touched.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import sys
import tempfile

from dotenv import find_dotenv, load_dotenv

# Anchors the transcript is EXPECTED to reference back (real continuations) and
# the prior-coverage phrase it must NOT re-reference (the control topic).
POSITIVE_ANCHORS = ("LayerLens", "Aperture")
CONTROL_PRIOR_PHRASE = "EU AI Act"

GENERIC_MEMORY_HINT = re.compile(
    r"\b(earlier|last week|previous episode|last episode|as we (?:just )?"
    r"(?:heard|covered|discussed)|previously|we covered|back in (?:august|"
    r"september))\b",
    re.IGNORECASE,
)


def _load_env() -> None:
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path)


# --- fabricated prior runs (written as real run folders) ---------------------

# Each entry is a real run folder (config.yaml + memory.json) in the scratch
# history tree, so the REAL eligibility logic in pipeline.memory.context_for
# decides what the current episode may reference.
PRIOR_RUNS = [
    {
        "folder": "28-08-2026",
        "window_end": "2026-08-28",
        "episode_date": "2026-08-28",
        "topics": {
            "agents": (
                "LayerLens, the open agent-observability toolkit, shipped its "
                "1.0 release in August 2026 with runtime tracing of tool calls "
                "and a measurements UI. The team said a plugin SDK and a stable "
                "tracing spec were next."
            ),
            "model_release": (
                "Aperture-3B, a small MoE model, launched in late August 2026 "
                "with 12B-class quality on a few billion active parameters and "
                "1.5x faster inference than dense peers."
            ),
        },
    },
    {
        "folder": "01-09-2026",
        "window_end": "2026-09-01",
        "episode_date": "2026-09-01",
        "topics": {
            "policy": (
                "EU AI Act enforcement: the first conformity obligations took "
                "effect and national regulators published staffing plans. Open: "
                "how member states will audit high-risk systems in practice."
            ),
        },
    },
    {
        # Window-end inside the CURRENT window: must be excluded as a source.
        "folder": "05-09-2026",
        "window_end": "2026-09-05",
        "episode_date": "2026-09-05",
        "topics": {
            "agents": (
                "Same-window recap of LayerLens 1.0. Should never be injected "
                "into the current episode (an episode is not its own memory)."
            ),
        },
    },
]


def _write_prior_runs(root: pathlib.Path) -> None:
    import yaml
    for run in PRIOR_RUNS:
        d = root / run["folder"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.yaml").write_text(
            yaml.safe_dump({"window": {"end": run["window_end"]}},
                           sort_keys=False),
            encoding="utf-8")
        (d / "memory.json").write_text(json.dumps({
            "episode_date": run["episode_date"],
            "topics": [{"topic": t, "summary": s}
                       for t, s in run["topics"].items()],
        }, indent=2), encoding="utf-8")


# --- fabricated current episode ----------------------------------------------

def _current_items() -> list:
    from . import RankedItem
    return [
        # POSITIVE (successor): LayerLens 1.0's open thread (plugin SDK) is now
        # shipped -> the transcript is expected to reference the prior episode.
        RankedItem(
            title="LayerLens 1.1 ships the plugin SDK and a stable tracing spec",
            url="https://layerlens.dev/blog/sdk-1dot1",
            date="2026-09-08",
            body=(
                "LayerLens released v1.1, delivering the plugin SDK the team "
                "teased at the 1.0 launch last month: third parties can now "
                "register custom tool tracers, and the tracing spec is frozen "
                "at v1. Two early SDK integrations are already in the wild."
            ),
            source="hn",
            score=0.90,
            judge_reason="Successor release of a previously covered toolkit.",
            topics={"agents": 1.0},
        ),
        # POSITIVE (successor): Aperture-3B gets a tuned variant.
        RankedItem(
            title="Aperture-3B-pro: an RL-tuned variant of the small MoE",
            url="https://aperture.models/aperture-3b-pro",
            date="2026-09-09",
            body=(
                "Aperture released 3B-pro, a post-trained variant of the small "
                "MoE that launched in August, lifting instruction-following and "
                "tool use while keeping its 1.5x inference-speed advantage."
            ),
            source="hn",
            score=0.88,
            judge_reason="New tuned variant of a model covered earlier.",
            topics={"model_release": 1.0},
        ),
        # CONTROL: prior policy memory (EU AI Act) is unrelated to this item ->
        # the transcript must discuss it standalone and NOT invent a reference.
        RankedItem(
            title="Japan court rules some AI-generated art is copyrightable",
            url="https://example.com/japan-ai-copyright",
            date="2026-09-09",
            body=(
                "A Tokyo district court found that AI-generated artwork with "
                "sufficient human curation can qualify for copyright, a first "
                "for the jurisdiction and a signal for global training-data "
                "policy."
            ),
            source="hn",
            score=0.85,
            judge_reason="Novel legal ruling on AI authorship.",
            topics={"policy": 1.0},
        ),
    ]


# --- input assembly ----------------------------------------------------------

def _build_combined_input(episode_date: str, mem_section: str,
                          items: list) -> str:
    """Assemble the transcript input exactly like the production producer:
    the INTRO block, the injected MEMORY block (same markers), one TOPIC block
    per source (excerpt-only, so no network fetch), and the RECAP block."""
    lines = ["=== INTRO ===",
             f"# AI News Digest — {episode_date}",
             "A pipeline-generated digest of this week's AI work.",
             "=== END INTRO ==="]
    if mem_section:
        lines.append("")
        lines.append(mem_section)
    for it in items:
        lines += ["",
                  f"=== TOPIC: {it.title} ===",
                  f"SOURCE: blog · URL: {it.url}",
                  f"EXCERPT: {' '.join(it.body.split())}",
                  "=== END TOPIC ==="]
    lines += ["", "=== RECAP ===", "Topics covered in this episode:"]
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. {it.title}")
    lines.append("=== END RECAP ===")
    return "\n".join(lines)


# --- reference scan ----------------------------------------------------------

def _strip_tags(text: str) -> str:
    return re.sub(r"</?Person\d+>", "", text or "")


def _scan(transcript: str) -> dict:
    lines = (transcript or "").splitlines()
    by_anchor = {
        anchor: [ln.strip() for ln in lines if anchor.lower() in ln.lower()]
        for anchor in POSITIVE_ANCHORS
    }
    control_hits = [
        ln.strip() for ln in lines
        if CONTROL_PRIOR_PHRASE.lower() in ln.lower()]
    generic = [ln.strip() for ln in lines if GENERIC_MEMORY_HINT.search(ln)]
    return {"by_anchor": by_anchor, "control_hits": control_hits,
            "generic": generic}


def _print_scan(scan: dict) -> int:
    """Print the scan and return an exit code (0 = expectations met)."""
    ok = True
    print("\n== cross-episode reference scan ==")
    for anchor in POSITIVE_ANCHORS:
        hits = scan["by_anchor"][anchor]
        status = f"{len(hits)} line(s) naming {anchor!r}"
        print(f"  EXPECTED reference ({anchor}): {status}")
        if not hits:
            ok = False
        for h in hits:
            print(f"    - {h}")
    control = scan["control_hits"]
    status = (f"{len(control)} line(s) re-referencing {CONTROL_PRIOR_PHRASE!r}"
              " (should be 0)")
    print(f"  CONTROL (no reference expected): {status}")
    for h in control:
        print(f"    - {h}")
    if control:
        ok = False
    hints = scan["generic"]
    if hints:
        print("  generic continuation-phrase lines to eyeball:")
        for h in hints:
            print(f"    ~ {h}")
    else:
        print("  generic continuation phrases: none spotted")
    return 0 if ok else 1


# --- main --------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    _load_env()
    ap = argparse.ArgumentParser(
        description="Fabricate cross-episode memory + generate + scan a "
                    "transcript for references.")
    ap.add_argument("--scratch", default=None,
                    help="data root for fabricated runs + outputs "
                         "(default: a fresh temp dir)")
    ap.add_argument("--mem-windows", type=int, default=2,
                    help="retention in episode-windows (1-4)")
    ap.add_argument("--no-llm", action="store_true",
                    help="assemble inputs only; do not call the LLM")
    args = ap.parse_args(argv)

    scratch = (pathlib.Path(args.scratch).resolve()
               if args.scratch else pathlib.Path(
                   tempfile.mkdtemp(prefix="memtest-")))
    scratch.mkdir(parents=True, exist_ok=True)
    root = scratch / "history"
    root.mkdir(parents=True, exist_ok=True)

    # Point the pipeline's store at the scratch tree; data/history untouched.
    from . import store as store_mod
    store_mod.ROOT = root

    _write_prior_runs(root)

    from .memory import context_for
    items = _current_items()
    episode_date = datetime.date.today().isoformat()
    ctx = context_for(items, mem_windows=args.mem_windows,
                      episode_date=episode_date, window_span_days=7)

    print(f"scratch dir: {scratch}")
    print("== memory context selected by the REAL eligibility logic ==")
    if not ctx:
        print("  (empty)")
    for topic in sorted(ctx):
        for entry in ctx[topic]:
            print(f"  {topic}: {entry}")
    # The same-window run must never be its own memory source.
    same_window_leak = any("05-09-2026" in e for e in ctx.get("agents", []))
    print(f"  [same-window self-memory leak: {same_window_leak}]")

    from podcast_engine.generator import SimplePodcastGenerator
    from podcast_engine import EngineConfig
    caps = EngineConfig(length="short", depth="brief")._overrides()
    gen = SimplePodcastGenerator(memory_context=ctx, items=items, **caps)

    mem_section = gen._memory_section()
    combined = _build_combined_input(episode_date, mem_section, items)
    preview = {"episode_date": episode_date,
               "memory_context": ctx,
               "injected_memory_block": mem_section}
    (scratch / "memory_context_preview.json").write_text(
        json.dumps(preview, indent=2), encoding="utf-8")

    print("\n== injected MEMORY block (transcript input) ==")
    print(mem_section or "  (none)")
    print("\n(topic blocks use excerpt-only content; no sources fetched)")

    if args.no_llm:
        print("\ndry run: stopped before the LLM call (--no-llm)")
        print(f"input blocks -> {scratch / 'memory_context_preview.json'}")
        return 0

    if not __import__("os").environ.get("SKAINET_API_KEY"):
        print("\nerror: SKAINET_API_KEY not set (required for the transcript "
              "LLM call); use --no-llm for a dry run", file=sys.stderr)
        return 2

    print("\ngenerating transcript (single short episode, no audio)...")
    transcript = gen.generate_transcript(combined)
    transcript_path = scratch / "memtest_transcript.md"
    transcript_path.write_text(transcript, encoding="utf-8")

    print("\n== generated transcript ==")
    print(_strip_tags(transcript))
    print(f"\ntranscript -> {transcript_path}")
    return _print_scan(_scan(transcript))


if __name__ == "__main__":
    raise SystemExit(main())
