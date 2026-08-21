# AI Weekly Podcast

A small, on-demand pipeline that turns a week of AI research and community news
into a podcast episode. Five stages, each a single Python module with a stable
contract, so any stage can be swapped without touching its neighbours.

```
collect ──> HN (points>100, 7d, batched LLM relevance gate, per-URL body fetch)
        ─-> arXiv (cs.AI, 7d, full abstracts, LLM relevance gate)
rank    ─-> unified rubric scores the whole pool; no fixed top-N slice
cluster ─-> MiniLM embeddings group top-50 into topics; LLM labels + cross-merges
plan    ─-> budget fill (~24-34 min) + narrative outline (hook, motif, signposts)
generate ─> topic-grouped digestible brief + NotebookLM Audio Overview
```

## Stages & contracts

| Stage | Input | Output |
|-------|-------|--------|
| `collect` | date (default: today) | `<data/DD-MM-YYYY>/collect.json` — every `Item`: title, url, date, body, source (`hn` or `arxiv`) |
| `rank` | collect.json | `<data/DD-MM-YYYY>/rank.json` — the **full** scored pool (sorted desc, no top-N slice) |
| `cluster` | rank.json | `<data/DD-MM-YYYY>/cluster.json` — `Topic[]` from the top-50 pool (MiniLM groups + LLM labels/merges) |
| `plan` | cluster.json | `<data/DD-MM-YYYY>/episode_plan.json` — budget fill + narrative outline |
| `generate` | episode_plan.json + cluster.json | `<data/DD-MM-YYYY>/podcast_brief.md` + `episode.json` manifest (+ `episode.mp3`) |

Each run writes all of its outputs into a folder named `data/DD-MM-YYYY/` (local
run date), so successive runs never overwrite each other. Running a single stage
in isolation reads from the most recent folder's previous-stage file.

Each stage also writes a human-reviewable `<stage>_report.md` next to its JSON
(the quality gates: arXiv gate pass-rate, rubric A/B comparison, cluster table,
merge diff, budget fill, narrative outline).

**Two rubric arms** (`RUBRIC_MODE` env, default `unified`): `unified` scores the
whole pool on one source-agnostic scale (major release ≈ must-study paper ≈
0.85); `separated` is the legacy two-track (PAPER_RUBRIC for arxiv,
NEWS_RUBRIC for hn). `--only rank --ab` runs both and writes a comparison
report so you can see which surfaces more community signal without collapsing
paper quality.

**HN prefiltering** happens server-side at `points>100` (community signal), then
a **batched LLM relevance gate** keeps only stories useful to AI researchers at
a company. **arXiv** is now gated too (recall-leaning, on title+abstract) so the
judge sees the plausibly-relevant survivors, not all ~987 cs.AI papers. Points
are used only to *select*, never emitted, so the judge weighs content, not hype.

**arXiv↔HN dedup** is deterministic and O(n): extract the arXiv ID from each HN
URL, hash-lookup against the collected arXiv set, and drop the HN twin (the arXiv
entry already carries the full abstract). No quadratic similarity search.

**Topic clustering** (`cluster`): `all-MiniLM-L6-v2` embeddings group the top-50
scored items into candidate clusters by cosine similarity, then one LLM call
per cluster labels it (title, why-it-matters, primary member) and one
cross-cluster merge call fuses same-event clusters — so a model release and a
paper about it can land in one mixed-source topic. The brief is grouped by
these topics, not by source.

**Dynamic length** (`plan`): the episode length is week-adaptive, not a fixed
top-10. The score distribution's knee caps the cluster pool; a minute-budget
knapsack fills topics to ~28–30 min (clamped to 24–34). A narrative planner then
writes the outline (hook, segment order, signposts, motif, transitions, outro).

## Requirements

- Python 3.11+
- An OpenAI-compatible API key (used by the relevance gate + ranking judge + clustering/plan LLM calls)
- `trafilatura` (main-text extraction for HN URL bodies)
- `sentence-transformers` (MiniLM embeddings for topic clustering)

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e .

cp .env.example .env    # then set SKAINET_API_KEY
```

## Usage

```bash
# run all five stages (collect → rank → cluster → plan → generate)
.venv/bin/python -m pipeline run

# or re-run a single stage from the previous stage's file
.venv/bin/python -m pipeline run --only collect
.venv/bin/python -m pipeline run --only rank
.venv/bin/python -m pipeline run --only cluster
.venv/bin/python -m pipeline run --only plan
.venv/bin/python -m pipeline run --only generate --no-audio

# A/B test the rubric (runs both arms, writes rank_report.md)
.venv/bin/python -m pipeline run --only rank --ab
```

The final step is intentionally flexible: open the run's `podcast_brief.md` in
Gemini Notebook, create an Audio Overview, and save the MP3 as
`episode.mp3` in the run folder. The `--no-audio` flag skips the (optional,
cred-gated) NotebookLM automation wrapper. Everything before that click is
automated.

## Validation

```bash
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest tests/ -q     # contract tests (no network, no LLM)
```

Each stage also writes a `<stage>_report.md` next to its JSON output for
human review (arXiv gate pass-rate, rubric A/B, cluster table, merge diff,
budget fill, narrative outline).

## Configuration

| Variable | Purpose |
|----------|---------|
| `SKAINET_API_KEY` | API key for the OpenAI-compatible judge backend (required) |
| `RUBRIC_MODE` | `unified` (default, one source-agnostic rubric) or `separated` (legacy two-track) |
| `JUDGE_MODEL` | Override the judge/cluster/plan model (`Qwen/Qwen3.8-27B` by default) |

The relevance gate model (`pipeline/collect.py`), the cluster embedding model
(`all-MiniLM-L6-v2`), the similarity threshold, and the budget envelope
(`24–34 min`) are constants in their respective modules.
