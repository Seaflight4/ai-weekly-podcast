# AI Weekly Podcast

A small, on-demand pipeline that turns a week of AI research and community news
into a podcast episode. Three stages, each a single Python module with a stable
contract, so any stage can be swapped without touching its neighbours.

```
collect ──> HN (points>100, 7d, batched LLM relevance gate, per-URL body fetch)
        ─-> arXiv (cs.AI, 7d, full abstracts) — HN→arXiv dedup is free & O(n)
rank    ─-> two judges: arXiv papers → deep dives (5), HN stories → quick briefs (5)
generate ─> write a podcast brief (you turn it into audio with Gemini Notebook)
```

## Stages & contracts

| Stage | Input | Output |
|-------|-------|--------|
| `collect` | date (default: today) | `data/collect.json` — every `Item`: title, url, date, body, source (`hn` or `arxiv`) |
| `rank` | `data/collect.json` | `data/rank.json` — `DEEP_N` deep dives + `BRIEF_N` quick briefs, each with `score` + `judge_reason` + `kind` |
| `generate` | `data/rank.json` | `data/podcast_brief.md` + `data/episode.json` manifest |

Each stage writes its output to `data/`, so it can be re-run in isolation from
its predecessor's file. The stage-3 contract is deliberately thin: it returns a
clean Markdown brief plus a manifest. The brief is designed to be dropped into
Gemini Notebook (formerly NotebookLM) → **Audio Overview** → download as
`data/episode.mp3`.

**HN prefiltering** happens server-side at `points>100` (community signal), then
a **batched LLM relevance gate** keeps only stories useful to AI researchers at
a company. Points/upvotes are used only to *select*, never emitted to the next
stage, so the judge weighs content, not hype.

**arXiv↔HN dedup** is deterministic and O(n): extract the arXiv ID from each HN
URL, hash-lookup against the collected arXiv set, and drop the HN twin (the arXiv
entry already carries the full abstract). No quadratic similarity search.

**Two-track judging**: `rank` never mixes the pools. arXiv papers are judged
against a *research* rubric and become **deep dives** (`DEEP_N`, default 5);
HN stories are judged against a *news* rubric (client signal, concrete
capabilities/pricing/incidents) and become **quick briefs** (`BRIEF_N`, default
5). Each pool is scored in one LLM request per batch (`BATCH_SIZE=25`,
empirically ~3× faster than one per story) and trimmed to its slot count, so
both sources always get airtime regardless of pool sizes (arXiv ≈ 987/wk vs
HN ≈ 60/wk).

## Requirements

- Python 3.11+
- An OpenAI-compatible API key (used by the relevance gate + ranking judge)
- `trafilatura` (main-text extraction for HN URL bodies)

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e .

cp .env.example .env    # then set SKAINET_API_KEY
```

## Usage

```bash
# run all three stages (collect → rank → generate)
.venv/bin/python -m pipeline run

# or re-run a single stage from the previous stage's file
.venv/bin/python -m pipeline run --only collect
.venv/bin/python -m pipeline run --only rank
.venv/bin/python -m pipeline run --only generate --no-audio
```

The final step is intentionally flexible: open `data/podcast_brief.md` in
Gemini Notebook, create an Audio Overview, and save the MP3 as
`data/episode.mp3`. The `--no-audio` flag skips the (optional, cred-gated)
NotebookLM automation wrapper. Everything before that click is automated.

## Configuration

| Variable | Purpose |
|----------|---------|
| `SKAINET_API_KEY` | API key for the OpenAI-compatible judge backend (required) |

The judge model, base URL, `DEEP_N`/`BRIEF_N`, and the batch size are constants
in `pipeline/rank.py`.
