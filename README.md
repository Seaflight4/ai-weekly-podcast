# AI Weekly Podcast

A small, on-demand pipeline that turns a week of AI research into a podcast
episode. Four stages, each a single Python module with a stable contract, so
any stage can be swapped without touching its neighbours.

```
collect ──> HF papers + Import AI + BensBites + The Batch + Last Week in AI
distil  ──> LLM dedup: collapse same-story items into groups, count consensus
rank    ──> LLM judge scores each group; keep the top 10
generate ─> write a podcast brief (you turn it into audio with Gemini Notebook)
```

## Stages & contracts

| Stage | Input | Output |
|-------|-------|--------|
| `collect` | date (default: today) | `data/collect.json` — title, url, date, body, source (papers: `hf-papers`, RSS newsletters: `rss:import-ai`, `rss:bensbites`, `rss:lwiai`, scraped: `batch`) |
| `distil` | `data/collect.json` | `data/distil.json` — every story group (near-duplicates collapsed), with `consensus` = distinct sources covering it |
| `rank` | `data/distil.json` | `data/rank.json` — top 10 with `score` + `judge_reason` |
| `generate` | `data/rank.json` | `data/podcast_brief.md` + `data/episode.json` manifest |

Each stage writes its output to `data/`, so it can be re-run in isolation from
its predecessor's file. The stage-4 contract is deliberately thin: it returns a
clean Markdown brief plus a manifest. The brief is designed to be dropped into
Gemini Notebook (formerly NotebookLM) → **Audio Overview** → download as
`data/episode.mp3`.

`distil` only *deduplicates* — it never chooses the final list. The pick is
entirely `rank`'s job, so the judge genuinely selects 10 from ~20–30 groups.
The consensus score (how many distinct newsletters/papers covered the story) is
passed to the judge as evidence to weigh, not as a hard rule.

## Requirements

- Python 3.11+
- An OpenAI-compatible API key (used by dedup + the ranking judge)

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e .

cp .env.example .env    # then set SKAINET_API_KEY
```

## Usage

```bash
# run all four stages
.venv/bin/python -m pipeline run

# or re-run a single stage from the previous stage's file
.venv/bin/python -m pipeline run --only collect
.venv/bin/python -m pipeline run --only distil
.venv/bin/python -m pipeline run --only rank
.venv/bin/python -m pipeline run --only generate
```

The final step is intentionally manual: open `data/podcast_brief.md` in
Gemini Notebook, create an Audio Overview, and save the MP3 as
`data/episode.mp3`. Everything before that click is automated.

## Configuration

| Variable | Purpose |
|----------|---------|
| `SKAINET_API_KEY` | API key for the OpenAI-compatible judge backend (required) |

The judge model and base URL are constants in `pipeline/rank.py`.
