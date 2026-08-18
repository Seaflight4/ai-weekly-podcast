# AI Weekly Podcast

A small, on-demand pipeline that turns a week of AI research into a podcast
episode. Three stages, each a single Python module with a stable contract, so
any stage can be swapped without touching its neighbours.

```
collect ──> 20 papers from Hugging Face Daily Papers
rank    ──> LLM judge scores each; keep the top 5
generate ─> write a podcast brief (you turn it into audio with Gemini Notebook)
```

## Stages & contracts

| Stage | Input | Output |
|-------|-------|--------|
| `collect` | date (default: today) | `data/collect.json` — title, url, date, body, source |
| `rank` | `data/collect.json` | `data/rank.json` — top 5 with `score` + `judge_reason` |
| `generate` | `data/rank.json` | `data/podcast_brief.md` + `data/episode.json` manifest |

Each stage writes its output to `data/`, so it can be re-run in isolation from
its predecessor's file. The stage-3 contract is deliberately thin: it returns a
clean Markdown brief plus a manifest. The brief is designed to be dropped into
Gemini Notebook (formerly NotebookLM) → **Audio Overview** → download as
`data/episode.mp3`.

## Requirements

- Python 3.11+
- An OpenAI-compatible API key (used by the ranking judge)

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e .

cp .env.example .env    # then set SKAINET_API_KEY
```

## Usage

```bash
# run all three stages
.venv/bin/python -m pipeline run

# or re-run a single stage from the previous stage's file
.venv/bin/python -m pipeline run --only collect
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
