# AI Weekly Podcast

A small, on-demand pipeline that turns a week of AI research and community news
into a podcast episode. Three stages, each a single Python module with a stable
contract, so any stage can be swapped without touching its neighbours.

```
collect ──> HN (points>100, 7d, batched LLM relevance gate, per-URL body fetch)
         ─-> arXiv (cs.AI, 7d, full abstracts, LLM relevance gate)
rank    ─-> unified rubric scores the whole pool; no fixed top-N slice
generate ─> top-N items fed to NotebookLM (format/length/tone from profile.md)
```

## Stages & contracts

| Stage | Input | Output |
|-------|-------|--------|
| `collect` | date (default: today) | `<data/DD-MM-YYYY>/collect.json` — every `Item`: title, url, date, body, source (`hn` or `arxiv`) |
| `rank` | collect.json | `<data/DD-MM-YYYY>/rank.json` — the **full** scored pool (sorted desc, no top-N slice) |
| `generate` | rank.json + profile.md | `<data/DD-MM-YYYY>/podcast_brief.md` + `episode.json` manifest (+ `episode.mp3`) |

Each run writes all of its outputs into a folder named `data/DD-MM-YYYY/` (local
run date), so successive runs never overwrite each other. Running a single stage
in isolation reads from the most recent folder's previous-stage file.

**HN prefiltering** happens server-side at `points>100` (community signal), then
a **batched LLM relevance gate** keeps only stories useful to AI researchers at
a company. **arXiv** is now gated too (recall-leaning, on title+abstract) so the
judge sees the plausibly-relevant survivors, not all ~987 cs.AI papers. Points
are used only to *select*, never emitted, so the judge weighs content, not hype.

**arXiv↔HN dedup** is deterministic and O(n): extract the arXiv ID from each HN
URL, hash-lookup against the collected arXiv set, and drop the HN twin (the arXiv
entry already carries the full abstract). No quadratic similarity search.

**Personalization** (`rank`): if a `profile.md` exists and `ALPHA < 1.0`, a
personal-match pass scores every item and
`final_score = ALPHA*score + (1-ALPHA)*personal_score`. Otherwise
`final_score = score` (Phase 1 behaviour). The profile carries both the
personal signal (topics, anti-topics, body) and the narrative style knobs
(format, length, tone, audience level, intro/outro/transition style) that
`generate` templates into the NotebookLM instructions.

**Audio generation** (`generate`): the top-N items (by `final_score`, N from
`profile.md`) are fed to NotebookLM as URL/text sources plus a brief, and one
`generate_audio` call produces the MP3. The `format`/`target_length` profile
fields map directly to NotebookLM's `AudioFormat`/`AudioLength` knobs. NotebookLM
writes the spoken script; the profile steers style without a per-run planner
LLM call.

## Requirements

- Python 3.11+
- An OpenAI-compatible API key (used by the relevance gate + ranking judge)
- `trafilatura` (main-text extraction for HN URL bodies)
- A NotebookLM account (for the audio step; optional with `--no-audio`)

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e .

cp .env.example .env    # then set SKAINET_API_KEY
```

## Usage

```bash
# run all three stages (collect -> rank -> generate)
.venv/bin/python -m pipeline run

# or re-run a single stage from the previous stage's file
.venv/bin/python -m pipeline run --only collect
.venv/bin/python -m pipeline run --only rank
.venv/bin/python -m pipeline run --only generate --no-audio

# re-run only the personal pass on a cached rank.json (fast iteration)
.venv/bin/python -m pipeline run --only rank --from-cache data/21-08-2026/rank.json

# mark an item as kept/skipped for the personal-match feedback loop
.venv/bin/python -m pipeline mark <url> kept|skipped
```

The `--no-audio` flag skips the (cred-gated) NotebookLM automation wrapper and
stops after writing `podcast_brief.md` + `episode.json`. Drop the brief into
NotebookLM manually if you want audio without the automation.

## Profile (`profile.md`)

The listener tunes two things in one file: the **personal signal** (what the
ranker blends into scores) and the **narrative style** (what `generate`
templates into the NotebookLM instructions). Per the no-prompt-chasing rule,
the prompts in the code are fixed; this file is the variable.

```yaml
---
# Personal signal (feeds the ranker's personal-match pass)
topics: [agent evals, inference cost]
anti_topics: [pure scaling]

# Narrative style (feeds the generate stage's NotebookLM instructions)
knowledge_level: researcher     # undergrad | researcher | expert
tone: dense                     # dense | conversational | casual
format: deep_dive               # deep_dive | brief | critique | debate  -> AudioFormat
target_length: default          # short | default | long               -> AudioLength
intro_style: theme-first        # theme-first | biggest-story | bullet
outro_style: links              # links | recap | teaser
transition_style: bridge        # next | bridge | motif
top_n: 20                       # how many ranked items to feed NotebookLM
---
What I'm working on this quarter: agent reliability, evals...
```

Invalid style values fall back to the default (first option listed) with no
error; `top_n` is clamped to [1, 100].

## Validation

```bash
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest tests/ -q     # contract tests (no network, no LLM)
```

## Configuration

| Variable | Purpose |
|----------|---------|
| `SKAINET_API_KEY` | API key for the OpenAI-compatible judge backend (required) |
| `JUDGE_MODEL` | Override the judge/rank model (`Qwen/Qwen3.8-27B` by default) |
| `ALPHA` | Blend weight: `1.0` = pure importance, `0.0` = pure personal (default `0.7`) |

The relevance gate model (`pipeline/collect.py`) and the batch sizes are
constants in their respective modules.
