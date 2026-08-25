# AI Weekly Podcast

A small, on-demand pipeline that turns a week of AI research and community news
into a podcast episode. Three stages, each a single Python module with a stable
contract, so any stage can be swapped without touching its neighbours.

```
collect ──> HN (points>100, 7d, batched LLM relevance gate, per-URL body fetch)
         ─-> arXiv (cs.AI, 7d, full abstracts, LLM relevance gate)
rank    ─-> unified rubric scores the whole pool (pure importance, no personal pass)
generate ─> threshold-selected items fed to NotebookLM (format/length/tone from profile.md)
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

**Ranking** (`rank`): a unified rubric scores the whole pool on general
importance. `final_score = score` — there is no personal-match pass. The
profile only controls narrative style (see below).

**Source selection** (`generate`): items with `final_score >= 0.8` are "must
include". Pace sets a floor and cap:
- `deep_dive`: 5–10 items (fewer, each in depth)
- `brief`: 10–20 items (more, each briefly)

On a weak week (few items ≥ 0.8), the floor pads with the next-highest-scored
items. On a strong week (many items ≥ 0.8), the cap cuts at the top. A week
where exactly N items land ≥ 0.8 (within floor/cap) keeps all N.

**Audio generation** (`generate`): the selected items are fed to NotebookLM
as URL/text sources plus a brief, and one `generate_audio` call produces the
MP3. `format`/`target_length` map to NotebookLM's `AudioFormat`/`AudioLength`
knobs. `knowledge_level` and `pace` template into the `instructions` string
(a soft steer; `pace` is being tested against the hard `format` knob).
NotebookLM writes the spoken script; the profile steers style without a
per-run planner LLM call.

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

# re-run rank from a cached rank.json (fast iteration)
.venv/bin/python -m pipeline run --only rank --from-cache data/21-08-2026/rank.json
```

The `--no-audio` flag skips the (cred-gated) NotebookLM automation wrapper and
stops after writing `podcast_brief.md` + `episode.json`. Drop the brief into
NotebookLM manually if you want audio without the automation.

## Profile (`profile.md`)

The listener tunes the **narrative style** — what `generate` templates
into the NotebookLM instructions and how many items are selected. Per the
no-prompt-chasing rule, the prompts in the code are fixed; this file is the
variable.

```yaml
---
# Narrative style (feeds the generate stage's NotebookLM instructions)
knowledge_level: researcher     # undergrad | researcher
pace: deep_dive                 # deep_dive | brief  — fewer topics in depth, or more topics briefly
format: deep_dive               # deep_dive | brief | critique | debate  -> AudioFormat (NotebookLM knob)
target_length: default          # short | default | long               -> AudioLength (NotebookLM knob)
---
```

| Field | Effect |
|-------|--------|
| `knowledge_level` | Soft instruction: `researcher` skips basics (no need to define attention/RLHF); `undergrad` briefly explains jargon first. |
| `pace` | Soft instruction + source-set size: `deep_dive` selects 5–10 items (each in depth); `brief` selects 10–20 items (each briefly). All items ≥ 0.8 are included; floor/cap apply on weak/strong weeks. |
| `format` | Hard NotebookLM knob (`AudioFormat`): `deep_dive`, `brief`, `critique`, or `debate` show format. |
| `target_length` | Hard NotebookLM knob (`AudioLength`): `short`, `default`, or `long`. |

`pace` is a soft instruction tested against the hard `format` knob —
e.g. `pace=brief` asks for breadth while `format=deep_dive` requests the
Deep Dive show format, to see which signal wins.

Invalid style values fall back to the default (first option listed) with no
error.

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

The relevance gate model (`pipeline/collect.py`) and the batch sizes are
constants in their respective modules.
