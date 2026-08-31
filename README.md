# AI Weekly Podcast

A small, on-demand pipeline that turns a week of AI research and community news
into a podcast episode. Three stages, each a single Python module with a stable
contract, so any stage can be swapped without touching its neighbours.

```
collect ──> HN (points>100, 7d, batched LLM relevance gate, per-URL body fetch)
         ─-> arXiv (cs.AI, 7d, full abstracts, LLM relevance gate)
rank    ─-> unified rubric scores the whole pool (pure importance, no personal pass)
generate ─> threshold-selected items fed to the vendored podcastfy audio backend
```

The `generate` stage feeds the selected items to a vendored **podcastfy** stack:
it re-fetches full arXiv PDFs and HN/web pages, generates a two-host transcript
via an OpenAI-compatible LLM (DeepSeek on the SkaiNet gateway — same
`SKAINET_API_KEY`), and synthesizes voice-cloned audio via the TNG qwen3 TTS
service. It also writes a ground-truth `transcript.md`, so the `transcribe`
stage is a no-op.

## Stages & contracts

| Stage | Input | Output |
|-------|-------|--------|
| `collect` | date (default: today) | `<data/default/DD-MM-YYYY>/collect.json` — every `Item`: title, url, date, body, source (`hn` or `arxiv`) |
| `rank` | collect.json | `<data/default/DD-MM-YYYY>/rank.json` — the **full** scored pool (sorted desc, no top-N slice) |
| `generate` | rank.json | `<data/default/DD-MM-YYYY>/podcast_brief.md` + `episode.json` manifest + `episode.mp3` + `transcript.md` |

Each run writes all of its outputs into a folder named
`data/default/DD-MM-YYYY/` (local run date), so successive runs never overwrite
each other. Running a single stage in isolation reads from the most recent
folder's previous-stage file. Personalized renders are written to
`data/personalized/DD-MM-YYYY/`.

**HN prefiltering** happens server-side at `points>100` (community signal), then
a **batched LLM relevance gate** keeps only stories useful to AI researchers at
a company. **arXiv** is now gated too (recall-leaning, on title+abstract) so the
judge sees the plausibly-relevant survivors, not all ~987 cs.AI papers. Points
are used only to *select*, never emitted, so the judge weighs content, not hype.

**arXiv↔HN dedup** is deterministic and O(n): extract the arXiv ID from each HN
URL, hash-lookup against the collected arXiv set, and drop the HN twin (the arXiv
entry already carries the full abstract). No quadratic similarity search.

**Ranking** (`rank`): a unified rubric scores the whole pool on general
importance. There is no personalization pass — every listener gets the same
ranking.

**Source selection** (`generate`): items with `score >= 0.8` are "must
include", with a fixed floor of 10 and cap of 20:
- Weak week (few items ≥ 0.8): pad with the next-highest-scored items down to 10.
- Strong week (many items ≥ 0.8): cut at 20, keeping the top-scored.
- Typical week: all items ≥ 0.8 qualify (within 10–20).

**Audio generation** (`generate`): the selected items are written into a
`podcast_brief.md` and fed to the vendored podcastfy backend. It re-fetches full
arXiv PDFs (10k head + 3k tail) and HN/web pages (22k) into a per-run cache,
generates a two-host transcript via DeepSeek on the SkaiNet gateway (reusing the
existing `SKAINET_API_KEY`), and synthesizes voice-cloned audio via the TNG qwen3
TTS service. It uses podcastfy's tuned defaults for transcript + TTS styling
and writes a ground-truth `transcript.md`, so the `transcribe` stage skips
Whisper. It needs `ffmpeg` on `PATH` (for pydub MP3 encoding).

## Requirements

- Python 3.11+
- An OpenAI-compatible API key (used by the relevance gate + ranking judge, and
  reused by the podcastfy backend's TNG TTS)
- `ffmpeg` on `PATH` (for pydub MP3 encoding)

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[test]"

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
.venv/bin/python -m pipeline run --only rank --from-cache data/default/21-08-2026/rank.json

# regenerate audio from a cached transcript (skip the LLM step)
.venv/bin/python -m pipeline run --only generate --transcript-in data/default/31-08-2026/transcript.md
```

The `--no-audio` flag skips the audio backend and stops after writing
`podcast_brief.md` + `episode.json`. The `--transcript-in` flag reuses a cached
transcript and goes straight to TTS — useful for retrying audio after a
transient TTS outage without paying the multi-minute LLM cost again.

## Local testing (for colleagues)

The fastest way to run the app locally is Docker, which bundles `ffmpeg` and
all Python deps. The service serves a single-page UI on port 8000.

```bash
git clone <repo> && cd learn-ai-podcast-pipeline
cp .env.example .env          # then set SKAINET_API_KEY
docker compose up              # http://localhost:8000
```

What you get on first open:
- **2 default Friday episodes** are committed under `data/default/` and show
  up immediately — no generation needed to see the app in action.
- The **weekly scheduler is off by default** (`SCHEDULE_ENABLED=false` in
  `.env.example`), so your machine won't auto-run a fresh episode every
  Friday. Toggle it on in the UI (top-right switch) or set
  `SCHEDULE_ENABLED=true` in `.env` if you want it.
- **Generate new episode** runs collect → rank → generate for today; the run
  panel shows a **progress bar with ETA** (fixed estimates: collect ~2m,
  rank ~1.5m, generate ~5m) plus the live log tail.
- **Personalize** (button on a default episode) lets you delete items from
  the brief and re-render audio into `data/personalized/<date>/`.

Local data (episodes, personalized renders, job logs) persists in the
`./data` volume mount.

## Validation

```bash
.venv/bin/python -m pytest tests/ -q     # contract tests (no network, no LLM)
```

## Configuration

| Variable | Purpose |
|----------|---------|
| `SKAINET_API_KEY` | API key for the OpenAI-compatible judge backend (required); also reused by the podcastfy backend's TNG TTS |
| `JUDGE_MODEL` | Override the judge/rank model (`Qwen/Qwen3.8-27B` by default) |
| `LLM_API_BASE` | OpenAI-compatible LLM endpoint for the podcastfy transcript LLM |
| `LLM_MODEL` | LLM model name for the podcastfy transcript LLM |
| `SCHEDULE_ENABLED` | `false` to start with the weekly scheduler disabled (default `true`); toggle at runtime via the UI switch |
| `SCHEDULE_CRON` | 5-field cron for the weekly auto-run (default `0 9 * * fri`) |

The relevance gate model (`pipeline/collect.py`) and the batch sizes are
constants in their respective modules.
