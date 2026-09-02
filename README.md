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
| `collect` | date (default: today) | `<data/history/DD-MM-YYYY>/collect.json` — every `Item`: title, url, date, body, source (`hn` or `arxiv`) |
| `rank` | collect.json | `<data/history/DD-MM-YYYY>/rank.json` — the **full** scored pool (sorted desc, no top-N slice) |
| `generate` | rank.json | `<data/history/DD-MM-YYYY>/podcast_brief.md` + `episode.json` manifest + `episode.mp3` + `transcript.md` |

Each run writes all of its outputs into a folder named
`data/history/DD-MM-YYYY/` (run anchor date = window end), so successive runs
never overwrite each other. Running a single stage in isolation reads from the
most recent folder's previous-stage file. `data/history/` is the **single
episode namespace**: everything the UI lists, no matter which flow created it.

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

**Source selection** (`generate`): the target count is derived from the run
config's word budget (see below) — e.g. medium + deep-dive ≈ 7 sources,
clamped to 4–30. The top `target_n` items by score above a quality floor
(`score >= 0.5`) air; a weak week is never padded with sub-floor junk. The
resolved config is stored in every run dir as `config.yaml` for reproducibility.

### Episode length & topic depth → duration

Length and depth are two independent knobs; the word budget makes them
conform:

- **Length** sets the *total* spoken-word budget: `length_minutes × WPM`
  (spoken words-per-minute is fixed at **165**).
- **Depth** sets the *per-source* budget: `depth_minutes × WPM`
  (`brief` = 1 min/source, `deep-dive` = 2 min/source).
- Intro + recap take **20%** of the total; source count falls out of the rest:
  `num_sources = (length − 20%) ÷ per-source-minutes`.

| Length | total / intro+recap | Brief (1 min/source) | Deep-dive (2 min/source) |
|---|---|---|---|
| Short 10 min | 1650 / 330 w | 8 × 165 w | 4 × 330 w |
| Medium 17.5 min | 2888 / 578 w | 14 × 165 w | 7 × 330 w |
| Long 30 min | 4950 / 990 w | 24 × 165 w | 12 × 330 w |

Each part is prompted with its word ceiling, given a matching per-call token
cap, and then deterministically trimmed to the last sentence boundary within
budget — so the finished episode lands near the requested length instead of
drifting (the old behaviour produced ~20+ min for a "medium" episode). Topic
depth only scales how much source *text* the LLM sees (`per_paper_chars` ×
`depth_factor`); it never scales the output word budget.

**Audio generation** (`generate`): the selected items are written into a
`podcast_brief.md` and fed to the vendored podcastfy backend. It re-fetches full
arXiv PDFs (10k head + 3k tail) and HN/web pages (22k) into a per-run cache,
generates a two-host transcript via DeepSeek on the SkaiNet gateway (reusing the
existing `SKAINET_API_KEY`), and synthesizes voice-cloned audio via the TNG qwen3
TTS service. Audience level, familiar topics, depth and source count from the
run config shape the transcript prompt and part budget. It needs `ffmpeg` on
`PATH` (for pydub MP3 encoding).

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
# run all three stages (collect -> rank -> generate) for the past week
.venv/bin/python -m pipeline run

# anchor the run at a different window end (window = past 7 days)
.venv/bin/python -m pipeline run --date 2026-08-28

# or re-run a single stage from the previous stage's file
.venv/bin/python -m pipeline run --only collect
.venv/bin/python -m pipeline run --only rank
.venv/bin/python -m pipeline run --only generate --no-audio

# re-run rank from a cached rank.json (fast iteration)
.venv/bin/python -m pipeline run --only rank --from-cache data/history/28-08-2026/rank.json

# regenerate audio from a cached transcript (skip the LLM step)
.venv/bin/python -m pipeline run --only generate --transcript-in data/history/28-08-2026/transcript.md
```

The `--no-audio` flag skips the audio backend and stops after writing
`podcast_brief.md` + `episode.json`. The `--transcript-in` flag reuses a cached
transcript and goes straight to TTS — useful for retrying audio after a
transient TTS outage without paying the multi-minute LLM cost again.

The audio backend may also automatically reuse an existing `transcript.md` in
the run dir, but **only when its stored fingerprint still matches the current
brief, selection and config** (stored under `.podcastfy-cache/transcript.fingerprint`).
A stale transcript — e.g. after a full re-run on the same date with a
different window/length, an edited brief, or changed config — is silently
regenerated instead, so a run's audio always reflects what was actually selected.

## The web app (for colleagues)

The fastest way to run the app locally is Docker, which bundles `ffmpeg` and
all Python deps. The service serves a single-page UI on port 8000.

```bash
git clone <repo> && cd learn-ai-podcast-pipeline
cp .env.example .env          # then set SKAINET_API_KEY
docker compose up              # http://localhost:8000
```

On first open the app shows a **Set up your podcast** dialog, pre-filled with
defaults (past 7 days, Researcher, no familiar topics, Medium 15–20 min,
Deep-dive). Saving writes `data/podcast_config.yaml` — the persistent config
that survives browser sessions and container restarts:

- **For you** (applies to every generation): knowledge level (audience) and
  familiar topics — the LLM tailors explanations to these.
- **Episode shape** (dialog defaults): rolling window in days, podcast length,
  topic depth. These pre-fill the *Generate new episode* dialog and are not
  written back.

Then, all in the browser:

- **Generate new episode** — confirm the pre-filled window/length/depth (with a
  live "derived sources · est. minutes" preview) and the run starts. The run
  panel shows a **progress bar with ETA** (fixed estimates: collect ~2m,
  rank ~1.5m, generate ~5m) plus the live log tail. One run at a time.
- **History** — every generated episode, newest first, with ready/draft badges.
  Click to play the audio and read the **Brief** / **Transcript** tabs.
- **Edit brief** — delete/restore sources from the brief and re-render the
  audio. The episode is **replaced in place** (its `episode.mp3`,
  `transcript.md` and manifest are regenerated; the new-length/depth come from
  that run's stored `config.yaml`, the knowledge level from the current
  persistent config).
- **Delete** — removes an episode from history entirely.
- **Settings** — edits `data/podcast_config.yaml` anytime.

Your episode history, config and job logs all persist locally in the `./data`
volume mount. `data/` is gitignored — history is per-machine, nothing is
committed. There is no scheduled auto-run: episodes are generated explicitly
(from the UI or the CLI above).

## Validation

```bash
.venv/bin/python -m pytest tests/ -q     # contract tests (no network, no LLM)
```

## Configuration

Environment variables (`.env`) only carry the API keys/models:

| Variable | Purpose |
|----------|---------|
| `SKAINET_API_KEY` | API key for the OpenAI-compatible judge backend (required); also reused by the podcastfy backend's TNG TTS |
| `JUDGE_MODEL` | Override the judge/rank model (`Qwen/Qwen3.8-27B` by default) |
| `LLM_API_BASE` | OpenAI-compatible LLM endpoint for the podcastfy transcript LLM |
| `LLM_MODEL` | LLM model name for the podcastfy transcript LLM |

Everything user-facing lives in `data/podcast_config.yaml` (edited via the
UI's setup/Settings dialog): `user.audience`, `user.familiar_topics`,
`podcast.window_days`, `podcast.length`, `podcast.depth`. The relevance gate
model (`pipeline/collect.py`) and the batch sizes are constants in their
respective modules.
