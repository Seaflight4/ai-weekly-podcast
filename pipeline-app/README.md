# AI Weekly Podcast (pipeline-app)

The on-demand podcast pipeline: it turns the past week of AI research and
community news into a podcast episode. Everything runs in your browser:
generate an episode, play it, and read or edit its content. Uses the shared
`podcast_engine` library at `../podcast-engine`.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (with Compose; Desktop/colima
  all work)
- A SkaiNet (OpenAI-compatible) API key for LLM calls and audio synthesis

## Quick start

```bash
cd pipeline-app
cp .env.example .env          # then set SKAINET_API_KEY
docker compose up --build     # then open http://localhost:8000 in your browser
```

To update the project later, pull the new code and rebuild:

```bash
git pull
docker compose up --build
```

Always use `--build`: a plain `docker compose up` reuses the last-built image,
so after pulling new code you would keep running the old version. With `--build`
Docker rebuilds only the changed layers (fast when dependencies are unchanged).

Your episodes, settings and job logs persist locally in the `../data` folder
(gitignored, so nothing is shared or committed).

## Using the app

On first open, the **Set up your podcast** dialog appears:

- **Audience** (knowledge level) and **familiar topics** — the episode is
  tailored to these.
- **Topics you're interested in** and **steering strength (α)** — interests
  steer which items make it into each episode (see *Steering by topic* below).
- **Episode shape** — rolling window in days, podcast length, topic depth per
  source, and the cross-episode memory lookback.

Save to keep these settings (they can be changed anytime under **Settings**).

Then, all in the browser:

- **Generate new episode** — confirm the window, length and depth, and the run
  starts. The panel tracks progress and shows the live log; one run at a time.
  Generating twice in one day keeps **both** episodes — history is
  time-stamped, so a new episode never overwrites an earlier one.
- **History** — every episode, newest first, with ready/draft badges. Click one
  to play the audio and read the **Brief**, **Transcript** and **Memory** tabs.
  Per-item topic labels appear **in the brief only** (each bullet carries them
  as `· topic_a,topic_b`), keeping the history list and hero clean.
- **Edit brief** — remove or restore sources, then re-render the audio. The
  episode is replaced in place with your selection.
- **Delete** — removes an episode from history entirely.

## Assigning topics: multi-label + steering

Every candidate item gets **1–3 coarse topic labels** (a data-derived ~20-bucket
taxonomy: agents, post_training, multimodal, business_economics, ai_for_science,
…) assigned by the rank judge in the SAME LLM pass that scores importance —
score, reason and labels share the item's full body, so there is no separate
labeler with a narrower context. All applied labels are **equal weight**
(`{id: 1.0}` for each), most salient first.

When you set **topics you're interested in** in Settings, each item's
importance score is blended with the cosine similarity of its label vector and
your profile — `final = (1-α)·importance + α·cosine(item, profile)` — and the
top sources are chosen from that blended order. α = 0 turns steering off
(default selection preserved). Because items can carry several labels, a paper
that is both `post_training` and `agents` matches a profile interested in
either. Labels on the aired episode also power the history topic data in each
run's `labels.json`.

Where labels live:

- `rank.json` — every item's multi-label vector (as judged).
- `labels.json` — the episode-level topic vector aggregated from the aired
  items (used for filtering and history).
- `podcast_brief.md` — per-item `· topic_a,topic_b` tokens, which stay visible
  in the Brief tab.

Episodes aired before judging carried labels can be backfilled with the cheap
offline labeler (also writes/annotates their `labels.json` and brief):

```bash
python -m pipeline run --only label             # all runs
python -m pipeline run --only label --date YYYY-MM-DD --force-label   # force re-label
```

The production labeler is validated against a big reference model with
`python -m pipeline.eval_labels --pool <items.json> --out data/eval/<name>`
(multi-label: exact label-SET match, top-label match, mean per-topic Cohen's
kappa, mean label-set IoU).

## Refreshing the taxonomy (re-runnable label generation)

The ~20-bucket taxonomy is derived from the news itself — a big model
free-labels a sampled 4-week corpus, the raw labels are clustered into the
canonical buckets, and the set is verified on a held-out split. Because content
drifts, this is packaged as a **function + CLI you can re-run every few months**
to adapt the label pool:

```bash
# 1. Sample a fresh corpus (last 4 weeks: arXiv cs.AI + Hacker News)
python -m pipeline.fetch_corpus --out data/eval/corpus__<date>

# 2. Derive + verify the new taxonomy; see the go/no-go report
python -m pipeline.derive_taxonomy --pool data/eval/corpus__<date>/corpus.json \
    --out data/eval/taxonomy__<date>

# 3. If report.json says accept, adopt it (no code edits):
python -m pipeline.derive_taxonomy --pool data/eval/corpus__<date>/corpus.json \
    --out data/eval/taxonomy__<date> --apply data/taxonomy.json
```

What you get under `--out`:

- `raw_labels.json` / `frequency.json` — the free-label pass.
- `canonical_labels.json` — the derived set + mapping.
- `taxonomy.runtime.json` — the artifact the pipeline loads.
- `report.json` — acceptance gates: hold-out **coverage gap** (`<= 5%` desired),
  **small-vs-big top-1 agreement** (`>= 0.6` desired), per-label prevalence +
  kappa, a light ablation, and corner-case mappings (`decision.accept`).

The active taxonomy is read by `pipeline/topics.py` at import from
`data/taxonomy.json` when present (overridable via `TAXONOMY_PATH`); otherwise
the curated built-in set is used. The frontend gets it from `GET /api/taxonomy`,
so a swap reaches the Settings checkboxes automatically. A degenerate or
malformed artifact is ignored (the pipeline falls back to the curated set).

You can also verify a **fixed candidate set** without re-deriving ([--canonicals]),
and the derivation can be driven programmatically:

```python
from pipeline import derive_taxonomy as dt
report = dt.derive(["data/eval/corpus__x/corpus.json"],
                   "data/eval/taxonomy__refresh_01", apply_to="data/taxonomy.json")
```

## Cross-episode memory

Each episode keeps a small per-topic memory summary (`memory.json`) of what it
covered and the open threads, written when the episode is generated. The next
episode may reference it back — but only for **direct continuations** (the
same product/model/paper/news thread with a new development or successor
release), never a same-topic-but-different-story bridge and never a fabricated
"last episode".

- **Retention is configurable** via "Episode memory lookback" (1–4 windows,
  default 2 ≈ the last 2 episodes / ~2 weeks for a 7-day window): `--mem-windows`
  on the CLI, `podcast.mem_windows` in `config.yaml`, or the Settings /
  "Generate new episode" dialogs. The current window is never its own memory
  source.
- **References are strict-only**: the transcript may acknowledge a prior item
  only when it is a genuine continuation of the exact same story.
- **Injection**: prior summaries matched to this episode's topics are fed to
  the transcript LLM as a brief "prior coverage sync" part, plus a grounded
  reference rule for each topic part. Items are also ordered topically so
  related topics sit adjacent (enabling "as we just heard with…" call-backs
  within the same episode).
- Your editable `podcast_brief.md` never contains memory internals — the
  **Memory** tab in History shows each episode's summary.
- Backfilled/offline: an episode generated before this feature has no
  `memory.json`; it simply isn't a memory source until regenerated.

## Source fault-tolerance

The collect stage runs each source branch in parallel and is per-source
resilient: a branch that fails (e.g. arXiv throttling with HTTP 429) is
dropped with a warning and the run continues with the surviving sources — the
episode then simply omits that source's content (e.g. an HN-only episode while
arXiv is rate-limited). arXiv 429s are retried with a short backoff (~40s
total) to give a transient throttle a chance, then degrades. `collect` only
aborts when **no** source yields any item (it refuses to write an empty
`collect.json`).

## Configuration

Environment variables (`.env`) only carry API keys/endpoints:

| Variable | Purpose |
|----------|---------|
| `SKAINET_API_KEY` | API key for the LLM judge/transcript backend (required); also used for TTS |
| `LLM_API_BASE` | OpenAI-compatible LLM endpoint for the podcast transcript |
| `TAXONOMY_PATH` | Optional: override the runtime taxonomy artifact path (default `data/taxonomy.json`) |

Everything user-facing (audience, familiar topics, topic prefs + α, window,
length, depth, memory lookback) is edited in the UI and stored in
`data/podcast_config.yaml`.

## Testing everything we implemented

### Unit tests (no network, no LLM)

```bash
cd pipeline-app
uv run python -m pytest tests -q        # 122 tests: pipeline + service + taxonomy refresh
```

### Cross-episode memory — end to end

The fast harness fabricates prior episodes + a current episode and scans the
generated transcript for real/reference/no-invented follow-ups:

```bash
cd pipeline-app
python -m pipeline.memtest --no-llm              # dry run: inspect the memory-context selection
python -m pipeline.memtest                       # real transcript LLM call, scans for references
python -m pipeline.memtest --mem-windows 3       # exercise a different retention setting
```

### Multi-label topics — verify the flow

1. Generate an episode (or run `python -m pipeline run --only label` to
   backfill), then confirm:
   - `rank.json`: items carry 1–3 topic ids (`"topics": {"post_training": 1.0,
     "agents": 1.0}`).
   - `podcast_brief.md`: bullets end with `· post_training,agents`.
   - `labels.json`: the episode topic vector across all buckets.
2. Quality-check the labeler against a big reference model:

   ```bash
   python -m pipeline.eval_labels --pool data/eval/batches/*.json \
       --out data/eval/labels__small__vs__big
   ```

### Steering — verify the blend

Set **topics you're interested in** and α>0 in Settings, generate, then check
`rank.json`: items matching your prefs have `personal_score` (cosine) and a
`final_score` between their importance and personal score, and the aired order
(`select_sources`) follows `final_score` (α=0 keeps pure importance order).
Unit coverage lives in `tests/test_pipeline.py` (`test_personal_match_*`,
`test_rank_steering_*`).

### Taxonomy refresh — verify before adopting

1. `python -m pipeline.fetch_corpus --out data/eval/corpus__<date>`
2. `python -m pipeline.derive_taxonomy --pool ... --out data/eval/taxonomy__<date>`
3. Read `data/eval/taxonomy__<date>/report.json`: `coverage_gap` ≤ 0.05 and
   `agreement_top1_small_vs_big` ≥ 0.60 → `decision.accept` is true. Preview
   `taxonomy.runtime.json` (ids/labels/descriptions).
4. Only then add `--apply data/taxonomy.json` and re-open the UI — the Settings
   checkboxes (populated from `GET /api/taxonomy`) and all downstream stages now
   use the new set. Revert by deleting `data/taxonomy.json`.

### UI — manual walkthrough

1. Open `http://localhost:8000`, complete the setup dialog.
2. **Generate new episode** → watch the progress panel → episode appears in
   History.
3. Open it: play audio; check the **Brief** tab shows per-item `· label_a,label_b`
   tokens; **Memory** tab shows per-topic summaries (from episode ≥ 2 onward).
4. **Edit brief** → delete an item → **Generate audio** → episode replaced in
   place (`selection_source` flips to `customized`).
5. **Settings**: topics checkboxes match the live taxonomy; changing topic prefs
   + α steers the next run.
6. **Delete** removes the episode and its folder.
