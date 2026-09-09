# AI Weekly Podcast

An on-demand pipeline that turns the past week of AI research and community news
into a podcast episode. Everything runs in your browser: generate an episode,
play it, and read or edit its content.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (with Compose; Desktop/colima
  all work)
- A SkaiNet (OpenAI-compatible) API key for LLM calls and audio synthesis

## Quick start

```bash
git clone <repo> && cd source-ranking
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

Your episodes, settings and job logs persist locally in the `./data` folder
(gitignored, so nothing is shared or committed).

## Using the app

On first open, the **Set up your podcast** dialog appears:

- **Audience** (knowledge level) and **familiar topics** — the episode is
  tailored to these.
- **Topics you're interested in** and **steering strength (α)** — interests
  steer which items make it into each episode (see *Steering by topic* below).
- **Episode shape** — rolling window in days, podcast length, and topic depth
  per source.

Save to keep these settings (they can be changed anytime under **Settings**).

Then, all in the browser:

- **Generate new episode** — confirm the window, length and depth, and the run
  starts. The panel tracks progress and shows the live log; one run at a time.
  Generating twice in one day keeps **both** episodes — history is
  time-stamped, so a new episode never overwrites an earlier one.
- **History** — every episode, newest first, with ready/draft badges. The topic
  filter box narrows the list to episodes covering a topic you care about, and
  each episode shows its topic chips. Click one to play the audio and read the
  **Brief** and **Transcript** tabs.
- **Edit brief** — remove or restore sources, then re-render the audio. The
  episode is replaced in place with your selection.
- **Delete** — removes an episode from history entirely.

## Steering by topic

Every candidate item gets topic labels from a cheap model during the rank stage,
across three axes: a headline **type** (model_release, research_findings,
evaluation, incident, business, …), a **technical** area (post_training,
agents_tool_use, inference_efficiency, safety_alignment, …) and an
**application** domain (coding, healthcare, finance, …). When you set
**topics you're interested in** in Settings, each item's importance score is
blended with its match to your profile —
`final = (1-α)·importance + α·match` — and the top sources are chosen from that
blended order. α = 0 turns steering off (default selection preserved). Labels
on the aired episode also power the history topic filter. Existing episodes can
be labeled for filtering via `python -m pipeline run --only label`.

The cheap labeler is validated against a big reference model with
`python -m pipeline.eval_labels --pool <items.json> --out data/eval/<name>`
(see `data/eval/labels__mistral_small__vs__qwen/eval_report.json`).

## Configuration

Environment variables (`.env`) only carry API keys/endpoints:

| Variable | Purpose |
|----------|---------|
| `SKAINET_API_KEY` | API key for the LLM judge/transcript backend (required); also used for TTS |
| `LLM_API_BASE` | OpenAI-compatible LLM endpoint for the podcast transcript |

Everything user-facing (audience, familiar topics, window, length, depth) is
edited in the UI and stored in `data/podcast_config.yaml`.
