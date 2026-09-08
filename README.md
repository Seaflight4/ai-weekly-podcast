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
- **Episode shape** — rolling window in days, podcast length, and topic depth
  per source.

Save to keep these settings (they can be changed anytime under **Settings**).

Then, all in the browser:

- **Generate new episode** — confirm the window, length and depth, and the run
  starts. The panel tracks progress and shows the live log; one run at a time.
  Generating twice in one day keeps **both** episodes — history is
  time-stamped, so a new episode never overwrites an earlier one.
- **History** — every episode, newest first, with ready/draft badges. Click one
  to play the audio and read the **Brief** and **Transcript** tabs.
- **Edit brief** — remove or restore sources, then re-render the audio. The
  episode is replaced in place with your selection.
- **Delete** — removes an episode from history entirely.

## Configuration

Environment variables (`.env`) only carry API keys/endpoints:

| Variable | Purpose |
|----------|---------|
| `SKAINET_API_KEY` | API key for the LLM judge/transcript backend (required); also used for TTS |
| `LLM_API_BASE` | OpenAI-compatible LLM endpoint for the podcast transcript |

Everything user-facing (audience, familiar topics, window, length, depth) is
edited in the UI and stored in `data/podcast_config.yaml`.
