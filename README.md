# Source → Podcast

A monorepo with two independently testable tools that share one library:

```
├── podcast-engine/    # SHARED generic engine: sources (URLs/PDFs) → transcript + episode.mp3
│   └── podcast_engine/   (the importable python package)
├── mcp-app/           # Standalone MCP server exposing the generate_podcast tool
└── pipeline-app/      # AI Weekly Podcast: rank AI news with an LLM judge, then generate an episode
```

Both apps import `podcast_engine` (single source of truth under
`podcast-engine/`) and persist their runtime state under `data/`. They have
**no dependency on each other** — build and run either one on its own.

## Quick start — podcast MCP (mcp-app)

```bash
git clone ssh://git@bitbucket.int.tngtech.com:122/air/source-to-podcast.git
cd mcp-app
cp .env.example .env      # then set SKAINET_API_KEY
docker compose up -d --build   # podcast-mcp on http://localhost:8001 (MCP over Streamable HTTP)
# restart opencode
# generate podcast from opencode CLI. For example: Use the `podcast_generate_podcast` tool to make a short podcast about the RAG Survey paper from https://arxiv.org/abs/2312.10997.
docker compose down # stop container
```

See `mcp-app/README.md` for the full product docs.

## Quick start — pipeline app (pipeline-app)

```bash
git clone ssh://git@bitbucket.int.tngtech.com:122/air/source-to-podcast.git
cd pipeline-app
cp .env.example .env      # then set SKAINET_API_KEY
docker compose up --build # then open http://localhost:8000 in your browser
docker compose down # stop container
```

See `pipeline-app/README.md` for the full product docs.

## Prerequisites (both apps)

- [Docker](https://docs.docker.com/get-docker/) (with Compose)
- A SkaiNet (OpenAI-compatible) API key (`SKAINET_API_KEY`) for LLM calls and
  TTS — see `.env.example` in each app folder
- TNG network/VPN access (the engine calls internal LLM + TTS services)

The Docker builds use the **repo root as build context**
(`docker build -f <app>/Dockerfile .`), so the shared `podcast-engine` library
is copied in from its single location. Each app has its own `pyproject.toml`,
`requirements.txt` and `docker-compose.yml`; the old root-level build files
were retired in the refactor.

## Layout

| Path | What |
|------|------|
| `podcast-engine/` | Shared library project (pyproject + `podcast_engine/` package): engine, sources, transcript LLM, TNG TTS backend. Edit once, used by both apps. |
| `mcp-app/mcp_server/` | MCP server (tool + paced job resource), `python -m mcp_server`. |
| `mcp-app/` | MCP Dockerfile, compose, pyproject, requirements, opencode.json, `.env.example` |
| `pipeline-app/pipeline/` | Ranking pipeline (collect → rank → generate), `python -m pipeline run`. |
| `pipeline-app/service/` + `static/` | FastAPI web app + browser UI over the pipeline. |
| `pipeline-app/` | Pipeline Dockerfile, compose, pyproject, requirements, `.env.example` |
| `data/` | Shared runtime volume (episodes, config, job logs) — gitignored. |
| `tests/` | Shared `podcast_engine` tests (app tests live in each app's `tests/`). |
