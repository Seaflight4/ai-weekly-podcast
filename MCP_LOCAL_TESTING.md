# Testing the podcast MCP server locally

Run the podcast MCP server on your own machine to test it as a tool in
opencode. This is the local, single-test setup — no shared/hosted server
required.

## Prerequisites

- Docker Desktop/colima (with Compose) and git
- [opencode](https://opencode.ai) CLI
- Access to the TNG network/VPN (the server calls internal LLM and TTS
  services: `chat.model.tngtech.com` and `tts.model.tngtech.com`)
- A `SKAINET_API_KEY`

## Steps

1. Clone the repo (runs from the `mcp-app/` folder — the MCP is standalone,
   no pipeline/ranking code involved):

   ```bash
   git clone ssh://git@bitbucket.int.tngtech.com:122/air/source-to-podcast.git
   cd mcp-app
   ```

2. Configure the key:

   ```bash
   cp .env.example .env
   # edit .env: set SKAINET_API_KEY=<your key>
   ```

3. Build and start the MCP server (builds the image with the repo root as
   build context, runs `python -m mcp_server` on port 8001):

   ```bash
   docker compose up -d --build
   ```

4. Confirm it is alive. A raw request returns a JSON `Missing session ID`
   error — that is the correct "I speak MCP" response (opencode's real
   handshake includes the session ID):

   ```bash
   docker compose ps                  # podcast-mcp Up
   curl http://localhost:8001/mcp     # -> {"jsonrpc":"2.0",...,"error":{"code":-32600,...}}
   ```

5. Add the MCP to opencode — nothing to type: the `mcp-app/` folder ships
   `opencode.json` with the `podcast` MCP pointing at `http://localhost:8001/mcp`.
   Launch opencode **inside `mcp-app/`**, starting the container **before**
   opencode, since opencode connects to MCP servers once, at session start.
   If opencode is already open, run `/mcp` (reconnect) or restart it.

6. Call it as a tool. In opencode the tool appears as
   `podcast_generate_podcast`. For example:

   > Use the `podcast_generate_podcast` tool to make a short podcast about the
   > RAG Survey paper from https://arxiv.org/abs/2312.10997

   It returns a `job_id` and a resource URI `podcast://jobs/{id}`. The job
   runs async (transcript LLM + TTS takes a few minutes). Reads on the job
   resource are paced: the first check returns ~1 minute after launch and
   later checks at most once per minute, so don't poll it rapidly — wait for
   each read to return (it blocks until it's time to check again). Check the
   result by asking opencode to read the job resource, or inspect the output
   folder:

   ```bash
   ls ../data/engine/<latest-timestamp>/      # transcript.md + episode.mp3
   ```

7. Stop when done:

   ```bash
   docker compose down
   ```

## Troubleshooting

- `podcast_generate_podcast` not in the tool list: the container was not up
  when opencode started; run `docker compose ps`, then `/mcp` or restart
  opencode.
- Job fails at the "TTS health" / LLM stage: `.env` key is wrong, or not on
  the TNG network.
- `curl` connection refused: compose not started, or wrong service name.

## Testing the pipeline app instead

The pipeline (ranking + web UI) lives in `pipeline-app/` and shares the same
`podcast_engine` library; it has no MCP code. See `pipeline-app/README.md`.
