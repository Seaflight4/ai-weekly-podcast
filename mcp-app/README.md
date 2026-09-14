# Podcast MCP server

Standalone MCP server exposing the `generate_podcast` tool. It runs the shared
`podcast-engine` library (repo-root `../podcast-engine`) — no ranking pipeline
involved.

## Run

Pre-requisites: Docker (with Compose), a `SKAINET_API_KEY`, and TNG
network/VPN access (the engine calls internal LLM + TTS services).

```bash
cd mcp-app
cp .env.example .env          # then set SKAINET_API_KEY
docker compose up -d --build  # podcast-mcp on http://localhost:8001
curl http://localhost:8001/mcp   # -> {"jsonrpc":"2.0",...,"error":{"code":-32600,...}} = alive
```

In opencode, the `podcast` MCP (`opencode.json`, `podcast_generate_podcast`
tool) points at `http://localhost:8001/mcp`. Start the container before
launching opencode in this directory.

Generated episodes + the sqlite job DB persist under `../data`.
