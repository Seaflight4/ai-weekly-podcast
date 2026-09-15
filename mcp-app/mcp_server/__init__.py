"""MCP server exposing the generate_podcast tool.

A thin network layer over :mod:`podcast_engine`. An LLM agent (the MCP
client) provides sources (URLs + base64-encoded PDFs) and a topic; the
server generates a podcast episode asynchronously and exposes the result
via a pollable resource.

All runtime state lives under this app's own ``data/`` folder (anchored to
the app directory, not the process CWD), so it works whether the server is
run from cmd or in Docker (where ``/app/data`` is the bind mount).
"""
from __future__ import annotations

import pathlib

APP_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_ROOT = APP_ROOT / "data"
