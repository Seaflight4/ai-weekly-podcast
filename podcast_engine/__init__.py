"""Generic podcast generation engine.

Turns a list of sources (documents, URLs, PDFs) into a finished podcast
episode (transcript + audio). Domain-agnostic: callers supply the topic,
audience, and framing. Both the AI-news pipeline (``pipeline/``) and the
MCP server (``mcp_server/``) consume this engine.

Entry point: :func:`podcast_engine.engine.generate_podcast`
"""
from .engine import EngineConfig, EpisodeResult, generate_podcast
from .sources import Source

__all__ = ["EngineConfig", "EpisodeResult", "Source", "generate_podcast"]
