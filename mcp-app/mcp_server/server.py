"""MCP server exposing the generate_podcast tool.

Transport: Streamable HTTP (default port 8001). Run with::

    python -m mcp_server

or programmatically::

    from mcp_server.server import main
    main()
"""
from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from podcast_engine import EngineConfig
from mcp_server import inputs, jobs, DATA_ROOT


server = MCPServer(
    "podcast-server",
    instructions=(
        "Generate podcast episodes from provided sources. "
        "Call generate_podcast with a topic and a list of sources "
        "(URLs or base64-encoded PDFs). The tool returns a job_id immediately. "
        "Job reads on the podcast://jobs/{job_id} resource are paced: the "
        "first read is only served ~1 minute after launch and later reads at "
        "most once per minute, so do NOT poll repeatedly in fast succession — "
        "wait for the read to return. It returns the finished transcript + "
        "audio path when the job is done."
    ),
)


@server.tool()
def generate_podcast(
    topic: str,
    sources: list[dict],
    audience: str = "a general audience interested in the topic.",
    familiar_topics: list[str] | None = None,
    length: str = "medium",
    depth: str = "deep-dive",
    podcast_name: str = "Podcast",
    host1_name: str = "Brian",
    host2_name: str = "Tina",
) -> str:
    """Generate a podcast episode from provided sources.

    Args:
        topic: Free-text description of the podcast topic (e.g. "RAG in
            enterprise search").
        sources: List of source entries. Each entry is either
            ``{"url": "https://..."}`` (a web page or arXiv paper) or
            ``{"pdf_base64": "<base64>", "filename": "report.pdf"}`` (an
            inline PDF document).
        audience: Free-text description of the target audience.
        familiar_topics: Topics the listener already knows (skipped in
            the explanation).
        length: Episode length preset: "short", "medium", or "long".
        depth: Per-source depth: "brief" or "deep-dive".
        podcast_name: Display name for the podcast.
        host1_name: Name of the first host.
        host2_name: Name of the second host.

    Returns:
        JSON string with job_id, status, and a resource URI to poll.
    """
    if not sources:
        return json.dumps({"error": "at least one source is required"})

    papers_dir = DATA_ROOT / "mcp/papers"
    engine_sources = inputs.normalize_sources(sources, papers_dir)

    config = EngineConfig(
        podcast_topic=topic,
        podcast_name=podcast_name,
        audience=audience,
        familiar_topics=familiar_topics or [],
        length=length,
        depth=depth,
        host1_name=host1_name,
        host2_name=host2_name,
    )

    job_id = jobs.submit(engine_sources, config)
    return json.dumps({
        "job_id": job_id,
        "status": "pending",
        "resource": f"podcast://jobs/{job_id}",
    })


@server.resource("podcast://jobs/{job_id}")
def get_job(job_id: str) -> str:
    """Check a podcast generation job for status and results.

    Reads are paced to ~one per minute: the first read is served only after
    ~1 minute from launch, and an in-flight job's read blocks until it is
    time to check again. A completed/failed job returns immediately.

    Returns JSON with:
      - status: "pending", "running", "completed", or "failed"
      - stage: current processing stage (when running)
      - transcript: full transcript text (when completed)
      - audio_path: filesystem path to episode.mp3 (when completed)
      - error: error message (when failed)
    """
    result = jobs.poll(job_id)
    return json.dumps(result, indent=2)


def main():
    """Entry point for ``python -m mcp_server``."""
    import os
    port = int(os.environ.get("MCP_PORT", "8001"))
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    server.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
