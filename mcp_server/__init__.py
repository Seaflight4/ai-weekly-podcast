"""MCP server exposing the generate_podcast tool.

A thin network layer over :mod:`podcast_engine`. An LLM agent (the MCP
client) provides sources (URLs + base64-encoded PDFs) and a topic; the
server generates a podcast episode asynchronously and exposes the result
via a pollable resource.
"""
