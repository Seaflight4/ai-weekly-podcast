"""Vendored podcastfy-long: long-form podcast transcript + TTS generation.

Reads a ``podcast_brief.md`` (the format emitted by ``pipeline.generate``),
fetches full arXiv PDFs and Hacker-News/web pages, generates a two-host
transcript via an OpenAI-compatible LLM, and synthesizes voice-cloned audio
via the TNG qwen3 TTS service.

The entry point is :class:`SimplePodcastGenerator` in :mod:`pipeline.podcastfy.generator`.
"""
