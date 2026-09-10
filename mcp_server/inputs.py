"""Normalize MCP tool arguments into engine Source objects.

Each source entry from the tool call is either:
  - ``{"url": "https://..."}`` — a web page or arXiv paper URL
  - ``{"pdf_base64": "...", "filename": "report.pdf"}`` — an inline PDF

URLs pointing at arxiv.org are classified as ``kind="arxiv"`` (the engine
downloads the PDF); all other URLs are ``kind="blog"`` (the engine fetches
the web page text). Base64 PDFs are decoded to a temp file and classified
as ``kind="pdf"`` (the engine reads the local file directly).
"""
from __future__ import annotations

import base64
import os
import pathlib
import re
import uuid

from podcast_engine import Source


_ARXIV_ABS_RE = re.compile(r"^https?://arxiv\.org/abs/")
_ARXIV_PDF_RE = re.compile(r"^https?://arxiv\.org/pdf/")


def normalize_sources(raw_sources: list[dict],
                      papers_dir: pathlib.Path | str) -> list[Source]:
    """Convert raw tool-arg source dicts into engine :class:`Source` objects.

    Args:
        raw_sources: List of ``{url}`` or ``{pdf_base64, filename, title?}``.
        papers_dir: Directory to write decoded base64 PDFs into.

    Returns:
        List of :class:`Source` objects ready for ``generate_podcast()``.
    """
    papers_dir = pathlib.Path(papers_dir)
    papers_dir.mkdir(parents=True, exist_ok=True)
    sources: list[Source] = []
    for raw in raw_sources:
        url = raw.get("url")
        pdf_b64 = raw.get("pdf_base64")
        title = raw.get("title", "")

        if url:
            sources.append(_from_url(url, title))
        elif pdf_b64 and raw.get("filename"):
            sources.append(_from_base64(pdf_b64, raw["filename"], title, papers_dir))
        else:
            raise ValueError(
                f"source must have 'url' or ('pdf_base64' + 'filename'); got keys: {list(raw.keys())}"
            )
    return sources


def _from_url(url: str, title: str) -> Source:
    """Classify a URL as arxiv or blog and build a Source."""
    if _ARXIV_ABS_RE.match(url):
        pdf_url = url.replace("https://arxiv.org/abs/", "https://arxiv.org/pdf/", 1)
        return Source(title=title or url, url=url, pdf_url=pdf_url,
                      excerpt="", kind="arxiv")
    elif _ARXIV_PDF_RE.match(url):
        return Source(title=title or url, url=url, pdf_url=url,
                      excerpt="", kind="arxiv")
    else:
        return Source(title=title or url, url=url, pdf_url=None,
                      excerpt="", kind="blog")


def _from_base64(pdf_base64: str, filename: str, title: str,
                 papers_dir: pathlib.Path) -> Source:
    """Decode a base64-encoded PDF to a local file and build a Source."""
    # Sanitize filename: keep the extension, replace problematic chars.
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    if not safe_name.endswith(".pdf"):
        safe_name += ".pdf"
    # Prefix with uuid to avoid collisions.
    dest = papers_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    dest.write_bytes(base64.b64decode(pdf_base64))
    return Source(
        title=title or filename,
        url=f"file://{dest}",
        pdf_url=None,
        excerpt="",
        kind="pdf",
        local_path=str(dest),
    )
