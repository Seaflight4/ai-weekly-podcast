"""Source ingestion for podcast generation.

Downloads arXiv PDFs and fetches web page text for blog sources so they can
be fed to the transcript-generation LLM. Provides the :class:`Source`
dataclass used by both the engine and the MCP server.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# Extracts an arXiv ID (e.g. 2608.15089) from an arxiv.org URL.
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]+)")


@dataclass
class Source:
    title: str
    url: str = ""
    pdf_url: Optional[str] = None
    excerpt: str = ""
    kind: str = "blog"  # "arxiv", "blog", or "pdf"
    local_path: Optional[str] = None  # already-on-disk PDF (kind="pdf")

    @property
    def arxiv_id(self) -> Optional[str]:
        if not self.pdf_url:
            return None
        m = _ARXIV_ID_RE.search(self.pdf_url)
        return m.group(1) if m else None


def download_arxiv_pdfs(sources: List[Source], target_dir: str = "papers") -> List[str]:
    """Download arXiv PDFs for sources that have a pdf_url.

    Skips downloads that already exist on disk. Returns a list of local
    file paths (in the same order as the input sources that had a pdf_url).
    """
    os.makedirs(target_dir, exist_ok=True)
    paths: List[str] = []
    for src in sources:
        if not src.pdf_url or not src.arxiv_id:
            continue
        dest = os.path.join(target_dir, f"{src.arxiv_id}.pdf")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            logger.info("Already have %s, skipping download", dest)
            paths.append(dest)
            continue
        logger.info("Downloading %s -> %s", src.pdf_url, dest)
        try:
            resp = requests.get(src.pdf_url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            paths.append(dest)
        except Exception as e:
            logger.warning("Failed to download %s: %s", src.pdf_url, e)
    return paths


def _slugify(url: str) -> str:
    """Build a filesystem-safe slug from a URL for caching fetched pages."""
    # Strip scheme and use host + path, replacing non-alphanumerics.
    cleaned = re.sub(r"^https?://", "", url)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", cleaned).strip("_")
    # Keep it bounded so pathological long URLs don't break the filesystem.
    return slug[:120] or "page"


def fetch_web_content(sources: List[Source], target_dir: str = "web") -> dict:
    """Fetch the main-article text for non-arXiv (blog) sources.

    Mirrors ``download_arxiv_pdfs``: one .txt file per blog source is
    written to ``target_dir`` and reused on subsequent runs. Returns a
    mapping of ``Source.url`` -> extracted plain-text content. Sources
    whose fetched content is empty or whose fetch fails are simply
    omitted from the returned dict (the caller then falls back to the
    brief excerpt).
    """
    os.makedirs(target_dir, exist_ok=True)
    fetched: dict = {}
    for src in sources:
        if src.kind == "arxiv" or src.kind == "pdf":
            continue
        dest = os.path.join(target_dir, f"{_slugify(src.url)}.txt")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            try:
                with open(dest, "r", encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    fetched[src.url] = content
                    logger.info("Using cached web content for %s", src.url)
                    continue
            except Exception:
                pass  # fall through to re-fetch
        logger.info("Fetching %s -> %s", src.url, dest)
        content = ""
        try:
            import trafilatura
            downloaded = trafilatura.fetch_url(src.url)
            if not downloaded:
                logger.warning("No content returned for %s", src.url)
                continue
            content = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                favor_recall=True,
            ) or ""
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", src.url, e)
            continue
        if not content.strip():
            logger.warning("Empty extraction for %s", src.url)
            continue
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)
        fetched[src.url] = content
    return fetched
