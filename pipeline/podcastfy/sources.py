"""Source ingestion for podcast generation.

Parses a podcast brief markdown file into structured source entries and
downloads arXiv PDFs so they can be fed to the transcript-generation LLM
via the existing PyMuPDF extraction path. Also fetches full main-article
text for blog (non-arXiv) sources via trafilatura.

Brief format (see pipeline/generate.py `_brief_text`):
    ## arXiv papers (N)

    - [Title](https://arxiv.org/abs/XXXX.XXXXX) · [PDF](https://arxiv.org/pdf/XXXX.XXXXX) — score 0.88 · post_training
      > Excerpt paragraph.

    ## Hacker News stories (N)

    - [Title](https://example.com/path) — score 0.85
      > Excerpt paragraph.

The trailing ``· <topic_id>`` is optional (legacy briefs and items the judge
failed to label omit it) and is not used for generation.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# Matches a markdown bullet with a title link, optional · [PDF](url), a score,
# and an optional trailing topic id (e.g. "— score 0.88 · agents").
_BULLET_RE = re.compile(
    r"^- \[(?P<title>[^\]]+)\]\((?P<url>[^)]+)\)"  # [Title](url)
    r"(?:\s*·\s*\[PDF\]\((?P<pdf_url>[^)]+)\))?"   # optional · [PDF](pdf_url)
    r"(?:\s*—\s*score\s*[\d.]+)?"                  # optional — score 0.XX
    r"(?:\s*·\s*(?P<label>[\w-]+))?"               # optional · topic_id
    r"\s*$",
    re.MULTILINE,
)
# Matches the blockquote excerpt immediately following a bullet (indented '> ...').
_QUOTE_RE = re.compile(r"^\s*>\s*(.+)$", re.MULTILINE)
# Extracts an arXiv ID (e.g. 2608.15089) from an arxiv.org URL.
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]+)")


@dataclass
class Source:
    title: str
    url: str
    pdf_url: Optional[str]
    excerpt: str
    kind: str  # "arxiv" or "blog"

    @property
    def arxiv_id(self) -> Optional[str]:
        if not self.pdf_url:
            return None
        m = _ARXIV_ID_RE.search(self.pdf_url)
        return m.group(1) if m else None


def parse_brief(brief_path: str) -> List[Source]:
    """Parse a podcast brief markdown file into a list of Source entries."""
    with open(brief_path, "r") as f:
        text = f.read()

    sources: List[Source] = []
    for m in _BULLET_RE.finditer(text):
        # Find the nearest following blockquote for the excerpt.
        after = text[m.end():]
        qm = _QUOTE_RE.match(after)
        excerpt = qm.group(1).strip() if qm else ""
        pdf_url = m.group("pdf_url")
        kind = "arxiv" if pdf_url else "blog"
        sources.append(
            Source(
                title=m.group("title").strip(),
                url=m.group("url").strip(),
                pdf_url=pdf_url,
                excerpt=excerpt,
                kind=kind,
            )
        )
    logger.info("Parsed %d sources from %s", len(sources), brief_path)
    return sources


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
        if src.kind == "arxiv":
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
