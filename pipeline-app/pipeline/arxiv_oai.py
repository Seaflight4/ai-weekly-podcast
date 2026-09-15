"""OAI-PMH metadata mirror for the arXiv collect stage.

The arXiv OAI-PMH endpoint (``oaipmh.arxiv.org``) is arXiv's *preferred* way to
keep an up-to-date copy of arXiv metadata and — unlike the search query API
(``export.arxiv.org/api/query``) — is not subject to its opaque 429 throttling
(verified from the shared corporate egress the query API throttles). This module
mirrors the ``cs:cs:AI`` metadata locally so ``collect`` reads a stable cache
instead of the throttled query endpoint.

Model:

- The mirror is a local cache of normalized metadata, one JSON file per arXiv
  id under ``MIRROR_ROOT/records/``, plus a ``state.json`` checkpoint.
- Sync is LAZY: ``ensure_coverage`` only deepens the harvest when the cached
  coverage horizon is newer than the window a caller needs. There is no
  scheduler; a stale or cold mirror self-heals on the next collect.
- OAI only filters by *last-modified* datestamp (header ``datestamp``), not by
  submission date, so the weekly window is applied locally: a record is kept
  when its **primary category is ``cs.AI``** (the first token of ``<categories>``,
  matching the old ``cat:cs.AI`` search) and its **first submitted date** (the
  earliest ``<version>/<date>``) falls inside the window.

Only metadata is mirrored (title, abstract, authors, categories, dates). Full
text is fetched by the engine per episode from ``arxiv.org/pdf`` — a different,
unthrottled endpoint, with its own graceful abstract fallback.

Withdrawal: OAI carries no reliable announce-type marker; a record whose
``<comments>`` mentions withdrawal is skipped (the same heuristic the query-API
path used). The query-API collector (``collect._collect_arxiv_queryapi``)
remains as a fallback for cold mirrors / mirror failure.
"""
from __future__ import annotations

import datetime
import email.utils
import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from . import DATA_ROOT

# --- endpoint + constants ----------------------------------------------------

OAI_BASE = "https://oaipmh.arxiv.org/oai"
OAI_SET = "cs:cs:AI"
OAI_METADATA_PREFIX = "arXivRaw"
OAI_PRIMARY_CATEGORY = "cs.AI"

OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/",
          "ar": "http://arxiv.org/OAI/arXivRaw/"}
AR_RAW = "http://arxiv.org/OAI/arXivRaw/"

OAI_PAGE_SLEEP = 1.0          # politeness between ListRecords pages
OAI_RETRIES = 3
OAI_BACKOFF_BASE = 5.0
OAI_BACKOFF_CAP = 30.0

# Mirror location (env-overridable, like PIPELINE_DATA_ROOT / TAXONOMY_PATH).
# Must live on the deployment's persistent data volume, not the ephemeral FS.
MIRROR_ROOT = pathlib.Path(
    os.environ.get("ARXIV_MIRROR", str(DATA_ROOT / "arxiv_mirror")))
RECORDS_DIR = MIRROR_ROOT / "records"
STATE_FILE = MIRROR_ROOT / "state.json"

# How far a fresh mirror walks back (weeks of cs.AI for bootstrap + memory
# windows), the safety buffer before a window, and the record retention horizon.
BACKFILL_DAYS = 56
COVERAGE_BUFFER_DAYS = 1
RETENTION_DAYS = 90


# --- helpers -----------------------------------------------------------------

def _parse_iso(s) -> datetime.date | None:
    if not s:
        return None
    try:
        return datetime.date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def _parse_rfc_date(s) -> datetime.date | None:
    # arXiv `<version>/<date>` is RFC 822-ish ("Wed, 27 Mar 2026 19:46:47 GMT").
    if not s:
        return None
    try:
        return email.utils.parsedate_to_datetime(s).date()
    except (TypeError, ValueError):
        pass
    return _parse_iso(s)


def _clean_text(s) -> str:
    """Strip embedded markup + collapse whitespace (arXiv abstracts/titles)."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def _safe_id(arid: str) -> str:
    """Filesystem-safe key for an arXiv id older scheme may include ``/``."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", arid)


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_name(STATE_FILE.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


# --- fetching + parsing ------------------------------------------------------

def _http_get(url: str) -> str:
    """GET with exponential backoff on arXiv's transient 429/5xx/network."""
    last = "unknown error"
    for attempt in range(OAI_RETRIES):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "ai-weekly-podcast-poc/0.1"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = f"network error: {e}"
        if attempt == OAI_RETRIES - 1:
            break
        wait = min(OAI_BACKOFF_BASE * (2 ** attempt), OAI_BACKOFF_CAP)
        print(f"      arxiv-oai: {last} on request — backing off {wait:.0f}s "
              f"({attempt + 1}/{OAI_RETRIES})")
        time.sleep(wait)
    raise RuntimeError(f"arxiv-oai: giving up after {OAI_RETRIES} retries ({last})")


def _records_url(from_ds: str | None = None, token: str | None = None) -> str:
    params: dict[str, str]
    if token:
        params = {"verb": "ListRecords", "resumptionToken": token}
    else:
        params = {"verb": "ListRecords", "set": OAI_SET,
                  "metadataPrefix": OAI_METADATA_PREFIX, "from": from_ds}
    return f"{OAI_BASE}?{urllib.parse.urlencode(params)}"


def _parse_feed(xml: str) -> tuple[list[dict], str | None, str]:
    """Parse a ListRecords response -> (records, resumption token, responseDate)."""
    feed = ET.fromstring(xml)
    ns = OAI_NS
    response_date = feed.findtext("oai:responseDate", default="", namespaces=ns) or ""
    records: list[dict] = []
    for rec in feed.findall(".//oai:record", ns):
        norm = _normalize(rec)
        if norm:
            records.append(norm)
    tok_el = feed.find(".//oai:resumptionToken", ns)
    token = None
    if tok_el is not None and (tok_el.text or "").strip():
        token = tok_el.text.strip()
    return records, token, response_date


def _withdrawn_comment(comment: str) -> bool:
    return "withdraw" in (comment or "").lower()


def _normalize(rec: ET.Element) -> dict | None:
    """One OAI-PMH record (arXivRaw prefix) -> a normalized JSON-able dict."""
    ns = OAI_NS
    meta = rec.find("oai:metadata", ns)
    if meta is None:
        return None
    raw = meta.find(f"{{{AR_RAW}}}arXivRaw")
    if raw is None:
        return None

    def text(tag: str, *alt: str) -> str:
        for name in (tag, *alt):
            el = raw.find(f"{{{AR_RAW}}}{name}")
            if el is not None and (el.text or "").strip():
                return el.text or ""
        return ""

    arid = text("id").strip()
    if not arid:
        return None
    categories = text("categories").strip()
    primary = re.split(r"[\s,]+", categories)[0] if categories else ""
    header_ds = rec.findtext("oai:header/oai:datestamp", default="", namespaces=ns)
    last_modified = _parse_iso(header_ds)

    submitted = None
    for v in raw.findall(f"{{{AR_RAW}}}version"):
        d = _parse_rfc_date(v.findtext(f"{{{AR_RAW}}}date") or "")
        if d and (submitted is None or d < submitted):
            submitted = d
    if submitted is None:
        submitted = last_modified

    comment = text("comments", "comment")
    return {
        "id": arid,
        "primary_category": primary,
        "categories": categories,
        "title": _clean_text(text("title")),
        "abstract": _clean_text(text("abstract")),
        "authors": _clean_text(text("authors")),
        "license": text("license"),
        "comments": comment,
        "withdrawn": _withdrawn_comment(comment),
        "first_submitted": submitted.isoformat() if submitted else "",
        "last_modified": last_modified.isoformat() if last_modified else "",
    }


def _write_record(rec: dict) -> None:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RECORDS_DIR / f"{_safe_id(rec['id'])}.json"
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, dest)


# --- sync + coverage ---------------------------------------------------------

def sync(from_ds: str, fetch=_http_get) -> dict:
    """Harvest ``OAI_SET`` records modified since ``from_ds`` and upsert them.

    Follows resumption tokens page-by-page (politely paced) and advances the
    ``state.json`` checkpoint on success. Idempotent: re-running the same
    ``from_ds`` just rewrites the same record files.
    """
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    token: str | None = None
    url = _records_url(from_ds)
    n_records = 0
    pages = 0
    server_date: datetime.date | None = None
    while True:
        xml = fetch(url)
        recs, token, resp_date = _parse_feed(xml)
        rd = _parse_iso(resp_date)
        if rd is not None and (server_date is None or rd > server_date):
            server_date = rd
        for rec in recs:
            _write_record(rec)
            n_records += 1
        pages += 1
        if pages % 10 == 0:
            print(f"      arxiv-oai: {pages} page(s), {n_records} record(s) so far")
        if not token:
            break
        time.sleep(OAI_PAGE_SLEEP)
        url = _records_url(token=token)

    state = _load_state()
    prev_from = _parse_iso(state.get("last_from"))
    parsed_from = _parse_iso(from_ds)
    if server_date is not None:
        newer = server_date - datetime.timedelta(days=1)
        if prev_from is None or newer > prev_from:
            state["last_from"] = newer.isoformat()
    else:
        state["last_from"] = prev_from.isoformat() if prev_from else from_ds
    if parsed_from is not None:
        deepest = _parse_iso(state.get("deepest_from"))
        state["deepest_from"] = min(deepest or parsed_from,
                                    parsed_from).isoformat()
    state["synced_at"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat()
    state["record_count"] = n_records
    _save_state(state)
    print(f"      arxiv-oai: synced {n_records} record(s) in {pages} page(s) "
          f"from {from_ds}")
    return {"pages": pages, "records": n_records, "from": from_ds}


def ensure_coverage(needed_from: datetime.date) -> dict:
    """Make sure the mirror covers records modified since ``needed_from``.

    No-op when the cached horizon (``deepest_from``) already reaches back that
    far; otherwise deepens the harvest (bootstrap walks back ``BACKFILL_DAYS``).
    Prunes expired records after a sync so the mirror stays bounded.
    """
    state = _load_state()
    deepest = _parse_iso(state.get("deepest_from"))
    if deepest is not None and deepest <= needed_from:
        return {"action": "cached", "state": state}

    from_ds = needed_from - datetime.timedelta(days=BACKFILL_DAYS) \
        if deepest is None else needed_from
    stats = sync(from_ds.isoformat())
    stats["action"] = "synced"
    now = datetime.date.today()
    if RECORDS_DIR.exists():
        removed = prune(now - datetime.timedelta(days=RETENTION_DAYS))
        if removed:
            print(f"      arxiv-oai: pruned {removed} expired record(s)")
    return stats


# --- reading + retention -----------------------------------------------------

def records_for_window(start: datetime.date, end: datetime.date,
                       primary_only: bool = True) -> list[dict]:
    """The mirror's cs.AI records with submission date in ``[start, end]``.

    Applies the ``cat:cs.AI``-equivalent filter locally: primary category must
    be ``cs.AI`` (OAI category sets include cross-lists otherwise), withdrawn
    records are dropped, and the window is matched on *first submitted* date.
    Returns newest-first.
    """
    if not RECORDS_DIR.exists():
        return []
    out: list[dict] = []
    for p in RECORDS_DIR.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if primary_only and rec.get("primary_category") != OAI_PRIMARY_CATEGORY:
            continue
        if rec.get("withdrawn"):
            continue
        sd = _parse_iso(rec.get("first_submitted"))
        if sd is None or not (start <= sd <= end):
            continue
        out.append(rec)
    out.sort(key=lambda r: r.get("first_submitted", ""), reverse=True)
    return out


def prune(cutoff: datetime.date) -> int:
    """Delete records whose newest (submitted|modified) date is older than cut."""
    if not RECORDS_DIR.exists():
        return 0
    removed = 0
    for p in RECORDS_DIR.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        last = _parse_iso(rec.get("last_modified"))
        first = _parse_iso(rec.get("first_submitted"))
        newest = max(x for x in (last, first) if x) if (last or first) else None
        if newest is not None and newest < cutoff:
            p.unlink(missing_ok=True)
            removed += 1
    return removed
