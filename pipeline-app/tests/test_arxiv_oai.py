"""Contract tests for the arXiv OAI metadata mirror (pipeline/arxiv_oai.py).

No network: fetching is monkeypatched; the mirror lives in a tmp dir.
"""
import datetime
import json
import pathlib
import sys

import pytest

_FILE = pathlib.Path(__file__).resolve()
# pipeline lives in the app folder (parents[1]).
sys.path.insert(0, str(_FILE.parents[1]))

from pipeline import arxiv_oai

OAI_NS = "http://www.openarchives.org/OAI/2.0/"
RAW_NS = "http://arxiv.org/OAI/arXivRaw/"


def _feed_xml(records, token=None, response="2026-09-15T00:00:00Z"):
    tok = f"<resumptionToken>{token}</resumptionToken>" if token else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<OAI-PMH xmlns="{OAI_NS}">'
        f"<responseDate>{response}</responseDate>"
        f"<request>http://oaipmh.arxiv.org/oai</request>"
        f"<ListRecords>{''.join(records)}{tok}</ListRecords>"
        "</OAI-PMH>"
    )


def _record(arid, versions, categories="cs.AI", datestamp="2026-09-14",
            comments=None, title="T", abstract="abstract"):
    vs = "".join(
        f"<version><date>{d}</date><size>1</size></version>"
        for d in versions)
    cm = f"<comments>{comments}</comments>" if comments is not None else ""
    return (
        f"<record><header><identifier>oai:arXiv.org:{arid}</identifier>"
        f"<datestamp>{datestamp}</datestamp><setSpec>cs:cs:AI</setSpec></header>"
        f"<metadata><arXivRaw xmlns=\"{RAW_NS}\">"
        f"<id>{arid}</id>{vs}"
        f"<title>{title}</title><authors>A B</authors>"
        f"<categories>{categories}</categories>"
        f"<abstract>{abstract}</abstract>{cm}"
        f"</arXivRaw></metadata></record>"
    )


def _set_mirror(tmp_path, monkeypatch):
    monkeypatch.setattr(arxiv_oai, "RECORDS_DIR", tmp_path / "records")
    monkeypatch.setattr(arxiv_oai, "STATE_FILE", tmp_path / "state.json")


def _write_record_files(tmp_path, recs):
    d = tmp_path / "records"
    d.mkdir(parents=True, exist_ok=True)
    for r in recs:
        (d / f"{r['id']}.json").write_text(json.dumps(r), encoding="utf-8")


# --- parsing / normalization -------------------------------------------------

def test_parse_feed_normalizes_arxivraw():
    xml = _feed_xml([_record(
        "2609.00001",
        ["Fri, 4 Sep 2026 09:30:00 GMT", "Thu, 10 Sep 2026 11:00:00 GMT"],
        categories="cs.AI cs.LG")])
    records, token, resp = arxiv_oai._parse_feed(xml)
    assert token is None
    assert len(records) == 1
    r = records[0]
    assert r["id"] == "2609.00001"
    assert r["primary_category"] == "cs.AI"          # FIRST token of categories
    assert r["categories"] == "cs.AI cs.LG"
    assert r["first_submitted"] == "2026-09-04"      # earliest <version>/<date>
    assert r["last_modified"] == "2026-09-14"        # header datestamp
    assert r["withdrawn"] is False
    assert r["abstract"] == "abstract"


def test_withdrawal_comment_is_flagged():
    xml = _feed_xml([_record("2609.00002", ["Fri, 4 Sep 2026 09:30:00 GMT"],
                             comments="Paper withdrawn by authors")])
    records, _, _ = arxiv_oai._parse_feed(xml)
    assert records[0]["withdrawn"] is True


# --- sync / paging / state ---------------------------------------------------

def test_sync_follows_resumption_token_and_writes_state(tmp_path, monkeypatch):
    _set_mirror(tmp_path, monkeypatch)
    page1 = _feed_xml([_record("2609.00001", ["Fri, 4 Sep 2026 09:30:00 GMT"])],
                      token="TOKENABC")
    page2 = _feed_xml([_record("2609.00002", ["Sat, 5 Sep 2026 09:30:00 GMT"])])
    urls = []

    def fake_fetch(url):
        urls.append(url)
        return page1 if len(urls) == 1 else page2

    stats = arxiv_oai.sync("2026-09-01", fetch=fake_fetch)
    assert stats["records"] == 2
    assert stats["pages"] == 2
    assert any("from=2026-09-01" in u for u in urls)
    assert any("resumptionToken=TOKENABC" in u for u in urls)
    assert (tmp_path / "records" / "2609.00001.json").exists()
    assert (tmp_path / "records" / "2609.00002.json").exists()
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["deepest_from"] == "2026-09-01"   # min ever harvested
    assert state["last_from"] == "2026-09-14"      # responseDate - 1 day
    assert state["record_count"] == 2


# --- coverage ----------------------------------------------------------------

def test_ensure_coverage_skips_when_already_covered(tmp_path, monkeypatch):
    _set_mirror(tmp_path, monkeypatch)
    (tmp_path / "state.json").write_text(
        json.dumps({"deepest_from": "2026-08-01"}), encoding="utf-8")
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("sync should not run when coverage suffices")

    monkeypatch.setattr(arxiv_oai, "sync", boom)
    out = arxiv_oai.ensure_coverage(datetime.date(2026, 9, 8))
    assert out["action"] == "cached"
    assert called["n"] == 0


def test_ensure_coverage_bootstraps_when_no_state(tmp_path, monkeypatch):
    _set_mirror(tmp_path, monkeypatch)
    captured = {}

    def fake_sync(from_ds, fetch=None):
        captured["from_ds"] = from_ds
        return {"records": 0, "pages": 1}

    monkeypatch.setattr(arxiv_oai, "sync", fake_sync)
    out = arxiv_oai.ensure_coverage(datetime.date(2026, 9, 8))
    assert out["action"] == "synced"
    expected = (datetime.date(2026, 9, 8)
                - datetime.timedelta(days=arxiv_oai.BACKFILL_DAYS)).isoformat()
    assert captured["from_ds"] == expected


def test_ensure_coverage_deepens_when_window_stale(tmp_path, monkeypatch):
    _set_mirror(tmp_path, monkeypatch)
    (tmp_path / "state.json").write_text(
        json.dumps({"deepest_from": "2026-09-20"}), encoding="utf-8")  # too shallow
    captured = {}

    def fake_sync(from_ds, fetch=None):
        captured["from_ds"] = from_ds
        arxiv_oai._save_state({"deepest_from": from_ds, "last_from": from_ds})
        return {"records": 0, "pages": 1}

    monkeypatch.setattr(arxiv_oai, "sync", fake_sync)
    out = arxiv_oai.ensure_coverage(datetime.date(2026, 9, 8))
    assert out["action"] == "synced"
    assert captured["from_ds"] == "2026-09-08"


# --- window filter / retention -----------------------------------------------

def test_records_for_window_filters_primary_and_date(tmp_path, monkeypatch):
    _set_mirror(tmp_path, monkeypatch)
    _write_record_files(tmp_path, [
        {"id": "1", "primary_category": "cs.AI", "first_submitted": "2026-09-05",
         "last_modified": "2026-09-14", "abstract": "a", "withdrawn": False},
        {"id": "2", "primary_category": "cs.AI", "first_submitted": "2026-09-01",
         "last_modified": "2026-09-14", "abstract": "a", "withdrawn": False},  # out of window
        {"id": "3", "primary_category": "cs.LG", "first_submitted": "2026-09-05",
         "last_modified": "2026-09-14", "abstract": "a", "withdrawn": False},  # cross-list, non-primary
        {"id": "4", "primary_category": "cs.AI", "first_submitted": "2026-09-05",
         "last_modified": "2026-09-14", "abstract": "a", "withdrawn": True},   # withdrawn
    ])
    out = arxiv_oai.records_for_window(datetime.date(2026, 9, 3),
                                       datetime.date(2026, 9, 9))
    assert [r["id"] for r in out] == ["1"]


def test_prune_removes_old_records(tmp_path, monkeypatch):
    _set_mirror(tmp_path, monkeypatch)
    _write_record_files(tmp_path, [
        {"id": "keep1", "first_submitted": "2026-09-05",
         "last_modified": "2026-09-14"},
        {"id": "old1", "first_submitted": "2026-01-05",
         "last_modified": "2026-01-06"},
    ])
    removed = arxiv_oai.prune(datetime.date(2026, 9, 1))
    assert removed == 1
    assert (tmp_path / "records" / "keep1.json").exists()
    assert not (tmp_path / "records" / "old1.json").exists()


# --- collect integration -----------------------------------------------------

def _import_collect():
    from pipeline import collect
    return collect


def test_collect_arxiv_mirror_path_builds_items(monkeypatch, tmp_path):
    collect = _import_collect()
    rec = {"id": "2609.00001", "title": "T", "abstract": "abs",
           "first_submitted": "2026-09-05", "primary_category": "cs.AI",
           "last_modified": "2026-09-14", "withdrawn": False}
    monkeypatch.setattr(collect.arxiv_oai, "ensure_coverage",
                        lambda needed: {"action": "cached"})
    monkeypatch.setattr(collect.arxiv_oai, "records_for_window",
                        lambda s, e: [rec])
    monkeypatch.setattr(collect, "_arxiv_relevant_titles", lambda items: list(items))
    out = collect._collect_arxiv("2026-09-08", datetime.date(2026, 9, 1))
    assert len(out) == 1
    assert out[0].source == "arxiv"
    assert out[0].url == "https://arxiv.org/abs/2609.00001"
    assert out[0].body == "abs"
    assert out[0].date == "2026-09-05"


def test_collect_arxiv_falls_back_when_mirror_fails(monkeypatch, tmp_path):
    collect = _import_collect()
    called = {"n": 0}

    def raise_coverage(needed):
        raise RuntimeError("sync failed")

    monkeypatch.setattr(collect.arxiv_oai, "ensure_coverage", raise_coverage)
    monkeypatch.setattr(collect, "_collect_arxiv_queryapi",
                        lambda date, start: (called.__setitem__("n", called["n"] + 1), [])[1])
    out = collect._collect_arxiv("2026-09-08", datetime.date(2026, 9, 1))
    assert out == []
    assert called["n"] == 1


def test_collect_arxiv_falls_back_when_mirror_empty(monkeypatch, tmp_path):
    collect = _import_collect()
    monkeypatch.setattr(collect.arxiv_oai, "ensure_coverage",
                        lambda needed: {"action": "cached"})
    monkeypatch.setattr(collect.arxiv_oai, "records_for_window",
                        lambda s, e: [])
    called = {"n": 0}

    def fake(date, start):
        called["n"] += 1
        return []

    monkeypatch.setattr(collect, "_collect_arxiv_queryapi", fake)
    out = collect._collect_arxiv("2026-09-08", datetime.date(2026, 9, 1))
    assert out == []
    assert called["n"] == 1
