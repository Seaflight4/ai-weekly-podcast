"""Shared pytest fixtures for the pipeline-app test suite.

Keeps every test hermetic w.r.t. the runtime taxonomy: before each test the
active ``pipeline.topics.TAXONOMY`` is pinned to the in-code curated
``DEFAULT_TAXONOMY`` and restored afterwards. A deployed ``data/taxonomy.json``
(a runtime refresh) changes which ids are valid, and the label/steering/config
tests were written against ``DEFAULT_TAXONOMY``'s ids (e.g. ``post_training``,
``model_release``) — without the pin those tests would silently break whenever
the deployed id set changes. Runtime taxonomy loading/adoption itself is
covered by the dedicated ``test_topics_*`` tests.
"""
import json
import pathlib
import sys

import pytest

_FILE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_FILE.parent))           # tests/
sys.path.insert(0, str(_FILE.parents[1]))       # pipeline-app/
sys.path.insert(0, str(_FILE.parents[2]))       # repo root
sys.path.insert(0, str(_FILE.parents[2] / "podcast-engine"))


@pytest.fixture(autouse=True)
def _pin_taxonomy_to_default(tmp_path):
    """Pin ``pipeline.topics.TAXONOMY`` to the in-code curated set per test."""
    from pipeline import topics as topics_mod
    pristine = topics_mod.TAXONOMY_PATH
    spec = {"taxonomy": [
        {"id": t["id"], "label": t["label"],
         "description": t.get("description", "")}
        for t in topics_mod.DEFAULT_TAXONOMY]}
    pin = tmp_path / "pinned_taxonomy.json"
    pin.write_text(json.dumps(spec), encoding="utf-8")
    topics_mod.reload_taxonomy(pin)
    try:
        yield
    finally:
        topics_mod.reload_taxonomy(pristine)
