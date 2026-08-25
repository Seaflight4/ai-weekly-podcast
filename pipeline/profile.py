"""Listener profile loader.

Reads `profile.md` at the repo root: YAML frontmatter with style fields
that drive the NotebookLM instruction template + the source-set size.

The pipeline ranks items purely by general importance; the profile only
controls narrative style (audience level, pace, format, length).
"""
from __future__ import annotations

from dataclasses import dataclass
import pathlib

PROFILE_PATH = pathlib.Path("profile.md")

# Allowed values per style field. The first entry is the default.
STYLE_FIELDS: dict[str, list[str]] = {
    "knowledge_level":  ["researcher", "undergrad"],
    "pace":             ["deep_dive", "brief"],
    "format":           ["deep_dive", "brief", "critique", "debate"],
    "target_length":    ["default", "short", "long"],
}


@dataclass
class Profile:
    knowledge_level: str = "researcher"
    pace: str = "deep_dive"
    format: str = "deep_dive"
    target_length: str = "default"


def load_profile(path: pathlib.Path | str | None = None) -> Profile:
    """Parse `profile.md` into a Profile. Returns a default Profile if the
    file is absent.

    Frontmatter is a minimal hand-rolled parse: each scalar key takes
    `key: value` inline. No external YAML dependency.
    """
    p = pathlib.Path(path) if path is not None else PROFILE_PATH
    if not p.exists():
        return Profile()

    text = p.read_text(encoding="utf-8")
    if text.startswith("---"):
        rest = text[3:]
        end = rest.find("\n---")
        frontmatter = rest[:end] if end != -1 else rest
    else:
        frontmatter = text

    scalars: dict[str, str] = {}
    for raw in frontmatter.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if ":" in stripped and not stripped.startswith("-"):
            k, v = stripped.split(":", 1)
            scalars[k.strip()] = v.strip().strip("\"'")

    def _pick(field_name: str) -> str:
        raw_val = scalars.get(field_name, "")
        allowed = STYLE_FIELDS[field_name]
        if raw_val and raw_val in allowed:
            return raw_val
        return allowed[0]

    return Profile(
        knowledge_level=_pick("knowledge_level"),
        pace=_pick("pace"),
        format=_pick("format"),
        target_length=_pick("target_length"),
    )
