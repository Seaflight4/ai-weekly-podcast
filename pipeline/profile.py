"""Listener profile loader.

Reads `profile.md` at the repo root: YAML frontmatter (`topics`, `anti_topics`)
+ a free-text prose body passed verbatim to the personal-match prompt.

This is the *only* content the listener tunes (per the no-prompt-chasing rule).
The personal-match prompt in `rank.py` is fixed; this file is the variable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pathlib

PROFILE_PATH = pathlib.Path("profile.md")


@dataclass
class Profile:
    topics: list[str] = field(default_factory=list)
    anti_topics: list[str] = field(default_factory=list)
    body: str = ""


def load_profile(path: pathlib.Path | str | None = None) -> Profile:
    """Parse `profile.md` into a Profile. Returns an empty Profile if the
    file is absent (personalization off — the personal pass gets no signal).

    `path` defaults to the current module-level `PROFILE_PATH` (resolved at
    call time so tests can monkeypatch it).

    Frontmatter is a minimal hand-rolled parse: only `topics:` and
    `anti_topics:` list keys, one list item per `- value` line until a blank
    line or non-indented line. No external YAML dependency for two list keys.
    """
    p = pathlib.Path(path) if path is not None else PROFILE_PATH
    if not p.exists():
        return Profile()

    text = p.read_text(encoding="utf-8")
    # Strip leading `---` comment lines from the frontmatter block so the
    # loader treats `#`-prefixed lines as comments, not list items.
    if text.startswith("---"):
        # find the closing `---` on its own line
        rest = text[3:]
        end = rest.find("\n---")
        if end != -1:
            frontmatter = rest[:end]
            body = rest[end + 4:].lstrip("\n")
        else:
            frontmatter, body = rest, ""
    else:
        frontmatter, body = text, ""

    topics: list[str] = []
    anti: list[str] = []
    current: list[str] | None = None
    for raw in frontmatter.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped.startswith("topics:"):
            current = topics
            inline = stripped[len("topics:"):].strip()
            if inline and inline not in ("[]",):
                current.append(inline.strip("- ").strip())
        elif stripped.startswith("anti_topics:"):
            current = anti
            inline = stripped[len("anti_topics:"):].strip()
            if inline and inline not in ("[]",):
                current.append(inline.strip("- ").strip())
        elif stripped.startswith("-"):
            if current is not None:
                current.append(stripped.lstrip("- ").strip())
        else:
            current = None

    return Profile(topics=topics, anti_topics=anti, body=body.strip())
