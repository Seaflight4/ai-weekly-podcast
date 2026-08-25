"""Listener profile loader.

Reads `profile.md` at the repo root: YAML frontmatter + a free-text prose
body passed verbatim to the personal-match prompt.

Two kinds of frontmatter keys:
  - Signal: `topics`, `anti_topics` (lists) — feed the personal-match pass.
  - Style:  scalar knobs (see `STYLE_FIELDS`) — drive the NotebookLM
    instruction template. Style replaces the deleted `plan` stage's frozen
    narrative prompt; the listener tunes style here, not in code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pathlib

PROFILE_PATH = pathlib.Path("profile.md")

# Allowed values per style field. The first entry is the default.
STYLE_FIELDS: dict[str, list[str]] = {
    "knowledge_level":  ["researcher", "undergrad", "expert"],
    "tone":             ["dense", "conversational", "casual"],
    "format":           ["deep_dive", "brief", "critique", "debate"],
    "target_length":    ["default", "short", "long"],
    "intro_style":      ["theme-first", "biggest-story", "bullet"],
    "outro_style":      ["links", "recap", "teaser"],
    "transition_style": ["bridge", "next", "motif"],
}
DEFAULT_TOP_N = 20


@dataclass
class Profile:
    # signal
    topics: list[str] = field(default_factory=list)
    anti_topics: list[str] = field(default_factory=list)
    body: str = ""
    # style (defaults = first entry in STYLE_FIELDS)
    knowledge_level: str = "researcher"
    tone: str = "dense"
    format: str = "deep_dive"
    target_length: str = "default"
    intro_style: str = "theme-first"
    outro_style: str = "links"
    transition_style: str = "bridge"
    top_n: int = DEFAULT_TOP_N


def load_profile(path: pathlib.Path | str | None = None) -> Profile:
    """Parse `profile.md` into a Profile. Returns an empty Profile if the
    file is absent (personalization off — the personal pass gets no signal,
    style falls back to defaults).

    Frontmatter is a minimal hand-rolled parse: list keys (`topics`,
    `anti_topics`) take one `- value` line per item; scalar keys take
    `key: value` inline. No external YAML dependency.
    """
    p = pathlib.Path(path) if path is not None else PROFILE_PATH
    if not p.exists():
        return Profile()

    text = p.read_text(encoding="utf-8")
    if text.startswith("---"):
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
    scalars: dict[str, str] = {}
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
            # scalar `key: value`
            if ":" in stripped:
                k, v = stripped.split(":", 1)
                scalars[k.strip()] = v.strip().strip("\"'")

    def _pick(field_name: str) -> str:
        raw_val = scalars.get(field_name, "")
        allowed = STYLE_FIELDS[field_name]
        if raw_val and raw_val in allowed:
            return raw_val
        return allowed[0]

    top_n = DEFAULT_TOP_N
    try:
        tn = int(scalars.get("top_n", DEFAULT_TOP_N))
        if 1 <= tn <= 100:
            top_n = tn
    except ValueError:
        pass

    return Profile(
        topics=topics, anti_topics=anti, body=body.strip(),
        knowledge_level=_pick("knowledge_level"),
        tone=_pick("tone"),
        format=_pick("format"),
        target_length=_pick("target_length"),
        intro_style=_pick("intro_style"),
        outro_style=_pick("outro_style"),
        transition_style=_pick("transition_style"),
        top_n=top_n,
    )
