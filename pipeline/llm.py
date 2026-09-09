"""Shared LLM client + JSON parser for all stages.

Centralizes the OpenAI-compatible client (lazy singleton), a single `_chat`
helper, and the robust JSON extractor used by every stage that talks to the
judge backend. Stages import from here instead of from `rank.py`.
"""
from __future__ import annotations

import json
import os
import re

from openai import OpenAI

SKAINET_BASE_URL = "https://chat.model.tngtech.com/v1/"

# Fixed judge/rank model. Model choices are pinned constants per stage (not
# configurable): collect gates use collect.RELEVANCE_MODEL, the transcript LLM
# uses podcastfy.generator.DEFAULT_MODEL, TTS is TNG qwen3, and the rank judge
# is this one.
JUDGE_MODEL = "Qwen/Qwen3.8-27B"


def _config() -> dict:
    """Read backend config lazily (not at import time) so a misconfigured env
    only errors when a stage actually runs."""
    key = os.environ.get("SKAINET_API_KEY")
    if not key:
        raise SystemExit("SKAINET_API_KEY not set (required for LLM calls)")
    return {
        "api_key": key,
        "base_url": SKAINET_BASE_URL,
        "default_model": JUDGE_MODEL,
    }


_client = None


def get_client() -> OpenAI:
    """Lazy singleton OpenAI client pointed at the SkAInet backend.

    A generous client-wide timeout prevents a stalled gateway request from
    hanging the process forever (a stuck request otherwise blocks with no
    bound); per-call ``timeout`` overrides can still be passed where a
    tighter bound is wanted.
    """
    global _client
    if _client is None:
        cfg = _config()
        _client = OpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            default_headers={"x-user-agent": "tng/practice-judge"},
            timeout=300.0,
        )
    return _client


def chat(user_msg: str, system_msg: str, model: str | None = None,
         temperature: float = 0.0) -> str:
    """One chat completion. `model` defaults to JUDGE_MODEL."""
    cfg = _config()
    resp = get_client().chat.completions.create(
        model=model or cfg["default_model"],
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def parse_json(raw: str) -> dict:
    """Extract the first valid JSON object from a model response.

    Strips markdown fences, then scans for a brace-matched object. Raises
    ValueError if no parseable object is found.
    """
    s = raw.strip()
    s = re.sub(r"```.*?```", "", s, flags=re.DOTALL)
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    start = s.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in model response:\n{raw}")
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                cand = s[start : i + 1]
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"found braces but no valid JSON object in model response:\n{raw}")
