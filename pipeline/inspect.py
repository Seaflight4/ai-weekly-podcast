"""Inspection reports — human-reviewable artifacts written next to each
stage's JSON output.

These are the *quality gates* for an LLM-driven (non-deterministic) pipeline.
Contract tests (see `tests/`) check JSON shapes and invariants; these reports
make the *content* reviewable so a human can spot drift, a misfiring gate, or a
collapsed cluster.

Each report writes `data/<run>/<stage>_report.md`. All writers are pure
functions over JSON-shaped data (no network, no LLM) so they can be re-run
cheaply on any cached artifact.
"""
from __future__ import annotations

from . import store
import json, pathlib

# --- helpers -------------------------------------------------------------

def _report_path(name: str) -> pathlib.Path:
    return store.run_dir() / name

def _write(name: str, body: str) -> pathlib.Path:
    p = _report_path(name)
    p.write_text(body, encoding="utf-8")
    print(f"      wrote {p.name}")
    return p

def _short(text: str, n: int = 120) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n - 1] + "…"

def _by_source(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for it in items:
        counts[it.get("source", "?")] = counts.get(it.get("source", "?"), 0) + 1
    return counts

def _ratio(counts: dict[str, int]) -> str:
    a = counts.get("arxiv", 0)
    h = counts.get("hn", 0)
    if h == 0:
        return f"{a}:0"
    from math import gcd
    g = gcd(a, h)
    return f"{a // g}:{h // g}" if g else f"{a}:{h}"

# --- 01: arXiv gate report ------------------------------------------------

def gate_report(collect_path: pathlib.Path, kept_sources: dict[str, int] | None = None) -> pathlib.Path:
    """Pass-rate + 5 kept / 5 dropped samples.

    `kept_sources` is optional context (the per-source counts after gating);
    without it we just describe the file on disk.
    """
    items = json.loads(collect_path.read_text())
    arxiv = [i for i in items if i.get("source") == "arxiv"]
    hn = [i for i in items if i.get("source") == "hn"]
    lines = [
        f"# arXiv gate report",
        "",
        f"- arXiv survivors: **{len(arxiv)}**",
        f"- HN survivors: **{len(hn)}**",
        f"- source ratio (arxiv:hn): **{_ratio(_by_source(items))}**",
        "",
        "## 5 arXiv survivors (samples)",
        "",
    ]
    for it in arxiv[:5]:
        lines.append(f"- **{_short(it.get('title', ''))}** — {_short(it.get('body', ''), 160)}")
    lines += ["", "## 5 HN survivors (samples)", ""]
    for it in hn[:5]:
        lines.append(f"- **{_short(it.get('title', ''))}** — {it.get('url', '')}")
    return _write("collect_report.md", "\n".join(lines))

# --- 02: rubric A/B report ------------------------------------------------

def rubric_ab_report(ranked_unified: list[dict], ranked_separated: list[dict],
                     top_n: int = 50) -> pathlib.Path:
    """Compare unified vs separated rubric arms on the same collect.json.

    Headline metric: paper:HN ratio in the top-N pool. Also reports per-source
    score histograms, top-N overlap, and which HN stories surface only under
    unified (the whole point of switching rubrics).
    """
    def _top(ranked, n):
        return sorted(ranked, key=lambda r: r.get("score", 0), reverse=True)[:n]

    u_top = _top(ranked_unified, top_n)
    s_top = _top(ranked_separated, top_n)
    u_counts = _by_source(u_top)
    s_counts = _by_source(s_top)

    def _hist(items, src):
        buckets = {"<0.4": 0, "0.4-0.69": 0, "0.7-0.89": 0, "0.9+": 0}
        for it in items:
            if it.get("source") != src:
                continue
            sc = it.get("score", 0)
            if sc >= 0.9:
                buckets["0.9+"] += 1
            elif sc >= 0.7:
                buckets["0.7-0.89"] += 1
            elif sc >= 0.4:
                buckets["0.4-0.69"] += 1
            else:
                buckets["<0.4"] += 1
        return buckets

    u_urls = {it["url"] for it in u_top}
    s_urls = {it["url"] for it in s_top}
    overlap = u_urls & s_urls
    only_unified = u_urls - s_urls
    only_separated = s_urls - u_urls
    unified_only_hn = [it for it in u_top if it["url"] in only_unified and it.get("source") == "hn"]

    lines = [
        "# Rubric A/B report",
        "",
        f"## Top-{top_n} source balance",
        "",
        "| arm | arxiv | hn | ratio |",
        "|-----|-------|----|-------|",
        f"| unified | {u_counts.get('arxiv', 0)} | {u_counts.get('hn', 0)} | {_ratio(u_counts)} |",
        f"| separated | {s_counts.get('arxiv', 0)} | {s_counts.get('hn', 0)} | {_ratio(s_counts)} |",
        "",
        "## Per-source score histograms (top-%d)" % top_n,
        "",
        "| arm | source | <0.4 | 0.4-0.69 | 0.7-0.89 | 0.9+ |",
        "|-----|--------|------|-----------|----------|------|",
    ]
    for arm, ranked in (("unified", u_top), ("separated", s_top)):
        for src in ("arxiv", "hn"):
            h = _hist(ranked, src)
            lines.append(f"| {arm} | {src} | {h['<0.4']} | {h['0.4-0.69']} | {h['0.7-0.89']} | {h['0.9+']} |")
    lines += [
        "",
        f"## Top-{top_n} overlap",
        "",
        f"- both arms: **{len(overlap)}** items",
        f"- unified-only: **{len(only_unified)}** items",
        f"- separated-only: **{len(only_separated)}** items",
        "",
        "## HN stories surfacing only under unified (the win condition)",
        "",
    ]
    for it in unified_only_hn:
        lines.append(f"- **{_short(it.get('title', ''))}** (score {it.get('score', 0):.2f}) — {it.get('url', '')}")
    if not unified_only_hn:
        lines.append("_(none — unified did not surface new HN stories in the top)_")
    return _write("rank_report.md", "\n".join(lines))

# --- Phase 2: personalization A/B report (ticket 08) ---------------------

def personal_ab_report(plain: list[dict], personal: list[dict],
                       top_n: int = 30) -> pathlib.Path:
    """Compare plain (ALPHA=1.0) vs. personal arms on the same cached pool.

    The **primary tuning digest**: shows which items each arm surfaced/demoted
    so the listener can see what personalization is doing. Per top-N item:
    rank, title, source, importance, personal, final score, which arm surfaced it.
    """
    p_top = sorted(plain, key=lambda r: r.get("final_score", r.get("score", 0)),
                   reverse=True)[:top_n]
    s_top = sorted(personal, key=lambda r: r.get("final_score", 0), reverse=True)[:top_n]
    p_urls = {r["url"] for r in p_top}
    s_urls = {r["url"] for r in s_top}
    overlap = p_urls & s_urls
    plain_only = p_urls - s_urls
    personal_only = s_urls - p_urls

    alpha = personal[0].get("alpha", 0.7) if personal else 0.7

    def _row(r, rank):
        sc = r.get("score", 0)
        ps = r.get("personal_score", 0)
        fs = r.get("final_score", sc)
        return (f"| {rank} | {_short(r.get('title', ''), 70)} | "
                f"{r.get('source', '?')} | {sc:.2f} | {ps:.2f} | {fs:.2f} |")

    lines = [
        "# Personalization A/B report",
        "",
        f"- ALPHA (personal arm): **{alpha}**",
        f"- top-N compared: **{top_n}**",
        f"- both arms: **{len(overlap)}** items",
        f"- plain-only: **{len(plain_only)}** items",
        f"- personal-only: **{len(personal_only)}** items",
        "",
        "## Personal arm top-N (the one you're tuning toward)",
        "",
        "| rank | title | src | imp | pers | final |",
        "|------|-------|-----|-----|------|-------|",
    ]
    for i, r in enumerate(s_top, 1):
        lines.append(_row(r, i))
    lines += [
        "",
        "## Plain arm top-N (importance only, Phase 1 baseline)",
        "",
        "| rank | title | src | imp | pers | final |",
        "|------|-------|-----|-----|------|-------|",
    ]
    for i, r in enumerate(p_top, 1):
        lines.append(_row(r, i))
    lines += [
        "",
        "## Items the personal arm surfaced (personal-only)",
        "",
    ]
    surfaced = [r for r in s_top if r["url"] in personal_only]
    if surfaced:
        for r in surfaced:
            sc = r.get("score", 0)
            ps = r.get("personal_score", 0)
            fs = r.get("final_score", sc)
            lines.append(f"- **{_short(r.get('title', ''), 80)}** — "
                         f"imp {sc:.2f} · pers {ps:.2f} · final {fs:.2f} — {r.get('url', '')}")
    else:
        lines.append("_(none — personal arm did not surface new items in the top)_")
    lines += [
        "",
        "## Items the personal arm demoted out of top-N (plain-only)",
        "",
    ]
    demoted = [r for r in p_top if r["url"] in plain_only]
    if demoted:
        for r in demoted:
            sc = r.get("score", 0)
            lines.append(f"- **{_short(r.get('title', ''), 80)}** — imp {sc:.2f} — {r.get('url', '')}")
    else:
        lines.append("_(none — personal arm kept every plain-top item)_")
    return _write("rank_report.md", "\n".join(lines))

# --- 04/05: cluster + merge report ---------------------------------------

def cluster_report(topics: list[dict]) -> pathlib.Path:
    """Cluster table with pure-paper / pure-HN / mixed flags + merge diff.

    `topics` is the *final* (post-merge) `cluster.json` content. The merge
    before/after is reconstructed from each topic's members when available.
    """
    lines = ["# Cluster report", "", "## Topics", ""]
    mixed = 0
    for t in topics:
        members = t.get("members", [])
        sources = {m.get("source") for m in members}
        if len(sources) > 1:
            flag = "mixed"
            mixed += 1
        elif sources == {"arxiv"}:
            flag = "pure-paper"
        elif sources == {"hn"}:
            flag = "pure-hn"
        else:
            flag = "empty"
        lines.append(f"### {t.get('id')} — {t.get('title', '(unlabeled)')}  [{flag}]")
        lines.append(f"- why: {_short(t.get('why', ''))}")
        lines.append(f"- aggregate score: {t.get('aggregate_score', 0):.2f}")
        lines.append(f"- primary: {t.get('primary_url', '')}")
        lines.append("- members:")
        for m in members:
            lines.append(f"  - [{m.get('source')}] score {m.get('score', 0):.2f} — {_short(m.get('title', ''))}")
        lines.append("")
    lines += [
        "## Cross-referencing metric",
        "",
        f"- mixed-source topics (paper + HN together): **{mixed}** of {len(topics)}",
    ]
    return _write("cluster_report.md", "\n".join(lines))

def merge_report(before: list[dict], after: list[dict]) -> pathlib.Path:
    """Before/after the cross-cluster LLM merge pass.

    `before` = MiniLM candidate clusters (pre-merge); `after` = final topics.
    Each `before` topic that is no longer top-level in `after` was merged.
    """
    after_ids = {t.get("id") for t in after}
    merged = [t for t in before if t.get("id") not in after_ids]
    lines = [
        "# Merge report",
        "",
        f"- candidate clusters (pre-merge): **{len(before)}**",
        f"- final topics (post-merge): **{len(after)}**",
        f"- merged away: **{len(merged)}**",
        "",
        "## Merged clusters (no longer top-level after the merge pass)",
        "",
    ]
    for t in merged:
        members = ", ".join(_short(m.get("title", ""), 60) for m in t.get("members", []))
        lines.append(f"- {t.get('id')}: {members}")
    if not merged:
        lines.append("_(none — the merge pass did not combine any candidate clusters)_")
    return _write("merge_report.md", "\n".join(lines))

# --- 03/06: budget report -------------------------------------------------

def budget_report(ranked: list[dict], plan: dict, knee_index: int | None = None) -> pathlib.Path:
    """Score distribution with the knee marked + fill-order table."""
    scores = sorted((r.get("score", 0) for r in ranked), reverse=True)
    lines = [
        "# Budget report",
        "",
        f"- scored items: **{len(scores)}**",
        f"- knee index (candidate pool cut): **{knee_index}**" if knee_index is not None else "- knee index: (not computed)",
        f"- target minutes: **{plan.get('target_minutes', 0)}**",
        f"- planned minutes: **{sum(s.get('minutes', 0) for s in plan.get('segments', []))}**",
        "",
        "## Score distribution (top 60)",
        "",
        "| rank | score |",
        "|------|-------|",
    ]
    for i, sc in enumerate(scores[:60], 1):
        mark = "  <-- knee" if knee_index is not None and i == knee_index + 1 else ""
        lines.append(f"| {i} | {sc:.2f}{mark} |")
    lines += ["", "## Fill order", "", "| # | topic | minutes | cumulative |", "|---|-------|---------|-------------|"]
    cum = 0.0
    for i, seg in enumerate(plan.get("segments", []), 1):
        cum += seg.get("minutes", 0)
        lines.append(f"| {i} | {seg.get('topic_id')} | {seg.get('minutes', 0):.1f} | {cum:.1f} |")
    return _write("budget_report.md", "\n".join(lines))

# --- 06/07: plan + brief report ------------------------------------------

def plan_report(plan: dict) -> pathlib.Path:
    segs = plan.get("segments", [])
    groups = plan.get("editorial_groups", [])
    lines = [
        "# Narrative plan report",
        "",
        f"- date: {plan.get('date')}",
        f"- target minutes: {plan.get('target_minutes')}",
        f"- planned minutes: **{sum(s.get('minutes', 0) for s in segs)}**",
        f"- motif: *{_short(plan.get('motif', ''), 160)}*",
    ]
    if groups:
        # Role distribution summary
        role_counts: dict[str, int] = {}
        for g in groups:
            role_counts[g.get("role", "?")] = role_counts.get(g.get("role", "?"), 0) + 1
        role_str = ", ".join(f"{r}:{n}" for r, n in sorted(role_counts.items()))
        lines.append(f"- editorial groups: **{len(groups)}** ({role_str})")
    lines += [
        "",
        "## Hook",
        "",
        plan.get("hook", ""),
    ]
    # Editorial groups section (ticket 13)
    if groups:
        lines += ["", "## Editorial meta-groups", ""]
        for g in groups:
            lines.append(f"### {g.get('label', '?')}  [role: {g.get('role', '?')}]")
            lines.append(f"- opener: {g.get('opener', '')}")
            tids = g.get("topic_ids", [])
            lines.append(f"- topics ({len(tids)}): {', '.join(tids)}")
            lines.append("")
    lines += [
        "",
        "## Segments",
        "",
    ]
    for i, seg in enumerate(segs, 1):
        grp = seg.get("editorial_group", "")
        grp_tag = f"  [{grp}]" if grp else ""
        lines.append(f"### {i}. {seg.get('topic_id')} ({seg.get('minutes', 0):.1f} min){grp_tag}")
        lines.append(f"- opening: {seg.get('opening', '')}")
        lines.append(f"- signposts: {'; '.join(seg.get('signposts', []))}")
        lines.append(f"- transition: {seg.get('transition_out', '')}")
        lines.append(f"- speakers: {', '.join(seg.get('speakers', []))}")
        lines.append("")
    lines += ["## Outro", "", plan.get("outro", "")]
    return _write("plan_report.md", "\n".join(lines))
