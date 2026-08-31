// AI Weekly Podcast — single-page frontend. No framework, no build step.

const $ = (sel) => document.querySelector(sel);
let activeDate = null;
let activeTab = "default";   // "default" | "personalized"
let pollTimer = null;
let scheduleCron = null;

// --- brief parsing ---------------------------------------------------------
// Brief format (see pipeline/generate.py _brief_text):
//   ## arXiv papers (N)
//   - [Title](url) · [PDF](pdf_url) — score 0.88
//     > Excerpt.
// We split into a header (everything up to the first ## section), sections,
// and items. Each item keeps its source section so we can re-render the
// brief in the same shape when items are deleted.

function parseBrief(md) {
  const lines = md.split("\n");
  const headerLines = [];
  const sections = []; // {title, raw, items: [{title, url, pdfUrl, score, excerpt, block: [lines]}]}
  let cur = null;
  let i = 0;
  // Header: everything before the first "## " line.
  while (i < lines.length && !lines[i].startsWith("## ")) {
    headerLines.push(lines[i]);
    i++;
  }
  for (; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith("## ")) {
      cur = { title: line, items: [] };
      sections.push(cur);
      continue;
    }
    const m = line.match(/^- \[([^\]]+)\]\(([^)]+)\)(?:\s*·\s*\[PDF\]\(([^)]+)\))?(?:\s*—\s*score\s*([\d.]+))?\s*$/);
    if (m && cur) {
      const item = {
        title: m[1], url: m[2], pdfUrl: m[3] || null,
        score: m[4] ? parseFloat(m[4]) : null,
        excerpt: "", removed: false,
      };
      // excerpt is the next non-empty line starting with "  > " or ">"
      if (i + 1 < lines.length) {
        const ex = lines[i + 1].match(/^\s*>\s*(.+)$/);
        if (ex) { item.excerpt = ex[1]; i++; }
      }
      cur.items.push(item);
    }
  }
  return { header: headerLines.join("\n"), sections };
}

function renderBrief(parsed) {
  const out = [parsed.header.trimEnd(), ""];
  for (const s of parsed.sections) {
    const kept = s.items.filter((it) => !it.removed);
    if (kept.length === 0) continue;
    out.push(s.title.replace(/\(\d+\)$/, `(${kept.length})`), "");
    for (const it of kept) {
      let line = `- [${it.title}](${it.url})`;
      if (it.pdfUrl) line += ` · [PDF](${it.pdfUrl})`;
      if (it.score !== null) line += ` — score ${it.score.toFixed(2)}`;
      out.push(line);
      if (it.excerpt) out.push(`  > ${it.excerpt}`);
    }
    out.push("");
  }
  return out.join("\n").trimEnd() + "\n";
}

// --- episodes list (tab-aware) --------------------------------------------

function apiBase() {
  return activeTab === "personalized" ? "/api/personalized" : "/api/episodes";
}

async function loadEpisodes() {
  const ul = $("#episodes");
  ul.innerHTML = "";
  const res = await fetch(apiBase());
  const eps = await res.json();
  if (eps.length === 0) {
    ul.innerHTML = activeTab === "personalized"
      ? '<li class="muted">No personalized renders yet.</li>'
      : '<li class="muted">No episodes yet. Generate one.</li>';
    return;
  }
  for (const ep of eps) {
    const li = document.createElement("li");
    li.dataset.date = ep.date;
    if (ep.date === activeDate) li.classList.add("active");
    const badge = ep.status === "ready" ? "ready" : ep.status === "draft" ? "draft" : "empty";
    li.innerHTML = `
      <div class="ep-date">${ep.date} <span class="badge badge-${badge}">${ep.status}</span></div>
      <div class="ep-meta">${ep.items} items${ep.has_audio ? " · audio" : ""}${ep.has_transcript ? " · transcript" : ""}</div>`;
    li.onclick = () => selectEpisode(ep.date);
    ul.appendChild(li);
  }
}

// --- episode detail (tab-aware) -------------------------------------------

async function selectEpisode(date) {
  activeDate = date;
  document.querySelectorAll("#episodes li").forEach((li) =>
    li.classList.toggle("active", li.dataset.date === date));
  const res = await fetch(`${apiBase()}/${date}`);
  if (!res.ok) { $("#detail").innerHTML = "<p>Not found.</p>"; return; }
  const run = await res.json();
  renderDetail(run);
}

function audioUrl(date) {
  return `${apiBase()}/${date}/audio`;
}

function renderDetail(run) {
  const sec = $("#detail");
  const audio = run.has_audio
    ? `<audio controls preload="metadata" src="${audioUrl(run.date)}"></audio>`
    : '<p class="muted">No audio for this run.</p>';
  // Only default-tab episodes can be personalized (personalized renders are
  // already curated; re-personalizing from a personalized render is out of scope).
  const personalizeBtn = activeTab === "default"
    ? '<button id="btn-personalize" class="primary">Personalize</button>' : "";
  // Personalized renders can be deleted; default episodes are shared, so no delete.
  const deleteBtn = activeTab === "personalized"
    ? '<button id="btn-delete" class="danger">Delete</button>' : "";
  sec.innerHTML = `
    <div class="detail-head">
      <h2>${run.date}</h2>
      <span class="muted">${run.items} items · ${run.selection_source || "auto"}</span>
      ${personalizeBtn}
      ${deleteBtn}
    </div>
    ${audio}
    <div class="tab-body markdown" id="brief"></div>`;
  $("#brief").innerHTML = renderMarkdown(run.brief || "(no brief)");
  if ($("#btn-personalize")) $("#btn-personalize").onclick = () => openPersonalize(run);
  if ($("#btn-delete")) $("#btn-delete").onclick = () => deletePersonalized(run.date);
}

async function deletePersonalized(date) {
  if (!confirm(`Delete the personalized render for ${date}? This cannot be undone.`)) return;
  const res = await fetch(`/api/personalized/${date}`, { method: "DELETE" });
  if (!res.ok) {
    if (res.status === 404) { alert("Already gone."); }
    else { alert("Delete failed: " + (await res.text())); }
    return;
  }
  activeDate = null;
  loadEpisodes();
  $("#detail").innerHTML = '<p class="muted">Select an episode on the left.</p>';
}

// Minimal, safe-enough markdown renderer for the brief.
// Handles: ## headings, bullet list items with links (incl. · [PDF](url)),
// blockquote excerpts, and paragraphs. Links are made clickable in a new tab.
function renderMarkdown(md) {
  const lines = md.split("\n");
  let html = "";
  let inList = false;
  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (let line of lines) {
    if (/^##\s/.test(line)) { closeList(); html += `<h2>${esc(line.replace(/^##\s/, ""))}</h2>`; continue; }
    const m = line.match(/^- \[([^\]]+)\]\(([^)]+)\)(?:\s*·\s*\[PDF\]\(([^)]+)\))?(?:\s*—\s*score\s*([\d.]+))?\s*$/);
    if (m) {
      if (!inList) { html += "<ul class=\"brief-items\">"; inList = true; }
      const [_, title, url, pdfUrl, score] = m;
      let item = `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(title)}</a>`;
      if (pdfUrl) item += ` · <a class="pdf" href="${esc(pdfUrl)}" target="_blank" rel="noopener">PDF</a>`;
      if (score) item += ` <span class="score">— score ${score}</span>`;
      html += `<li>${item}</li>`;
      continue;
    }
    if (/^\s*>\s/.test(line)) {
      closeList();
      html += `<blockquote>${esc(line.replace(/^\s*>\s/, ""))}</blockquote>`;
      continue;
    }
    if (line.trim() === "") { closeList(); continue; }
    closeList();
    html += `<p>${esc(line)}</p>`;
  }
  closeList();
  return html;
}

// --- personalization -------------------------------------------------------

async function openPersonalize(run) {
  if (!run.brief) { alert("No brief for this run."); return; }
  const parsed = parseBrief(run.brief);
  // match rank.json to get judge_reason per URL
  const rank = run.rank || [];
  const byUrl = new Map(rank.map((r) => [r.url, r]));
  for (const s of parsed.sections) {
    for (const it of s.items) {
      const r = byUrl.get(it.url);
      if (r) { it.judge_reason = r.judge_reason; it.source = r.source; }
    }
  }
  renderPersonalize(parsed, run);
}

function renderPersonalize(parsed, run) {
  const sec = $("#detail");
  const allItems = parsed.sections.flatMap((s) => s.items);
  const count = () => allItems.filter((it) => !it.removed).length;
  sec.innerHTML = `
    <div class="detail-head">
      <h2>Personalize — ${run.date}</h2>
      <span class="muted" id="perso-count">${count()} items kept</span>
    </div>
    <p class="muted">Delete items to drop them from the audio. "Generate audio" re-renders episode.mp3 with your selection.</p>
    <div class="perso-list" id="perso-list"></div>
    <div class="perso-actions-row">
      <button id="btn-generate" class="primary">Generate audio</button>
      <button id="btn-cancel">Cancel</button>
    </div>`;

  function rerender() {
    const list = $("#perso-list");
    list.innerHTML = "";
    for (const it of allItems) {
      const div = document.createElement("div");
      div.className = "perso-item" + (it.removed ? " removed" : "");
      const reason = it.judge_reason ? `<div class="reason">${esc(it.judge_reason)}</div>` : "";
      const src = it.source ? `<span class="source-tag">${it.source}</span>` : "";
      const score = it.score !== null ? ` <span class="score">${it.score.toFixed(2)}</span>` : "";
      div.innerHTML = `
        <div class="main">
          <div class="title"><a href="${esc(it.url)}" target="_blank" rel="noopener">${esc(it.title)}</a>${src}${score}</div>
          ${reason}
        </div>
        <div class="perso-actions">
          <button class="${it.removed ? "" : "danger"}">${it.removed ? "restore" : "delete"}</button>
        </div>`;
      div.querySelector("button").onclick = () => {
        it.removed = !it.removed;
        div.classList.toggle("removed", it.removed);
        div.querySelector("button").textContent = it.removed ? "restore" : "delete";
        div.querySelector("button").classList.toggle("danger", !it.removed);
        $("#perso-count").textContent = `${count()} items kept`;
      };
      list.appendChild(div);
    }
  }
  rerender();

  $("#btn-cancel").onclick = () => selectEpisode(run.date);
  $("#btn-generate").onclick = async () => {
    const kept = count();
    if (kept === 0) { alert("Keep at least one item."); return; }
    if (!confirm(`Render audio with ${kept} items into the personalized library?`)) return;
    const brief_md = renderBrief(parsed);
    $("#btn-generate").disabled = true;
    const res = await fetch(`/api/runs/${run.date}/generate`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ brief_markdown: brief_md }),
    });
    if (res.status === 409) { alert("A run is already active. Wait for it to finish."); $("#btn-generate").disabled = false; return; }
    if (!res.ok) { alert("Failed to start: " + (await res.text())); $("#btn-generate").disabled = false; return; }
    const job = await res.json();
    // Switch to the personalized tab so the run panel's completion handler
    // selects the freshly-rendered episode from the personalized library.
    switchTab("personalized");
    openRunPanel(job);
  };
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// --- run panel (live log) --------------------------------------------------

async function openRunPanel(job) {
  const panel = $("#run-panel");
  panel.classList.remove("hidden");
  $("#run-title").textContent = `${job.kind} run ${job.id} — ${job.status}`;
  $("#run-log").textContent = "";
  if (pollTimer) clearInterval(pollTimer);
  const poll = async () => {
    const res = await fetch(`/api/runs/${job.id}`);
    if (!res.ok) return;
    const j = await res.json();
    $("#run-title").textContent = `${j.kind} run ${j.id} — ${j.status}`;
    renderProgress(j.progress);
    $("#run-log").textContent = j.log_tail || "";
    const log = $("#run-log");
    log.scrollTop = log.scrollHeight;
    if (j.status === "done" || j.status === "failed") {
      clearInterval(pollTimer); pollTimer = null;
      loadEpisodes();
      if (j.date) selectEpisode(j.date);
    }
  };
  poll();
  pollTimer = setInterval(poll, 1500);
}

function fmtDur(sec) {
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60), s = sec % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function renderProgress(p) {
  if (!p) return;
  const bar = $("#progress-fill");
  const meta = $("#progress-meta");
  if (bar) bar.style.width = `${Math.round((p.fraction || 0) * 100)}%`;
  if (meta) {
    const eta = p.eta_sec > 0 ? ` · ~${fmtDur(p.eta_sec)} left` : "";
    meta.textContent = `${p.stage || "running"}… · ${fmtDur(p.elapsed_sec || 0)} elapsed${eta}`;
  }
}
$("#run-close").onclick = () => { $("#run-panel").classList.add("hidden"); if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } };

// --- new episode button -----------------------------------------------------

$("#btn-new").onclick = async () => {
  if (!confirm("Run collect → rank → generate now? This takes several minutes.")) return;
  const res = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  if (res.status === 409) { alert("A run is already active."); return; }
  if (!res.ok) { alert("Failed to start: " + (await res.text())); return; }
  const job = await res.json();
  openRunPanel(job);
};

// --- tabs (Default / My library) ------------------------------------------

function switchTab(tab) {
  activeTab = tab;
  activeDate = null;
  document.querySelectorAll(".list-tabs button").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === tab));
  loadEpisodes();
  $("#detail").innerHTML = '<p class="muted">Select an episode on the left.</p>';
}
document.querySelectorAll(".list-tabs button").forEach((b) =>
  b.onclick = () => switchTab(b.dataset.tab));

// --- schedule ---------------------------------------------------------------

async function loadSchedule() {
  const res = await fetch("/api/schedule");
  const s = await res.json();
  scheduleCron = s.cron;
  $("#schedule-info").textContent = s.enabled
    ? `auto-run: ${s.next_fire ? new Date(s.next_fire).toLocaleString() : s.cron}`
    : "auto-run: off";
  $("#schedule-toggle").checked = s.enabled;
}
$("#schedule-toggle").onchange = async (e) => {
  await fetch("/api/schedule", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled: e.target.checked }) });
  loadSchedule();
};

// --- init ------------------------------------------------------------------

loadEpisodes();
loadSchedule();
// refresh episode list periodically so scheduled runs show up.
setInterval(loadEpisodes, 10000);
