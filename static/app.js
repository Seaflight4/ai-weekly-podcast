// AI Weekly Podcast — single-page frontend. No framework, no build step.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
let activeDate = null;
let pollTimer = null;
let confirmResolve = null;
// Persistent podcast config (data/podcast_config.yaml). Null until loaded;
// the app forces the setup dialog when the file doesn't exist yet.
let podcastConfig = null;
let firstRun = true;

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
  const sections = [];
  let cur = null;
  let i = 0;
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

// --- formatting helpers -----------------------------------------------------

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtClock(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return null;
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60), r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

// --- episodes list (single history) ----------------------------------------

async function loadEpisodes() {
  const ul = $("#episodes");
  ul.innerHTML = "";
  const res = await fetch("/api/episodes");
  const eps = await res.json();
  $("#episode-count").textContent = String(eps.length);
  if (eps.length === 0) {
    ul.innerHTML = '<li class="muted empty-hint">No episodes yet. Generate one above.</li>';
    return;
  }
  for (const ep of eps) {
    const li = document.createElement("li");
    li.dataset.date = ep.date;
    if (ep.date === activeDate) li.classList.add("active");
    const badge = ep.status === "ready" ? "ready" : ep.status === "draft" ? "draft" : "empty";
    const dur = fmtClock(ep.duration_sec);
    const meta = [];
    meta.push(`${ep.items} items`);
    if (dur) meta.push(`· ${dur}`);
    else meta.push(`· audio`);
    if (ep.has_transcript) meta.push("· transcript");
    li.innerHTML = `
      <div class="ep-date">${ep.date_label || ep.date} <span class="badge badge-${badge}">${ep.status}</span></div>
      <div class="ep-meta">${meta.join(" ")}</div>`;
    li.onclick = () => selectEpisode(ep.date);
    ul.appendChild(li);
  }
}

function showEmptyState() {
  $("#empty-state").classList.remove("hidden");
  $("#detail-body").classList.add("hidden");
  $("#detail-body").innerHTML = "";
}

function showDetailBody() {
  $("#empty-state").classList.add("hidden");
  $("#detail-body").classList.remove("hidden");
}

// --- episode detail ---------------------------------------------------------

async function selectEpisode(date) {
  activeDate = date;
  $$("#episodes li").forEach((li) =>
    li.classList.toggle("active", li.dataset.date === date));
  const res = await fetch(`/api/episodes/${date}`);
  if (!res.ok) { showEmptyState(); return; }
  const run = await res.json();
  renderDetail(run);
}

function renderDetail(run) {
  showDetailBody();
  const body = $("#detail-body");
  const audio = run.has_audio
    ? `<audio controls preload="metadata" src="/api/episodes/${run.date}/audio?t=${Date.now()}"></audio>`
    : '<p class="muted">No audio for this run.</p>';
  const dur = fmtClock(run.duration_sec);
  const durChip = dur ? `<span class="chip" title="Measured duration">${dur}</span>` : "";
  const srcChip = run.selection_source
    ? `<span class="chip chip-${esc(run.selection_source)}">${esc(run.selection_source)}</span>` : "";
  body.innerHTML = `
    <div class="hero">
      <div class="detail-head">
        <h2>${run.date_label || run.date}</h2>
        <span class="chip">${run.items} items</span>
        ${srcChip}
        ${durChip}
      </div>
      ${audio}
      <div class="hero-actions">
        <button id="btn-personalize" class="ghost">Edit brief</button>
        <button id="btn-delete" class="danger">Delete episode</button>
      </div>
    </div>
    <div class="tabs" id="detail-tabs" role="tablist">
      <button data-dtab="brief" class="active" role="tab">Brief</button>
      ${run.has_transcript ? `<button data-dtab="transcript" role="tab">Transcript</button>` : ""}
    </div>
    <div class="tab-body markdown" id="brief"></div>
    <div class="tab-body markdown hidden" id="transcript"></div>`;
  $("#brief").innerHTML = renderMarkdown(run.brief || "(no brief)");
  if (run.has_transcript) {
    $("#transcript").innerHTML = renderTranscript(run.transcript || "(no transcript)");
  }
  $$("#detail-tabs button").forEach((b) =>
    b.onclick = () => switchDetailTab(b.dataset.dtab));
  $("#btn-personalize").onclick = () => openPersonalize(run);
  $("#btn-delete").onclick = () => deleteEpisode(run);
}

function switchDetailTab(tab) {
  $$("#detail-tabs button").forEach((b) =>
    b.classList.toggle("active", b.dataset.dtab === tab));
  $("#brief").classList.toggle("hidden", tab !== "brief");
  const tr = $("#transcript");
  if (tr) tr.classList.toggle("hidden", tab !== "transcript");
}

// Render a podcast transcript into a dialogue. Lines look like:
//   <Person1>...</Person1>  or  <Person2>...</Person2>
function renderTranscript(md) {
  const lines = md.split("\n");
  let html = "";
  const flushPara = (buf) => { if (buf.length) html += `<p>${esc(buf.join(" "))}</p>`; buf.length = 0; };
  const para = [];
  for (let line of lines) {
    const m = line.match(/^\s*<(Person\d+)>\s*(.*?)\s*<\/\1>\s*$/);
    if (m) {
      flushPara(para);
      const [, who, text] = m;
      const name = who === "Person1" ? "Brian" : "Tina";
      html += `<div class="trn turn"><span class="trn-speaker trn-${who}" title="${esc(who)}">${esc(name.slice(0, 1))}</span><span class="trn-text"><span class="trn-name">${esc(name)}</span>${esc(text)}</span></div>`;
      continue;
    }
    if (line.trim() === "") { flushPara(para); continue; }
    if (/^#{1,6}\s/.test(line)) { flushPara(para); html += renderMarkdown(line); continue; }
    para.push(line);
  }
  flushPara(para);
  return html;
}

async function deleteEpisode(run) {
  const date = run.date;
  const label = run.date_label || run.date;
  const ok = await askConfirm({
    title: "Delete episode",
    message: `Delete the episode for ${label} from history? This cannot be undone.`,
    confirmLabel: "Delete",
    danger: true,
  });
  if (!ok) return;
  const res = await fetch(`/api/episodes/${date}`, { method: "DELETE" });
  if (!res.ok) {
    if (res.status === 404) { showToast("That episode is already gone.", "error"); }
    else { showToast("Delete failed: " + (await res.text()), "error"); }
    return;
  }
  activeDate = null;
  loadEpisodes();
  showToast(`Deleted episode ${label}.`);
  showEmptyState();
}

// Minimal, safe-enough markdown renderer for the brief.
function renderMarkdown(md) {
  const lines = md.split("\n");
  let html = "";
  let inList = false;
  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (let line of lines) {
    if (/^##\s/.test(line)) { closeList(); html += `<h2>${esc(line.replace(/^##\s/, ""))}</h2>`; continue; }
    const m = line.match(/^- \[([^\]]+)\]\(([^)]+)\)(?:\s*·\s*\[PDF\]\(([^)]+)\))?(?:\s*—\s*score\s*([\d.]+))?\s*$/);
    if (m) {
      if (inList) { html += "</ul>"; inList = false; }
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

// --- brief editing + re-render (replaces the episode in place) --------------

async function openPersonalize(run) {
  if (!run.brief) { showToast("No brief for this run.", "error"); return; }
  const parsed = parseBrief(run.brief);
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
  showDetailBody();
  const body = $("#detail-body");
  const allItems = parsed.sections.flatMap((s) => s.items);
  const count = () => allItems.filter((it) => !it.removed).length;
  body.innerHTML = `
    <div class="detail-head perso-head">
      <h2>Edit brief — ${run.date_label || run.date}</h2>
      <span class="count-chip" id="perso-count">${count()} kept</span>
    </div>
    <p class="muted">Delete items to drop them from the audio. "Generate audio" re-renders this episode's mp3 with your selection, replacing it in history.</p>
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
      const src = it.source ? `<span class="tag tag-src">${esc(it.source)}</span>` : "";
      const score = it.score !== null ? ` <span class="tag tag-score">${it.score.toFixed(2)}</span>` : "";
      const badge = it.removed ? '<span class="tag tag-removed">removed</span>' : "";
      div.innerHTML = `
        <div class="main">
          <div class="title"><a href="${esc(it.url)}" target="_blank" rel="noopener">${esc(it.title)}</a>${src}${score}${badge}</div>
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
        $("#perso-count").textContent = `${count()} kept`;
      };
      list.appendChild(div);
    }
  }
  rerender();

  $("#btn-cancel").onclick = () => selectEpisode(run.date);
  $("#btn-generate").onclick = async () => {
    const kept = count();
    if (kept === 0) { showToast("Keep at least one item.", "error"); return; }
    const ok = await askConfirm({
      title: "Re-render audio",
      message: `Re-render the audio for ${run.date_label || run.date} with ${kept} items? The existing episode is replaced.`,
      confirmLabel: "Generate audio",
    });
    if (!ok) return;
    const brief_md = renderBrief(parsed);
    $("#btn-generate").disabled = true;
    const res = await fetch(`/api/runs/${run.date}/generate`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ brief_markdown: brief_md }),
    });
    if (res.status === 409) { showToast("A run is already active. Wait for it to finish.", "error"); $("#btn-generate").disabled = false; return; }
    if (!res.ok) { showToast("Failed to start: " + (await res.text()), "error"); $("#btn-generate").disabled = false; return; }
    const job = await res.json();
    openRunPanel(job);
  };
}

// --- in-app confirm + toasts -----------------------------------------------

function askConfirm({ title, message, confirmLabel = "Confirm", danger = false }) {
  return new Promise((resolve) => {
    confirmResolve = resolve;
    $("#confirm-title").textContent = title || "Are you sure?";
    $("#confirm-message").textContent = message || "";
    const ok = $("#btn-confirm-ok");
    ok.textContent = confirmLabel;
    ok.classList.toggle("danger", danger);
    ok.classList.toggle("primary", !danger);
    openModal("modal-confirm");
  });
}

function settleConfirm(result) {
  if (confirmResolve) { confirmResolve(result); confirmResolve = null; }
  closeModals();
}

function showToast(message, type = "info") {
  const box = $("#toasts");
  const div = document.createElement("div");
  div.className = `toast toast-${type}`;
  div.textContent = message;
  box.appendChild(div);
  setTimeout(() => {
    div.classList.add("toast-out");
    setTimeout(() => div.remove(), 250);
  }, 3200);
}

// --- run panel (live log) ---------------------------------------------------

async function openRunPanel(job) {
  const panel = $("#run-panel");
  panel.classList.remove("hidden");
  $("#run-title").textContent = `${job.kind} run ${job.id} — ${job.status}`;
  renderStages(job, 0);
  if (pollTimer) clearInterval(pollTimer);
  const poll = async () => {
    const res = await fetch(`/api/runs/${job.id}`);
    if (!res.ok) return;
    const j = await res.json();
    $("#run-title").textContent = `${j.kind} run ${j.id} — ${j.status}`;
    renderProgress(j.progress);
    renderStages(j, j.progress.stage_index);
    if (j.status === "done" || j.status === "failed") {
      clearInterval(pollTimer); pollTimer = null;
      $("#run-panel").classList.add(j.status === "done" ? "done" : "failed");
      loadEpisodes();
      if (j.status === "failed") showRunError(j);
      if (j.date) selectEpisode(j.date);
    }
  };
  poll();
  pollTimer = setInterval(poll, 1500);
}

function showRunError(job) {
  const tail = (job.log_tail || "").split("\n").slice(-30).join("\n").trim();
  $("#error-message").textContent = tail || "[no log output]";
  openModal("modal-error");
}

function renderStages(j, current) {
  const dots = $("#progress-stages");
  if (!dots) return;
  const labels = ["", "collecting", "ranking", "generating"];
  const n = j.progress.stage_count || 3;
  dots.innerHTML = Array.from({ length: n }, (_, i) => {
    const idx = i + 1;
    const cls = idx < current ? "done" : idx === current ? "active" : "";
    return `<span class="stage-dot ${cls}" title="${labels[idx] || ""}"></span>`;
  }).join("");
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
    meta.textContent = `${p.stage || "running"}… · ${fmtDur(p.elapsed_sec || 0)} elapsed`;
  }
}
$("#run-close").onclick = () => {
  $("#run-panel").classList.add("hidden");
  $("#run-panel").classList.remove("done", "failed");
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
};

// --- modal plumbing ----------------------------------------------------------

function openModal(id) {
  $("#modal-backdrop").classList.remove("hidden");
  $$(".modal").forEach((m) => m.classList.add("hidden"));
  $("#" + id).classList.remove("hidden");
}

function closeModals() {
  $("#modal-backdrop").classList.add("hidden");
  if (confirmResolve) { confirmResolve(false); confirmResolve = null; }
}

$("#modal-backdrop").addEventListener("click", (e) => {
  if (e.target.id === "modal-backdrop") closeModals();
});
$$("[data-close-modal]").forEach((b) =>
  b.addEventListener("click", () => closeModals()));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("#modal-backdrop").classList.contains("hidden")) {
    closeModals();
  }
});

$("#btn-confirm-cancel").onclick = () => settleConfirm(false);
$("#btn-confirm-ok").onclick = () => settleConfirm(true);
$("#btn-error-ok").onclick = () => closeModals();

// --- persistent config (setup on first run, Settings afterwards) ------------

const CFG_LENGTH_MIN = { short: 10, medium: 17.5, long: 30 };
const CFG_DEPTH_MIN = { brief: 1.0, "deep-dive": 2.0 };
const CFG_INTRO_RECAP = 0.20;

function updateDerivedMeta(lengthSel, depthSel, sourcesId, minutesId) {
  // Mirrors pipeline/config.py budget(): topic time = length minus the 20%
  // intro/recap slice, divided by per-source depth minutes.
  const n = Math.round(CFG_LENGTH_MIN[$(lengthSel).value] * (1 - CFG_INTRO_RECAP) / CFG_DEPTH_MIN[$(depthSel).value]);
  $(sourcesId).textContent = Math.max(4, Math.min(30, n));
  $(minutesId).textContent = Math.round(CFG_LENGTH_MIN[$(lengthSel).value]);
}

function fillSettingsForm() {
  $("#set-audience").value = podcastConfig.user.audience;
  $("#set-familiar").value = (podcastConfig.user.familiar_topics || []).join(", ");
  $("#set-window-days").value = podcastConfig.podcast.window_days;
  $("#set-length").value = podcastConfig.podcast.length;
  $("#set-depth").value = podcastConfig.podcast.depth;
  updateDerivedMeta("#set-length", "#set-depth", "#set-sources", "#set-minutes");
}

function readSettingsForm() {
  return {
    user: {
      audience: $("#set-audience").value,
      familiar_topics: $("#set-familiar").value.split(",").map((s) => s.trim()).filter(Boolean),
    },
    podcast: {
      window_days: parseInt($("#set-window-days").value, 10),
      length: $("#set-length").value,
      depth: $("#set-depth").value,
    },
  };
}

async function loadConfig() {
  const res = await fetch("/api/config");
  const data = await res.json();
  podcastConfig = data.config;
  firstRun = data.first_run;
  if (firstRun) {
    // First use: the user must write the config file before anything runs.
    $("#settings-title").textContent = "Set up your podcast";
    $("#btn-settings-cancel").classList.add("hidden");
    fillSettingsForm();
    openModal("modal-settings");
  }
}

function openSettings() {
  $("#settings-title").textContent = "Podcast settings";
  $("#btn-settings-cancel").classList.remove("hidden");
  fillSettingsForm();
  openModal("modal-settings");
}

$("#btn-settings").onclick = openSettings;
$("#btn-settings-cancel").onclick = () => { if (!firstRun) closeModals(); };
["#set-length", "#set-depth"].forEach((sel) =>
  $(sel).addEventListener("change", () =>
    updateDerivedMeta("#set-length", "#set-depth", "#set-sources", "#set-minutes")));

$("#btn-settings-save").onclick = async () => {
  const body = readSettingsForm();
  const res = await fetch("/api/config", {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) { showToast("Save failed: " + (await res.text()), "error"); return; }
  const data = await res.json();
  podcastConfig = data.config;
  firstRun = false;
  closeModals();
  showToast("Settings saved.");
};

// --- generate new episode dialog ---------------------------------------------

function isoDate(d) { return d.toISOString().slice(0, 10); }

function openGenerate() {
  if (firstRun || !podcastConfig) {
    showToast("Set up your podcast config first.", "info");
    openSettings();
    return;
  }
  const end = new Date();
  const start = new Date(end);
  start.setDate(start.getDate() - podcastConfig.podcast.window_days);
  $("#gen-window-start").value = isoDate(start);
  $("#gen-window-end").value = isoDate(end);
  $("#gen-length").value = podcastConfig.podcast.length;
  $("#gen-depth").value = podcastConfig.podcast.depth;
  updateDerivedMeta("#gen-length", "#gen-depth", "#gen-sources", "#gen-minutes");
  openModal("modal-generate");
}

$("#btn-new").onclick = openGenerate;
$("#btn-empty-new").onclick = openGenerate;
$("#btn-generate-cancel").onclick = closeModals;
["#gen-length", "#gen-depth"].forEach((sel) =>
  $(sel).addEventListener("change", () =>
    updateDerivedMeta("#gen-length", "#gen-depth", "#gen-sources", "#gen-minutes")));

$("#btn-generate-run").onclick = async () => {
  const body = {
    podcast: {
      window_start: $("#gen-window-start").value,
      window_end: $("#gen-window-end").value,
      length: $("#gen-length").value,
      depth: $("#gen-depth").value,
    },
  };
  if (!body.podcast.window_start || !body.podcast.window_end) {
    showToast("Window start and end are required.", "error"); return;
  }
  $("#btn-generate-run").disabled = true;
  const res = await fetch("/api/runs", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  $("#btn-generate-run").disabled = false;
  if (res.status === 409) { showToast("A run is already active.", "error"); return; }
  if (!res.ok) { showToast("Failed to start: " + (await res.text()), "error"); return; }
  closeModals();
  const job = await res.json();
  openRunPanel(job);
};

// --- init ------------------------------------------------------------------

loadConfig();
loadEpisodes();
showEmptyState();
// refresh episode list periodically; manual runs appear here too.
setInterval(loadEpisodes, 10000);
