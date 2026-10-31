/* arXiv Paper Curator — chat UI.
   Classic script (not a module) so inline-referenced handlers stay on window. */

const API = "/api/v1";
let mode = "ask";
let busy = false;

const $ = (id) => document.getElementById(id);
const thread = $("thread"), input = $("input"), sendBtn = $("send"), main = $("main");

/* ══════════════════════════════════════════════════════════════════
   Storage — conversations live in this browser.

   The chat id IS the thread_id. That's the whole trick: resuming a chat
   replays its transcript locally AND makes the backend load the real
   checkpointed messages for that thread, so follow-ups keep working.
   ══════════════════════════════════════════════════════════════════ */
const IDX_KEY = "apc.chats.v1";
const CHAT_KEY = (id) => "apc.chat." + id;
const MAX_CHATS = 50;

// Safari private mode throws on setItem. Detect once and degrade to memory.
const canStore = (() => {
  try { localStorage.setItem("apc.t", "1"); localStorage.removeItem("apc.t"); return true; }
  catch { return false; }
})();
const memStore = {};

function readJSON(key, fallback) {
  try {
    const raw = canStore ? localStorage.getItem(key) : memStore[key];
    return raw ? JSON.parse(raw) : fallback;
  } catch { return fallback; }   // one corrupt write must not brick the app
}

function writeJSON(key, value) {
  const raw = JSON.stringify(value);
  if (!canStore) { memStore[key] = raw; return; }
  try {
    localStorage.setItem(key, raw);
  } catch (e) {
    // Quota hit — drop the oldest chat and try once more, then give up quietly.
    if (e && (e.name === "QuotaExceededError" || e.code === 22)) {
      const idx = loadIndex();
      const oldest = idx.chats.sort((a, b) => a.updatedAt - b.updatedAt)[0];
      if (oldest) {
        localStorage.removeItem(CHAT_KEY(oldest.id));
        idx.chats = idx.chats.filter((c) => c.id !== oldest.id);
        try { localStorage.setItem(IDX_KEY, JSON.stringify(idx)); } catch {}
      }
      try { localStorage.setItem(key, raw); return; } catch {}
    }
    console.warn("Could not save conversation; continuing without persistence.", e);
  }
}

function loadIndex() {
  const idx = readJSON(IDX_KEY, null);
  return idx && Array.isArray(idx.chats) ? idx : { version: 1, chats: [] };
}
function saveIndex(idx) { writeJSON(IDX_KEY, idx); }
function loadChat(id) { return readJSON(CHAT_KEY(id), { id, messages: [] }); }
function saveChat(chat) { writeJSON(CHAT_KEY(chat.id), chat); }

function makeThreadId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

let currentThreadId = makeThreadId();
let chatPersisted = false;   // created lazily on the first user message

/** Create the sidebar entry on first send, so stray "New chat" clicks leave nothing behind. */
function ensureChat(firstText) {
  if (chatPersisted) return;
  const idx = loadIndex();
  const title = firstText.trim().replace(/\s+/g, " ").slice(0, 48) +
                (firstText.trim().length > 48 ? "…" : "");
  idx.chats.push({
    id: currentThreadId, title, mode,
    createdAt: Date.now(), updatedAt: Date.now(), turns: 0,
  });
  if (idx.chats.length > MAX_CHATS) {
    idx.chats.sort((a, b) => b.updatedAt - a.updatedAt);
    idx.chats.splice(MAX_CHATS).forEach((c) => {
      if (canStore) localStorage.removeItem(CHAT_KEY(c.id)); else delete memStore[CHAT_KEY(c.id)];
    });
  }
  saveIndex(idx);
  saveChat({ id: currentThreadId, messages: [] });
  chatPersisted = true;
  renderSidebar();
}

function appendTurn(userText, botText, meta) {
  if (!chatPersisted) return;
  const chat = loadChat(currentThreadId);
  chat.messages.push({ role: "user", text: userText });
  chat.messages.push({ role: "bot", text: botText, meta: meta || null });
  saveChat(chat);

  const idx = loadIndex();
  const entry = idx.chats.find((c) => c.id === currentThreadId);
  if (entry) {
    entry.updatedAt = Date.now();
    entry.turns = Math.ceil(chat.messages.length / 2);
    saveIndex(idx);
  }
  renderSidebar();
}

/* ══════════════════════════════════════════════════════════════════
   Sidebar
   ══════════════════════════════════════════════════════════════════ */
function relTime(ts) {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return Math.floor(s / 60) + "m ago";
  if (s < 86400) return Math.floor(s / 3600) + "h ago";
  return Math.floor(s / 86400) + "d ago";
}

function renderSidebar() {
  const list = $("convoList");
  const chats = loadIndex().chats.slice().sort((a, b) => b.updatedAt - a.updatedAt);
  if (!chats.length) {
    list.innerHTML = `<div class="convo-empty">No conversations yet. Ask something to start one.</div>`;
    return;
  }
  list.innerHTML = chats.map((c) => `
    <div class="convo${c.id === currentThreadId ? " active" : ""}" data-id="${escAttr(c.id)}">
      <span class="convo-title">${esc(c.title || "Untitled")}</span>
      <span class="convo-meta">${relTime(c.updatedAt)} · ${c.turns || 0} turn${c.turns === 1 ? "" : "s"}
        · <span class="convo-mode">${c.mode === "agentic_ask" ? "agentic" : "fast"}</span></span>
      <button class="convo-del" title="Delete conversation">×</button>
    </div>`).join("");
}

function openChat(id) {
  const idx = loadIndex();
  const entry = idx.chats.find((c) => c.id === id);
  if (!entry) return;

  currentThreadId = id;
  chatPersisted = true;
  setMode(entry.mode || "ask");

  thread.innerHTML = "";
  const chat = loadChat(id);
  for (const m of chat.messages) {
    if (m.role === "user") {
      addMsg("user", esc(m.text));
    } else {
      const bubble = addMsg("bot", "");
      bubble.innerHTML =
        (m.meta ? tracePanel(m.meta.reasoning_steps || [], m.meta.search_mode, true) : "") +
        md(m.text || "") +
        citeCards(m.meta) +
        metaRow(m.meta || {}, m.meta?.ms || 0, m.meta?.search_mode === "agentic" ? "agentic_ask" : "ask");
    }
  }
  renderSidebar();
  closeDrawer();
  main.scrollTop = main.scrollHeight;
}

function newChat() {
  currentThreadId = makeThreadId();
  chatPersisted = false;
  thread.innerHTML = WELCOME_HTML;
  wireChips();
  renderSidebar();
  closeDrawer();
  input.focus();
}

function deleteChat(id) {
  const idx = loadIndex();
  idx.chats = idx.chats.filter((c) => c.id !== id);
  saveIndex(idx);
  if (canStore) localStorage.removeItem(CHAT_KEY(id)); else delete memStore[CHAT_KEY(id)];
  if (id === currentThreadId) newChat(); else renderSidebar();
}

/* ══════════════════════════════════════════════════════════════════
   Rendering helpers
   ══════════════════════════════════════════════════════════════════ */
function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function escAttr(s) { return esc(s).replace(/"/g, "&quot;"); }

function md(s) {
  s = esc(s);
  s = s.replace(/```([\s\S]*?)```/g, (_, c) => `<pre><code>${c.trim()}</code></pre>`);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/^\s*[-•] (.+)$/gm, "<li>$1</li>").replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`);
  return s.split(/\n{2,}/).map((p) => (p.startsWith("<") ? p : `<p>${p.replace(/\n/g, "<br>")}</p>`)).join("");
}

/* Agent trace ─────────────────────────────
   reasoning_steps are plain strings built server-side by _extract_reasoning_steps.
   Match on their known prefixes to pick an icon; unknown steps fall back to a
   bullet rather than breaking, so adding a step server-side degrades gracefully. */
const STEP_ICONS = [
  [/^Validated query scope/i, "🛡"],
  [/^Fetched .* from arXiv|^Searched arXiv/i, "📥"],
  [/^Retrieved documents/i, "🔎"],
  [/^Graded documents/i, "⚖️"],
  [/^Rewritten query/i, "✏️"],
  [/^Generated answer/i, "✍️"],
];
function stepIcon(text) {
  for (const [re, icon] of STEP_ICONS) if (re.test(text)) return icon;
  return "•";
}

/** Split "Graded documents (1 relevant)" into label + parenthetical detail. */
function splitStep(text) {
  const m = text.match(/^(.*?)\s*\(([^()]*)\)\s*$/);
  return m ? { label: m[1], detail: m[2] } : { label: text, detail: "" };
}

function tracePanel(steps, searchMode, collapsed) {
  if (!steps || !steps.length) return "";
  const rows = steps.map((s) => {
    const { label, detail } = splitStep(s);
    return `<li class="tstep">
      <span class="tstep-icon">${stepIcon(s)}</span>
      <span class="tstep-body">
        <span class="tstep-label">${esc(label)}</span>
        ${detail ? `<span class="tstep-detail"> — ${esc(detail)}</span>` : ""}
      </span>
    </li>`;
  }).join("");
  return `<div class="trace${collapsed ? " collapsed" : ""}">
    <div class="trace-head">
      <span class="trace-caret">▾</span>
      <span>How I answered</span>
      <span class="trace-count">${steps.length} step${steps.length === 1 ? "" : "s"}</span>
    </div>
    <ol class="trace-steps">${rows}</ol>
  </div>`;
}

/** Placeholder trace shown while the request is in flight. */
function tracePending(label) {
  return `<div class="trace running">
    <div class="trace-head"><span class="spin"></span><span>${esc(label)}</span></div>
  </div>`;
}

/* Citations ───────────────────────────────
   sources_detailed carries {arxiv_id, title, authors, url, relevance_score} and was
   previously ignored entirely. Dedupe by arxiv_id: grading appends one SourceItem per
   retrieved chunk, so the same paper routinely appears 3x. */
function citeCards(data) {
  const detailed = (data && data.sources_detailed) || [];
  if (detailed.length) {
    const seen = new Set();
    const cards = [];
    for (const s of detailed) {
      const key = s.arxiv_id || s.url;
      if (!key || seen.has(key)) continue;
      seen.add(key);
      const authors = Array.isArray(s.authors) ? s.authors : [];
      const who = authors.length
        ? esc(authors.slice(0, 2).join(", ")) + (authors.length > 2 ? " et al." : "")
        : "";
      const bits = [who, s.arxiv_id ? "arXiv:" + esc(s.arxiv_id) : "",
                    typeof s.relevance_score === "number" ? s.relevance_score.toFixed(2) : ""]
                   .filter(Boolean).join(" · ");
      cards.push(`<a class="cite" href="${escAttr(s.url || "#")}" target="_blank" rel="noopener">
        <div class="cite-title">${esc(s.title || s.arxiv_id || "Untitled")}</div>
        ${bits ? `<div class="cite-sub">${bits}</div>` : ""}
      </a>`);
    }
    return cards.length ? `<div class="cites">${cards.join("")}</div>` : "";
  }
  // Fast mode has no sources_detailed — fall back to flat URL chips.
  const flat = (data && data.sources) || [];
  if (!flat.length) return "";
  return `<div class="meta">` + flat.slice(0, 4).map((u) =>
    `<a class="tag src" href="${escAttr(u)}" target="_blank" rel="noopener">📎 ${esc(String(u).split("/").pop().replace(".pdf", ""))}</a>`
  ).join("") + `</div>`;
}

/** `reqMode` is captured at send time — reading the live toggle would mislabel
    a response if the user flips modes mid-request. */
function metaRow(data, ms, reqMode) {
  let h = `<div class="meta">`;
  h += `<span class="tag">${esc(data.search_mode || reqMode)}</span>`;
  h += `<span class="tag">${reqMode === "agentic_ask" ? "attempts" : "chunks"}: ${data.chunks_used ?? "?"}</span>`;
  if (ms) h += `<span class="tag">${(ms / 1000).toFixed(1)}s</span>`;
  return h + `</div>`;
}

function addMsg(role, html) {
  $("welcome")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="avatar">${role === "user" ? "🧑" : "📄"}</div><div class="bubble">${html}</div>`;
  thread.appendChild(el);
  scrollToEnd(true);
  return el.querySelector(".bubble");
}

/** Only auto-scroll when the user is already near the bottom, so reading
    earlier messages mid-request doesn't yank the viewport. */
function scrollToEnd(force) {
  const near = main.scrollHeight - main.scrollTop - main.clientHeight < 120;
  if (force || near) main.scrollTop = main.scrollHeight;
}

/* ══════════════════════════════════════════════════════════════════
   Send
   ══════════════════════════════════════════════════════════════════ */
async function ask(text) {
  if (busy || !text.trim()) return;
  busy = true; sendBtn.disabled = true;

  const reqMode = mode;                       // capture, don't read later
  ensureChat(text);
  addMsg("user", esc(text));
  const bubble = addMsg("bot", tracePending(
    reqMode === "agentic_ask" ? "Running the agent…" : "Searching + answering…"));
  const t0 = performance.now();

  const payload = { query: text, thread_id: currentThreadId };
  if (reqMode === "ask") {
    payload.top_k = parseInt($("topk").value) || 5;
    payload.use_hybrid = $("hybrid").checked;
  }

  try {
    const r = await fetch(`${API}/${reqMode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const ms = performance.now() - t0;
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      bubble.innerHTML = `<p>⚠️ <strong>${r.status}</strong> — ${esc(err.detail || "request failed")}</p>`;
    } else {
      const data = await r.json();
      const steps = data.reasoning_steps?.length
        ? data.reasoning_steps
        : [`Searched + answered (${data.chunks_used ?? 0} chunks)`];
      const answer = data.answer || "(no answer)";
      bubble.innerHTML =
        tracePanel(steps, data.search_mode, false) +
        md(answer) +
        citeCards(data) +
        metaRow(data, ms, reqMode);
      appendTurn(text, answer, {
        search_mode: data.search_mode, chunks_used: data.chunks_used,
        ms,
        sources: data.sources || [], sources_detailed: data.sources_detailed || [],
        reasoning_steps: steps,
      });
    }
  } catch {
    bubble.innerHTML = `<p>⚠️ Can't reach the API. Start it with:</p><pre><code>uv run uvicorn src.main:app --port 8000</code></pre>`;
  }
  scrollToEnd(false);
  busy = false; sendBtn.disabled = false; input.focus();
}

/* ══════════════════════════════════════════════════════════════════
   Wiring
   ══════════════════════════════════════════════════════════════════ */
const WELCOME_HTML = `
  <div class="welcome" id="welcome">
    <div class="big">📄</div>
    <h2>Ask the papers anything</h2>
    <p>Answers are grounded in a daily-ingested arXiv corpus — hybrid BM25 + vector retrieval,
       with an agentic mode that can fetch brand-new papers from arXiv on demand.</p>
    <div class="chips">
      <button class="chip">What is multi-head attention?</button>
      <button class="chip">How reliable are LLMs at probabilistic reasoning?</button>
      <button class="chip">Find new papers on mixture of experts routing</button>
    </div>
  </div>`;

function wireChips() {
  document.querySelectorAll(".chip").forEach((c) =>
    c.addEventListener("click", () => ask(c.textContent.trim())));
}

function setMode(m) {
  mode = m;
  document.querySelectorAll("#seg button").forEach((b) => b.classList.toggle("active", b.dataset.mode === m));
  $("knobs").style.opacity = m === "ask" ? 1 : 0.35;
}

$("seg").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (btn) setMode(btn.dataset.mode);
});

// One delegated listener for the whole list, not one per row.
$("convoList").addEventListener("click", (e) => {
  const del = e.target.closest(".convo-del");
  const row = e.target.closest(".convo");
  if (!row) return;
  if (del) { deleteChat(row.dataset.id); return; }
  if (row.dataset.id !== currentThreadId) openChat(row.dataset.id);
});

$("newChat").addEventListener("click", newChat);

// Trace panels collapse on click (delegated — panels are created dynamically).
thread.addEventListener("click", (e) => {
  const head = e.target.closest(".trace-head");
  if (head) head.parentElement.classList.toggle("collapsed");
});

/* mobile drawer */
function closeDrawer() { $("sidebar").classList.remove("open"); $("scrim").classList.remove("show"); }
$("menuBtn").addEventListener("click", () => {
  $("sidebar").classList.toggle("open");
  $("scrim").classList.toggle("show");
});
$("scrim").addEventListener("click", closeDrawer);

sendBtn.addEventListener("click", () => { ask(input.value); input.value = ""; resize(); });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input.value); input.value = ""; resize(); }
});
function resize() { input.style.height = "auto"; input.style.height = input.scrollHeight + "px"; }
input.addEventListener("input", resize);

/* health check (5s timeout — never hang on "checking…") */
(async () => {
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(5000) });
    $("dot").classList.toggle("ok", r.ok);
    $("statusText").textContent = r.ok ? "API connected" : `API error ${r.status}`;
  } catch (e) {
    $("statusText").textContent = e.name === "TimeoutError" ? "API timeout" : "API offline";
  }
})();

/* boot — resume the most recent conversation if there is one */
(function boot() {
  if (!canStore) $("sidebarFoot").textContent = "History unavailable in this browser mode.";
  const chats = loadIndex().chats.slice().sort((a, b) => b.updatedAt - a.updatedAt);
  if (chats.length) openChat(chats[0].id);
  else { renderSidebar(); wireChips(); }
  input.focus();
})();
