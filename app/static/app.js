// Single-page client for the RAG API. No framework, no build step.
(() => {
  const $ = (sel) => document.querySelector(sel);
  const thread = $("#thread");
  const form = $("#composer");
  const input = $("#question");
  const sendBtn = $("#send");
  const scope = $("#scope");
  const sourcesEl = $("#sources");

  const state = {
    docs: [],
    turns: [],      // { el, data, kind }
    active: -1,
    busy: false,
    pane: "chat",   // mobile only
    sidePane: "sources",
  };

  const STOP = new Set("a an the and or but if of to in on at for from by with about as into is are was were be been can could should would may might must will i me my we our you your it its they them their this that these those what which who when where why how there not no any all some do does did have has had get".split(" "));

  // --- helpers ---------------------------------------------------------------

  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function api(path, opts = {}) {
    const res = await fetch(`api/${path}`, opts);
    let body = null;
    try { body = await res.json(); } catch { /* empty body */ }
    if (!res.ok) {
      let detail = body && body.detail;
      if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join("; ");
      throw new Error(detail || `Request failed (${res.status})`);
    }
    return body;
  }

  const postJSON = (path, data) => api(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
  });

  function where(src) {
    const parts = [];
    if (src.page) parts.push(`p. ${src.page}`);
    if (src.section && src.section !== "Introduction") parts.push(src.section);
    return parts.join(" · ") || "Introduction";
  }

  function fileHref(src) {
    return `api/documents/${encodeURIComponent(src.doc_id)}/file${src.page ? `#page=${src.page}` : ""}`;
  }

  // Inline markup for answers: paragraphs, "-" / "1." lists, **bold**, and [n] citation chips.
  function renderAnswer(text, turnIndex) {
    const inline = (s) => esc(s)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\[(\d+)\]/g, (_, n) => `<button type="button" class="cite" data-turn="${turnIndex}" data-n="${n}" aria-label="Source ${n}">${n}</button>`);
    const blocks = text.trim().split(/\n\s*\n/);
    return blocks.map((block) => {
      const lines = block.split("\n").filter((l) => l.trim());
      if (lines.every((l) => /^\s*[-*•]\s+/.test(l))) {
        return `<ul>${lines.map((l) => `<li>${inline(l.replace(/^\s*[-*•]\s+/, ""))}</li>`).join("")}</ul>`;
      }
      if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l))) {
        return `<ol>${lines.map((l) => `<li>${inline(l.replace(/^\s*\d+[.)]\s+/, ""))}</li>`).join("")}</ol>`;
      }
      // paragraph that leads into a list ("...based on:\n- a\n- b")
      const firstItem = lines.findIndex((l) => /^\s*[-*•]\s+/.test(l));
      if (firstItem > 0 && lines.slice(firstItem).every((l) => /^\s*[-*•]\s+/.test(l))) {
        return `<p>${inline(lines.slice(0, firstItem).join(" "))}</p><ul>${lines.slice(firstItem).map((l) => `<li>${inline(l.replace(/^\s*[-*•]\s+/, ""))}</li>`).join("")}</ul>`;
      }
      return `<p>${inline(lines.join(" "))}</p>`;
    }).join("");
  }

  // Highlight the sentences the answer quoted; otherwise highlight the question's keywords.
  function highlight(text, highlights, query) {
    const ranges = [];
    for (const h of highlights || []) {
      const at = text.indexOf(h);
      if (at >= 0) ranges.push([at, at + h.length]);
    }
    if (!ranges.length && query) {
      const words = [...new Set(query.toLowerCase().match(/[a-z0-9]{4,}/g) || [])].filter((w) => !STOP.has(w));
      if (words.length) {
        const re = new RegExp(`\\b(${words.map((w) => w.replace(/s$/, "")).join("|")})\\w*`, "gi");
        let m;
        while ((m = re.exec(text))) ranges.push([m.index, m.index + m[0].length]);
      }
    }
    ranges.sort((a, b) => a[0] - b[0]);
    let out = "", pos = 0;
    for (const [s, e] of ranges) {
      if (s < pos) continue;
      out += esc(text.slice(pos, s)) + `<mark>${esc(text.slice(s, e))}</mark>`;
      pos = e;
    }
    return out + esc(text.slice(pos));
  }

  // --- status & library ------------------------------------------------------

  async function loadHealth() {
    const el = $("#status");
    try {
      const h = await api("health");
      el.className = "status ok";
      const mode = h.answer_mode === "gemini" ? `Gemini · ${h.model}` : "Extractive mode";
      $("#status-text").textContent = `${mode} · ${h.documents} docs · ${h.chunks} chunks`;
      $("#status").title = `Embeddings: ${h.embedding_model}\nVector store: ${h.vector_store}`;
      $("#about-foot").textContent = `Running now: ${h.answer_mode === "gemini" ? `Gemini (${h.model})` : "extractive answerer (no API key set)"} · ${h.embedding_model} · ${h.vector_store}`;
      $("#drop-hint").textContent = `PDF, Markdown or text up to ${h.max_upload_mb} MB · drag here or click`;
    } catch {
      el.className = "status err";
      $("#status-text").textContent = "API unreachable";
    }
  }

  async function loadDocs() {
    try {
      const { documents } = await api("documents");
      state.docs = documents;
      renderDocs();
    } catch (err) {
      $("#doclist").innerHTML = `<li class="error">${esc(err.message)}</li>`;
    }
  }

  function renderDocs(newId) {
    const docs = state.docs;
    $("#lib-count").textContent = docs.length || "";
    $("#doclist").innerHTML = docs.map((d) => {
      const kind = d.kind === "pdf" ? "pdf" : d.kind === "markdown" ? "md" : "txt";
      const size = d.pages ? `${plural(d.pages, "page")} · ` : "";
      const badge = d.source === "sample" ? `<span class="badge">sample</span>` : `<span class="badge ${d.doc_id === newId ? "new" : ""}">uploaded</span>`;
      return `<li class="doc">
        <div class="doc-icon">${kind}</div>
        <div class="doc-body">
          <div class="doc-title">${esc(d.title)}${badge}</div>
          <div class="doc-meta">${size}${plural(d.chunks, "chunk")}</div>
          <div class="doc-actions">
            <button type="button" class="linkish" data-ask="${esc(d.doc_id)}">Ask this doc</button>
            <a class="linkish" href="api/documents/${encodeURIComponent(d.doc_id)}/file" target="_blank" rel="noopener">Open</a>
            ${d.source === "upload" ? `<button type="button" class="linkish danger" data-del="${esc(d.doc_id)}">Remove</button>` : ""}
          </div>
        </div>
      </li>`;
    }).join("");

    const current = scope.value;
    scope.innerHTML = `<option value="">All documents</option>` +
      docs.map((d) => `<option value="${esc(d.doc_id)}">${esc(d.title)}</option>`).join("");
    scope.value = docs.some((d) => d.doc_id === current) ? current : "";
  }

  $("#doclist").addEventListener("click", async (e) => {
    const askId = e.target.closest("[data-ask]")?.dataset.ask;
    if (askId) {
      scope.value = askId;
      showPane("chat");
      input.focus();
      return;
    }
    const delId = e.target.closest("[data-del]")?.dataset.del;
    if (delId) {
      const doc = state.docs.find((d) => d.doc_id === delId);
      if (!confirm(`Remove "${doc ? doc.title : "this document"}" from the index?`)) return;
      try {
        await api(`documents/${encodeURIComponent(delId)}`, { method: "DELETE" });
        setUploadMsg(`Removed "${doc ? doc.title : delId}".`, "ok");
        await Promise.all([loadDocs(), loadHealth()]);
      } catch (err) {
        setUploadMsg(err.message, "err");
      }
    }
  });

  // --- upload ----------------------------------------------------------------

  const drop = $("#dropzone");
  const fileInput = $("#file-input");

  function setUploadMsg(text, kind = "") {
    const el = $("#upload-msg");
    el.textContent = text;
    el.className = `upload-msg ${kind}`;
  }

  async function upload(file) {
    if (!file) return;
    if (!/\.(pdf|md|markdown|txt)$/i.test(file.name)) {
      setUploadMsg("Only PDF, Markdown (.md) and text (.txt) files are supported.", "err");
      return;
    }
    drop.classList.add("busy");
    setUploadMsg(`Reading, chunking and embedding "${file.name}"…`);
    const body = new FormData();
    body.append("file", file);
    try {
      const res = await api("documents", { method: "POST", body });
      const d = res.document;
      if (res.duplicate) {
        setUploadMsg(`"${d.title}" is already indexed.`, "ok");
      } else {
        let msg = `Indexed "${d.title}": ${d.pages ? `${plural(d.pages, "page")}, ` : ""}${plural(d.chunks, "chunk")}. You can ask about it now.`;
        if (res.evicted.length) msg += ` Removed the oldest upload to make room.`;
        setUploadMsg(msg, "ok");
      }
      await Promise.all([loadDocs(), loadHealth()]);
      renderDocs(d.doc_id);
    } catch (err) {
      setUploadMsg(err.message, "err");
    } finally {
      drop.classList.remove("busy");
      fileInput.value = "";
    }
  }

  fileInput.addEventListener("change", () => upload(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));
  // Stop a file dropped elsewhere on the page from navigating away.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  // --- conversation ----------------------------------------------------------

  function addUser(text) {
    $("#welcome")?.remove();
    const el = document.createElement("div");
    el.className = "msg user";
    el.innerHTML = `<div class="bubble">${esc(text)}</div>`;
    thread.appendChild(el);
  }

  function addPending() {
    const el = document.createElement("div");
    el.className = "msg bot";
    el.innerHTML = `<div class="card"><div class="thinking"><i></i><i></i><i></i><span>Retrieving passages…</span></div></div>`;
    thread.appendChild(el);
    thread.scrollTop = thread.scrollHeight;
    return el;
  }

  function renderTurn(index) {
    const turn = state.turns[index];
    const { data, kind } = turn;
    let html;
    if (kind === "error") {
      html = `<div class="card"><p class="answer error">${esc(data.message)}</p></div>`;
    } else if (kind === "search") {
      const n = data.results.length;
      html = `<div class="card">
        <p class="answer">${n ? `Found <strong>${n}</strong> matching passage${n > 1 ? "s" : ""}, ranked by relevance. They're listed in the Sources panel.` : "No passages matched closely enough. Try different wording."}</p>
        <div class="meta"><span class="tag">semantic search</span><span>no generation</span></div>
      </div>`;
    } else {
      const cited = data.sources.filter((s) => s.cited);
      const modeLabel = data.mode === "gemini" ? `gemini · ${data.model}` : data.mode === "extractive" ? "extractive" : "refused";
      html = `<div class="card">
        <div class="answer ${data.found ? "" : "notfound"}">${renderAnswer(data.answer, index)}</div>
        ${data.note ? `<div class="note">${esc(data.note)}</div>` : ""}
        ${cited.length ? `<div class="cited-list">${cited.map((s) => `<button type="button" data-turn="${index}" data-n="${s.n}"><b>${s.n}</b><span>${esc(s.title)} · ${esc(where(s))}</span></button>`).join("")}</div>` : ""}
        <div class="meta"><span class="tag">${esc(modeLabel)}</span><span>${data.sources.length} retrieved · ${cited.length} cited</span><span>${data.elapsed_ms} ms</span></div>
      </div>`;
    }
    turn.el.innerHTML = html;
  }

  function setActive(index) {
    state.active = index;
    state.turns.forEach((t, i) => t.el.classList.toggle("active", i === index));
    renderSources();
  }

  function renderSources() {
    const turn = state.turns[state.active];
    const list = !turn || turn.kind === "error" ? [] : turn.kind === "search" ? turn.data.results : turn.data.sources;
    $("#sources-empty").hidden = list.length > 0;
    $("#src-count").textContent = list.length || "";
    $("#m-src-count").textContent = list.length || "";
    if (!list.length) {
      sourcesEl.innerHTML = turn && turn.kind !== "error"
        ? `<p class="empty">No passages passed the relevance cut-off for this question.</p>` : "";
      $("#sources-empty").hidden = !!sourcesEl.innerHTML;
      return;
    }
    const query = turn.kind === "search" ? turn.data.query : turn.question;
    const card = (s, dim) => `
      <article class="src ${dim ? "dim" : ""}" id="src-${state.active}-${s.n}">
        <div class="src-top">
          <div class="src-n">${s.n}</div>
          <div>
            <div class="src-title">${esc(s.title)}</div>
            <div class="src-loc">${esc(where(s))}</div>
          </div>
        </div>
        <div class="src-text ${dim ? "clamp" : ""}">${highlight(s.text, s.highlights, query)}</div>
        <div class="src-foot">
          <span class="bar" title="Cosine similarity"><i style="width:${Math.max(4, Math.round(s.similarity * 100))}%"></i></span>
          <span>${(s.similarity * 100).toFixed(0)}% match</span>
          <span class="push"><a href="${fileHref(s)}" target="_blank" rel="noopener">Open${s.page ? ` p. ${s.page}` : ""} ↗</a></span>
        </div>
      </article>`;
    if (turn.kind === "search") {
      sourcesEl.innerHTML = `<p class="src-heading">Search results</p>` + list.map((s) => card(s, false)).join("");
    } else {
      const cited = list.filter((s) => s.cited);
      const other = list.filter((s) => !s.cited);
      sourcesEl.innerHTML =
        (cited.length ? `<p class="src-heading">Cited in the answer</p>${cited.map((s) => card(s, false)).join("")}` : "") +
        (other.length ? `<p class="src-heading">${cited.length ? "Also retrieved" : "Retrieved"}</p>${other.map((s) => card(s, true)).join("")}` : "");
    }
  }

  function focusSource(turnIndex, n) {
    if (state.active !== turnIndex) setActive(turnIndex);
    showSide("sources");
    showPane("sources");
    document.querySelectorAll(".cite.on").forEach((c) => c.classList.remove("on"));
    document.querySelectorAll(`.cite[data-turn="${turnIndex}"][data-n="${n}"]`).forEach((c) => c.classList.add("on"));
    const el = document.getElementById(`src-${turnIndex}-${n}`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    el.classList.remove("flash");
    void el.offsetWidth;
    el.classList.add("flash");
    setTimeout(() => el.classList.remove("flash"), 1600);
  }

  thread.addEventListener("click", (e) => {
    const chip = e.target.closest("[data-n][data-turn]");
    if (chip) {
      e.stopPropagation();
      focusSource(Number(chip.dataset.turn), Number(chip.dataset.n));
      return;
    }
    const sug = e.target.closest(".suggestions button");
    if (sug) {
      input.value = sug.textContent;
      submit();
      return;
    }
    const msg = e.target.closest(".msg.bot");
    if (msg) {
      const index = state.turns.findIndex((t) => t.el === msg);
      if (index >= 0) {
        setActive(index);
        if (window.matchMedia("(max-width: 860px)").matches) showPane("sources");
      }
    }
  });

  async function submit() {
    const text = input.value.trim();
    if (!text || state.busy) return;
    const mode = form.querySelector('input[name="mode"]:checked').value;
    const docIds = scope.value ? [scope.value] : null;
    state.busy = true;
    sendBtn.disabled = true;
    input.value = "";
    autosize();
    addUser(text);
    const el = addPending();
    const turn = { el, question: text, kind: mode === "search" ? "search" : "answer", data: null };
    state.turns.push(turn);
    const index = state.turns.length - 1;
    try {
      turn.data = mode === "search"
        ? await postJSON("search", { query: text, doc_ids: docIds })
        : await postJSON("ask", { question: text, doc_ids: docIds });
    } catch (err) {
      turn.kind = "error";
      turn.data = { message: err.message || "Something went wrong." };
    }
    renderTurn(index);
    setActive(index);
    if (turn.kind !== "error") showSide("sources");
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    state.busy = false;
    sendBtn.disabled = false;
    input.focus();
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); submit(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
  });
  function autosize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
    input.style.overflowY = input.scrollHeight > 160 ? "auto" : "hidden";
  }
  input.addEventListener("input", autosize);
  form.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", () => {
    input.placeholder = r.value === "search" && r.checked
      ? "Search passages, e.g. “medical certificate”"
      : "Ask about attendance, exams, conduct, the library…";
  }));

  // --- panes -----------------------------------------------------------------

  function showSide(name) {
    state.sidePane = name;
    document.querySelectorAll(".side-tabs button").forEach((b) => {
      const on = b.dataset.pane === name;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on);
    });
    $("#pane-sources").classList.toggle("active", name === "sources");
    $("#pane-library").classList.toggle("active", name === "library");
  }

  // Mobile: one pane at a time (chat, sources or library).
  function showPane(name) {
    state.pane = name;
    document.querySelectorAll(".mobile-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.pane === name));
    $("#pane-chat").classList.toggle("active", name === "chat");
    $(".side").classList.toggle("show", name !== "chat");
    if (name !== "chat") showSide(name);
  }

  document.querySelectorAll(".side-tabs button").forEach((b) => b.addEventListener("click", () => showSide(b.dataset.pane)));
  document.querySelectorAll(".mobile-tabs button").forEach((b) => b.addEventListener("click", () => showPane(b.dataset.pane)));

  $("#about-btn").addEventListener("click", () => $("#about").showModal());

  loadHealth();
  loadDocs();
})();
