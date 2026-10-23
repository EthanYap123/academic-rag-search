// In-browser stand-in for the FastAPI backend, used by the static GitHub Pages demo.
//
// It mirrors app/ingest.py and app/rag.py: the same chunking rules, the same
// embedding model (all-MiniLM-L6-v2, run by transformers.js instead of ONNX
// Runtime in Python), the same retrieval re-ranking and the same extractive
// answerer. The sample handbooks arrive pre-chunked and pre-embedded in
// index.json (built by scripts/build_demo.py); uploads are processed here and
// live only in this browser tab.
(() => {
  const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
  const PDFJS_URL = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.min.mjs";
  const PDFJS_WORKER = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs";
  const MODEL = "Xenova/all-MiniLM-L6-v2";
  const NOT_FOUND_MESSAGE = "I couldn't find this in the indexed documents, so I won't guess. " +
    "Try rephrasing the question, or upload a document that covers it.";
  const ALLOWED = [".pdf", ".md", ".markdown", ".txt"];

  let cfg = null;
  const docs = new Map();   // doc_id -> document info
  let chunks = [];          // { chunk_id, doc_id, title, page, section, text, v: Float32Array }
  const fileUrls = new Map();

  // --- embeddings -------------------------------------------------------------

  let extractorPromise = null;
  const DemoAPI = { modelReady: false };

  function extractor() {
    if (!extractorPromise) {
      extractorPromise = import(TRANSFORMERS_URL).then(async ({ pipeline, env }) => {
        env.allowLocalModels = false;
        const pipe = await pipeline("feature-extraction", MODEL, { dtype: "q8" });
        DemoAPI.modelReady = true;
        return pipe;
      });
      extractorPromise.catch(() => { extractorPromise = null; });
    }
    return extractorPromise;
  }

  async function embed(texts) {
    const pipe = await extractor();
    const out = [];
    for (let i = 0; i < texts.length; i += 32) {
      const res = await pipe(texts.slice(i, i + 32), { pooling: "mean", normalize: true });
      for (const row of res.tolist()) out.push(Float32Array.from(row));
    }
    return out;
  }

  const sentenceCache = new Map();
  async function embedCached(texts) {
    const missing = [...new Set(texts.filter((t) => !sentenceCache.has(t)))];
    if (missing.length) {
      const vecs = await embed(missing);
      missing.forEach((t, i) => sentenceCache.set(t, vecs[i]));
    }
    return texts.map((t) => sentenceCache.get(t));
  }

  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

  // --- text helpers (port of app/ingest.py) -----------------------------------

  const NUMBERED_HEADING = /^(?:section\s+)?(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{1,80})$/i;
  const PAGE_FOOTER = /^(page\s+)?\d+(\s*(of|\/)\s*\d+)?$/i;
  const BULLET = /^\s*(?:[•\-*–]|\(?[a-z]\)|\d+\))\s+/;
  const BULLET_GLYPHS = /[\x7f●▪◦‣⁃■○]/g;
  const ABBREVIATIONS = ["e.g.", "i.e.", "etc.", "Mr.", "Ms.", "Dr.", "No.", "vs.", "approx."];
  const SENTENCE_END = /(?<=[.!?:])\s+(?=[A-Z0-9"'(•])/;
  const MD_HEADING = /^(#{1,6})\s+(.*)$/;

  function clean(text) {
    return text.replace(/ /g, " ").replace(/­/g, "").replace(BULLET_GLYPHS, "•")
      .replace(/(\w)-\n(\w)/g, "$1$2").replace(/[ \t]+/g, " ");
  }

  function isHeading(line) {
    line = line.trim();
    if (!line || line.length > 90 || /[.,;]$/.test(line)) return false;
    const words = line.split(/\s+/);
    if (NUMBERED_HEADING.test(line)) {
      return words.length <= 12 && !/^[a-z]/.test(words[1]);
    }
    const letters = (line.match(/\p{L}/gu) || []).length;
    return letters >= 4 && line === line.toUpperCase() && /\p{Lu}/u.test(line) && words.length <= 8;
  }

  function splitSentences(text) {
    let protectedText = text;
    ABBREVIATIONS.forEach((abbr, i) => { protectedText = protectedText.split(abbr).join(`\0${i}\0`); });
    return protectedText.split(SENTENCE_END).map((part) => {
      ABBREVIATIONS.forEach((abbr, i) => { part = part.split(`\0${i}\0`).join(abbr); });
      return part.trim();
    }).filter(Boolean);
  }

  function linesToUnits(lines) {
    const units = [];
    let buf = [];
    let inBullet = false;
    let bulletNext = false;
    const flush = () => {
      if (buf.length) {
        const text = buf.join(" ");
        if (inBullet) units.push(text); else units.push(...splitSentences(text));
        buf = [];
      }
      inBullet = false;
    };
    for (let line of lines) {
      line = line.trim();
      if (!line) { flush(); continue; }
      if (["•", "-", "*", "–"].includes(line)) { bulletNext = true; continue; }
      if (bulletNext || BULLET.test(line)) {
        flush();
        buf.push("• " + line.replace(BULLET, ""));
        inBullet = true;
        bulletNext = false;
        continue;
      }
      if (inBullet && /[.!?]$/.test(buf[buf.length - 1])) flush();
      buf.push(line);
    }
    flush();
    return units;
  }

  function joinUnits(units) {
    let out = "";
    for (const unit of units) {
      if (!out) out = unit;
      else if (unit.startsWith("•") || out.endsWith(":") || out.split("\n").pop().startsWith("•")) out += "\n" + unit;
      else out += " " + unit;
    }
    return out;
  }

  function chunkUnits(text) {
    const units = [];
    for (let line of text.split("\n")) {
      line = line.trim();
      if (line.startsWith("•")) units.push(line);
      else if (line) units.push(...splitSentences(line));
    }
    return units;
  }

  function pack(units, page, section) {
    const out = [];
    let current = [];
    let size = 0;
    let fresh = 0;
    for (const unit of units) {
      if (fresh && size + unit.length > cfg.chunk_chars) {
        out.push({ text: joinUnits(current), page, section });
        const tail = [];
        let tailSize = 0;
        for (let i = current.length - 1; i >= 0; i--) {
          if (tailSize + current[i].length > cfg.chunk_overlap) break;
          tail.unshift(current[i]);
          tailSize += current[i].length + 1;
        }
        current = tail; size = tailSize; fresh = 0;
      }
      current.push(unit);
      size += unit.length + 1;
      fresh += 1;
    }
    if (fresh) out.push({ text: joinUnits(current), page, section });
    return out;
  }

  function runningLines(pages) {
    if (pages.length < 2) return new Set();
    const counts = new Map();
    for (const lines of pages) {
      for (const l of new Set(lines.map((x) => x.trim()).filter(Boolean))) counts.set(l, (counts.get(l) || 0) + 1);
    }
    const needed = pages.length < 5 ? pages.length : pages.length * 0.6;
    return new Set([...counts].filter(([l, n]) => n >= needed && !isHeading(l)).map(([l]) => l));
  }

  async function sha1Id(bytes) {
    const hash = await crypto.subtle.digest("SHA-1", bytes);
    return [...new Uint8Array(hash)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 12);
  }

  function titleFromFilename(name) {
    return name.replace(/\.[^.]+$/, "").replace(/[-_]/g, " ").replace(/\w\S*/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase());
  }

  async function parsePdf(bytes) {
    const pdfjs = await import(PDFJS_URL);
    pdfjs.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
    const pdf = await pdfjs.getDocument({ data: bytes.slice() }).promise;
    const rawPages = [];
    for (let p = 1; p <= pdf.numPages; p++) {
      const content = await (await pdf.getPage(p)).getTextContent();
      const text = content.items.map((it) => (it.str || "") + (it.hasEOL ? "\n" : "")).join("");
      rawPages.push(clean(text).split("\n"));
    }
    let title = "";
    try { title = ((await pdf.getMetadata()).info.Title || "").trim(); } catch { /* no metadata */ }
    const noise = runningLines(rawPages);
    let section = "Introduction";
    const out = [];
    rawPages.forEach((lines, i) => {
      const page = i + 1;
      let buffer = [];
      const emit = () => { if (buffer.length) { out.push(...pack(linesToUnits(buffer), page, section)); buffer = []; } };
      for (const line of lines) {
        const s = line.trim();
        if (!s || noise.has(s) || PAGE_FOOTER.test(s)) continue;
        if (!title && page === 1) { title = s; continue; }
        if (page === 1 && s === title) continue;
        if (isHeading(s)) { emit(); section = s; continue; }
        buffer.push(s);
      }
      emit();
    });
    return { title, kind: "pdf", pages: pdf.numPages, chunks: out };
  }

  function parseMarkdown(bytes, kind) {
    const text = clean(new TextDecoder().decode(bytes));
    let title = "";
    let section = "Introduction";
    const out = [];
    let buffer = [];
    const emit = () => { if (buffer.length) { out.push(...pack(linesToUnits(buffer), null, section)); buffer = []; } };
    for (const line of text.split("\n")) {
      const m = line.trim().match(MD_HEADING);
      if (m) {
        emit();
        const heading = m[2].trim().replace(/^#+|#+$/g, "").trim();
        if (m[1].length === 1 && !title) title = heading; else section = heading;
        continue;
      }
      buffer.push(line.replace(/[*_`]{1,3}/g, ""));
    }
    emit();
    return { title, kind, pages: 0, chunks: out };
  }

  // --- retrieval and answering (port of app/rag.py) ---------------------------

  const STOPWORDS = new Set(`a an the and or but if of to in on at for from by with about as into is are was were be been
    being do does did can could should would may might must will shall i me my we our you your he she
    it its they them their this that these those what which who whom whose when where why how there
    here not no any all some each every more most other such than too very just also only own same so
    get got have has had having please tell know need want`.split(/\s+/));

  function terms(text) {
    const words = text.toLowerCase().match(/[a-z0-9]+/g) || [];
    return new Set(words.filter((w) => !STOPWORDS.has(w) && w.length > 1)
      .map((w) => (w.length > 3 ? w.replace(/s+$/, "") : w)));
  }

  function keywordOverlap(qTerms, text) {
    if (!qTerms.size) return 0;
    const t = terms(text);
    let n = 0;
    for (const w of qTerms) if (t.has(w)) n++;
    return n / qTerms.size;
  }

  async function retrieve(question, k, docIds, minSimilarity) {
    const pool = docIds && docIds.length ? chunks.filter((c) => docIds.includes(c.doc_id)) : chunks;
    if (!pool.length) return [];
    const [qv] = await embed([question]);
    const qTerms = terms(question);
    return pool
      .map((c) => ({ c, sim: Math.round(dot(qv, c.v) * 1e4) / 1e4 }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, k * 3)
      .filter((h) => h.sim >= minSimilarity)
      .map((h) => ({ ...h, score: h.sim + 0.15 * keywordOverlap(qTerms, `${h.c.section} ${h.c.text}`) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((h, i) => ({
        n: i + 1, chunk_id: h.c.chunk_id, doc_id: h.c.doc_id, title: h.c.title, page: h.c.page || null,
        section: h.c.section, text: h.c.text, similarity: h.sim, cited: false, highlights: [],
      }));
  }

  async function extractiveAnswer(question, sources) {
    const cand = [];
    for (const src of sources) {
      for (let s of chunkUnits(src.text)) {
        s = s.trim();
        if (s.length >= 15) cand.push([src.n, s]);
      }
    }
    if (!cand.length) return "";
    const [qv] = await embed([question]);
    const svs = await embedCached(cand.map((c) => c[1]));
    const qTerms = terms(question);
    const chunkSim = Object.fromEntries(sources.map((s) => [s.n, s.similarity]));
    const scores = cand.map(([n, s], i) => 0.6 * dot(svs[i], qv) + 0.4 * chunkSim[n] + 0.2 * keywordOverlap(qTerms, s));

    const best = Math.max(...scores);
    const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
    const picked = [];
    for (const i of order.slice(0, 8)) {
      if (picked.length === 3 || scores[i] < Math.max(0.3, best * 0.85)) break;
      picked.push(i);
    }
    for (const i of [...picked]) {
      const j = i + 1;
      if (j < cand.length && cand[j][0] === cand[i][0] && cand[j][1].endsWith(":") && !picked.includes(j)) picked.push(j);
    }
    let expanded = [];
    for (const i of [...picked].sort((a, b) => cand[a][0] - cand[b][0] || a - b)) {
      expanded.push(i);
      if (cand[i][1].endsWith(":")) {
        let j = i + 1;
        while (j < cand.length && cand[j][0] === cand[i][0] && cand[j][1].startsWith("•") && expanded.length < 12) {
          expanded.push(j);
          j++;
        }
      }
    }
    const rank = new Map();
    for (const i of picked) if (!rank.has(cand[i][0])) rank.set(cand[i][0], rank.size);
    const r = (i) => (rank.has(cand[i][0]) ? rank.get(cand[i][0]) : 99);
    expanded = [...new Set(expanded)].sort((a, b) => r(a) - r(b) || a - b);

    for (const src of sources) src.highlights = expanded.filter((i) => cand[i][0] === src.n).map((i) => cand[i][1]);

    let out = "";
    for (const i of expanded) {
      const [n, s] = cand[i];
      if (s.startsWith("•")) out += `\n- ${s.slice(1).trim()} [${n}]`;
      else out += (out ? "\n\n" : "") + `${s} [${n}]`;
    }
    return out.trim();
  }

  async function ask({ question, top_k, doc_ids }) {
    const started = performance.now();
    question = (question || "").trim();
    const sources = await retrieve(question, top_k || cfg.top_k, doc_ids, cfg.min_similarity);
    const result = (answer, found, mode) => ({
      question, answer, found, mode, sources, model: null, note: null,
      elapsed_ms: Math.round(performance.now() - started),
    });
    if (!sources.length) return result(NOT_FOUND_MESSAGE, false, "none");
    const text = await extractiveAnswer(question, sources);
    if (!text) return result(NOT_FOUND_MESSAGE, false, "extractive");
    const used = new Set([...text.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
    for (const s of sources) s.cited = used.has(s.n);
    return result(text, true, "extractive");
  }

  // --- documents --------------------------------------------------------------

  function docList() {
    return [...docs.values()]
      .sort((a, b) => (a.source !== "sample") - (b.source !== "sample") || a.added_at - b.added_at)
      .map(({ file, ...d }) => d);
  }

  function removeDoc(docId) {
    docs.delete(docId);
    chunks = chunks.filter((c) => c.doc_id !== docId);
    if (fileUrls.has(docId)) { URL.revokeObjectURL(fileUrls.get(docId)); fileUrls.delete(docId); }
  }

  async function upload(file) {
    const name = file.name || "upload";
    const ext = (name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
    if (!ALLOWED.includes(ext)) throw new Error("Only PDF, Markdown (.md) and plain-text (.txt) files can be uploaded.");
    if (file.size > cfg.max_upload_mb * 1024 * 1024) throw new Error(`File is larger than ${cfg.max_upload_mb} MB.`);
    if (!file.size) throw new Error("The file is empty.");
    const bytes = new Uint8Array(await file.arrayBuffer());
    const docId = await sha1Id(bytes);
    if (docs.has(docId)) return { document: docList().find((d) => d.doc_id === docId), duplicate: true, evicted: [] };

    let parsed;
    try {
      parsed = ext === ".pdf" ? await parsePdf(bytes) : parseMarkdown(bytes, ext === ".txt" ? "text" : "markdown");
    } catch (err) {
      console.warn("could not parse upload", err);
      throw new Error("This file could not be read. Is it a valid, unencrypted PDF?");
    }
    const parts = parsed.chunks.filter((c) => c.text.length >= 25);
    if (!parts.length) throw new Error("No text could be extracted. Scanned PDFs need OCR before upload.");
    const title = (parsed.title || titleFromFilename(name)).slice(0, 120);
    const vectors = await embed(parts.map((c) => `${title}. ${c.section}. ${c.text}`));
    parts.forEach((c, i) => chunks.push({
      chunk_id: `${docId}:${i}`, doc_id: docId, title, page: c.page, section: c.section, text: c.text, v: vectors[i],
    }));
    const type = ext === ".pdf" ? "application/pdf" : "text/plain;charset=utf-8";
    fileUrls.set(docId, URL.createObjectURL(new Blob([bytes], { type })));
    const info = {
      doc_id: docId, title, filename: name, kind: parsed.kind, pages: parsed.pages, chunks: parts.length,
      source: "upload", added_at: Date.now() / 1000,
    };
    docs.set(docId, info);

    const evicted = [];
    const uploads = docList().filter((d) => d.source === "upload");
    for (const old of uploads.slice(0, Math.max(0, uploads.length - cfg.max_uploaded_docs))) {
      removeDoc(old.doc_id);
      evicted.push(old.title);
    }
    return { document: info, duplicate: false, evicted };
  }

  // --- request router used by app.js ------------------------------------------

  const ready = fetch("index.json").then((r) => r.json()).then((data) => {
    cfg = data.config;
    for (const d of data.documents) docs.set(d.doc_id, d);
    chunks = data.chunks.map((c) => ({ ...c, v: Float32Array.from(c.v) }));
  });
  // Start fetching the model straight away so the first question is quick.
  ready.then(() => extractor()).catch((err) => console.warn("model preload failed", err));

  DemoAPI.fileUrl = (docId) => fileUrls.get(docId) || docs.get(docId)?.file || "#";

  DemoAPI.request = async (path, opts = {}) => {
    await ready;
    const method = (opts.method || "GET").toUpperCase();
    const body = typeof opts.body === "string" ? JSON.parse(opts.body) : opts.body;
    if (path === "health") {
      return {
        status: "ok", demo: true, answer_mode: "extractive", model: null,
        embedding_model: "all-MiniLM-L6-v2 (transformers.js, in your browser)",
        vector_store: "in-browser index", documents: docs.size, chunks: chunks.length,
        max_upload_mb: cfg.max_upload_mb,
      };
    }
    if (path === "ask") {
      if (!body.question || body.question.trim().length < 2) throw new Error("Question is too short.");
      return ask(body);
    }
    if (path === "search") {
      const results = await retrieve(body.query.trim(), body.top_k || 8, body.doc_ids, cfg.min_similarity - 0.1);
      return { query: body.query, results };
    }
    if (path === "documents" && method === "GET") return { documents: docList() };
    if (path === "documents" && method === "POST") return upload(body.get("file"));
    const del = path.match(/^documents\/([^/]+)$/);
    if (del && method === "DELETE") {
      const id = decodeURIComponent(del[1]);
      const doc = docs.get(id);
      if (!doc) throw new Error("Document not found.");
      if (doc.source === "sample") throw new Error("Sample documents can't be removed.");
      removeDoc(id);
      return { deleted: id };
    }
    throw new Error(`Unknown endpoint: ${path}`);
  };

  window.DemoAPI = DemoAPI;
})();
