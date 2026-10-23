"""Retrieval and grounded answer generation.

Pipeline for a question:
  1. Embed the question and pull candidate chunks from ChromaDB.
  2. Re-rank with a small keyword-overlap bonus and drop chunks below MIN_SIMILARITY.
  3. Answer only from those chunks:
       - with Gemini when GEMINI_API_KEY is set (strict prompt, numbered sources,
         every claim must cite [n]; answers without valid citations are rejected), or
       - with the extractive answerer, which quotes the most relevant sentences verbatim.
  4. Return the answer with numbered citations (title, page, section, snippet).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field

import httpx
import numpy as np

from . import config
from .ingest import chunk_units
from .store import Hit, VectorStore

log = logging.getLogger("rag")

NOT_FOUND = "NOT_FOUND"
NOT_FOUND_MESSAGE = (
    "I couldn't find this in the indexed documents, so I won't guess. "
    "Try rephrasing the question, or upload a document that covers it."
)

STOPWORDS = set(
    """a an the and or but if of to in on at for from by with about as into is are was were be been
    being do does did can could should would may might must will shall i me my we our you your he she
    it its they them their this that these those what which who whom whose when where why how there
    here not no any all some each every more most other such than too very just also only own same so
    get got have has had having please tell know need want""".split()
)


@dataclass
class Source:
    n: int
    chunk_id: str
    doc_id: str
    title: str
    page: int | None
    section: str
    text: str
    similarity: float
    cited: bool = False
    highlights: list[str] = field(default_factory=list)


@dataclass
class Answer:
    question: str
    answer: str
    found: bool
    mode: str  # "gemini" | "extractive"
    sources: list[Source]
    model: str | None = None
    note: str | None = None
    elapsed_ms: int = 0

    def to_dict(self):
        return asdict(self)


def terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w.rstrip("s") if len(w) > 3 else w for w in words if w not in STOPWORDS and len(w) > 1}


def keyword_overlap(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & terms(text)) / len(query_terms)


# --- retrieval ------------------------------------------------------------------

def retrieve(store: VectorStore, question: str, k: int = config.TOP_K,
             doc_ids: list[str] | None = None, min_similarity: float = config.MIN_SIMILARITY) -> list[Hit]:
    candidates = store.query(question, k * 3, doc_ids)
    q_terms = terms(question)
    scored = []
    for hit in candidates:
        if hit.similarity < min_similarity:
            continue
        score = hit.similarity + 0.15 * keyword_overlap(q_terms, f"{hit.section} {hit.text}")
        scored.append((score, hit))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [hit for _, hit in scored[:k]]


def to_sources(hits: list[Hit]) -> list[Source]:
    return [
        Source(i + 1, h.chunk_id, h.doc_id, h.title, h.page, h.section, h.text, h.similarity)
        for i, h in enumerate(hits)
    ]


def cite_label(source: Source) -> str:
    where = f"page {source.page}" if source.page else "no page numbers"
    return f"{source.title}, {where}, section \"{source.section}\""


# --- extractive answerer --------------------------------------------------------

def extractive_answer(store: VectorStore, question: str, sources: list[Source]) -> str:
    """Quote the sentences that best answer the question, each tagged with its source."""
    candidates: list[tuple[int, str]] = []
    for src in sources:
        for sentence in chunk_units(src.text):
            sentence = sentence.strip()
            if len(sentence) >= 15:
                candidates.append((src.n, sentence))
    if not candidates:
        return ""

    q_vec = store.embedder.embed([question])[0]
    s_vecs = store.embedder.embed_cached([s for _, s in candidates])
    sims = s_vecs @ q_vec
    q_terms = terms(question)
    chunk_sim = {s.n: s.similarity for s in sources}
    # A sentence scores well if it matches the question AND sits in a chunk that matches it.
    scores = np.array([
        0.6 * sims[i] + 0.4 * chunk_sim[n] + 0.2 * keyword_overlap(q_terms, s)
        for i, (n, s) in enumerate(candidates)
    ])

    best = float(scores.max())
    order = np.argsort(-scores)
    picked: list[int] = []
    for i in order[:8]:
        if len(picked) == 3 or scores[i] < max(0.3, best * 0.85):
            break
        picked.append(int(i))
    # A picked sentence directly followed by a list intro ("...based on:") brings the list along.
    for i in list(picked):
        j = i + 1
        if j < len(candidates) and candidates[j][0] == candidates[i][0] \
                and candidates[j][1].endswith(":") and j not in picked:
            picked.append(j)
    # A sentence that ends with ":" introduces a list; pull in the items that follow it.
    expanded: list[int] = []
    for i in sorted(picked, key=lambda j: (candidates[j][0], j)):
        expanded.append(i)
        if candidates[i][1].endswith(":"):
            j = i + 1
            while j < len(candidates) and candidates[j][0] == candidates[i][0] \
                    and candidates[j][1].startswith("•") and len(expanded) < 12:
                expanded.append(j)
                j += 1

    # Keep the best source first, and sentences from one source in document order.
    source_rank = {}
    for i in picked:
        source_rank.setdefault(candidates[i][0], len(source_rank))
    expanded = sorted(dict.fromkeys(expanded), key=lambda j: (source_rank.get(candidates[j][0], 99), j))

    for src in sources:
        src.highlights = [candidates[i][1] for i in expanded if candidates[i][0] == src.n]

    lines: list[str] = []
    for i in expanded:
        n, sentence = candidates[i]
        if sentence.startswith("•"):
            lines.append(f"\n- {sentence[1:].strip()} [{n}]")
        else:
            lines.append(("\n\n" if lines else "") + f"{sentence} [{n}]")
    return "".join(lines).strip()


# --- Gemini answerer ------------------------------------------------------------

SYSTEM_PROMPT = """You answer questions about institutional policy documents.

Rules:
- Use ONLY the numbered sources provided. Do not use outside knowledge, and do not guess.
- Every sentence that states a fact must end with the number of the source it comes from, like [2]. Use [1][3] when two sources support it.
- Quote numbers, deadlines and amounts exactly as they appear in the sources.
- If the sources do not contain the answer, reply with exactly: NOT_FOUND
- If the sources only partly answer the question, answer that part and say what the documents do not cover.
- Be concise: a short paragraph or a few bullet points. No preamble."""


def build_prompt(question: str, sources: list[Source]) -> str:
    blocks = [f"[{s.n}] {cite_label(s)}\n{s.text}" for s in sources]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def clean_citations(answer: str, max_n: int) -> tuple[str, set[int]]:
    """Normalise "[1, 2]" to "[1][2]" and drop numbers that don't match a source."""
    used: set[int] = set()

    def repl(m):
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        valid = [n for n in nums if 1 <= n <= max_n]
        used.update(valid)
        return "".join(f"[{n}]" for n in valid)

    return CITATION.sub(repl, answer), used


def gemini_answer(question: str, sources: list[Source]) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": build_prompt(question, sources)}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
    }
    resp = httpx.post(url, json=body, headers={"x-goog-api-key": config.GEMINI_API_KEY}, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()


# --- entry point ----------------------------------------------------------------

def answer_question(store: VectorStore, question: str, k: int = config.TOP_K,
                    doc_ids: list[str] | None = None) -> Answer:
    started = time.perf_counter()
    question = question.strip()
    sources = to_sources(retrieve(store, question, k, doc_ids))

    def done(answer: Answer) -> Answer:
        answer.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return answer

    if not sources:
        return done(Answer(question, NOT_FOUND_MESSAGE, False, "none", []))

    note = None
    if config.GEMINI_API_KEY:
        try:
            text = gemini_answer(question, sources)
            if text.strip() == NOT_FOUND:
                return done(Answer(question, NOT_FOUND_MESSAGE, False, "gemini", sources, config.GEMINI_MODEL))
            text, used = clean_citations(text, len(sources))
            if used:
                for s in sources:
                    s.cited = s.n in used
                return done(Answer(question, text, True, "gemini", sources, config.GEMINI_MODEL))
            note = "The model's answer had no valid citations, so it was replaced with quoted passages."
        except Exception as exc:  # network, quota, bad key: fall back rather than fail
            log.warning("Gemini call failed: %s", exc)
            note = "The language model was unavailable, so this answer quotes the passages directly."

    text = extractive_answer(store, question, sources)
    if not text:
        return done(Answer(question, NOT_FOUND_MESSAGE, False, "extractive", sources))
    _, used = clean_citations(text, len(sources))
    for s in sources:
        s.cited = s.n in used
    return done(Answer(question, text, True, "extractive", sources, note=note))


def search(store: VectorStore, query: str, k: int = 8, doc_ids: list[str] | None = None) -> list[Source]:
    # Keyword-style queries ("medical certificate") score lower than full questions,
    # and search results are only ranked, not answered from, so the cut-off is looser.
    return to_sources(retrieve(store, query, k, doc_ids, min_similarity=config.MIN_SIMILARITY - 0.1))
