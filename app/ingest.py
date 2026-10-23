"""Document parsing and chunking.

PDFs are read page by page so every chunk keeps the page it came from. Section
headings (numbered like "3.2 Late Submission", or Markdown "#" headings) are
tracked as the text is walked, so each chunk also knows its section. Chunks never
cross a page or a section boundary, which keeps citations exact.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from . import config


@dataclass
class Chunk:
    text: str
    page: int | None
    section: str
    index: int = 0


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    filename: str
    kind: str  # "pdf" | "markdown" | "text"
    pages: int
    chunks: list[Chunk] = field(default_factory=list)


# --- text helpers -----------------------------------------------------------

NUMBERED_HEADING = re.compile(r"^(?:section\s+)?(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{1,80})$", re.I)
PAGE_FOOTER = re.compile(r"^(page\s+)?\d+(\s*(of|/)\s*\d+)?$", re.I)
BULLET = re.compile(r"^\s*(?:[•\-\*–]|\(?[a-z]\)|\d+\))\s+")
ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "Mr.", "Ms.", "Dr.", "No.", "vs.", "approx.")
SENTENCE_END = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9\"'(•])")


def doc_id_for(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:12]


BULLET_GLYPHS = re.compile("[\x7f●▪◦‣⁃■○]")


def clean(text: str) -> str:
    text = text.replace(" ", " ").replace("­", "")
    text = BULLET_GLYPHS.sub("•", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    return re.sub(r"[ \t]+", " ", text)


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90 or line.endswith((".", ",", ";")):
        return False
    if NUMBERED_HEADING.match(line):
        # "3 days late" or "10 marks deducted" are sentences, not headings
        words = line.split()
        return len(words) <= 12 and not words[1][0].islower()
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 4 and line.isupper() and len(line.split()) <= 8


def split_sentences(text: str) -> list[str]:
    protected = text
    for i, abbr in enumerate(ABBREVIATIONS):
        protected = protected.replace(abbr, f"\x00{i}\x00")
    parts = SENTENCE_END.split(protected)
    out = []
    for part in parts:
        for i, abbr in enumerate(ABBREVIATIONS):
            part = part.replace(f"\x00{i}\x00", abbr)
        part = part.strip()
        if part:
            out.append(part)
    return out


def lines_to_units(lines: list[str]) -> list[str]:
    """Join wrapped lines back into sentences; bullets always start a new unit."""
    units: list[str] = []
    buf: list[str] = []
    in_bullet = False
    bullet_next = False  # PDFs often put the bullet glyph on its own line

    def flush():
        nonlocal in_bullet
        if buf:
            text = " ".join(buf)
            # A list item stays one unit even if it has two sentences.
            units.extend([text] if in_bullet else split_sentences(text))
            buf.clear()
        in_bullet = False

    for line in lines:
        line = line.strip()
        if not line:
            flush()
            continue
        if line in ("•", "-", "*", "–"):
            bullet_next = True
            continue
        if bullet_next or BULLET.match(line):
            flush()
            buf.append("• " + BULLET.sub("", line))
            in_bullet, bullet_next = True, False
            continue
        if in_bullet and buf[-1].endswith((".", "!", "?")):
            flush()  # a list item that ended on a full line; this line starts new text
        buf.append(line)
    flush()
    return units


def join_units(units: list[str]) -> str:
    """Sentences are joined with spaces; list items go on their own line."""
    out = ""
    for unit in units:
        if not out:
            out = unit
        elif unit.startswith("•") or out.endswith(":") or out.split("\n")[-1].startswith("•"):
            out += "\n" + unit
        else:
            out += " " + unit
    return out


def chunk_units(text: str) -> list[str]:
    """Inverse of join_units: list items whole, other lines split into sentences."""
    units = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("•"):
            units.append(line)
        elif line:
            units.extend(split_sentences(line))
    return units


def pack(units: list[str], page: int | None, section: str) -> list[Chunk]:
    """Greedily pack sentences into chunks of ~CHUNK_CHARS, carrying a small overlap."""
    chunks: list[Chunk] = []
    current: list[str] = []
    size = 0
    fresh = 0  # sentences in `current` that are not overlap from the previous chunk
    for unit in units:
        if fresh and size + len(unit) > config.CHUNK_CHARS:
            chunks.append(Chunk(join_units(current), page, section))
            # keep trailing sentences (up to CHUNK_OVERLAP chars) as context for the next chunk
            tail: list[str] = []
            tail_size = 0
            for s in reversed(current):
                if tail_size + len(s) > config.CHUNK_OVERLAP:
                    break
                tail.insert(0, s)
                tail_size += len(s) + 1
            current, size, fresh = tail, tail_size, 0
        current.append(unit)
        size += len(unit) + 1
        fresh += 1
    if fresh:
        chunks.append(Chunk(join_units(current), page, section))
    return chunks


# --- parsers ----------------------------------------------------------------

def _running_lines(pages: list[list[str]]) -> set[str]:
    """Lines repeated on most pages (running headers/footers) are noise."""
    if len(pages) < 2:
        return set()
    counts = Counter(line for lines in pages for line in {l.strip() for l in lines if l.strip()})
    needed = len(pages) if len(pages) < 5 else len(pages) * 0.6
    return {line for line, n in counts.items() if n >= needed and not is_heading(line)}


def parse_pdf(data: bytes, filename: str) -> ParsedDocument:
    from io import BytesIO

    reader = PdfReader(BytesIO(data))
    raw_pages = [clean(page.extract_text() or "").split("\n") for page in reader.pages]
    noise = _running_lines(raw_pages)

    meta_title = ""
    if reader.metadata and reader.metadata.title:
        meta_title = str(reader.metadata.title).strip()

    title = meta_title
    section = "Introduction"
    chunks: list[Chunk] = []
    for page_no, lines in enumerate(raw_pages, start=1):
        buffer: list[str] = []

        def emit():
            if buffer:
                chunks.extend(pack(lines_to_units(buffer), page_no, section))
                buffer.clear()

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped in noise or PAGE_FOOTER.match(stripped):
                continue
            if not title and page_no == 1:
                title = stripped  # first line of the document
                continue
            if page_no == 1 and stripped == title:
                continue
            if is_heading(stripped):
                emit()
                section = stripped
                continue
            buffer.append(stripped)
        emit()

    return _finish(data, filename, title, "pdf", len(reader.pages), chunks)


MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_markdown(data: bytes, filename: str, kind: str = "markdown") -> ParsedDocument:
    text = clean(data.decode("utf-8", errors="replace"))
    title = ""
    section = "Introduction"
    chunks: list[Chunk] = []
    buffer: list[str] = []

    def emit():
        if buffer:
            chunks.extend(pack(lines_to_units(buffer), None, section))
            buffer.clear()

    for line in text.split("\n"):
        m = MD_HEADING.match(line.strip())
        if m:
            emit()
            heading = m.group(2).strip().strip("#").strip()
            if len(m.group(1)) == 1 and not title:
                title = heading
            else:
                section = heading
            continue
        buffer.append(re.sub(r"[*_`]{1,3}", "", line))
    emit()
    return _finish(data, filename, title, kind, 0, chunks)


def _finish(data, filename, title, kind, pages, chunks) -> ParsedDocument:
    chunks = [c for c in chunks if len(c.text) >= 25]
    for i, c in enumerate(chunks):
        c.index = i
    title = title or Path(filename).stem.replace("-", " ").replace("_", " ").title()
    return ParsedDocument(doc_id_for(data), title[:120], filename, kind, pages, chunks)


def parse_file(data: bytes, filename: str) -> ParsedDocument:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(data, filename)
    if ext in (".md", ".markdown"):
        return parse_markdown(data, filename)
    if ext == ".txt":
        return parse_markdown(data, filename, kind="text")
    raise ValueError(f"Unsupported file type: {ext}")
