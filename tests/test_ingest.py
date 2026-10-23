from pathlib import Path

from app import config
from app.ingest import chunk_units, is_heading, pack, parse_file, split_sentences

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def parse(name):
    path = SAMPLES / name
    return parse_file(path.read_bytes(), path.name)


def test_pdf_title_pages_and_sections():
    doc = parse("academic-policy-handbook.pdf")
    assert doc.title == "Academic Policy Handbook 2026"
    assert doc.pages == 3
    late = [c for c in doc.chunks if c.section == "3.1 Late Submission of Coursework"]
    assert late and late[0].page == 2
    assert "10 percent" in late[0].text


def test_running_headers_and_page_numbers_are_stripped():
    for name in ("examination-regulations.pdf", "student-conduct-code.pdf"):
        for chunk in parse(name).chunks:
            assert "Northbridge Polytechnic |" not in chunk.text
            assert "Page 1 of" not in chunk.text


def test_pdf_bullets_stay_whole():
    doc = parse("examination-regulations.pdf")
    chunk = next(c for c in doc.chunks if c.section.startswith("3 "))
    units = chunk_units(chunk.text)
    bullets = [u for u in units if u.startswith("•")]
    assert len(bullets) == 3
    assert "Programmable and graphing calculators" in bullets[1]


def test_markdown_sections_have_no_page():
    doc = parse("internship-programme-faq.md")
    assert doc.title == "Internship Programme FAQ"
    graded = next(c for c in doc.chunks if c.section == "How is the internship graded?")
    assert graded.page is None
    assert graded.text.count("•") == 3


def test_chunks_respect_size_and_overlap():
    sentences = [f"Sentence number {i} talks about the attendance rule in some detail." for i in range(60)]
    chunks = pack(sentences, 4, "2 Attendance")
    assert len(chunks) > 1
    assert all(len(c.text) <= config.CHUNK_CHARS + 80 for c in chunks)
    assert all(c.page == 4 and c.section == "2 Attendance" for c in chunks)
    # consecutive chunks share a sentence of overlap
    last = chunks[0].text.split(". ")[-1]
    assert last.rstrip(".") in chunks[1].text


def test_heading_detection():
    assert is_heading("3.1 Late Submission of Coursework")
    assert is_heading("EXAMINATION RULES")
    assert not is_heading("3 days after the deadline the work receives zero marks.")
    assert not is_heading("10 percent is deducted")


def test_sentence_split_keeps_abbreviations():
    parts = split_sentences("Bring items, e.g. pens and rulers. Phones are not allowed.")
    assert parts == ["Bring items, e.g. pens and rulers.", "Phones are not allowed."]
