import pytest

from app import config, rag


def ask(store, question):
    return rag.answer_question(store, question)


def test_answer_cites_the_right_page_and_section(store):
    a = ask(store, "What is the penalty for submitting coursework late?")
    assert a.found and a.mode == "extractive"
    assert "10 percent" in a.answer
    top = next(s for s in a.sources if s.cited)
    assert top.title == "Academic Policy Handbook 2026"
    assert top.page == 2
    assert top.section == "3.1 Late Submission of Coursework"
    assert f"[{top.n}]" in a.answer


def test_every_answer_sentence_is_cited_and_quoted_verbatim(store):
    a = ask(store, "What happens if I miss an exam because I was sick?")
    assert a.found
    by_n = {s.n: s for s in a.sources}
    for line in filter(None, (l.strip() for l in a.answer.split("\n"))):
        quote, n = line.lstrip("- ").rsplit(" [", 1)
        source = by_n[int(n.rstrip("]"))]
        assert quote.lstrip("• ") in source.text  # extractive answers never invent text


def test_list_answers_include_the_whole_list(store):
    a = ask(store, "How is the internship graded?")
    assert "supervisor's evaluation (40 percent)" in a.answer
    assert "final report and presentation (40 percent)" in a.answer


def test_unrelated_questions_are_refused(store):
    a = ask(store, "What is the capital of France?")
    assert not a.found
    assert a.answer == rag.NOT_FOUND_MESSAGE
    assert a.sources == []


def test_document_filter(store):
    doc = next(d for d in store.documents() if d.title.startswith("IT and Library"))
    a = rag.answer_question(store, "What are the rules?", doc_ids=[doc.doc_id])
    assert all(s.doc_id == doc.doc_id for s in a.sources)


def test_clean_citations():
    text, used = rag.clean_citations("Fact one [1, 2]. Fact two [9]. Fact three [3].", 3)
    assert text == "Fact one [1][2]. Fact two . Fact three [3]."
    assert used == {1, 2, 3}


# --- Gemini path, with the network call replaced ------------------------------

@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    replies = {}

    def fake(question, sources):
        prompt = rag.build_prompt(question, sources)
        assert "[1]" in prompt and "page" in prompt
        reply = replies["text"]
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(rag, "gemini_answer", fake)
    return replies


def test_gemini_answer_with_valid_citations(store, gemini):
    gemini["text"] = "Late work loses 10 percent per day, up to three days [1]."
    a = ask(store, "What is the penalty for late coursework?")
    assert a.mode == "gemini" and a.found
    assert [s.n for s in a.sources if s.cited] == [1]


def test_gemini_answer_without_citations_is_rejected(store, gemini):
    gemini["text"] = "You will probably be fine."
    a = ask(store, "What is the penalty for late coursework?")
    assert a.mode == "extractive"
    assert "no valid citations" in a.note


def test_gemini_not_found_is_refused(store, gemini):
    gemini["text"] = "NOT_FOUND"
    a = ask(store, "What is the penalty for late coursework?")
    assert not a.found and a.answer == rag.NOT_FOUND_MESSAGE


def test_gemini_failure_falls_back(store, gemini):
    gemini["text"] = RuntimeError("quota exceeded")
    a = ask(store, "What is the penalty for late coursework?")
    assert a.found and a.mode == "extractive"
    assert "unavailable" in a.note
