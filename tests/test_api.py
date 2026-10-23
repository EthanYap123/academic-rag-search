from conftest import FIXTURES


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    assert h["answer_mode"] == "extractive"
    assert h["documents"] == 5 and h["chunks"] > 20


def test_ui_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Academic" in res.text
    for asset in ("app.css", "app.js", "favicon.svg"):
        assert client.get(f"/{asset}").status_code == 200


def test_ask_and_search(client):
    a = client.post("/api/ask", json={"question": "How many books can I borrow?"}).json()
    assert a["found"] and "eight items" in a["answer"]
    assert any(s["cited"] and s["page"] == 2 for s in a["sources"])

    s = client.post("/api/search", json={"query": "medical certificate"}).json()
    assert len(s["results"]) >= 2
    assert s["results"][0]["section"] == "4 Absence from an Examination"


def test_validation(client):
    assert client.post("/api/ask", json={"question": "x"}).status_code == 422
    assert client.post("/api/ask", json={"question": "a" * 501}).status_code == 422


def test_upload_ask_open_delete(client):
    pdf = (FIXTURES / "scholarship-guide.pdf").read_bytes()
    res = client.post("/api/documents", files={"file": ("scholarship-guide.pdf", pdf, "application/pdf")})
    assert res.status_code == 201
    doc = res.json()["document"]
    assert doc["title"] == "Scholarship and Financial Aid Guide" and doc["pages"] == 2

    again = client.post("/api/documents", files={"file": ("copy.pdf", pdf, "application/pdf")}).json()
    assert again["duplicate"]

    a = client.post("/api/ask", json={"question": "How much is the bursary?"}).json()
    cited = [s for s in a["sources"] if s["cited"]]
    assert cited[0]["title"] == doc["title"] and cited[0]["page"] == 2

    f = client.get(f"/api/documents/{doc['doc_id']}/file")
    assert f.status_code == 200 and f.headers["content-type"] == "application/pdf"

    assert client.delete(f"/api/documents/{doc['doc_id']}").status_code == 200
    ids = [d["doc_id"] for d in client.get("/api/documents").json()["documents"]]
    assert doc["doc_id"] not in ids


def test_upload_markdown(client):
    md = b"# Hostel Rules\n\n## Quiet Hours\n\nQuiet hours in the hostel are from 11 pm to 7 am every day.\n"
    doc = client.post("/api/documents", files={"file": ("hostel.md", md, "text/markdown")}).json()["document"]
    a = client.post("/api/ask", json={"question": "When are the hostel quiet hours?"}).json()
    assert any(s["cited"] and s["section"] == "Quiet Hours" for s in a["sources"])
    client.delete(f"/api/documents/{doc['doc_id']}")


def test_upload_rejections(client):
    bad_type = client.post("/api/documents", files={"file": ("notes.docx", b"hello", "application/octet-stream")})
    assert bad_type.status_code == 400
    broken = client.post("/api/documents", files={"file": ("broken.pdf", b"%PDF-1.4 not really", "application/pdf")})
    assert broken.status_code == 422
    empty = client.post("/api/documents", files={"file": ("empty.txt", b"", "text/plain")})
    assert empty.status_code == 400


def test_samples_cannot_be_deleted(client):
    sample = client.get("/api/documents").json()["documents"][0]
    assert sample["source"] == "sample"
    assert client.delete(f"/api/documents/{sample['doc_id']}").status_code == 403
    assert client.delete("/api/documents/nope").status_code == 404
