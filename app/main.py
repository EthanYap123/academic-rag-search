"""FastAPI app: JSON API under /api, and the single-page UI served from /."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, rag
from .ingest import doc_id_for, parse_file
from .store import DocInfo, VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

STATIC_DIR = Path(__file__).parent / "static"
store: VectorStore | None = None


def seed_samples(vs: VectorStore) -> None:
    """Index everything in data/samples, and drop sample entries whose file changed or went away."""
    current: set[str] = set()
    for path in sorted(config.SAMPLES_DIR.iterdir()):
        if path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
            continue
        data = path.read_bytes()
        doc_id = doc_id_for(data)
        current.add(doc_id)
        if vs.get(doc_id):
            continue
        info = vs.add(parse_file(data, path.name), "sample", path)
        log.info("indexed sample %s (%d chunks)", info.title, info.chunks)
    for doc in vs.documents():
        if doc.source == "sample" and doc.doc_id not in current:
            vs.delete(doc.doc_id)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global store
    store = VectorStore()
    seed_samples(store)
    for doc in store.stale:  # re-index uploads left from an older index version
        path = Path(doc.path)
        if doc.source == "upload" and path.exists():
            store.add(parse_file(path.read_bytes(), doc.filename), "upload", path)
    threading.Thread(target=store.warm_cache, daemon=True).start()
    log.info("ready: %d documents, %d chunks, answer mode %s",
             len(store.documents()), store.chunk_count(), answer_mode())
    yield


app = FastAPI(title="Academic Resource RAG Search Engine", lifespan=lifespan)


def answer_mode() -> str:
    return "gemini" if config.GEMINI_API_KEY else "extractive"


def doc_json(d: DocInfo) -> dict:
    return {
        "doc_id": d.doc_id, "title": d.title, "filename": d.filename, "kind": d.kind,
        "pages": d.pages, "chunks": d.chunks, "source": d.source, "added_at": d.added_at,
    }


# --- API ------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=config.TOP_K, ge=1, le=10)
    doc_ids: list[str] | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=8, ge=1, le=20)
    doc_ids: list[str] | None = None


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "answer_mode": answer_mode(),
        "model": config.GEMINI_MODEL if config.GEMINI_API_KEY else None,
        "embedding_model": "all-MiniLM-L6-v2 (ONNX, local)",
        "vector_store": "ChromaDB (cosine)",
        "documents": len(store.documents()),
        "chunks": store.chunk_count(),
        "max_upload_mb": config.MAX_UPLOAD_MB,
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    return rag.answer_question(store, req.question, req.top_k, req.doc_ids or None).to_dict()


@app.post("/api/search")
def search(req: SearchRequest):
    sources = rag.search(store, req.query, req.top_k, req.doc_ids or None)
    return {"query": req.query, "results": [s.__dict__ for s in sources]}


@app.get("/api/documents")
def list_documents():
    return {"documents": [doc_json(d) for d in store.documents()]}


@app.post("/api/documents", status_code=201)
def upload_document(file: UploadFile = File(...)):
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Only PDF, Markdown (.md) and plain-text (.txt) files can be uploaded.")
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    if not data:
        raise HTTPException(400, "The file is empty.")

    existing = store.get(doc_id_for(data))
    if existing:
        return {"document": doc_json(existing), "duplicate": True, "evicted": []}

    try:
        parsed = parse_file(data, name)
    except Exception as exc:
        log.warning("could not parse %s: %s", name, exc)
        raise HTTPException(422, "This file could not be read. Is it a valid, unencrypted PDF?")
    if not parsed.chunks:
        raise HTTPException(422, "No text could be extracted. Scanned PDFs need OCR before upload.")

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / f"{parsed.doc_id}{ext}"
    path.write_bytes(data)
    info = store.add(parsed, "upload", path)

    # Keep the public demo bounded: the oldest uploads make way for new ones.
    evicted = []
    uploads = [d for d in store.documents() if d.source == "upload"]
    for old in uploads[: max(0, len(uploads) - config.MAX_UPLOADED_DOCS)]:
        remove_document(old.doc_id)
        evicted.append(old.title)
    return {"document": doc_json(info), "duplicate": False, "evicted": evicted}


def remove_document(doc_id: str) -> DocInfo | None:
    info = store.delete(doc_id)
    if info and info.source == "upload":
        Path(info.path).unlink(missing_ok=True)
    return info


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    doc = store.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    if doc.source == "sample":
        raise HTTPException(403, "Sample documents can't be removed.")
    remove_document(doc_id)
    return {"deleted": doc_id}


@app.get("/api/documents/{doc_id}/file")
def document_file(doc_id: str):
    doc = store.get(doc_id)
    if not doc or not Path(doc.path).exists():
        raise HTTPException(404, "Document not found.")
    media = "application/pdf" if doc.kind == "pdf" else "text/plain; charset=utf-8"
    return FileResponse(doc.path, media_type=media, filename=doc.filename,
                        content_disposition_type="inline")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
