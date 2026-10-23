"""Vector store: ChromaDB (persistent, cosine space) plus a small JSON registry of documents.

Embeddings come from all-MiniLM-L6-v2 run locally through ONNX Runtime, so
indexing and search cost nothing and need no API key.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from . import config
from .ingest import ParsedDocument


@dataclass
class DocInfo:
    doc_id: str
    title: str
    filename: str
    kind: str
    pages: int
    chunks: int
    source: str  # "sample" | "upload"
    added_at: float
    path: str


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    page: int | None
    section: str
    text: str
    similarity: float


class Embedder:
    def __init__(self):
        self._fn = ONNXMiniLM_L6_V2()
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vectors = np.asarray(self._fn(texts), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def embed_cached(self, texts: list[str]) -> np.ndarray:
        """Embed with a memo, for sentences that come up again across questions."""
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            if len(self._cache) > 20000:
                self._cache.clear()
            for text, vec in zip(missing, self.embed(missing)):
                self._cache[text] = vec
        return np.stack([self._cache[t] for t in texts]) if texts else self.embed([])


def embedding_text(title: str, section: str, text: str) -> str:
    # Prefixing the title and section helps short chunks match questions phrased around them.
    return f"{title}. {section}. {text}"


class VectorStore:
    def __init__(self, data_dir: Path | None = None, embedder: Embedder | None = None):
        self.data_dir = Path(data_dir or config.DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.data_dir / "documents.json"
        self.embedder = embedder or Embedder()
        self._lock = threading.Lock()
        self._client = chromadb.PersistentClient(
            path=str(self.data_dir / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            config.COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self._docs: dict[str, DocInfo] = {}
        # Documents indexed by an older version of the chunker; the caller re-ingests them.
        self.stale: list[DocInfo] = []
        self._load_registry()

    # --- registry -------------------------------------------------------------

    def _load_registry(self):
        if not self.registry_path.exists():
            return
        raw = json.loads(self.registry_path.read_text("utf-8"))
        docs = [DocInfo(**d) for d in (raw.get("documents", []) if isinstance(raw, dict) else raw)]
        if isinstance(raw, dict) and raw.get("version") == config.INDEX_VERSION:
            self._docs = {d.doc_id: d for d in docs}
            return
        self.stale = docs
        self._client.delete_collection(config.COLLECTION)
        self._col = self._client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
        self._save_registry()

    def _save_registry(self):
        data = {"version": config.INDEX_VERSION, "documents": [asdict(d) for d in self._docs.values()]}
        self.registry_path.write_text(json.dumps(data, indent=2), "utf-8")

    def documents(self) -> list[DocInfo]:
        return sorted(self._docs.values(), key=lambda d: (d.source != "sample", d.added_at))

    def get(self, doc_id: str) -> DocInfo | None:
        return self._docs.get(doc_id)

    def chunk_count(self) -> int:
        return self._col.count()

    # --- writes ---------------------------------------------------------------

    def add(self, doc: ParsedDocument, source: str, path: Path) -> DocInfo:
        if not doc.chunks:
            raise ValueError("No readable text was found in this file (is it a scanned image?).")
        texts = [embedding_text(doc.title, c.section, c.text) for c in doc.chunks]
        vectors = self.embedder.embed(texts)
        ids = [f"{doc.doc_id}:{c.index}" for c in doc.chunks]
        metadatas = [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "page": c.page or 0,
                "section": c.section,
                "index": c.index,
            }
            for c in doc.chunks
        ]
        with self._lock:
            self._col.delete(where={"doc_id": doc.doc_id})
            for start in range(0, len(ids), 500):
                end = start + 500
                self._col.add(
                    ids=ids[start:end],
                    embeddings=vectors[start:end].tolist(),
                    documents=[c.text for c in doc.chunks[start:end]],
                    metadatas=metadatas[start:end],
                )
            info = DocInfo(
                doc.doc_id, doc.title, doc.filename, doc.kind, doc.pages,
                len(doc.chunks), source, time.time(), str(path),
            )
            self._docs[doc.doc_id] = info
            self._save_registry()
        return info

    def delete(self, doc_id: str) -> DocInfo | None:
        with self._lock:
            info = self._docs.pop(doc_id, None)
            if info:
                self._col.delete(where={"doc_id": doc_id})
                self._save_registry()
        return info

    def warm_cache(self):
        """Pre-embed every stored sentence so the first answers are fast."""
        from .ingest import chunk_units

        data = self._col.get(include=["documents"])
        sentences = [s for doc in data["documents"] for s in chunk_units(doc)]
        for start in range(0, len(sentences), 64):
            self.embedder.embed_cached(sentences[start:start + 64])

    # --- reads ----------------------------------------------------------------

    def query(self, text: str, k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        if self._col.count() == 0:
            return []
        vector = self.embedder.embed([text])[0].tolist()
        where = None
        if doc_ids:
            where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}
        res = self._col.query(
            query_embeddings=[vector],
            n_results=min(k, self._col.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(
                Hit(
                    chunk_id=cid,
                    doc_id=meta["doc_id"],
                    title=meta["title"],
                    page=meta["page"] or None,
                    section=meta["section"],
                    text=doc,
                    similarity=round(1.0 - float(dist), 4),
                )
            )
        return hits
