"""Build the static, in-browser demo (deployed to GitHub Pages).

The sample handbooks are parsed, chunked and embedded here by the same Python
pipeline the server uses, and written to index.json. In the browser,
demo-api.js stands in for the FastAPI backend: it embeds questions with the
same model (all-MiniLM-L6-v2 via transformers.js) and runs the same retrieval
and extractive answering logic.

    python scripts/build_demo.py [output_dir]   # default: _site
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.ingest import parse_file  # noqa: E402
from app.store import Embedder, embedding_text  # noqa: E402

STATIC = ROOT / "app" / "static"


def main(out: Path):
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(STATIC, out)
    (out / "files").mkdir()

    html = (out / "index.html").read_text("utf-8")
    html = html.replace('<script src="app.js"></script>',
                        '<script src="demo-api.js"></script>\n  <script src="app.js"></script>')
    (out / "index.html").write_text(html, "utf-8")
    shutil.copy2(ROOT / "app" / "demo" / "demo-api.js", out / "demo-api.js")

    embedder = Embedder()
    docs, chunks = [], []
    for path in sorted(config.SAMPLES_DIR.iterdir()):
        if path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
            continue
        shutil.copy2(path, out / "files" / path.name)
        doc = parse_file(path.read_bytes(), path.name)
        vectors = embedder.embed([embedding_text(doc.title, c.section, c.text) for c in doc.chunks])
        docs.append({
            "doc_id": doc.doc_id, "title": doc.title, "filename": doc.filename, "kind": doc.kind,
            "pages": doc.pages, "chunks": len(doc.chunks), "source": "sample", "added_at": 0,
            "file": f"files/{path.name}",
        })
        for c, v in zip(doc.chunks, vectors):
            chunks.append({
                "chunk_id": f"{doc.doc_id}:{c.index}", "doc_id": doc.doc_id, "title": doc.title,
                "page": c.page, "section": c.section, "text": c.text,
                "v": [round(float(x), 5) for x in v],
            })

    index = {
        "config": {
            "chunk_chars": config.CHUNK_CHARS, "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": config.TOP_K, "min_similarity": config.MIN_SIMILARITY,
            "max_upload_mb": config.MAX_UPLOAD_MB, "max_uploaded_docs": config.MAX_UPLOADED_DOCS,
        },
        "documents": docs,
        "chunks": chunks,
    }
    (out / "index.json").write_text(json.dumps(index, separators=(",", ":")), "utf-8")
    (out / ".nojekyll").write_text("")
    print(f"demo built in {out}: {len(docs)} documents, {len(chunks)} chunks")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_site")
