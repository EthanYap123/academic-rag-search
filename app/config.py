"""Runtime settings, read once from environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
SAMPLES_DIR = ROOT / "data" / "samples"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

COLLECTION = "academic_docs"
# Bump when parsing or chunking changes, so existing indexes are rebuilt on startup.
INDEX_VERSION = 2

# Chunking: aim for ~180 words per chunk, with one or two sentences of overlap.
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 160))

# Retrieval
TOP_K = int(os.getenv("TOP_K", 5))
# Chunks with cosine similarity below this are treated as unrelated to the question.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", 0.30))

# Generation (optional). Without a key the extractive answerer is used.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Upload limits for the public demo
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 10))
MAX_UPLOADED_DOCS = int(os.getenv("MAX_UPLOADED_DOCS", 25))
ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}
