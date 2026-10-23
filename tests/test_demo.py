import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_demo  # noqa: E402


def test_demo_build(tmp_path):
    out = tmp_path / "site"
    build_demo.main(out)

    html = (out / "index.html").read_text("utf-8")
    assert html.index("demo-api.js") < html.index("app.js")

    index = json.loads((out / "index.json").read_text("utf-8"))
    assert len(index["documents"]) == 5
    assert index["config"]["min_similarity"] > 0
    for doc in index["documents"]:
        assert (out / doc["file"]).exists()
    chunk = index["chunks"][0]
    assert len(chunk["v"]) == 384
    assert abs(sum(x * x for x in chunk["v"]) - 1) < 1e-3  # unit-length vectors
    assert {"chunk_id", "doc_id", "title", "page", "section", "text"} <= chunk.keys()
