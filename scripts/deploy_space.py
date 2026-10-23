"""Publish this repository to a Hugging Face Docker Space.

Needs HF_TOKEN (a write token). Optional: GEMINI_API_KEY is stored as a Space
secret, and SPACE_NAME overrides the Space name (default: academic-rag-search).

    python scripts/deploy_space.py
"""

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE_HEADER = """---
title: Academic RAG Search
emoji: 📚
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Grounded answers over policy handbooks, with page citations
---

"""
INCLUDE = ["app", "data/samples", "scripts", "Dockerfile", ".dockerignore", "requirements.txt", "README.md"]


def main():
    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    repo_id = f"{user}/{os.getenv('SPACE_NAME', 'academic-rag-search')}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for item in INCLUDE:
            src = ROOT / item
            dst = stage / item
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        readme = stage / "README.md"
        readme.write_text(SPACE_HEADER + readme.read_text("utf-8"), "utf-8")
        api.upload_folder(
            folder_path=stage, repo_id=repo_id, repo_type="space",
            commit_message="Deploy from GitHub", delete_patterns=["app/**", "scripts/**", "data/**"],
        )

    if os.getenv("GEMINI_API_KEY"):
        api.add_space_secret(repo_id, "GEMINI_API_KEY", os.environ["GEMINI_API_KEY"])

    print(f"Deployed: https://huggingface.co/spaces/{repo_id}")
    print(f"App URL:  https://{repo_id.replace('/', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
