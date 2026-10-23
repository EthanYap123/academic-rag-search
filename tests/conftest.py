import os
import sys
import tempfile
from pathlib import Path

import pytest

# Point the app at a throwaway data directory before it is imported.
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="rag-test-")
os.environ.pop("GEMINI_API_KEY", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def store(client):
    from app import main

    return main.store
