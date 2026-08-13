"""Initialize a persistent ChromaDB client at the project root.

Usage:
    uv run python scripts/init_chroma.py
"""

from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma"


def init_chroma() -> chromadb.ClientAPI:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    client.heartbeat()
    print(f"Chroma path: {CHROMA_PATH}")
    return client


if __name__ == "__main__":
    init_chroma()
