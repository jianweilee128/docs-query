"""Initialize a persistent ChromaDB client."""

import chromadb

from config.settings import CHROMA_PATH


def init_chroma() -> chromadb.ClientAPI:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    client.heartbeat()
    print(f"Chroma path: {CHROMA_PATH}")
    return client
