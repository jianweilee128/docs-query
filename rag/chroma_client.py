"""Initialize a persistent ChromaDB client."""

import threading

import chromadb

from config.settings import CHROMA_PATH

_client: chromadb.ClientAPI | None = None
_lock = threading.Lock()


def init_chroma() -> chromadb.ClientAPI:
    """Return the process-wide client, building it exactly once."""
    global _client
    if _client is None:
        # Without the lock, concurrent first calls each build a client and race
        # on creating the default tenant.
        with _lock:
            if _client is None:
                CHROMA_PATH.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(CHROMA_PATH))
                client.heartbeat()
                print(f"Chroma path: {CHROMA_PATH}")
                _client = client
    return _client


def reset_chroma() -> None:
    global _client
    with _lock:
        _client = None
