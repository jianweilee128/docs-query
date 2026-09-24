import threading
from functools import lru_cache

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config.settings import (
    BATCH_SIZE,
    COLLECTION_NAME,
    EMBED_MODEL,
    OPENAI_API_KEY,
)
from rag.chroma_client import init_chroma
from rag.chunking import chunk_text


@lru_cache(maxsize=1)
def get_embedding_function() -> OpenAIEmbeddingFunction:
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY missing — set it in .env")
    return OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL,
    )


_collections: dict = {}
_collections_lock = threading.Lock()


def get_collection(target_collection: str | None = None):
    # Resolve the default first so None and "docs" share one cache entry.
    name = target_collection or COLLECTION_NAME
    if name not in _collections:
        with _collections_lock:
            if name not in _collections:
                _collections[name] = init_chroma().get_or_create_collection(
                    name=name,
                    embedding_function=get_embedding_function(),
                )
    return _collections[name]


def upsert_chunks(
    chunks: list[dict],
    target_collection: str | None = None,
    batch_size: int | None = None,
    drop_stale: bool = True,
) -> int:
    """Upsert by stable chunk id; optionally delete ids no longer in the corpus."""
    collection = get_collection(target_collection)
    size = batch_size if batch_size is not None else BATCH_SIZE

    new_ids = [chunk["id"] for chunk in chunks]
    if drop_stale:
        existing = collection.get(include=[])["ids"]
        stale = list(set(existing) - set(new_ids))
        if stale:
            # delete in batches — Chroma can choke on huge id lists
            for start in range(0, len(stale), size):
                collection.delete(ids=stale[start : start + size])
            print(f"Removed {len(stale)} stale chunks")

    for start in range(0, len(chunks), size):
        batch = chunks[start : start + size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["document"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"Upserted {min(start + size, len(chunks))}/{len(chunks)}")

    return collection.count()


if __name__ == "__main__":
    chunks = chunk_text()
    print(f"Found {len(chunks)} chunks")
    print(f"Sample id: {chunks[0]['id']}")
    print(f"Sample metadata: {chunks[0]['metadata']}")

    count = upsert_chunks(chunks)
    print(f"Chroma count: {count}")
