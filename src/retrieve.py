import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from store import get_collection


def retrieve_chunks(
    query: str,
    target_collection: str | None = None,
    n_results: int = 5,
) -> list[dict[str, str]]:
    """Return [{id, document}, ...] for citation tagging."""
    collection = get_collection(target_collection)
    results = collection.query(query_texts=[query], n_results=n_results)

    ids = results["ids"][0] if results["ids"] else []
    documents = results["documents"][0] if results["documents"] else []

    return [
        {"id": chunk_id, "document": doc}
        for chunk_id, doc in zip(ids, documents)
        if doc
    ]


def retrieve_text(
    query: str,
    target_collection: str | None = None,
    n_results: int = 5,
) -> list[str]:
    return [
        c["document"]
        for c in retrieve_chunks(query, target_collection, n_results)
    ]


if __name__ == "__main__":
    hits = retrieve_chunks("how do I create a component?", "docs")
    for hit in hits:
        print(f"\n--- {hit['id']} ---\n{hit['document'][:200]}")
