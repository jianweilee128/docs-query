from config.settings import COLLECTION_NAME, DOC_VERSION, TOP_K
from rag.store import get_collection


def retrieve_chunks(
    query: str,
    target_collection: str | None = None,
    n_results: int | None = None,
    where: dict | None = None,
    latest_only: bool = True,
) -> list[dict]:
    """Return [{id, document, metadata}, ...]. Optionally filter by metadata."""
    collection = get_collection(target_collection)

    filters = dict(where or {})
    if latest_only and "doc_version" not in filters:
        filters["doc_version"] = DOC_VERSION

    query_kwargs: dict = {
        "query_texts": [query],
        "n_results": n_results if n_results is not None else TOP_K,
        "include": ["documents", "metadatas"],
    }
    if len(filters) == 1:
        query_kwargs["where"] = filters
    elif len(filters) > 1:
        query_kwargs["where"] = {"$and": [{k: v} for k, v in filters.items()]}

    results = collection.query(**query_kwargs)

    ids = results["ids"][0] if results["ids"] else []
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    chunks = []
    for chunk_id, doc, meta in zip(ids, documents, metadatas):
        if not doc:
            continue
        chunks.append(
            {
                "id": chunk_id,
                "document": doc,
                "metadata": meta or {},
            }
        )
    return chunks


def retrieve_text(
    query: str,
    target_collection: str | None = None,
    n_results: int | None = None,
    where: dict | None = None,
) -> list[str]:
    return [
        c["document"]
        for c in retrieve_chunks(query, target_collection, n_results, where)
    ]


if __name__ == "__main__":
    hits = retrieve_chunks("how do I create a component?", COLLECTION_NAME)
    for hit in hits:
        meta = hit["metadata"]
        print(
            f"\n--- {hit['id']} | {meta.get('heading')} | v={meta.get('doc_version')} ---"
        )
        print(hit["document"][:200])
