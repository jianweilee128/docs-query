from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config.settings import (
    BATCH_SIZE,
    COLLECTION_NAME,
    EMBED_MODEL,
    OPENAI_API_KEY,
)
from rag.chroma_client import init_chroma
from rag.chunking import chunk_text


def get_embedding_function() -> OpenAIEmbeddingFunction:
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY missing — set it in .env")
    return OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL,
    )


def get_collection(target_collection: str | None = None):
    chroma = init_chroma()
    name = target_collection or COLLECTION_NAME
    return chroma.get_or_create_collection(
        name=name,
        embedding_function=get_embedding_function(),
    )


def add_texts(
    texts: list[str],
    target_collection: str | None = None,
    batch_size: int | None = None,
) -> int:
    collection = get_collection(target_collection)
    size = batch_size if batch_size is not None else BATCH_SIZE

    for start in range(0, len(texts), size):
        batch = texts[start : start + size]
        ids = [f"chunk-{start + i}" for i in range(len(batch))]
        collection.add(ids=ids, documents=batch)
        print(f"Stored {min(start + size, len(texts))}/{len(texts)}")

    return collection.count()


if __name__ == "__main__":
    chunks = chunk_text()
    print(f"Found {len(chunks)} chunks")

    count = add_texts(chunks)
    print(f"Chroma count: {count}")
