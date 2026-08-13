import os

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

from chunking import chunk_text

load_dotenv()

COLLECTION_NAME = "angular-docs"
CHROMA_PATH = "./chroma"
EMBED_MODEL = "text-embedding-3-small"


def store_chunks(chunks: list[str], batch_size: int = 100) -> chromadb.Collection:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY missing — set it in .env")

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBED_MODEL,
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Rebuild so re-runs stay in sync with current chunking
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    ids = [f"chunk-{i}" for i in range(len(chunks))]
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(ids=ids[start:end], documents=chunks[start:end])
        print(f"Stored {min(end, len(chunks))}/{len(chunks)}")

    return collection


if __name__ == "__main__":
    chunks = chunk_text()
    print(f"Found {len(chunks)} chunks")
    print(f"max={max(map(len, chunks))} min={min(map(len, chunks))}")

    collection = store_chunks(chunks)
    print(f"Chroma count: {collection.count()}")
