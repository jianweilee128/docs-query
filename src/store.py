import os
import sys
from pathlib import Path

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from chunking import chunk_text
from init_chroma import init_chroma

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "docs"
EMBED_MODEL = "text-embedding-3-small"


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
    batch_size: int = 100,
) -> int:
    collection = get_collection(target_collection)

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        ids = [f"chunk-{start + i}" for i in range(len(batch))]
        # Chroma embeds via OpenAIEmbeddingFunction
        collection.add(ids=ids, documents=batch)
        print(f"Stored {min(start + batch_size, len(texts))}/{len(texts)}")

    return collection.count()


if __name__ == "__main__":
    chunks = chunk_text()
    print(f"Found {len(chunks)} chunks")

    count = add_texts(chunks)
    print(f"Chroma count: {count}")
