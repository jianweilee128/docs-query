import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from chunking import chunk_text
from init_chroma import init_chroma

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "docs"
EMBED_MODEL = "text-embedding-3-small"


def get_collection(target_collection: str | None = None):
    chroma = init_chroma()
    name = target_collection or COLLECTION_NAME
    return chroma.get_or_create_collection(name=name)


def embed_texts(texts: list[str], openai_client: OpenAI) -> list[list[float]]:
    response = openai_client.embeddings.create(
        input=texts,
        model=EMBED_MODEL,
    )
    # OpenAI may not return items in request order — sort by index
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def add_texts(
    texts: list[str],
    openai_client: OpenAI,
    target_collection: str | None = None,
    batch_size: int = 100,
) -> int:
    collection = get_collection(target_collection)

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings = embed_texts(batch, openai_client)
        ids = [f"chunk-{start + i}" for i in range(len(batch))]
        collection.add(ids=ids, documents=batch, embeddings=embeddings)
        print(f"Stored {min(start + batch_size, len(texts))}/{len(texts)}")

    return collection.count()


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY missing — set it in .env")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    chunks = chunk_text()
    print(f"Found {len(chunks)} chunks")

    count = add_texts(chunks, openai_client)
    print(f"Chroma count: {count}")
