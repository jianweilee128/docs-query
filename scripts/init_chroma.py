"""Initialize a persistent ChromaDB client.

Usage:
    uv run python scripts/init_chroma.py
"""

from rag.chroma_client import init_chroma

if __name__ == "__main__":
    init_chroma()
