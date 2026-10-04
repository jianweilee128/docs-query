#!/bin/sh
set -e

if [ -z "$OPENAI_API_KEY" ]; then
  echo "OPENAI_API_KEY is required (embeddings, and chat unless LLM_BASE_URL is set)." >&2
  exit 1
fi

if [ ! -f /app/chroma/chroma.sqlite3 ]; then
  echo "No Chroma index found — ingesting Angular docs (one-time)."
  python -m rag.store
fi

exec "$@"
