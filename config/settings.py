"""Central project settings. Secrets stay in .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Paths
DATA_PATH = ROOT / "data" / "llms-full.txt"
CHROMA_PATH = ROOT / "chroma"
PROMPT_GENERATE_PATH = ROOT / "prompts" / "generate.md"
EVAL_QUESTIONS_PATH = ROOT / "eval" / "questions.json"
EVAL_RESULTS_DIR = ROOT / "eval" / "results"

# Chroma
COLLECTION_NAME = "docs"

# Chunking
CHUNK_SIZE = 1000

# Embeddings / store
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Retrieve / generate
TOP_K = 5
CHAT_MODEL = "gpt-4.1-mini"
