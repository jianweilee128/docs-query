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
PROMPT_JUDGE_PATH = ROOT / "prompts" / "judge.md"
EVAL_QUESTIONS_PATH = ROOT / "eval" / "questions.json"
EVAL_GOLD_PATH = ROOT / "eval" / "gold_labels.json"
EVAL_RESULTS_DIR = ROOT / "eval" / "results"

# Chroma
COLLECTION_NAME = "docs"
CORPUS_NAME = "angular"
DOC_VERSION = "current"

# Chunking
CHUNK_SIZE = 1000

# Embeddings / store (still OpenAI unless you also point this at a local embedder)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
BATCH_SIZE = 100
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Chat / judge. Empty LLM_BASE_URL → official OpenAI.
# Ollama: LLM_BASE_URL=http://localhost:11434/v1  CHAT_MODEL=qwen2.5
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", CHAT_MODEL)

# Retrieve / generate
TOP_K = 5

# Eval
EVAL_WORKERS = 8
