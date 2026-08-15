import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrieve import retrieve_chunks

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = "gpt-4.1-mini"
PROMPT_PATH = ROOT / "prompts" / "generate.md"


def build_prompt(query: str, chunks: list[dict[str, str]]) -> str:
    context = "\n\n---\n\n".join(
        f"[{chunk['id']}]\n{chunk['document']}" for chunk in chunks
    )
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(query=query, context=context)


def generate_answer(
    query: str,
    target_collection: str | None = None,
    n_results: int = 5,
) -> tuple[str, list[dict[str, str]]]:
    """Return (answer_with_citations, retrieved_chunks) for spot-checking."""
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY missing — set it in .env")

    chunks = retrieve_chunks(
        query, target_collection=target_collection, n_results=n_results
    )
    if not chunks:
        return "No relevant documentation chunks were retrieved.", []

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = build_prompt(query, chunks)
    response = client.responses.create(model=CHAT_MODEL, input=prompt)
    return response.output_text, chunks


if __name__ == "__main__":
    question = "how do I create a component?"
    print(f"Q: {question}\n")
    answer, chunks = generate_answer(question, target_collection="docs")
    print(answer)
    print("\n--- retrieved for spot-check ---")
    for chunk in chunks:
        print(f"{chunk['id']}: {chunk['document'][:120].replace(chr(10), ' ')}...")
