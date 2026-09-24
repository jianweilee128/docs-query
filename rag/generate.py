from config.settings import (
    CHAT_MODEL,
    COLLECTION_NAME,
    PROMPT_GENERATE_PATH,
    TOP_K,
)
from rag.llm import complete, empty_usage
from rag.retrieve import retrieve_chunks


def build_prompt(query: str, chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        heading = (chunk.get("metadata") or {}).get("heading", "")
        label = f"[{chunk['id']}]" + (f" {heading}" if heading else "")
        parts.append(f"{label}\n{chunk['document']}")
    context = "\n\n---\n\n".join(parts)
    template = PROMPT_GENERATE_PATH.read_text(encoding="utf-8")
    return template.format(query=query, context=context)


def generate_answer(
    query: str,
    target_collection: str | None = None,
    n_results: int | None = None,
    where: dict | None = None,
) -> tuple[str, list[dict], dict]:
    """Return (answer_with_citations, retrieved_chunks, generation_usage)."""
    chunks = retrieve_chunks(
        query,
        target_collection=target_collection,
        n_results=n_results if n_results is not None else TOP_K,
        where=where,
    )
    if not chunks:
        return (
            "No relevant documentation chunks were retrieved.",
            [],
            empty_usage(CHAT_MODEL),
        )

    prompt = build_prompt(query, chunks)
    result = complete(prompt, model=CHAT_MODEL)
    return result.text, chunks, result.usage_only()


if __name__ == "__main__":
    question = "how do I create a component?"
    print(f"Q: {question}\n")
    answer, chunks, usage = generate_answer(question, target_collection=COLLECTION_NAME)
    print(answer)
    print(
        f"\n{usage['latency_s']}s | in={usage['prompt_tokens']} "
        f"out={usage['completion_tokens']} | ${usage['cost_usd']:.6f}"
    )
    print("\n--- retrieved for spot-check ---")
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        print(
            f"{chunk['id']} ({meta.get('heading')}): "
            f"{chunk['document'][:120].replace(chr(10), ' ')}..."
        )
