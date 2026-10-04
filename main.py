"""CLI entry: ask a question against the Angular docs RAG."""

from config.settings import COLLECTION_NAME
from rag.generate import generate_answer


def main() -> None:
    print("Angular docs RAG. Empty line or quit/exit to stop.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question or question.lower() in {"quit", "exit", ":q"}:
            return
        answer, _chunks, _usage = generate_answer(
            question, target_collection=COLLECTION_NAME
        )
        print(answer)
        print()


if __name__ == "__main__":
    main()
