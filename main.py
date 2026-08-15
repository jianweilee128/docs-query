"""CLI entry: ask a question against the Angular docs RAG."""

from config.settings import COLLECTION_NAME
from rag.generate import generate_answer


def main() -> None:
    question = input("> ").strip()
    answer, _chunks = generate_answer(question, target_collection=COLLECTION_NAME)
    print(answer)


if __name__ == "__main__":
    main()
