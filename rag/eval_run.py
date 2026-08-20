"""Stage 6 — run the eval set and score answers.

Usage:
    uv run python -m rag.eval_run
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from config.settings import COLLECTION_NAME, EVAL_WORKERS, ROOT
from rag.generate import generate_answer

QUESTIONS_PATH = ROOT / "eval" / "questions.json"
RESULTS_DIR = ROOT / "eval" / "results"

ABSTAIN_HINTS = (
    "not in the documentation",
    "not in the docs",
    "do not contain",
    "doesn't contain",
    "does not contain",
    "not enough information",
    "cannot find",
    "aren't in the excerpts",
    "are not in the excerpts",
    "no relevant",
)


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def looks_like_abstain(answer: str) -> bool:
    lower = answer.lower()
    return any(hint in lower for hint in ABSTAIN_HINTS)


def score_case(case: dict, answer: str) -> dict:
    lower = answer.lower()
    missing = [
        phrase
        for phrase in case.get("must_include", [])
        if phrase.lower() not in lower
    ]
    forbidden_hits = [
        phrase
        for phrase in case.get("must_not_include", [])
        if phrase.lower() in lower
    ]

    if case.get("expect_abstain"):
        passed = looks_like_abstain(answer) and not missing
        reason = "abstain ok" if passed else "expected abstain / not-in-docs"
    else:
        passed = not missing and not forbidden_hits and not looks_like_abstain(answer)
        reason = "keywords ok" if passed else "missing keywords or unexpected abstain"

    return {
        "id": case["id"],
        "type": case["type"],
        "passed": passed,
        "reason": reason,
        "missing": missing,
        "forbidden_hits": forbidden_hits,
        "question": case["question"],
        "answer": answer,
    }

def score_context(case, chunks: list[dict]) -> dict:
    """Did the retrieved chunks contain the information needed to answer the question?"""
    haystacks = "\n".join([c["document"] for c in chunks]).lower()
    anchors = case.get("must_include", [])
    missing = [a for a in anchors if a.lower() not in haystacks]
    return {
        "anchors": len(anchors),
        "missing": missing,
        "recall": round((len(anchors) - len(missing)) / len(anchors), 3) if anchors else None
    }


def diagnose_context(context: dict, generation: dict) -> str:
    if not context["anchors"]:
        return "na"
    if context["missing"]:
        return "retrieval miss"
    return "pass" if generation["passed"] else "generation miss"


def run_case(case: dict) -> dict:
    """Answer and score one case. Never raises — a broken case scores as a failure."""
    try:
        answer, chunks = generate_answer(
            case["question"],
            target_collection=COLLECTION_NAME,
        )
    except Exception as exc:
        return {
            "id": case["id"],
            "type": case["type"],
            "passed": False,
            "reason": f"error: {type(exc).__name__}: {exc}",
            "verdict": "error",
            "missing": [],
            "forbidden_hits": [],
            "context": None,
            "retrieved_ids": [],
            "question": case["question"],
            "answer": "",
        }

    generation = score_case(case, answer)
    context = score_context(case, chunks)
    return {
        **generation,
        "verdict": diagnose_context(context, generation),
        "context": context,
        "retrieved_ids": [c["id"] for c in chunks],
    }


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def context_summary(results: list[dict]) -> dict:
    """Split the score into retrieval health and generation health."""
    # Abstain cases have no anchors to look for, so they can't be scored here.
    scored = [r for r in results if r["context"] and r["context"]["anchors"]]
    hits = [r for r in scored if not r["context"]["missing"]]
    misses = [r for r in scored if r["context"]["missing"]]
    generated = [r for r in hits if r["passed"]]

    return {
        "verdicts": dict(Counter(r["verdict"] for r in results)),
        "context_scored": len(scored),
        "context_hit_rate": round(len(hits) / len(scored), 3) if scored else None,
        "context_recall_mean": mean([r["context"]["recall"] for r in scored]),
        "recall_on_misses": mean([r["context"]["recall"] for r in misses]),
        "generation_given_context": round(len(generated) / len(hits), 3) if hits else None,
    }


def run_eval(workers: int | None = None) -> Path:
    cases = load_questions()
    count = workers if workers is not None else EVAL_WORKERS
    results: list[dict | None] = [None] * len(cases)

    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = {pool.submit(run_case, case): i for i, case in enumerate(cases)}
        for done, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            graded = future.result()
            results[index] = graded
            status = "PASS" if graded["passed"] else "FAIL"
            # One print per line keeps worker output from interleaving.
            print(
                f"[{done}/{len(cases)}] {graded['id']}: {status} "
                f"[{graded['verdict']}] ({graded['reason']})"
            )

    results = [r for r in results if r is not None]
    passed = sum(1 for r in results if r["passed"])
    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "score": round(passed / len(results), 3) if results else 0.0,
        **context_summary(results),
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"run-{stamp}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nScore: {summary['passed']}/{summary['total']} ({summary['score']:.0%})")
    print("\nVerdicts:")
    for name, total in sorted(summary["verdicts"].items()):
        print(f"  {name:<16} {total}")
    print(
        f"\nRetrieval  hit rate {summary['context_hit_rate']} "
        f"over {summary['context_scored']} scored | "
        f"mean recall {summary['context_recall_mean']} | "
        f"recall on misses {summary['recall_on_misses']}"
    )
    print(f"Generation given context {summary['generation_given_context']}")
    print(f"\nWrote {out}")
    return out


if __name__ == "__main__":
    run_eval()
