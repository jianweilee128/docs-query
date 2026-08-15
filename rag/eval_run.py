"""Stage 6 — run the eval set and score answers.

Usage:
    uv run python -m rag.eval_run
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config.settings import COLLECTION_NAME, ROOT
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


def run_eval() -> Path:
    cases = load_questions()
    results = []

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}: {case['question']}")
        answer, _chunks = generate_answer(
            case["question"],
            target_collection=COLLECTION_NAME,
        )
        graded = score_case(case, answer)
        status = "PASS" if graded["passed"] else "FAIL"
        print(f"  -> {status} ({graded['reason']})")
        results.append(graded)

    passed = sum(1 for r in results if r["passed"])
    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "score": round(passed / len(results), 3) if results else 0.0,
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"run-{stamp}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nScore: {summary['passed']}/{summary['total']} ({summary['score']:.0%})")
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    run_eval()
