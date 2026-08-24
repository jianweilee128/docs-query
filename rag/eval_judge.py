"""Score frozen gold labels with the LLM judge and report agreement.

Usage:
    uv run python -m rag.eval_judge
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from config.settings import (
    EVAL_GOLD_PATH,
    EVAL_RESULTS_DIR,
    EVAL_WORKERS,
    JUDGE_MODEL,
    LLM_BASE_URL,
    PROMPT_JUDGE_PATH,
)
from rag.llm import complete, get_llm_client

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_gold() -> dict:
    return json.loads(EVAL_GOLD_PATH.read_text(encoding="utf-8"))


def build_judge_prompt(question: str, answer: str, expect_abstain: bool) -> str:
    template = PROMPT_JUDGE_PATH.read_text(encoding="utf-8")
    return template.format(
        expect_abstain=str(expect_abstain).lower(),
        question=question,
        answer=answer,
    )


def parse_judge_output(text: str) -> dict:
    raw = text.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_RE.search(raw)
        if not match:
            raise ValueError(f"judge did not return JSON: {raw[:200]!r}")
        data = json.loads(match.group(0))

    if "passed" not in data:
        raise ValueError(f"judge JSON missing 'passed': {data!r}")
    passed = data["passed"]
    if isinstance(passed, str):
        passed = passed.strip().lower() in {"true", "pass", "yes", "1"}
    return {
        "passed": bool(passed),
        "reason": str(data.get("reason") or "").strip(),
    }


def judge_answer(
    question: str,
    answer: str,
    expect_abstain: bool = False,
    client: OpenAI | None = None,
) -> dict:
    openai_client = client or get_llm_client()
    prompt = build_judge_prompt(question, answer, expect_abstain)
    try:
        text = complete(
            prompt,
            model=JUDGE_MODEL,
            temperature=0,
            json_mode=True,
            client=openai_client,
        )
    except Exception:
        # Some Ollama builds reject response_format; the prompt already asks for JSON.
        text = complete(
            prompt,
            model=JUDGE_MODEL,
            temperature=0,
            json_mode=False,
            client=openai_client,
        )
    graded = parse_judge_output(text)
    graded["model"] = JUDGE_MODEL
    return graded


def cohens_kappa(human: list[bool], predicted: list[bool]) -> float | None:
    n = len(human)
    if n == 0:
        return None
    po = sum(h == p for h, p in zip(human, predicted)) / n
    human_pos = sum(human)
    pred_pos = sum(predicted)
    pe = (human_pos * pred_pos + (n - human_pos) * (n - pred_pos)) / (n * n)
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return round((po - pe) / (1 - pe), 3)


def confusion(human: list[bool], predicted: list[bool]) -> dict[str, int]:
    tp = tn = fp = fn = 0
    for h, p in zip(human, predicted):
        if h and p:
            tp += 1
        elif not h and not p:
            tn += 1
        elif not h and p:
            fp += 1
        else:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def validate_judge(workers: int | None = None) -> Path:
    gold = load_gold()
    items = gold["items"]
    client = get_llm_client()
    count = workers if workers is not None else min(EVAL_WORKERS, len(items))
    rows: list[dict | None] = [None] * len(items)

    def _one(item: dict) -> dict:
        judged = judge_answer(
            item["question"],
            item["answer"],
            expect_abstain=bool(item.get("expect_abstain")),
            client=client,
        )
        return {
            "id": item["id"],
            "type": item["type"],
            "synthetic": bool(item.get("synthetic")),
            "human_passed": item["human_passed"],
            "keyword_passed": item.get("keyword_passed"),
            "judge_passed": judged["passed"],
            "judge_reason": judged["reason"],
            "notes": item.get("notes", ""),
            "agree": judged["passed"] == item["human_passed"],
        }

    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = {pool.submit(_one, item): i for i, item in enumerate(items)}
        for done, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            row = future.result()
            rows[index] = row
            mark = "AGREE" if row["agree"] else "DISAGREE"
            print(
                f"[{done}/{len(items)}] {row['id']}: {mark} "
                f"(human={'pass' if row['human_passed'] else 'fail'}, "
                f"judge={'pass' if row['judge_passed'] else 'fail'})"
            )

    results = [r for r in rows if r is not None]
    human = [r["human_passed"] for r in results]
    judged = [r["judge_passed"] for r in results]
    n = len(results)
    matrix = confusion(human, judged)
    agreed = sum(1 for r in results if r["agree"])
    keyword_agreed = sum(
        1 for r in results if r["keyword_passed"] == r["human_passed"]
    )
    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "model": JUDGE_MODEL,
        "base_url": LLM_BASE_URL,
        "gold": str(EVAL_GOLD_PATH).replace("\\", "/"),
        "source_run": gold.get("source_run"),
        "n": n,
        "agreement": round(agreed / n, 3) if n else None,
        "agreed": agreed,
        "disagreed": n - agreed,
        "kappa": cohens_kappa(human, judged),
        "confusion": matrix,
        "keyword_agreement": round(keyword_agreed / n, 3) if n else None,
        "keyword_kappa": cohens_kappa(
            human, [bool(r["keyword_passed"]) for r in results]
        ),
        "disagreements": [r for r in results if not r["agree"]],
        "results": results,
    }

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = EVAL_RESULTS_DIR / f"judge-agreement-{stamp}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"\nJudge vs human:    {agreed}/{n} ({summary['agreement']:.0%})  "
        f"kappa={summary['kappa']}"
    )
    print(
        f"Keyword vs human:  {keyword_agreed}/{n} ({summary['keyword_agreement']:.0%})  "
        f"kappa={summary['keyword_kappa']}"
    )
    print(
        "Confusion          TP={tp} TN={tn} FP={fp} (lenient) FN={fn} (strict)".format(
            **matrix
        )
    )
    if summary["disagreements"]:
        print("\nDisagreements:")
        for row in summary["disagreements"]:
            print(
                f"  {row['id']}: human={'pass' if row['human_passed'] else 'fail'} "
                f"judge={'pass' if row['judge_passed'] else 'fail'} "
                f"— {row['judge_reason']}"
            )
    print(f"\nWrote {out}")
    return out


if __name__ == "__main__":
    validate_judge()
