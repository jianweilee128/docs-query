"""Diff pass/fail between two eval runs.

Usage:
    uv run python scripts/_diff_runs.py <run_a.json> <run_b.json>
"""

import json
import sys
from pathlib import Path


def load(path: str) -> tuple[dict, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data, {r["id"]: r for r in data["results"]}


def main(path_a: str, path_b: str) -> None:
    meta_a, a = load(path_a)
    meta_b, b = load(path_b)
    label_a = meta_a.get("model") or "A"
    label_b = meta_b.get("model") or "B"
    print(f"A = {label_a} ({meta_a['passed']}/{meta_a['total']})")
    print(f"B = {label_b} ({meta_b['passed']}/{meta_b['total']})\n")

    for case_id in sorted(set(a) | set(b)):
        row_a, row_b = a.get(case_id), b.get(case_id)
        if not row_a or not row_b or row_a["passed"] == row_b["passed"]:
            continue
        winner = label_a if row_a["passed"] else label_b
        loser_row = row_b if row_a["passed"] else row_a
        print(
            f"{case_id:<14} {row_a['type']:<8} only {winner} passes "
            f"| loser verdict={loser_row['verdict']}"
        )

    print("\nfailures by run:")
    for label, rows in ((label_a, a), (label_b, b)):
        failed = sorted(r["id"] for r in rows.values() if not r["passed"])
        print(f"  {label}: {len(failed)} -> {', '.join(failed)}")

    print("\nretrieval misses (should match if the swap was clean):")
    for label, rows in ((label_a, a), (label_b, b)):
        miss = sorted(r["id"] for r in rows.values() if r["verdict"] == "retrieval miss")
        print(f"  {label}: {', '.join(miss)}")

    print("\ngeneration misses:")
    for label, rows in ((label_a, a), (label_b, b)):
        miss = sorted(
            r["id"] for r in rows.values() if r["verdict"] == "generation miss"
        )
        print(f"  {label}: {', '.join(miss)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
