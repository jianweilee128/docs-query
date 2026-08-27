"""Print the summary + usage block of an eval run file.

Usage:
    uv run python scripts/_show_usage.py eval/results/run-20260826-133739.json
"""

import json
import sys
from pathlib import Path


def main(path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, value in data.items():
        if key in {"results", "disagreements"}:
            continue
        print(f"{key}: {value}")

    rows = [r for r in data.get("results", []) if r.get("usage")]
    if not rows:
        return

    print("\nslowest 5 calls:")
    for row in sorted(rows, key=lambda r: r["usage"]["latency_s"], reverse=True)[:5]:
        usage = row["usage"]
        print(
            f"  {row['id']:<14} {usage['latency_s']:>8}s  "
            f"in={usage['prompt_tokens']:<6} out={usage['completion_tokens']}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
