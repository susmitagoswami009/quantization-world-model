"""
Parses raw model output from results/<precision>_raw.jsonl into correct/incorrect
and writes results/<precision>_scored.jsonl + prints a per-subskill accuracy summary.

Multiple-choice parsing strategy:
1. Look for a standalone letter (A-D) at the start of the output, or immediately
   after common preambles ("The answer is B", "Answer: C").
2. If no clean letter is found, mark as "unparsed" (NOT automatically wrong) so it
   can be manually audited — quantized models sometimes answer correctly but in a
   malformed format, and that's a different failure mode than a wrong answer.

Usage:
    python score.py --in results/int4_raw.jsonl --out results/int4_scored.jsonl
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

LETTER_RE = re.compile(r"\b([A-D])\b")


def parse_answer(raw_output: str | None) -> str | None:
    if not raw_output:
        return None
    match = LETTER_RE.search(raw_output.strip()[:50])
    return match.group(1) if match else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.infile).read_text(encoding="utf-8").splitlines() if line.strip()]

    subskill_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "unparsed": 0, "error": 0})

    scored = []
    for r in records:
        if r.get("error"):
            r["parsed_answer"] = None
            r["correct"] = None
            subskill_stats[r["subskill"]]["error"] += 1
        else:
            parsed = parse_answer(r["raw_output"])
            r["parsed_answer"] = parsed
            if parsed is None:
                r["correct"] = None
                subskill_stats[r["subskill"]]["unparsed"] += 1
            else:
                r["correct"] = parsed == r["correct_answer"]
                subskill_stats[r["subskill"]]["correct" if r["correct"] else "wrong"] += 1
        scored.append(r)

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r) for r in scored) + "\n", encoding="utf-8")

    print(f"\n--- {args.infile} ---")
    for subskill, stats in sorted(subskill_stats.items()):
        total = sum(stats.values())
        acc = stats["correct"] / total if total else 0
        print(f"{subskill:30s} acc={acc:.2%}  correct={stats['correct']} wrong={stats['wrong']} "
              f"unparsed={stats['unparsed']} error={stats['error']} (n={total})")


if __name__ == "__main__":
    main()
