"""
Combines all results/<precision>_scored.jsonl files into one degradation-curve plot
and a summary table: accuracy and relative-accuracy-retained (vs FP16) per sub-skill
per precision.

Usage:
    python analyze.py --results-dir results --out-dir results
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PRECISION_ORDER = ["fp16", "int8", "int4", "int3"]


def load_scored(results_dir: Path) -> pd.DataFrame:
    rows = []
    for precision in PRECISION_ORDER:
        f = results_dir / f"{precision}_scored.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_scored(results_dir)
    if df.empty:
        print("No scored results found yet.")
        return

    # accuracy = correct / (correct + wrong); unparsed/error excluded from denominator
    # but reported separately since they're a distinct failure mode from wrong answers
    scored_mask = df["correct"].notna()
    acc_table = (
        df[scored_mask]
        .groupby(["subskill", "precision"])["correct"]
        .mean()
        .unstack("precision")
        .reindex(columns=[p for p in PRECISION_ORDER if p in df["precision"].unique()])
    )
    acc_table.to_csv(out_dir / "accuracy_table.csv")
    print("Accuracy table:\n", acc_table, "\n")

    if "fp16" in acc_table.columns:
        retained = acc_table.div(acc_table["fp16"], axis=0)
        retained.to_csv(out_dir / "relative_accuracy_retained.csv")
        print("Relative accuracy retained vs FP16:\n", retained, "\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    for subskill in acc_table.index:
        ax.plot(acc_table.columns, acc_table.loc[subskill], marker="o", label=subskill)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Accuracy")
    ax.set_title("Sub-skill accuracy across quantization levels")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "degradation_curves.png", dpi=150)
    print(f"Saved plot to {out_dir / 'degradation_curves.png'}")


if __name__ == "__main__":
    main()
