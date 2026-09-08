"""
Runs the full sub-skill x precision sweep against a llama.cpp multimodal GGUF model
and logs every response to results/<precision>_raw.jsonl.

Usage:
    python eval_harness.py --precision int4 \
        --model models/int4/qwen2vl-2b-int4.gguf \
        --mmproj models/int4/mmproj-f16.gguf \
        --llama-cli tools/llama.cpp/build/bin/llama-mtmd-cli.exe \
        --items data/processed/items.jsonl \
        --out results/int4_raw.jsonl

Each item is scored separately by scripts/score.py — this script only collects raw
model output, it does not judge correctness (keeps the two concerns separate so a
scoring-logic bug never requires re-running inference).
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def build_prompt(item: dict) -> str:
    choices_block = "\n".join(item["choices"])
    return (
        f"{item['question']}\n{choices_block}\n"
        "Answer with only the letter of the correct choice."
    )


def run_one(llama_cli: str, model: str, mmproj: str, image_path: str, prompt: str) -> tuple[str, float]:
    cmd = [
        llama_cli,
        "-m", model,
        "--mmproj", mmproj,
        "--image", image_path,
        "-p", prompt,
        "-n", "64",
        "--temp", "0",
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start
    return result.stdout.strip(), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--llama-cli", required=True)
    ap.add_argument("--items", default="data/processed/items.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap items for a dry run")
    args = ap.parse_args()

    items = [json.loads(line) for line in Path(args.items).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip items already present in an existing output file (id
    # match is enough since one run always covers a single precision). Long sweeps
    # on this hardware can get interrupted (session teardown, laptop sleep) — this
    # makes reruns pick up where they left off instead of redoing finished work.
    already_done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("error") is None:
                    already_done.add(rec["id"])
    if already_done:
        print(f"Resuming: {len(already_done)} items already done, skipping those.")
    items = [it for it in items if it["id"] not in already_done]

    with out_path.open("a", encoding="utf-8") as out_f:
        for i, item in enumerate(items, 1):
            prompt = build_prompt(item)
            image_path = item["images"][0]
            print(f"[{i}/{len(items)}] {item['id']} ({args.precision})")
            try:
                raw_output, elapsed = run_one(args.llama_cli, args.model, args.mmproj, image_path, prompt)
                record = {
                    "id": item["id"],
                    "subskill": item["subskill"],
                    "precision": args.precision,
                    "correct_answer": item["correct_answer"],
                    "raw_output": raw_output,
                    "latency_sec": round(elapsed, 2),
                    "error": None,
                }
            except Exception as e:
                record = {
                    "id": item["id"],
                    "subskill": item["subskill"],
                    "precision": args.precision,
                    "correct_answer": item["correct_answer"],
                    "raw_output": None,
                    "latency_sec": None,
                    "error": str(e),
                }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
