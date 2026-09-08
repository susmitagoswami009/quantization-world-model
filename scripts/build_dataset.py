"""
Builds data/processed/items.jsonl entirely from CLEVRER (zechen-nlp/clevrer on HF).

Scope note (see PROGRESS.md 2026-08-22): originally planned to draw from PhysBench,
PIQA, and CLEVRER. In practice:
  - PhysBench's images are only distributed as a 3.7 GB zip, impractical to pull for
    ~100 items on this hardware/connection, and the pre-formatted parquet mirror
    (lmms-lab-eval/PhysBench) turned out to have empty metadata fields.
  - PIQA is text-only, which would not exercise the model's visual pathway and would
    be inconsistent with the rest of a multimodal study.
  - CLEVRER alone provides real ground-truth, closed-vocabulary, static-frame-
    compatible items across all 5 target subskills once descriptive questions are
    converted from short-answer to multiple-choice (see build_mc_choices below).
This is a deliberate, documented pivot, not a silent scope cut.

Video source: individual validation clips are directly downloadable (no need for the
multi-GB zip) at:
    http://data.csail.mit.edu/clevrer/videos/validation/video_{lo}-{hi}/video_{id}.mp4
where {lo}-{hi} is the enclosing 1000-range folder (e.g. video_10000-11000).

Subskill <- CLEVRER split mapping:
    physical_state_prediction <- predictive   ("which event will happen next")
    causal_reasoning          <- explanatory  ("what is/isn't responsible for X")
    object_permanence         <- counterfactual ("what would NOT happen if object
                                  removed" - tests whether the model tracks an
                                  object's continued causal role, our proxy for
                                  permanence given no direct occlusion-tracking
                                  split exists)
    spatial_relation           <- descriptive, filtered to spatial-keyword questions
    temporal_consistency       <- descriptive, filtered to temporal-keyword questions
"""

import json
import os
import random
import re

os.environ.setdefault("HF_HOME", r"E:\quantization-world-model\tools\hf_cache")

from pathlib import Path

import cv2
import httpx
import pandas as pd
from huggingface_hub import hf_hub_download

random.seed(42)

ROOT = Path(r"E:\quantization-world-model")
VIDEO_DIR = ROOT / "data" / "raw" / "clevrer_videos"
IMG_DIR = ROOT / "data" / "processed" / "images"
OUT_PATH = ROOT / "data" / "processed" / "items.jsonl"
ATTRIB_PATH = ROOT / "docs" / "ATTRIBUTION_LOG.md"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

ITEMS_PER_SUBSKILL = 20  # reduced from the original 30-40 target; see PROGRESS.md

COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
SHAPES = ["cube", "sphere", "cylinder"]
MATERIALS = ["metal", "rubber"]
NUMBERS = [str(n) for n in range(5)]

SPATIAL_KEYWORDS = ["collide", "collides", "collision"]
TEMPORAL_KEYWORDS = ["first", "second", "last", "before", "after", "earlier", "later",
                      "when the video begins", "when the video ends", "order", "then",
                      "enters", "exits"]


def load_split(split: str) -> pd.DataFrame:
    path = hf_hub_download("zechen-nlp/clevrer", f"{split}/validation-00000-of-00001.parquet", repo_type="dataset")
    return pd.read_parquet(path)


def video_download_url(video_rel_path: str) -> str:
    # video_rel_path like "validation_videos/video_10000.mp4"
    vid_id = int(re.search(r"video_(\d+)\.mp4", video_rel_path).group(1))
    lo = (vid_id // 1000) * 1000
    hi = lo + 1000
    return f"http://data.csail.mit.edu/clevrer/videos/validation/video_{lo}-{hi}/video_{vid_id}.mp4"


def ensure_video(video_rel_path: str) -> Path:
    vid_id = re.search(r"(video_\d+)\.mp4", video_rel_path).group(1)
    local_path = VIDEO_DIR / f"{vid_id}.mp4"
    if not local_path.exists():
        url = video_download_url(video_rel_path)
        with httpx.stream("GET", url, timeout=60, follow_redirects=True) as r:
            r.raise_for_status()
            with local_path.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    return local_path


def extract_frame(video_path: Path, item_id: str, position: float = 0.7) -> str:
    """Extracts one frame at `position` (fraction through the clip) as a jpg."""
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = max(0, min(n_frames - 1, int(n_frames * position)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame from {video_path}")
    out_path = IMG_DIR / f"{item_id}.jpg"
    cv2.imwrite(str(out_path), frame)
    return str(out_path.relative_to(ROOT)).replace("\\", "/")


def extract_two_frames(video_path: Path, item_id: str) -> list[str]:
    early = extract_frame(video_path, f"{item_id}_a", position=0.25)
    late = extract_frame(video_path, f"{item_id}_b", position=0.85)
    return [early, late]


def parse_mc_from_choices(row) -> tuple[list[str], str]:
    """For predictive/counterfactual/explanatory rows that already have a `choices` dict."""
    choices = row["choices"]
    letters = "ABCDEFGH"
    opts = [f"{letters[i]}. {c}" for i, c in enumerate(choices["choice"])]
    gpt_answer = row["conversations"]["value"][1].strip()
    correct_letter = gpt_answer[0].upper()
    return opts, correct_letter


def build_mc_choices(answer: str) -> tuple[list[str], str]:
    """For descriptive rows: answer is a short string from a closed vocabulary.
    Builds a 4-choice (or 2-choice for yes/no) MCQ with same-category distractors."""
    answer = answer.strip().lower()
    if answer in ("yes", "no"):
        pool = ["yes", "no"]
    elif answer in COLORS:
        pool = COLORS
    elif answer in SHAPES:
        pool = SHAPES
    elif answer in MATERIALS:
        pool = MATERIALS
    elif answer in NUMBERS:
        pool = NUMBERS
    else:
        return None, None  # unrecognized answer type, skip this item

    distractors = [c for c in pool if c != answer]
    random.shuffle(distractors)
    n_distractors = min(3, len(distractors))
    options = [answer] + distractors[:n_distractors]
    random.shuffle(options)
    letters = "ABCD"
    opts_fmt = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
    correct_letter = letters[options.index(answer)]
    return opts_fmt, correct_letter


def append_items(items: list[dict]):
    with OUT_PATH.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def append_attribution(rows: list[str]):
    with ATTRIB_PATH.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")


def build_choice_split(split: str, subskill: str, n: int, id_prefix: str):
    df = load_split(split)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    built = 0
    items = []
    for _, row in df.iterrows():
        if built >= n:
            break
        try:
            opts, correct = parse_mc_from_choices(row)
        except Exception:
            continue
        item_id = f"{id_prefix}_{built:03d}"
        try:
            video_path = ensure_video(row["video"])
            img_rel = extract_frame(video_path, item_id, position=0.85)
        except Exception as e:
            print(f"  skip {item_id}: {e}")
            continue
        items.append({
            "id": item_id,
            "subskill": subskill,
            "images": [img_rel],
            "question": row["question"],
            "choices": opts,
            "correct_answer": correct,
            "source_benchmark": "CLEVRER",
            "source_id": f"{split}/{row['question_id']}/{row['video']}",
            "license": "CLEVRER research/non-commercial license (see clevrer.csail.mit.edu)",
            "notes": f"single frame @85% through clip; split={split}",
        })
        built += 1
    append_items(items)
    print(f"{subskill}: built {len(items)} items")
    return len(items)


def build_descriptive_subset(subskill: str, keywords: list[str], n: int, id_prefix: str):
    df = load_split("descriptive")
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    built = 0
    items = []
    for _, row in df.iterrows():
        if built >= n:
            break
        question = row["question"]
        q_lower = question.lower()
        if not any(kw in q_lower for kw in keywords):
            continue
        # keep categories clean: exclude anything matching the *other* keyword set
        other_keywords = TEMPORAL_KEYWORDS if keywords is SPATIAL_KEYWORDS else SPATIAL_KEYWORDS
        if any(kw in q_lower for kw in other_keywords):
            continue
        answer = row["conversations"]["value"][1].strip()
        opts, correct = build_mc_choices(answer)
        if opts is None:
            continue
        item_id = f"{id_prefix}_{built:03d}"
        try:
            video_path = ensure_video(row["video"])
            img_rel = extract_frame(video_path, item_id, position=0.85)
        except Exception as e:
            print(f"  skip {item_id}: {e}")
            continue
        items.append({
            "id": item_id,
            "subskill": subskill,
            "images": [img_rel],
            "question": question,
            "choices": opts,
            "correct_answer": correct,
            "source_benchmark": "CLEVRER",
            "source_id": f"descriptive/{row['question_id']}/{row['video']}",
            "license": "CLEVRER research/non-commercial license (see clevrer.csail.mit.edu)",
            "notes": f"single frame @85% through clip; answer converted to MCQ from short-answer",
        })
        built += 1
    append_items(items)
    print(f"{subskill}: built {len(items)} items")
    return len(items)


if __name__ == "__main__":
    if OUT_PATH.exists():
        OUT_PATH.unlink()

    counts = {}
    counts["physical_state_prediction"] = build_choice_split(
        "predictive", "physical_state_prediction", ITEMS_PER_SUBSKILL, "physstate")
    counts["causal_reasoning"] = build_choice_split(
        "explanatory", "causal_reasoning", ITEMS_PER_SUBSKILL, "causal")
    counts["object_permanence"] = build_choice_split(
        "counterfactual", "object_permanence", ITEMS_PER_SUBSKILL, "permanence")
    counts["spatial_relation"] = build_descriptive_subset(
        "spatial_relation", SPATIAL_KEYWORDS, ITEMS_PER_SUBSKILL, "spatial")
    counts["temporal_consistency"] = build_descriptive_subset(
        "temporal_consistency", TEMPORAL_KEYWORDS, ITEMS_PER_SUBSKILL, "temporal")

    print("\n=== Final counts ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  TOTAL: {sum(counts.values())}")
