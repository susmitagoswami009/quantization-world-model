# Dataset Item Schema

Every item in `data/processed/items.jsonl` is one JSON object per line with this shape:

```json
{
  "id": "physstate_003",
  "subskill": "physical_state_prediction",
  "images": ["data/processed/images/physstate_003.jpg"],
  "question": "If the ball continues rolling, what will happen when it reaches the edge of the table?",
  "choices": ["A. It stops", "B. It falls off", "C. It bounces back", "D. It speeds up"],
  "correct_answer": "B",
  "source_benchmark": "PhysBench",
  "source_id": "physbench_original_id_here",
  "license": "<license of source benchmark>",
  "notes": ""
}
```

## Field notes

- `subskill` must be one of exactly:
  `physical_state_prediction`, `object_permanence`, `causal_reasoning`,
  `spatial_relation`, `temporal_consistency`
- `images`: 1-3 static frames (the backbone is image-based, not video-native — for
  `temporal_consistency` items from CLEVRER, sample 2-3 representative frames rather
  than passing full video).
- `choices` / `correct_answer`: multiple-choice only. Free-text scoring is unreliable
  at low precision (INT3/INT4 outputs degrade into malformed text before they degrade
  into wrong answers) — multiple choice lets the scorer separate "wrong answer" from
  "broken output format".
- `source_id` + `license`: required for every item — this is the attribution trail for
  the dataset deliverable and the paper's data section. Do not add an item without it.

## Target counts

30-40 items per sub-skill = 150-200 items total. See PROGRESS.md for the current
count and whether solo-execution scope has trimmed this target.
