# Source Benchmark Attribution Log

Every item pulled into `data/processed/items.jsonl` must have a row here. This is the
attribution trail required for the dataset deliverable and the paper's data section,
and it decides whether images can be re-hosted in this repo or must stay linked to
the original source (license-dependent).

| Source benchmark | License | Items used | Re-hostable? | Notes |
|---|---|---|---|---|
| PhysBench | apache-2.0 (`USC-PSI-Lab/PhysBench`) | 0 — not used | N/A | Official images only ship as a 3.7 GB zip, impractical to pull for ~100 items here. Pre-formatted mirror (`lmms-lab-eval/PhysBench`) checked and found to have empty metadata fields (media_path/task_type/ability_type all blank) — unusable. Dropped in favor of CLEVRER; see PROGRESS.md. |
| PIQA | unknown (`ybisk/piqa`, checking source repo) | 0 — not used | N/A | Text-only benchmark (no images) — inconsistent with testing a model's visual pathway alongside the rest of the study. Dropped in favor of CLEVRER; see PROGRESS.md. |
| CLEVRER | Research/non-commercial use (CLEVRER project license, clevrer.csail.mit.edu) — questions/annotations via `zechen-nlp/clevrer` on Hugging Face, videos via official `data.csail.mit.edu/clevrer` host | pending — see items.jsonl once build_dataset.py runs | No — link to original video source, do not re-host extracted frames beyond what's needed for this course project | All 5 subskills sourced from here: predictive→physical_state_prediction, explanatory→causal_reasoning, counterfactual→object_permanence, descriptive (spatial-keyword filtered)→spatial_relation, descriptive (temporal-keyword filtered)→temporal_consistency. Descriptive-split short answers converted to MCQ using CLEVRER's fixed closed vocabulary (8 colors, 3 shapes, 2 materials, yes/no, counts 0-4) for same-category distractors. |

Update this table as items are pulled in Phase 1. Do not add an item to
`items.jsonl` until its source row exists here.
