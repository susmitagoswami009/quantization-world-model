# Quantization vs. World-Modeling in Small Multimodal LLMs

Investigating whether GGUF quantization (FP16 → INT8 → INT4 → INT3) silently damages
a multimodal model's physical "world modeling" ability, broken into five measurable
sub-skills, tested entirely on CPU-only hardware.

## Research Question

Does compressing a small vision-language model degrade physical reasoning uniformly,
or does one sub-skill fail first / collapse sharply at a critical bit-width?

## Sub-skills tested

1. Physical state prediction
2. Object permanence / containment
3. Causal reasoning
4. Spatial relation reasoning
5. Temporal consistency

Each scored independently at each precision level (20 items/sub-skill, 100 total —
see PROGRESS.md for why this is smaller than the original 30-40 target), producing
accuracy and relative-accuracy-retained curves vs. the FP16 baseline. All 100 items
are sourced from CLEVRER (see docs/ATTRIBUTION_LOG.md for why PhysBench and PIQA
were dropped).

## Key finding

Degradation is not uniform across sub-skills. Causal reasoning and spatial relation
reasoning both hold steady through INT8, then drop sharply between INT4 and INT3
(75%→55% and 60%→50% respectively), while temporal consistency stays comparatively
robust (80%→80%→80%→75%) across the full precision range. See
`results/degradation_curves.png` and `docs/paper_draft.md` for full results,
including an explicit statistical-power caveat (n=20/cell).

## Backbone model

**Qwen2-VL-2B-Instruct**, quantized to FP16 (baseline), INT8, INT4, INT3 via
llama.cpp / GGUF (`mtmd` multimodal support). Chosen over Moondream2 because it has
an actively maintained llama.cpp GGUF conversion path for both the language model and
the vision projector (`mmproj`), giving a clean, controlled precision ladder.

## Hardware

CPU-only laptop: AMD Ryzen 5 5500U (6c/12t), ~5.86 GB total RAM. No GPU. This is a
deliberate constraint — it mirrors the real deployment conditions (robotics, mobile,
embedded) the research is meant to inform. FP16 (4.4 GB of weights) was confirmed to
run successfully on this hardware — no Colab fallback was needed; the entire
pipeline, including the full 400-call evaluation sweep, ran locally.

## Repo layout

```
quantization-world-model/
├── README.md              this file
├── PROGRESS.md            running log of what's been done, session to session
├── models/                GGUF model weights (fp16/int8/int4/int3) — gitignored, large
├── data/
│   ├── raw/                downloaded CLEVRER video clips (source; see ATTRIBUTION_LOG.md)
│   └── processed/          normalized {id, subskill, image, question, choices, answer}
├── scripts/                quantization, dataset assembly, eval harness, scoring, analysis
├── results/                per-run CSV/JSON logs + degradation-curve plots
├── notebooks/              optional demo notebook (stretch deliverable)
└── docs/                   paper draft, slide outline, attribution log
```

## Status

See [PROGRESS.md](PROGRESS.md) for the current state and next steps.

## Deliverables

- [x] Evaluation pipeline (this repo — `scripts/`)
- [x] Assembled sub-skill dataset with source attribution (`data/processed/`, `docs/ATTRIBUTION_LOG.md`)
- [x] **Course submission paper** (`docs/paper_jmlr.docx`) — JMLR format, structured
      around the course grading rubric (Idea / Problem Exploration+Design /
      Evaluation / Report Quality). **This is the one to submit for the viva.**
      Needs your name/institution filled in.
- [x] IEEE two-column paper draft (`docs/paper_ieee.docx`) — earlier draft, kept
      for reference (e.g. if you still want an arXiv/ICCIT version later), but
      superseded by `paper_jmlr.docx` for the course submission.
- [x] Slide deck for course defense (`docs/slide_deck.pptx`)
- [x] (Stretch) demo notebook comparing outputs across precision levels
      (`notebooks/demo_quantization_degradation.ipynb` — genuinely executed
      end-to-end, not just generated; runs live inference on a real example item)
