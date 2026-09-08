# Slide Deck Outline (course defense) — filled with real results

1. **Title** — Does Quantization Break World Modeling? A Sub-Skill Analysis of
   Compressed Multimodal LLMs on CPU-Only Hardware
2. **Motivation** — quantized multimodal models deployed exactly where physical
   reasoning matters most (robotics/mobile/embedded) and compute is scarcest
3. **The gap** — quantization studies ≠ world-modeling studies, nobody's crossed them
4. **Research question** — does compression damage physical reasoning uniformly,
   or does one sub-skill fail first?
5. **The 5 sub-skills** — physical state prediction, object permanence, causal
   reasoning, spatial relation, temporal consistency — one example item each
6. **Method** — Qwen2-VL-2B-Instruct, FP16→INT8→INT4→INT3 via llama.cpp/GGUF,
   CPU-only laptop (5.86 GB RAM, no GPU) — deliberate, mirrors real deployment
7. **Dataset** — 100 items (20/subskill), all CLEVRER-derived, real ground truth,
   real extracted video frames — explain the PhysBench/PIQA → CLEVRER pivot briefly
8. **Results — the degradation curves** — results/degradation_curves.png, the
   central slide
9. **Key finding** — causal_reasoning and spatial_relation both cliff sharply
   between INT4→INT3 (75%→55%, 60%→50%); temporal_consistency stays flat
   (80%→80%→80%→75%) — degradation is sub-skill-dependent, not uniform
10. **Deployment implications** — don't quantize below INT4 if the application
    depends on causal or spatial relational reasoning; temporal-ordering-heavy
    applications tolerate more aggressive compression
11. **Honesty slide: statistical caveat** — n=20/cell, ~±22pp 95% CI — this is a
    suggestive pilot pattern, not a statistically confirmed effect. Include this,
    don't hide it — it's a strength of the presentation, not a weakness.
12. **Limitations** — small n, single backbone, static-frame proxy for video,
    synthetic-domain dataset, vision tower not quantized
13. **Conclusion / Q&A**

Status: ready to build as .pptx — see docs/paper_draft.md for full narrative detail
backing each slide.
