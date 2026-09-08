# Paper Outline (IEEE two-column, target: arXiv → ICCIT 2026)

Working title: *Does Quantization Break World Modeling? A Sub-Skill Analysis of
Compressed Multimodal LLMs on CPU-Only Hardware*

## I. Introduction
- Motivation: quantized multimodal models are deployed exactly where physical
  reasoning matters most (robotics, mobile, embedded) and exactly where compute is
  most limited.
- Gap: prior work studies quantization's effect on general reasoning (math/logic)
  OR world-modeling in full-precision models, never both together.
- Contribution: a 5-subskill, 4-precision degradation study, run entirely on
  consumer CPU hardware to mirror real deployment conditions.

## II. Related Work
- Quantization methods for LLMs (GPTQ, AWQ, GGUF k-quants) and their reported
  effect on general benchmarks (MMLU, GSM8K, etc.)
- World-modeling / physical reasoning benchmarks for (V)LMs: PhysBench, PIQA,
  CLEVRER, and what they each actually measure.
- The gap this paper fills: no existing study crosses these two lines.

## III. Method
- Backbone: Qwen2-VL-2B-Instruct, chosen for CPU feasibility and llama.cpp GGUF
  multimodal support.
- Precision levels: FP16 (baseline), INT8, INT4, INT3 (k-quants: Q8_0, Q4_K_M,
  Q3_K_M) via llama.cpp. Only the language model is quantized; the vision tower
  (mmproj) is held at F16 across all runs — isolates LM quantization level as the
  single controlled variable, and matches standard practice since GGUF vision-
  projector quantization is experimental/unsupported in most tooling.
- Sub-skill taxonomy and why these five (state prediction, object permanence,
  causal reasoning, spatial relation, temporal consistency) — justify as
  decomposing "world modeling" into independently-testable primitives rather than
  one blended score.
- Dataset construction: sampling procedure from PhysBench/PIQA/CLEVRER, item
  normalization to multiple-choice, target n per subskill (see data/processed/SCHEMA.md).
- Evaluation protocol: exact scoring methodology, handling of unparsed/malformed
  low-precision outputs as a distinct failure category from wrong answers.
- Hardware: CPU-only, AMD Ryzen 5 5500U, 5.86 GB RAM — explicitly framed as
  representative of real deployment constraints, not a limitation to apologize for.

## IV. Results
- Per-subskill accuracy table across 4 precision levels (results/accuracy_table.csv)
- Relative-accuracy-retained curves vs FP16 (results/relative_accuracy_retained.csv)
- Degradation curve figure (results/degradation_curves.png) — the central figure.
- Key finding to fill in once data exists: which subskill degrades first / most
  sharply, whether decay is gradual or a cliff at a specific bit-width.

## V. Discussion
- Deployment implications: concrete guidance in the style of "don't quantize below
  INT4 if your application depends on [subskill]."
- Why the divergence across subskills (not just "accuracy went down") is the
  actual finding — connect to what each subskill structurally requires from the
  model (e.g., causal reasoning may depend on more distributed/fragile weight
  interactions than simple spatial pattern matching).

## VI. Limitations
- Single backbone model — findings may not generalize across architectures.
- Small per-cell item counts (30-40) — flag confidence intervals, note which
  differences are likely signal vs. noise.
- Static-frame proxy for temporal_consistency items (image-based backbone, not
  video-native).
- FP16 baseline may be CPU-infeasible on this specific hardware — document actual
  outcome here once known.

## VII. Conclusion

## Appendix / supplementary
- Full item list attribution (docs/ATTRIBUTION_LOG.md)
- Reproducibility: exact llama.cpp commit hash + quantize commands used (log from
  scripts/quantize step)

---
Status: outline only, no results yet. Fill in as Phases 4-5 complete — see PROGRESS.md.
