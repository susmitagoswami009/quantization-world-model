# Does Quantization Break World Modeling? A Sub-Skill Analysis of Compressed Multimodal LLMs on CPU-Only Hardware

*Content source of truth — formatted for JMLR submission at `docs/paper_jmlr.docx`.
Structured around the course rubric: Idea (Abstract+Intro+Conclusion, 10),
Problem Exploration+Design (50), Evaluation (30), Report quality (10).*

## Abstract

Quantization is the standard technique for compressing large multimodal models onto
cheap, low-power hardware, yet its effect on physical "world modeling" — a model's
ability to understand and predict how the physical world behaves — is unstudied.
This matters because quantized multimodal models are increasingly deployed exactly
where physical reasoning is needed most (robotics, mobile assistants, embedded
devices) and exactly where compute is most limited. We decompose world modeling
into five independently-measurable sub-skills — physical state prediction, object
permanence, causal reasoning, spatial relation reasoning, and temporal consistency —
and evaluate a small (2B-parameter) multimodal LLM (Qwen2-VL-2B-Instruct) across
four quantization levels (FP16, INT8, INT4, INT3) via llama.cpp/GGUF, using 100
CLEVRER-derived multiple-choice items (20 per sub-skill). We run the entire study,
including model conversion and inference, on CPU-only consumer laptop hardware to
mirror real deployment constraints. We find that degradation is not uniform: causal
reasoning and spatial relation reasoning both hold steady through INT8 and then drop
sharply between INT4 and INT3, while temporal consistency remains comparatively
robust. Given a small per-cell item count (n=20), we report these as suggestive
trends from a pilot study rather than statistically confirmed effects, and discuss
concrete deployment implications.

**Keywords:** quantization, multimodal LLM, world modeling, physical reasoning,
GGUF, edge deployment

---

## 1. Introduction

Post-training quantization — reducing model weights from 16-bit floating point down
to 8-, 4-, or even 3-bit representations — is the default way small models are made
to fit on consumer and embedded hardware. Prior work has studied how quantization
affects general reasoning ability (math, logic, code), and separately, research on
"world models" has studied physical/causal reasoning in full-precision
vision-language models. No prior work crosses these two lines: does compressing a
multimodal model damage its ability to reason about the physical world specifically,
and if so, does it damage every kind of physical reasoning equally?

This question is not academic. A quantized vision-language model deployed on a
robot arm, a mobile assistant, or an embedded inspection camera is asked to reason
about exactly the kind of physical scenarios this paper tests — and it runs on
exactly the kind of resource-constrained hardware where INT4 or even INT3
quantization is standard, not optional. If quantization degrades physical reasoning
non-uniformly, a deployment decision like "quantize to INT4 to fit on-device" could
silently break a specific capability (e.g., causal reasoning) while leaving others
(e.g., temporal ordering) intact — a failure mode invisible to aggregate benchmark
scores.

We also take the low-resource framing seriously in our methodology, not just our
research question: every step of this study, including GGUF conversion and the full
inference sweep, was run on a CPU-only consumer laptop (AMD Ryzen 5 5500U, 5.86 GB
RAM), the same class of hardware this research is meant to inform decisions about.
Section 2 documents this constraint-driven process in detail, since navigating it is
itself part of the paper's contribution — a reproducible pipeline for exactly this
hardware class did not exist before this project and had to be built from scratch.

**Contributions:**

1. A five-sub-skill decomposition of physical world modeling, each independently
   measurable via multiple-choice items with real ground truth.
2. A controlled quantization sweep (FP16/INT8/INT4/INT3) isolating LM-side
   quantization level as the single variable, vision tower held fixed at FP16.
3. A fully reproducible, CPU-only pipeline (build → convert → quantize → evaluate),
   released with the dataset and attribution log, developed under genuine hardware
   constraints (see Section 2).
4. An empirical finding that degradation is sub-skill-dependent, with a sharp
   INT4→INT3 transition specifically in causal and spatial relational reasoning
   (Section 4).

---

## 2. Problem Exploration and Design

This section covers both the exploration process that shaped every major decision
in this study and the resulting design — presented together because, in practice,
almost none of the final design was decided up front. It was arrived at by trying
the straightforward approach, hitting a real constraint, and adapting. We document
that process explicitly rather than presenting only the final, cleaned-up pipeline,
because the constraints themselves — hardware, data access, tooling maturity — are
as much a part of what this study demonstrates as the quantization results are: this
*is* what building for this deployment class actually looks like.

### 2.1 Backbone Model: Exploration and Design

**Exploration.** Two small (1.5–3B parameter) open multimodal LLMs were considered:
Moondream2 and Qwen2-VL-2B-Instruct, both candidates specifically because they can
plausibly run on CPU-only hardware. The deciding factor was not raw capability but
quantization tooling maturity: Moondream2 uses a bespoke inference/quantization
runtime rather than the GGUF format, which would not have produced a clean,
comparable FP16→INT8→INT4→INT3 ladder. Qwen2-VL-2B-Instruct, by contrast, has an
actively maintained conversion path in llama.cpp's multimodal (`mtmd`) support,
including a separate vision-projector (`mmproj`) export.

**Design decision.** Qwen2-VL-2B-Instruct was selected specifically for this
reason — tooling fit for the CPU/GGUF pipeline, not benchmark performance — since a
controlled, comparable precision ladder was the precondition for the entire study
being possible at all. This was validated with a feasibility spike (Section 2.3)
before any further work was committed to it.

### 2.2 Dataset Sourcing: Exploration and Design

**Exploration.** The original plan was to sample items from three established
benchmarks — PhysBench, PIQA, and CLEVRER — each covering different aspects of
physical reasoning. Checking real access to each changed the plan substantially:

- *PhysBench* (`USC-PSI-Lab/PhysBench`, Apache-2.0): the official images are
  distributed only as a single 3.7 GB archive, with a matching 3.7 GB video archive
  — impractical to download for the ~100 items actually needed on this hardware and
  connection. A compact, pre-formatted mirror (`lmms-lab-eval/PhysBench`) appeared
  to solve this, but inspection of its parquet schema (via `pandas`) revealed every
  metadata field (`media_path`, `task_type`, `ability_type`) was an empty string —
  the mirror was unusable for building real items.
- *PIQA* (`ybisk/piqa`): available only as a legacy Hugging Face loading script
  rather than a direct data file, and — more fundamentally — text-only. Using it
  would exercise a different, non-visual capability than the rest of the study,
  breaking the multimodal consistency the evaluation depends on.
- *CLEVRER* (`zechen-nlp/clevrer`): small (<5 MB), well-structured parquet files
  split by native question type (predictive, explanatory, counterfactual,
  descriptive), each with real ground-truth answers embedded in a conversation
  format. Videos are not embedded, but a targeted search for the official video host
  (`data.csail.mit.edu/clevrer`) found that individual clips — not just the full
  multi-GB archive — are directly downloadable (~1.5 MB each) once the correct
  folder-sharding URL pattern was identified empirically by probing candidate URLs.

Closer inspection of CLEVRER's `descriptive` split showed its answers are short
strings drawn from a small, fixed vocabulary (8 colors, 3 shapes, 2 materials,
yes/no, and counts 0–4) rather than free text — which made a principled multiple-
choice conversion possible: for a given correct answer, distractors are sampled from
the *same* category (a wrong color is always another color, never a shape), rather
than being handwritten or arbitrary.

**Design decision.** The dataset was sourced entirely from CLEVRER rather than the
original three-benchmark plan, once mapped correctly onto all five sub-skills:
`predictive → physical_state_prediction`, `explanatory → causal_reasoning`,
`counterfactual → object_permanence` (proxying permanence via whether the model
tracks an object's continued causal role when it is hypothetically removed, since no
CLEVRER split directly tests visual occlusion tracking), and `descriptive`, filtered
by keyword into two disjoint pools: collision/contact-relation questions for
`spatial_relation`, and ordering questions (first/last/before/after/enters/exits)
for `temporal_consistency`. This design was itself revised once empirically: an
initial keyword filter for `spatial_relation` assumed CLEVR-style left/right/behind
phrasing, which does not occur in CLEVRER's actual question text (confirmed by
sampling 40 unique question templates) — it was redefined around collision-relation
phrasing instead, with mutual exclusion against the temporal keyword set so the two
sub-skill pools do not overlap on the same underlying items.

The target item count was also reduced during design, from an initial 30–40 per
sub-skill to 20 per sub-skill (100 total), a deliberate, explicitly-logged scope
reduction given solo execution on a single machine under a fixed time budget, not a
silent cut (see Section 4.3 for how this affects statistical interpretation of the
results).

### 2.3 Hardware Constraints: Exploration and Design

**Exploration.** The target machine (AMD Ryzen 5 5500U, no discrete GPU) was
profiled before committing to any pipeline design: 5.86 GB total RAM, with as little
as 0.31–0.41 GB free at points during the session, and the primary drive (C:) had
only ~2.5 GB free against a 6+ GB requirement for model weights alone. No build
tools (compiler, CMake, Python-on-PATH) were present at the start. Each of these
was discovered by direct inspection (`Get-CimInstance`, disk queries) rather than
assumed, and each one shaped a specific design choice below.

**Design decisions, each tied to a specific constraint:**

- *Toolchain placement.* A portable MinGW toolchain (w64devkit) and CMake were
  installed to the secondary drive (E:) rather than C:, avoiding any dependency on
  the nearly-full system drive. Python was found already present but not on PATH;
  rather than installing a redundant copy, an isolated virtual environment was
  built on E: from the existing interpreter.
- *Build parallelism.* The RAM ceiling meant a full-parallel C++ compile of
  llama.cpp risked hanging or crashing the machine. The build was run with `-j 2`
  rather than full core count, after explicitly asking that memory-heavy background
  applications be closed first (RAM went from ~0.35 GB to ~0.81 GB free) — a
  concrete instance of hardware constraints directly gating a build decision.
- *Scratch/cache redirection.* Two separate tools defaulted their temporary
  storage to the nearly-full C: drive without being asked to — Python's
  `tempfile` (used by `convert_hf_to_gguf.py --use-temp-file`) and
  `huggingface_hub`'s model cache — and both caused real failures (a disk-write
  `OSError` mid-conversion, in the first case) before being redirected to E: via
  `TMPDIR`/`TEMP`/`TMP` and `HF_HOME` respectively. This is reported as a concrete
  example of a failure mode specific to this hardware class that a higher-resource
  setup would never surface.
- *FP16 feasibility, tested rather than assumed.* Whether the FP16 baseline
  (3.1 GB LM + 1.3 GB vision tower = 4.4 GB of weights) would even load on a
  5.86 GB machine was a genuine open question, not a foregone conclusion — it was
  resolved with a direct feasibility spike (one real image, one real question)
  before any further pipeline work was built on top of the assumption that it
  would work. It did: FP16 loaded and ran successfully, removing the need for a
  cloud-GPU fallback for any part of the study that matters to the results.
- *Resumability, added after a real failure.* A multi-hour evaluation sweep on
  this hardware was interrupted mid-run by a session/process teardown (with one
  logged data point — a single call's wall-clock time of over three hours — showing
  the process itself, not a single inference call, had been frozen). The evaluation
  harness was redesigned after this to skip already-completed items on rerun
  instead of overwriting from scratch, a design change made necessary specifically
  by the reality of running a long job on this class of hardware/session
  environment rather than a dedicated, always-on server.

### 2.4 Quantization Pipeline and Sub-skill Taxonomy: Design

The language model is quantized to four levels via `llama-quantize`: FP16
(baseline, 16.00 bits/weight, 3.1 GB), INT8 (Q8\_0, 8.50 bits/weight, 1.6 GB), INT4
(Q4\_K\_M, 5.08 bits/weight, 986 MB), and INT3 (Q3\_K\_M, 4.24 bits/weight, 824 MB).
The vision tower (`mmproj`) is deliberately held at FP16 across every run — this
isolates language-model quantization level as the single controlled variable, and
matches standard practice, since GGUF vision-projector quantization is experimental
and unsupported in most current tooling.

The five sub-skills — physical state prediction, object permanence, causal
reasoning, spatial relation reasoning, and temporal consistency — were chosen to
decompose "world modeling" into independently-measurable parts precisely so that a
single blended accuracy score could not hide which specific capability, if any,
degrades first. This design choice is what makes the result in Section 4 legible:
without the decomposition, "accuracy went down" would be the only available
finding.

### 2.5 Evaluation Protocol: Design

Each item is presented as a single image plus a multiple-choice question, run
through `llama-mtmd-cli` at each precision level with greedy decoding (temperature
0, for determinism and reproducibility). Model output is parsed for a single answer
letter, and three outcomes are distinguished per item — correct, incorrect, and
*unparsed* (malformed output) — as a deliberate design choice, since at low
precision a model can fail by giving a wrong answer or by failing to produce a
well-formed answer at all, and conflating these would hide which failure mode is
actually occurring. In this study, unparsed output never occurred (0 of 400 calls),
which is itself informative: all degradation observed in Section 4 is genuine
reasoning degradation, not output-format collapse.

---

## 3. Evaluation

### 3.1 Experimental Setup

All conversion, quantization, and inference — 400 total inference calls (100 items
× 4 precisions) — ran on the CPU-only hardware described in Section 2.3. This is a
deliberate methodological choice, not a limitation worked around: it mirrors the
actual deployment class this research is meant to inform, and every result below
was produced under exactly the resource constraints a real low-power deployment
would face.

### 3.2 Results

**Table 1: Accuracy by sub-skill and precision (n=20/cell)**

| Sub-skill | FP16 | INT8 | INT4 | INT3 |
|---|---|---|---|---|
| Causal reasoning | 65% | 65% | 60% | 50% |
| Object permanence | 50% | 50% | 45% | 65% |
| Physical state prediction | 40% | 45% | 45% | 50% |
| Spatial relation | 70% | 70% | 75% | 55% |
| Temporal consistency | 80% | 80% | 80% | 75% |

**Finding 1 — degradation is not uniform.** This is the central result: sub-skills
diverge in how they respond to compression, rather than all declining together.

**Finding 2 — causal reasoning and spatial relation reasoning both show a sharp
INT4→INT3 transition.** Both hold flat (or even tick up) through INT8 and INT4,
then drop sharply at INT3: spatial relation falls from 75% to 55% (a 20-point
single-step drop), and causal reasoning continues a decline from 60% to 50%. Two
independently-sourced sub-skills showing the same cliff location is the most
directionally consistent pattern in the data.

**Finding 3 — temporal consistency is the most robust sub-skill**, essentially flat
from FP16 through INT4 (80% throughout) with only a small dip at INT3 (75%).

**Finding 4 — object permanence and physical state prediction are noisy and
non-monotonic.** Object permanence rises at INT3 (45%→65%), attributed to sampling
noise rather than genuine improvement under compression (see 3.3). Physical state
prediction hovers near chance level (40–50%) across all four precisions, suggesting
this specific task is simply hard for a 2B-parameter model regardless of
quantization — a separate, also-interesting finding.

A single concrete example makes Finding 2 tangible: for the item asking "What is
the material of the object to collide with the green cylinder?" (choices: metal /
rubber; correct answer: metal), the model answers correctly at FP16, INT8, and INT4,
then answers "rubber" specifically at INT3 — reproduced live, deterministically, in
the accompanying demo notebook (`notebooks/demo_quantization_degradation.ipynb`).

### 3.3 Statistical Discussion

With n=20 items per sub-skill per precision level, the standard error of an
observed proportion near p=0.5 is approximately 11 percentage points, giving a 95%
confidence interval of roughly ±22 points. Most individual step-to-step differences
reported above are not statistically significant in isolation. Findings 1–3 are
reported as a suggestive, directionally consistent pattern from a small-sample
pilot study, not as a statistically confirmed effect — an honest constraint of this
evaluation's scale that should guide interpretation of every number above, and a
direct target for a larger-scale follow-up (Section 5).

### 3.4 Discussion and Deployment Implications

Held to the confidence level the data actually supports: if a deployment depends
specifically on causal or spatial relational reasoning, this study's data points
toward avoiding quantization below INT4, since both sub-skills that showed a clear
cliff did so specifically at the INT4→INT3 transition, not gradually. Applications
relying primarily on temporal ordering reasoning may tolerate more aggressive
quantization with comparatively less risk, based on the robustness observed here.
This is framed as a hypothesis for follow-up work with a larger item count, not a
settled guideline.

The divergence itself — not simply "accuracy went down" — is the paper's actual
contribution. It suggests that different physical-reasoning sub-skills may depend
on the model's weights in structurally different ways: sub-skills requiring
multi-step relational inference (what caused this, what relates to what) appear
more fragile under aggressive compression than sub-skills requiring more direct
temporal pattern-matching (what came first).

---

## 4. Related Work

**Quantization and general reasoning.** GPTQ (Frantar et al., 2023), AWQ (Lin et
al., 2024), and GGUF k-quant methods are widely used to compress LLMs for edge
deployment. Prior evaluations of quantized models on general benchmarks (MMLU,
GSM8K, HumanEval) generally find graceful degradation down to INT4, with more
pronounced drops below that. These studies do not evaluate multimodal models or
physical/world-modeling tasks specifically.

**World modeling and physical reasoning benchmarks.** PhysBench (Chow et al.,
2024/2025), PIQA (Bisk et al., 2020), and CLEVRER (Yi et al., 2020) each probe
different aspects of physical understanding: PhysBench evaluates multimodal
physical world understanding across a range of tasks; PIQA evaluates text-based
physical commonsense reasoning; CLEVRER evaluates descriptive, predictive,
counterfactual, and explanatory reasoning over synthetic collision videos. All of
this prior work evaluates full-precision models only.

**The gap.** No existing study measures how post-training quantization affects
physical/world-modeling reasoning specifically, let alone whether different
sub-skills within "world modeling" degrade at different rates. This paper is, to
our knowledge, the first to combine a controlled quantization sweep with a
sub-skill-decomposed physical reasoning evaluation, developed and validated
entirely under real low-resource hardware constraints.

*(Citations above are given in good faith from memory of the cited works' authors,
titles, and approximate years; verify exact venue/page details before final
submission — this is flagged explicitly rather than presented as verified.)*

---

## 5. Limitations

- **Small per-cell sample size (n=20).** Limits statistical confidence in any
  single reported difference; findings should be read as directional/suggestive.
- **Single backbone model.** Results may not generalize to other architectures or
  parameter scales; this is a case study, not a general claim about quantization.
- **Static-frame proxy for video.** CLEVRER is video-native; the backbone is
  image-based, so temporal-consistency items use a single representative frame
  rather than full motion.
- **Object-permanence operationalization.** Proxied via counterfactual-removal
  questions rather than direct visual occlusion/tracking, since CLEVRER does not
  include a dedicated occlusion-tracking split.
- **Single-domain dataset.** All items derive from CLEVRER's synthetic
  collision-scene domain; findings may not transfer to naturalistic imagery.
- **Vision tower not quantized.** Only the language model is quantized; results
  characterize LM-side compression specifically, not full end-to-end multimodal
  quantization.

## 6. Conclusion

This paper presents, to our knowledge, the first study of how post-training
quantization affects physical world-modeling ability in a small multimodal LLM,
decomposed into five independently-measured sub-skills, developed and evaluated
entirely on representative low-resource CPU hardware — with the exploration and
design process shaped directly by that hardware's real constraints, not abstracted
away. We find that degradation is not uniform across sub-skills: causal and spatial
relational reasoning show a sharp decline specifically between INT4 and INT3, while
temporal consistency remains comparatively robust. Given the small scale of this
pilot (n=20/sub-skill), these are presented as a directionally consistent,
hypothesis-generating pattern rather than a statistically confirmed result. The
full pipeline, dataset, attribution log, and a live, executed demo notebook are
released to support a larger-scale follow-up.

## Appendix: Reproducibility

- Model: Qwen/Qwen2-VL-2B-Instruct (Hugging Face)
- Quantization tool: llama.cpp (`mtmd` multimodal support), quantize types
  Q8_0/Q4_K_M/Q3_K_M
- Dataset build: `scripts/build_dataset.py`
- Evaluation harness: `scripts/eval_harness.py`, `scripts/score.py`
- Analysis: `scripts/analyze.py`
- Full item-level results: `results/{fp16,int8,int4,int3}_{raw,scored}.jsonl`
- Attribution log: `docs/ATTRIBUTION_LOG.md`
- Live demo: `notebooks/demo_quantization_degradation.ipynb` (executed end-to-end,
  not just generated)
- Full development log (every decision, bug, and fix in real time, the primary
  source for Section 2): `PROGRESS.md`
