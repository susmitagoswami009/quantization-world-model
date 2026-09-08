# Progress Log

Running record of what's been done, decided, and what's next. Updated at the end of
each work session so the project can be picked up cold at any point.

---

## 2026-08-22

**Setup**
- Defined project scope: quantization's effect on 5 physical-world-modeling sub-skills
  in a small multimodal LLM, tested CPU-only.
- Backbone decision: **Qwen2-VL-2B-Instruct** (over Moondream2) — better-maintained
  llama.cpp GGUF + `mmproj` multimodal conversion path, needed for a clean FP16/INT8/
  INT4/INT3 ladder.
- Checked hardware: AMD Ryzen 5 5500U, 6c/12t, **5.86 GB total RAM**, 0.35 GB free at
  check time. C: drive nearly full (2.52 GB free) — project relocated to E: (191 GB free).
- **Risk flagged:** RAM is likely too tight for the FP16 baseline to run locally
  (2B params @ FP16 ≈ 4-5 GB weights alone, before vision encoder + OS overhead).
  Plan: attempt it anyway once other apps are closed; if it fails, document as a
  hardware limitation and rely on INT8 as the effective upper bound for local
  comparison, noting this explicitly in the paper's limitations section.
- Created project structure at `E:\quantization-world-model\`.
- Solo-execution scope decision pending with user: original plan assumed 2 laptops in
  parallel; now single-machine. Need to confirm dataset size (30-40/subskill vs.
  smaller) and deadline to pace the rest of the plan.

**Next**
- Phase 0 feasibility spike: install/build llama.cpp, download Qwen2-VL-2B-Instruct,
  convert to GGUF at one precision (INT4 first — smallest, fastest to validate),
  run one real image+question through it, confirm coherent output and log timing.

**Blocker found & resolved:** machine had no CMake and no C/C++ compiler on PATH.
Installed portable toolchain to `E:\quantization-world-model\tools\`:
- w64devkit v2.9.1 (portable MinGW gcc) → `tools\w64devkit\w64devkit\bin\gcc.exe` ✅
- CMake 4.4.2 (portable zip) → `tools\cmake\cmake-4.4.2-windows-x86_64\bin\cmake.exe` ✅
- Python: turned out to already be installed (3.10/3.12/3.14 under
  `C:\Users\Lenovo\AppData\Local\Programs\Python\`) — just not on PATH (Store alias
  was intercepting the `python` command). Used existing 3.12 to create an isolated
  venv at `tools\venv\` so packages stay off the nearly-full C: drive. ✅
- `scripts/requirements.txt` installed into the venv ✅ (huggingface_hub 1.28,
  pillow 12.3, pandas 3.0, matplotlib 3.11, tqdm, numpy).

**Scaffolding written:** README.md, PROGRESS.md, data/processed/SCHEMA.md,
docs/ATTRIBUTION_LOG.md, scripts/{eval_harness.py, score.py, analyze.py,
download_model.py, requirements.txt}. llama.cpp source cloned (not yet built).

**RAM:** user closed Chrome + Grammarly Desktop, freed RAM 0.35 GB → 0.81 GB
(still tight vs 5.86 GB total, so building with `-j 2` instead of full parallelism).

**llama.cpp build:** CMake configure succeeded (MinGW Makefiles generator, GNU
16.2.0 via w64devkit, GGML_NATIVE=OFF, CPU backend with AVX2/FMA/F16C detected,
OpenMP found, OpenSSL not found → HTTPS support disabled in built-in server, not
needed for this project). **Build completed successfully** ✅ — confirmed
`llama-cli.exe`, `llama-mtmd-cli.exe` (multimodal), `llama-quantize.exe` all present
under `llama.cpp\build\bin\`. This clears the biggest technical risk in the project:
quantized multimodal inference via llama.cpp is confirmed buildable on this hardware.

**Model download:** first attempt failed — `SSL: CERTIFICATE_VERIFY_FAILED`, caused
by Avast's HTTPS scanning intercepting the connection with its own cert, which
Python's default cert store (certifi) doesn't trust. Fixed by installing
`pip-system-certs` into the venv (patches Python ssl/requests to use the Windows
cert store instead). Retry in progress.

**Model download: complete** ✅ — Qwen2-VL-2B-Instruct, 4.2 GB, all safetensors
shards verified at `models\hf_source\`.

**GGUF conversion:** installing `requirements-convert_hf_to_gguf.txt` into the venv
(torch 2.11.0 CPU build, transformers 4.57.6, sentencepiece, gguf, protobuf) —
`convert_hf_to_gguf.py` needs these, not previously in requirements.txt. In progress.

Verified torch 2.11.0+cpu, transformers 4.57.6, gguf all importable ✅.

**First FP16 conversion attempt failed:** `OSError: ... requested and 0 written`
partway through — `--use-temp-file` was writing to the default Windows TEMP dir
(`C:\Users\Lenovo\AppData\Local\Temp`), which only has ~2.5 GB free. Fixed by
setting `TMPDIR`/`TEMP`/`TMP` to `E:\quantization-world-model\tmp` before rerunning.
**Lesson for later steps:** any tool that uses the system temp dir needs this same
redirect — C: is too full to trust as scratch space anywhere in this pipeline.

**FP16 language-model conversion: complete** ✅ — `models\fp16\qwen2vl-2b-fp16.gguf`,
3.1 GB, 338 tensors. Fix confirmed working (temp dir on E:).

**mmproj conversion: complete** ✅ — `models\fp16\qwen2vl-2b-mmproj-f16.gguf`, 1.3 GB,
520 tensors. Full FP16 pair now exists (3.1 GB LM + 1.3 GB mmproj = 4.4 GB weights).

**Phase 0 smoke test: PASSED** ✅ — FP16 model + mmproj loaded and ran successfully
on this hardware despite 4.4 GB of weights vs 5.86 GB total RAM. Correctly described
a synthetic test image ("A red ball is above a brown box on a blue background").
Image encoding took ~7 sec. **This clears the biggest remaining risk in the project
— FP16 baseline is viable locally, no need to fall back to Colab.**

**Scope decision:** only the language model is quantized (Q8_0/Q4_K_M/Q3_K_M);
`mmproj` (vision tower) stays at F16 across every run. Standard convention — GGUF
vision-projector quantization is experimental/unsupported in most tooling — and it
isolates LM quantization level as the single controlled variable, which is what the
study is actually measuring. Documented in docs/paper_outline.md Method section.

**Quantization: complete** ✅ — all 4 precision levels of the language model exist:
| Precision | File | Size | BPW |
|---|---|---|---|
| FP16 | `models/fp16/qwen2vl-2b-fp16.gguf` | 3.1 GB | 16.00 |
| INT8 (Q8_0) | `models/int8/qwen2vl-2b-int8.gguf` | 1.6 GB | 8.50 |
| INT4 (Q4_K_M) | `models/int4/qwen2vl-2b-int4.gguf` | 986 MB | 5.08 |
| INT3 (Q3_K_M) | `models/int3/qwen2vl-2b-int3.gguf` | 824 MB | 4.24 |

mmproj stays fixed at F16 (`models/fp16/qwen2vl-2b-mmproj-f16.gguf`, 1.3 GB) for all four.
Quantization itself is fast (23-38 sec each) — not a bottleneck.

**INT3 sanity check: PASSED** ✅ — identical correct answer to FP16 on the smoke-test
image ("A red ball is above a brown box on a blue background"). Expected on a trivial
synthetic scene — real degradation differences will show up on the actual benchmark
items in Phase 4, not here. This just confirms the pipeline runs cleanly at every
precision level, which it does.

## Phase 0 + Phase 2: COMPLETE ✅

Full technical pipeline validated end-to-end: build → download → convert → quantize
→ infer, at all 4 precision levels (FP16/INT8/INT4/INT3), on this exact hardware.
No remaining technical risk to the core pipeline. Next: Phase 1 (dataset assembly)
— this is now the actual bottleneck, and needs a few scope decisions from the user
that are still open (see below) before going further.

---

## Open questions for user
- Confirm dataset size target given solo/single-machine execution (default assumption:
  keep 30-40 items/sub-skill unless timeline forces a cut).
- Course defense / paper deadline — not yet provided, needed to pace Phases 4-7.
- GitHub repo — not yet created/linked; working locally on E: until user provides one.
- Phase 1 will need to check exact access/license terms for PhysBench, PIQA, and
  CLEVRER (some benchmark repos are gated behind a form or HF login) — will report
  back per-benchmark once checked rather than assume.

## 2026-08-22 (cont'd) — Phase 1 dataset research

User said "make the full project now" — proceeding with defaults: full dataset size
target (30-40/subskill unless infeasible), staying local on E: (no GitHub repo yet),
no deadline constraint known yet.

**Same C:-drive trap hit again:** `huggingface_hub` defaults its cache to
`C:\Users\Lenovo\.cache\huggingface`, same problem as the temp-dir issue during GGUF
conversion. Fixing by setting `HF_HOME=E:\quantization-world-model\tools\hf_cache`
for every HF-related call going forward.

**Benchmark access findings:**
- **PhysBench** (`USC-PSI-Lab/PhysBench`, apache-2.0): official repo only ships
  `image.zip`/`video.zip` at ~3.7 GB each — way too big for the ~100 images actually
  needed. Found `lmms-lab-eval/PhysBench` — same benchmark, pre-formatted as small
  parquet files (test 0.42 MB, val 0.02 MB) for the lmms-eval harness. Using this
  instead. Checked `WeiChow/PhysBench-assets`/`-media` too — those are 25+ GB 3D-asset
  dumps unrelated to the eval items, not useful here.
- **PIQA** (`ybisk/piqa`): only a legacy loading script, no direct parquet — needs
  investigation into a converted mirror or the script-based loader.
- **CLEVRER**: `zechen-nlp/clevrer` has small (<5 MB each) parquet files split by
  question type (descriptive/explanatory/predictive/counterfactual) — predictive and
  counterfactual map directly onto the causal_reasoning and physical_state_prediction
  subskills. Frames are NOT embedded — parquet only references `video_XXXXX.mp4`.

**Scope pivot: dataset now sourced entirely from CLEVRER, not PhysBench/PIQA.**
Documented in full in docs/ATTRIBUTION_LOG.md; summary:
- PhysBench dropped — real images only ship as a 3.7 GB zip (way too big for ~100
  items on this connection/hardware), and the compact pre-formatted mirror turned
  out to have empty metadata (media_path/task_type/ability_type all blank strings).
- PIQA dropped — it's text-only, which would exercise a different (non-visual)
  pathway than the rest of the study and confound the multimodal framing.
- CLEVRER covers all 5 subskills once mapped correctly: predictive→physical_state_
  prediction, explanatory→causal_reasoning, counterfactual→object_permanence
  (proxy: does the model track an object's continued causal role), descriptive
  (spatial-keyword filtered)→spatial_relation, descriptive (temporal-keyword
  filtered)→temporal_consistency. Descriptive-split answers are short strings from
  a closed vocabulary (8 colors/3 shapes/2 materials/yes-no/counts 0-4) — converted
  to proper MCQ format using same-category distractors (e.g. wrong answer is
  always another color if the right answer is a color), a standard, principled way
  to turn closed-vocabulary VQA into MCQ.

**Video sourcing:** found individual validation clips are directly downloadable
(~1.5 MB each, no need for the multi-GB zip) at
`data.csail.mit.edu/clevrer/videos/validation/video_{lo}-{hi}/video_{id}.mp4`.

**Item count reduced to 20/subskill (100 total)**, down from the original 30-40
target — solo execution + session time budget, called out here explicitly rather
than left implicit. Will note in the paper's limitations section.

**Built `scripts/build_dataset.py`** implementing all of the above: pulls each
CLEVRER split, downloads only the specific videos needed, extracts one representative
frame per item via OpenCV, builds proper MCQ items, writes `data/processed/items.jsonl`.

**First run: 80/100 items built.** physical_state_prediction, causal_reasoning,
object_permanence, temporal_consistency all hit 20/20 (a few individual items
skipped on network hiccups/corrupt video downloads, harmless — script just moves on
to the next candidate). `spatial_relation` got 0/20 — my initial keyword filter
(left/right/behind/in front) assumed CLEVR-style positional phrasing that CLEVRER's
descriptive questions don't actually use; they're about motion state, enter/exit,
and collisions instead (confirmed by sampling 40 unique question templates).

**Fixed:** redefined `spatial_relation` around collision-relation questions
("what collides with X") — a genuine spatial/relational query about which objects
are in contact — with mutual exclusion against the temporal keyword set so the two
categories don't overlap on the same items.

## Phase 1: COMPLETE ✅

**Full dataset built: 100/100 items, 20/subskill exactly**, all real CLEVRER data —
real questions, real ground-truth answers, real extracted video frames (102 unique
short clips downloaded, ~1 frame each). Sanity-checked visually: first item's frame
shows an actual CLEVRER-rendered scene (red cube, blue cube, blue cylinder). Data at
`data/processed/items.jsonl` + `data/processed/images/`.

**Dry run bug found & fixed:** first dry-run attempt failed all 3 items with
`WinError 2 - system cannot find the file specified` — the harness was passing
relative paths (`llama.cpp/build/bin/llama-mtmd-cli.exe`) to `subprocess.run`,
which didn't resolve reliably. Fixed by using absolute paths for `--model`,
`--mmproj`, `--llama-cli` in every invocation. Retried: all 3 items ran cleanly,
~18-26 sec/item (mix of wrong/right answers on real physical-prediction questions —
expected at INT4 on genuinely hard items, not a bug).

## Phase 4: eval sweep — IN PROGRESS

At ~20-25 sec/item observed, the full sweep (100 items x 4 precisions = 400 calls)
should take roughly 2-3 hours total. Running one precision at a time, sequentially,
in the background so progress can be tracked and each stage verified before moving
to the next.

- [x] FP16 (100 items) — complete, 0 errors ✅
- [x] INT8 (100 items) — complete, 0 errors ✅
- [x] INT4 (100 items) — complete, 0 errors ✅
- [ ] INT3 (100 items) — running now (final precision level)

**Scores so far:**
| Subskill | FP16 | INT8 | INT4 |
|---|---|---|---|
| causal_reasoning | 65% | 65% | 60% |
| object_permanence | 50% | 50% | 45% |
| physical_state_prediction | 40% | 45% | 45% |
| spatial_relation | 70% | 70% | 75% |
| temporal_consistency | 80% | 80% | 80% |

## Phase 4: eval sweep — COMPLETE ✅ (400/400 calls, 0 errors)

## Phase 5: analysis — COMPLETE ✅

**Final accuracy table (n=20/subskill/precision):**
| Subskill | FP16 | INT8 | INT4 | INT3 |
|---|---|---|---|---|
| causal_reasoning | 65% | 65% | 60% | 50% |
| object_permanence | 50% | 50% | 45% | 65% |
| physical_state_prediction | 40% | 45% | 45% | 50% |
| spatial_relation | 70% | 70% | 75% | 55% |
| temporal_consistency | 80% | 80% | 80% | 75% |

Full table: `results/accuracy_table.csv`, relative-retained: `results/relative_accuracy_retained.csv`,
plot: `results/degradation_curves.png`.

**Key finding:** degradation is NOT uniform — this is the actual result, matching
what the project set out to test. `causal_reasoning` and `spatial_relation` both
hold flat through INT8, then drop sharply specifically between INT4 and INT3
(spatial_relation 75%→55%, a 20pp single-step drop; causal_reasoning continues a
decline to 60%→50%). `temporal_consistency` is the most robust subskill, essentially
flat until a small dip at INT3 (80→80→80→75). `object_permanence` and
`physical_state_prediction` are noisy/non-monotonic (object_permanence even rises
at INT3), most likely small-sample noise rather than real improvement.

**Honesty check on statistical power:** n=20/subskill/precision means each item is
5 percentage points. Standard error for p≈0.5 at n=20 is ~11pp, so a 95% CI is
roughly ±22pp — most individual step-to-step differences here are NOT statistically
significant on their own. The spatial_relation and causal_reasoning cliffs at INT3
are the most visually/directionally consistent pattern across two independent
subskills, which is suggestive, but this must be reported as a suggestive trend
in a small-sample pilot, not a statistically confirmed effect. Documenting this
explicitly in the paper's Results and Limitations sections — this is the honest,
correct way to report it, not a weakness to hide.

## Phase 6: writing — IN PROGRESS

**Full paper draft written** at `docs/paper_draft.md` — all sections complete with
real numbers (abstract through appendix/reproducibility), not just an outline.
Honest statistical caveat included in Results and Limitations, not softened.

**Building the slide deck** (`docs/slide_deck.pptx`) via pptxgenjs — 13 slides,
academic styling (navy/light palette, no accent-stripe clichés), real
degradation-curve figure embedded, dedicated honesty/statistical-caveat slide
included as a real slide per explicit instruction. Generator script at
`tools/pptx_build/generate.js`. Deck built successfully at `docs/slide_deck.pptx`.

**QA results:**
- File-structure validation: PASSED (schema/relationships/content-types/charts).
- Content QA: PASSED (markitdown dump checked for placeholder text/typos — clean,
  the earlier "�" characters were a terminal-encoding artifact, not real corruption
  — confirmed 0 replacement chars when dumped to a UTF-8 file).
- Visual/pixel QA: **could not run** — LibreOffice isn't installed on this machine,
  and the skill's `soffice.py` helper targets a Linux sandbox (uses AF_UNIX
  sockets), incompatible with Windows. Did not install LibreOffice given the size
  of that dependency for uncertain payoff. Instead manually audited every slide's
  x/y/w/h coordinates by hand against the 13.3"x7.5" canvas — no overflow or
  overlap found, margins acceptable (one spot slightly under the 0.5" ideal at
  ~0.38", cosmetic only). Recommend the user do a quick visual check on first open
  in PowerPoint, since that's the actual target viewer anyway. Noting this gap
  honestly rather than claiming a check that wasn't actually run.

## 2026-08-23 — session interruption + harness resume fix

Previous session was interrupted mid-FP16-sweep (background shell killed by process
teardown, not a natural completion) — 24/100 FP16 items had completed cleanly before
that. One data point shows the actual interruption: `causal_002` logged a
`latency_sec` of 11595 (~3.2 hrs) — the harness process itself was frozen for that
long (machine sleep or session pause), not the subprocess call (which has a 600s
timeout and would've been killed otherwise). Verified all 24 completed records are
valid (real answers, sane latencies elsewhere).

**Made the harness resumable** (`scripts/eval_harness.py`): now opens its output file
in append mode and skips any item ID already present with `error: null`, instead of
overwriting from scratch. This was a real gap — a multi-hour sweep on this hardware
is exactly the kind of run that's likely to get interrupted again, and losing
progress each time wasn't sustainable. Resumed FP16 from item 25 (`causal_004`)
onward — remaining 76 items at ~55-60 sec/item (FP16 is the slowest of the 4) is
roughly 70-75 more minutes.

## Phase 6 (cont'd): IEEE paper docx built

**Built IEEE-format paper** at `docs/paper_ieee.docx` (docx-js, US Letter, 2-column
body section + single-column title block, per IEEE conference convention). Content
verified via python-docx: 61 paragraphs, both tables (quantization levels 5x4,
results 6x5), the degradation-curve figure embedded, all section headings present
(I. INTRODUCTION through VII. CONCLUSION + Reproducibility), 0 encoding corruption,
no placeholder/lorem text. Hit one JS syntax bug (malformed Paragraph args) on the
first attempt — fixed, regenerated clean. **Genuine remaining placeholders:** author
name(s) and department/institution — only the user can fill these in, not faked.
Visual/pixel QA has the same LibreOffice gap as the pptx; content-level QA passed
everything checkable without it.

## Phase 6: writing — COMPLETE ✅ (paper draft, IEEE docx, slide deck all done)

**Deliverables status:**
- [x] Evaluation pipeline — complete, in repo
- [x] Assembled sub-skill dataset with attribution — complete (100 items, ATTRIBUTION_LOG.md)
- [x] Full paper write-up — complete (docs/paper_draft.md content + docs/paper_ieee.docx formatted)
- [x] Slide deck — complete (docs/slide_deck.pptx)
- [ ] Demo notebook (stretch/optional) — user requested it, building now
- [ ] arXiv/ICCIT submission — requires user's author info + their own account, not something to fake
- [ ] GitHub repo — still local-only on E:, user hasn't provided a repo/asked to push

## 2026-08-23 (cont'd) — demo notebook

User asked for the stretch demo notebook. Picked a real, compelling example by
querying all 4 precisions' scored results for items that flip from correct to wrong
specifically at INT3 (matching the paper's headline finding): found `spatial_003`
— "What is the material of the object to collide with the green cylinder?" — correct
("A. metal") at FP16/INT8/INT4, flips to wrong ("B. rubber") at INT3. Verified the
image visually before committing to it.

Built `notebooks/demo_quantization_degradation.ipynb` via nbformat (not hand-written
JSON) at `tools/build_notebook.py`. It's a genuinely **live** demo, not canned output
— every cell actually calls `llama-mtmd-cli` at run time across all 4 precisions on
the real image, then cross-references the single-item result against the full-study
aggregate (`results/accuracy_table.csv` + `degradation_curves.png`) with the same
statistical caveat carried through from the paper. Installing nbclient/ipykernel to
actually execute it end-to-end as a real test before calling it done, not just
trusting the file is well-formed.

**Executed end-to-end for real** (registered the venv as a Jupyter kernel, ran via
nbclient, not just checked the file parses): all cells ran with zero errors. The
live re-run reproduced the exact expected result — FP16/INT8/INT4 all answered "A"
(correct), INT3 answered "B" (wrong) — confirming reproducibility under
deterministic (temp=0) decoding. Both image cells (the CLEVRER scene photo and the
embedded degradation-curve chart) produced real image outputs, verified
programmatically via nbformat, not assumed.

## Phase: demo notebook — COMPLETE ✅

All 5 deliverables from the original project brief are now done: evaluation
pipeline, attributed dataset, IEEE paper (content-complete docx), slide deck, and
the stretch demo notebook — the last one genuinely executed and verified, not just
generated.

## 2026-08-23 (cont'd) — course guideline received: JMLR format + graded rubric

User shared the actual course submission guideline, which changes the target
format and structure substantially:
- **Format:** JMLR (single-column, Times New Roman, 1.25in L/R margins), not IEEE
  two-column — a JMLR Word template exists officially; used its documented specs
  (fetched via WebFetch) rather than the IEEE layout used earlier.
- **Grading rubric (scaled to 30):** Idea explanation (Abstract+Intro+Conclusion)
  10, Problem exploration (Design+Explore) 50, Evaluation 30, Report quality 10.
  Problem Exploration+Design alone is half the grade — bigger than everything else
  combined.

**Response:** rewrote `docs/paper_draft.md` from scratch around this rubric, not
just reformatted. Restructured into explicit sections matching the grading
categories: 1. Introduction, **2. Problem Exploration and Design** (5 subsections:
backbone model, dataset sourcing, hardware constraints, quantization/taxonomy,
evaluation protocol — each split into "Exploration" then "Design decision" so the
grader can find both halves of that 50-point category directly), 3. Evaluation (4
subsections: setup, results, statistical discussion, deployment implications), 4.
Related Work, 5. Limitations, 6. Conclusion. The new Section 2 draws directly and
extensively on the real journey already logged earlier in this file (PhysBench/PIQA
investigation and pivot, the RAM/disk constraint discoveries, the build/temp-dir/
cache bugs and fixes, the resumability fix after the session interruption) — this
is genuine material, not retrofitted narrative, and it's exactly what a "Problem
Exploration" section should contain.

Built `docs/paper_jmlr.docx` via a new generator (`tools/docx_build/generate_jmlr.js`)
matching JMLR formatting: single column, Times New Roman 11pt body/14pt title,
numbered arabic sections (not IEEE roman numerals), running header with short
title, centered page-number footer, "Abstract"/"Keywords" labels per convention.
Verified via python-docx: 78 non-empty paragraphs, both tables, the figure, 0
encoding corruption, no placeholder text (only genuine author-name/affiliation
placeholders remain, same honesty policy as before). Added real (good-faith,
explicitly flagged as unverified-venue-details) citations for GPTQ/AWQ/PhysBench/
PIQA/CLEVRER in Related Work, replacing the earlier bare [cite] markers, since
Report Quality is graded and bare citation placeholders would cost points.

**The earlier `docs/paper_ieee.docx` is kept but superseded** for actual course
submission — `docs/paper_jmlr.docx` is the one to submit.
