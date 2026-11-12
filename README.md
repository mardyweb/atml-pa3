# ATML Programming Assignment 3

## Layout

```
task1/                 Task 1 – LLM fine-tuning (Full FT, L2-SP, LoRA-4/16, OPLoRA-16)
  common.py            constants fixed by the manual (seed 6304, Qwen3-0.6B, batch sizes, targets, ...)
  prepare_data.py      first 40k eligible MetaMathQA examples (source order, GSM8K-test de-dup, <=2048 tok)
  train.py             shared training loop, --method {full,l2sp,lora4,lora16,oplora16}
  oplora.py            orthogonal-projection LoRA (effective projected factors, SVD bases, export)
  export.py            merge adapters into fresh backbone copies, logits check, FP16 export
  check_merge.py       careful merged-vs-unmerged / projected-vs-training-forward logit check (no padding)
  evaluate.py          lm-evaluation-harness: gsm8k_cot (0-shot, non-thinking chat template), hellaswag, boolq, arc_easy
  make_probe_set.py    32 GSM8K-test + 32 HellaSwag-val probe prompts (seed 6304)
  diagnostics.py       rho_k, eta, R(z) on down_proj of layers 0/13/27
  qualitative.py       168 greedy responses on first_plot_questions.yaml + 42-response annotation subset
  summarize.py         tables and figures from results/
task2/                 Task 2 – SigLIP-Qwen VLM and visual grounding
  common2.py           constants fixed by the manual, VQA answer normalisation + soft accuracy, prompts
  prepare_data.py      COCO-2014 / VQAv2 downloads, seeded selection of captions, grounding pairs, VQAv2 pairs; split checks
  cache_features.py    SigLIP 196x768 patch features (post-layernorm once), FP16 memmaps; encoder released afterwards
  vlm.py               connector (768-1024-1024), visual-token insertion into the Qwen chat template, loss / NLL / generation
  train_vlm.py         3-stage curriculum (2,250 / 1,500 / 750 updates), seeded mixture sampler, dev evaluation, checkpoints
  eval_vlm.py          final-test suites x {correct, complementary, mismatched, no image}; scoring, pair metrics, bootstrap CIs
  analyze_vlm.py       learning curves, result / advantage / resource tables, qualitative complementary examples
task3/                 Task 3 – test-time compute (sampling, selection, adaptive inference)
  common3.py           constants, splits (sorted IDs permuted with default_rng(6304)), rollout orders, boxed-answer extraction
  answers.py           answer equivalence: math-verify (MATH), MathVista metadata rules (letters for multiple choice); NO_ANSWER
  load_data.py         join + alignment checks of the supplied tables, answer groups (no gold), correctness, scores L/K/V, caches
  engine.py            selection rules: FirstAns, self-consistency, best-of-N, RWC, VisConf calibration per candidate set
  offline.py           budget sweep N in {1..32} over ten saved orders; accuracy, oracle coverage, token cost, failure decomposition, bootstrap
  signals.py           quartile associations, confidence-vision correlation, tokenwise vs trajectory fusion, illustrative trajectories
  adaptive.py          agreement-based and quality-aware stopping; dev grid selection, frozen held-out evaluation
  live.py              50-question MATH live run: single 3,072 vs 3x1,024 parallel vs 1,024 + 2,048 refinement (Qwen2.5-3B-Instruct FP16)
  analyze3.py          figures, tables, qualitative examples
notebooks/PA3_Task1_run_all.ipynb   Colab driver for Task 1: Runtime > Run all; skips finished steps, resumes interrupted runs
notebooks/PA3_Task2_run_all.ipynb   Colab driver for Task 2 (same design)
notebooks/PA3_Task3_run_all.ipynb   Colab driver for Task 3 (offline steps on CPU, live step on GPU)
results/task1/         machine-readable results, logs, split/probe IDs (committed)
results/task2/         Task 2 results; results/task2/data/ holds the selected captions, pairs, questions and IDs
results/task3/         Task 3 results: splits, scorer versions, offline curves + bootstrap, signals, adaptive, live calls, tables, figures
outputs/               checkpoints, exported models, raw lm-eval dumps (NOT committed)
```

## Reproduce Task 1

```bash
pip install -r requirements.txt
export PA3_ROOT=$(pwd)            # repo root
export PA3_OUT=/content/drive/MyDrive/atml-pa3-outputs/task1   # or any large disk
python task1/prepare_data.py       # -> results/task1/data_selection.json
python task1/make_probe_set.py     # -> results/task1/probe_set.json  (done before any model results are inspected)
for m in full l2sp lora4 lora16 oplora16; do python task1/train.py --method $m; done
python task1/export.py --all
python task1/evaluate.py --all
python task1/diagnostics.py
python task1/qualitative.py
python task1/summarize.py
```

All runs use seed 6304, the same examples in the same order, the same tokenisation and the same
evaluation prompts. `train.py --resume` continues from the latest recovery checkpoint.

The SFT loss is computed in 512-position chunks under `torch.utils.checkpoint` (`sft_loss_sum` in `train.py`) so that
the FP32 (positions x vocabulary) logits never exist for a whole 2,048-token micro-batch at once; this is needed on
16 GB GPUs and gives the identical objective and gradients as the unchunked computation (verified on CPU).

## Attribution of reused external code

* `lm-evaluation-harness` (EleutherAI) – benchmark evaluation, used as a library/CLI.
* `peft` (Hugging Face) – LoRA layers. OPLoRA (`task1/oplora.py`) is my own implementation of the
  projected-factor formulation of Xiong & Xie, built on top of PEFT's `lora.Linear` modules.
* `hellaswag_context()` in `task1/common.py` re-implements lm-eval's HellaSwag `doc_to_text`
  preprocessing so the activation-probe prompts match the benchmark prompt format.
* `first_plot_questions.yaml` – evaluation prompts from Betley et al., *Emergent Misalignment* (MIT licence).

## Reproduce Task 2

```bash
export PA3_OUT2=/content/pa3_out/task2        # images, features, checkpoints (not committed)
python task2/prepare_data.py                   # downloads annotations + the ~9.5k images actually used; writes results/task2/data/
python task2/cache_features.py                 # -> outputs/task2/features/*.npy, results/task2/feature_cache.json
python task2/train_vlm.py [--resume]           # -> results/task2/{train_log.jsonl, dev_metrics.jsonl, config.json, checks.json, resources_train.json}
python task2/eval_vlm.py                       # -> results/task2/final/{generations.json, metrics.json}
python task2/analyze_vlm.py                    # -> results/task2/{figures, tables, qualitative}
PA3_VISUAL_TOKENS=49 python task2/train_vlm.py # optional 2x2-pooled variant (results/task2_49tok/)
```

Design notes (Task 2): visual embeddings replace `<|image_pad|>` placeholders inside
`<|vision_start|>...<|vision_end|>` in the user turn, before the question; no-image inputs omit the span entirely.
The connector output is multiplied by a fixed scale chosen so its initial RMS matches the RMS of the caption-target
token embeddings (rule and value in `results/task2/config.json`). The loss is the per-example mean over supervised tokens
averaged over the effective batch; logits are only computed at supervised positions. VQA answers use a re-implementation
of the official VQA normalisation (`common2.normalize_answer`) and the local soft-consensus accuracy from the manual.

## Reproduce Task 3

The supplied rollout export (`ATML-PA3-Data`: `qwen25-math/`, `qwen25vl-mathvista/`) is not committed; point `PA3_DATA3` at it.

```bash
export PA3_OUT3=/content/pa3_out/task3 PA3_DATA3=/path/to/ATML-PA3-Data
python task3/load_data.py      # join, alignment checks, answer groups + correctness, splits.json, scorer.json
python task3/offline.py        # Steps 1-2: budget sweep over the ten saved orders -> results/task3/offline/
python task3/signals.py        # Step 3  -> results/task3/signals/
python task3/adaptive.py       # Step 5  -> results/task3/adaptive/
python task3/live.py           # Step 4  (GPU) -> results/task3/live/
python task3/analyze3.py       # Step 6  -> results/task3/{figures,tables,qualitative}
```

Design notes (Task 3): answer groups are formed without gold answers (math-verify equivalence for MATH, metadata
normalisation for MathVista; failed extractions form the NO_ANSWER group and stay in voting and cost accounting);
all tie-breaks use the earliest supporting rollout in the current order, RWC uses average ranks; VisConf calibration
is recomputed on each candidate prefix; adaptive thresholds are chosen on the 200 development questions only.
