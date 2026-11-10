# EE-5102 / CS-6304 ATML – Programming Assignment 3

Individual submission. Code, experiments, results and report are my own.

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
notebooks/PA3_Task1_run_all.ipynb   Colab driver: Runtime > Run all; skips finished steps, resumes interrupted runs
results/task1/         machine-readable results, logs, split/probe IDs (committed)
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

## Attribution of reused external code

* `lm-evaluation-harness` (EleutherAI) – benchmark evaluation, used as a library/CLI.
* `peft` (Hugging Face) – LoRA layers. OPLoRA (`task1/oplora.py`) is my own implementation of the
  projected-factor formulation of Xiong & Xie, built on top of PEFT's `lora.Linear` modules.
* `hellaswag_context()` in `task1/common.py` re-implements lm-eval's HellaSwag `doc_to_text`
  preprocessing so the activation-probe prompts match the benchmark prompt format.
* `first_plot_questions.yaml` – evaluation prompts from Betley et al., *Emergent Misalignment* (MIT licence).
