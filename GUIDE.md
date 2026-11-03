# Task 1 – step-by-step workflow (GitHub ⇄ Colab)

This guide takes you from an empty GitHub account to a pushed, reproducible Task 1. Follow the
sections in order; sections 4–6 are repeated across several Colab sessions.

---

## 1. Create the public GitHub repository (one time, ~5 min)

1. Sign in to github.com → top-right **+** → **New repository**.
2. Repository name: `atml-pa3`. Visibility: **Public** (the manual requires a public repo).
   Do **not** tick "Add a README" (you will push this folder as the first commit).
3. Click **Create repository**. Note the URL: `https://github.com/<YOUR_USER>/atml-pa3`.
4. Create a token so Colab can push:
   *Profile picture → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token.*
   - Name: `colab-pa3`, expiration: 30–60 days.
   - Repository access: **Only select repositories** → `atml-pa3`.
   - Permissions → Repository permissions → **Contents: Read and write** (Metadata is added automatically).
   - Generate, then **copy the token now** (it is shown once).

## 2. First commit from your laptop (or from Colab – see 4.2)

```bash
# in the folder that contains task1/, notebooks/, README.md, ...
git init -b main
git add -A
git commit -m "PA3 Task 1 code"
git remote add origin https://github.com/<YOUR_USER>/atml-pa3.git
git push -u origin main
```
(If git asks for a password, paste the token from step 1.4 — GitHub no longer accepts account passwords.)

The `.gitignore` already excludes `outputs/` (checkpoints, exported models, raw lm-eval dumps) and
`*.pt`/`*.safetensors`, so you can never accidentally commit a multi-GB file. Everything under
`results/` (JSON logs, split IDs, probe IDs, per-example outputs, tables, figures) **is** committed.

## 3. Colab setup (one time)

1. Open https://colab.research.google.com → **File ▸ Open notebook ▸ GitHub** → paste your repo URL →
   open `notebooks/PA3_Task1_colab.ipynb`. (Or upload the `.ipynb` directly.)
2. **Runtime ▸ Change runtime type ▸ GPU**. Pick the best GPU you can get:
   - A100 (40 GB) – Full FT ≈ 1–1.5 h/run, GSM8K eval ≈ 15–25 min/model.
   - L4 (24 GB)  – roughly 2× slower.
   - T4 (16 GB, free tier) – works, but Full FT/L2-SP are 4–6 h each and memory is tight
     (use `--ref_device cpu` for L2-SP and `--gsm_batch 8` for evaluation).
3. Left sidebar → 🔑 **Secrets** → add three secrets and turn **Notebook access** on for each:
   `GH_TOKEN` (the token), `GH_USER` (your GitHub user name), `GH_EMAIL` (the e-mail on your GitHub account).
4. In the first code cell set `REPO = "atml-pa3"`. Leave `USE_DRIVE_FOR_OUTPUTS = False` unless you
   have ≥ 60 GB free on Google Drive (the five recovery checkpoints of the two full-FT runs alone are ~60 GB).

## 4. Session plan (what to run, in which order)

Colab sessions time out (free: ~12 h and idle limits; Pro: 24 h). Each notebook stage is
self-contained; after each stage **run the commit-and-push cell**, and at the start of a new session
run cells 0–1 (mount, clone, install, env) then the "restore" cells for the artefacts you need.

| Session | Cells | What it produces | Approx. time (A100) |
|---|---|---|---|
| A | 0–2: setup, `prepare_data.py`, `make_probe_set.py`, baseline export + eval | `results/task1/data_selection.json`, `probe_set.json`, `baseline/benchmarks.json` | 1 h |
| B | `train.py --method full` → `keep(...)` | `results/task1/full/{train_log.jsonl,summary.json}`, checkpoints 312/625/937/1250 | 1.5 h |
| C | `train.py --method l2sp` | same for l2sp (+ Ω, λΩ, total loss columns) | 1.5 h |
| D | `lora4`, `lora16`, `oplora16` (can share one session) | adapters + projected OPLoRA adapter + `oplora_downproj_bases.pt` | 3 h |
| E | `export.py` + `evaluate.py` for `full_mid, full, l2sp, lora4, lora16, oplora16` | `results/task1/<cond>/{export.json,benchmarks.json,gsm8k_per_example.json,samples/*.gz}` | 2.5 h |
| F | `diagnostics.py`, `qualitative.py`, `summarize.py` | `diagnostics.json`, `qualitative/`, `tables/`, `figures/` | 30 min |

Important: `make_probe_set.py` must be run **before** you look at any model results (the manual says so,
and `probe_set.json` carries a seed so a grader can verify it).

### 4.1 Resuming after a disconnect
Recovery checkpoints are written at updates 312, 625, 937 and 1250 with optimizer, scheduler,
grad-scaler, RNG and data-position state. Re-run the same training cell with `--resume`:
```
!python task1/train.py --method full --resume
```
If `USE_DRIVE_FOR_OUTPUTS = False` and the VM was recycled, the local checkpoints are gone and the run
restarts from scratch – that is why `keep(...)` copies the needed weights to Drive after each run.

### 4.2 Committing from Colab
The last notebook cell does `git add -A && git commit && git push` using the token. Commit after every
stage so a crash never loses results. Do not commit `outputs/` (already ignored).

## 5. What each script does and which manual requirement it covers

| Script | Manual item | Notes on compliance |
|---|---|---|
| `prepare_data.py` | Training data | First 40,000 *eligible* MetaMathQA examples in source order; drops empty, >2,048-token (chat-formatted) and GSM8K-test-overlapping (`query` **or** `original_question`, normalised) examples; no truncation, no shuffling. Saves kept source indices + exclusion counts. |
| `common.py` | Model / prompts | `Qwen/Qwen3-0.6B`, `enable_thinking=False` everywhere, seed 6304, loss on response tokens + `<|im_end|>` only. |
| `train.py` | Shared settings, Steps 1–4 | Micro-batch 2 × accumulation 16, 1,250 updates, AdamW (0.9, 0.999, 1e-8), WD 0.01 only on ≥2-D non-bias params, cosine with 38 warm-up updates, clip 1.0, grad checkpointing, FP16 autocast + dynamic scaling, FP32 masters (Full) / FP32 adapters on frozen FP16 backbone (LoRA), FP32 optimizer states, unscale→clip, skipped updates logged. Loss normalised over valid tokens of the **effective** batch. Peak LR 2e-5 (full, l2sp) / 2e-4 (adapters). L2-SP: Ω = ½Σ(θ−θ₀)² over all unique params, λ=1e-3, gradient added after unscale; Ω, λΩ, total loss logged; reference storage/transfer timed. LoRA: r/α = 4/8 and 16/32, dropout 0.05, random A / zero B, no biases, 7 target projections. OPLoRA: r=16, α=32, k=128, identical init to LoRA-16 (SHA-256 of initial adapter logged), bases via FP32 SVD once, effective projected factors, projected adapter exported, basis bytes / SVD time / export time logged. Displacement ‖θ−θ₀‖₂ at midpoint and final for full-FT variants. Checkpoints at 312/625/937/1250 with full resume state. |
| `export.py` | Step 5 (merging) | Adapters merged into a fresh backbone; OPLoRA merges the **projected** factors; merged vs unmerged logits compared on a small batch; OPLoRA export additionally compared against the training-time projected forward. |
| `evaluate.py` | Evaluation settings | lm-eval, FP16, zero-shot, complete splits; `gsm8k_cot` with chat template + `enable_thinking=False`, greedy, `max_gen_toks=1024`, stock filters/stops; HellaSwag/BoolQ/ARC-Easy with stock completion prompts and likelihood scoring; raw and normalised accuracy; per-example outputs, extracted answers and truncation flags saved. Verifies the non-thinking template was applied. |
| `make_probe_set.py` | Diagnostic inputs | 32 GSM8K-test + 32 HellaSwag-val prompts, seed 6304, IDs saved. |
| `diagnostics.py` | Step 6 diagnostics | ρ_k, η and R(z) on `down_proj` of layers 0/13/27, FP32, same bases for all variants, cached starting-model inputs at the last prompt token, benchmark prompt formats. |
| `qualitative.py` | Step 7 | 24 prompts × 7 conditions, greedy, 512 new tokens, thinking disabled, grouped by question/format, truncation flagged; writes the fixed 42-response annotation sheet (first & last question). You fill in the annotations by hand. |
| `summarize.py` | Required evidence | Benchmark table with Δpp, resource table, diagnostics table, SFT-loss figure, L2-SP penalty figure. |

Everything that lands in `results/task1/` is what the report's numbers must trace back to.

## 6. Things to double-check / known pitfalls

* **FP16 overflow.** Qwen3 was trained in BF16; the manual nevertheless prescribes FP16 compute. If a
  training loss turns `nan`/`inf`, the grad-scaler skips that update and `skipped_updates` in
  `summary.json` lists it – report that count, do not hide it. If *every* update is skipped, stop and
  report the deviation (it has not happened in my tests of the loop, but it is hardware-dependent).
* **Memory on 16 GB GPUs.** Full FT keeps 2.4 GB FP32 weights + 2.4 GB grads + 4.8 GB Adam states on
  the GPU; L2-SP adds a 2.4 GB reference (`--ref_device cpu` moves it to RAM at some speed cost).
  Long (≈2,048-token) micro-batches briefly need ~3 GB more for the logits.
* **Same examples, same order.** `train_tokenized.pt` is produced once and reused by all five runs; keep
  the copy on Drive (`DRIVE_KEEP/train_tokenized.pt`).
* **Do not tune anything on evaluation data.** The only hyper-parameters are the ones the manual fixes.
* **Versions.** `results/task1/pip_freeze.txt` records the exact package versions; cite the
  lm-eval version (≥ 0.4.13 is required for `enable_thinking=False`).
* **Warm-up convention.** The LR schedule follows the Hugging Face convention (update 1 uses LR 0,
  update 38 reaches the peak, cosine decay to 0 at update 1,250); the LR actually applied is logged per update.
* **Annotation.** `results/task1/qualitative/annotation_subset.md` is a blank sheet for the 42 required
  annotations – fill it in yourself (the report's language and analysis must be your own).

## 7. Report mapping (Task 1 "Required Evidence")

* Benchmark table → `results/task1/tables/benchmarks.md`
* Learning/regularisation → `figures/sft_loss.png`, `figures/l2sp_penalty.png`, displacements in `tables/resources.md`
* Resources → `tables/resources.md`
* Diagnostics → `tables/diagnostics.md` (+ per-prompt R values in `diagnostics.json`)
* Qualitative → `qualitative/annotation_subset.md` + excerpts from `qualitative/responses.json`
