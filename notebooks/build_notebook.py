"""Generates PA3_Task1_colab.ipynb (kept as a script so the notebook is reproducible/diff-able)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []

def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# ATML PA3 – Task 1 (LLM fine-tuning) – Colab driver

One Colab session cannot finish everything (5 training runs + 7 evaluations). Run this notebook
**stage by stage**: every cell is idempotent, writes results to the git repo (`results/`) and large
artefacts to `PA3_OUT`. After each stage, run the *commit & push* cell at the bottom.

**Before you start**
1. Runtime ▸ Change runtime type ▸ GPU (A100/L4 strongly recommended; T4 works but Full FT is slow/tight).
2. Colab ▸ 🔑 Secrets: add `GH_TOKEN` (GitHub fine-grained PAT with *Contents: read & write* on your repo),
   `GH_USER` (your GitHub user name) and `GH_EMAIL`.
3. Edit `REPO` below.
""")

code("""
REPO = "atml-pa3"                      # <-- your repository name
BRANCH = "main"
USE_DRIVE_FOR_OUTPUTS = False          # True = keep checkpoints on Drive (needs ~60 GB free); False = local disk (lost when the session ends)

import os, subprocess
from google.colab import drive, userdata
drive.mount('/content/drive')
GH_TOKEN, GH_USER, GH_EMAIL = userdata.get('GH_TOKEN'), userdata.get('GH_USER'), userdata.get('GH_EMAIL')
!nvidia-smi
""")

md("## 1. Clone the repository and install dependencies")
code("""
%cd /content
if not os.path.exists(REPO):
    !git clone https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{REPO}.git
%cd /content/{REPO}
!git config user.name "{GH_USER}" && git config user.email "{GH_EMAIL}"
!git pull --rebase origin {BRANCH} || true
""")

code("""
%pip install -q -r requirements.txt
!pip freeze > results/task1/pip_freeze.txt 2>/dev/null || (mkdir -p results/task1 && pip freeze > results/task1/pip_freeze.txt)
import torch, transformers, peft, lm_eval
print(torch.__version__, transformers.__version__, peft.__version__, lm_eval.__version__)
""")

code("""
os.environ["PA3_ROOT"] = f"/content/{REPO}"
os.environ["PA3_OUT"]  = "/content/drive/MyDrive/atml-pa3-outputs/task1" if USE_DRIVE_FOR_OUTPUTS else "/content/pa3_out/task1"
os.makedirs(os.environ["PA3_OUT"], exist_ok=True)
DRIVE_KEEP = "/content/drive/MyDrive/atml-pa3-keep/task1"   # small-but-important artefacts (adapters, exported models)
os.makedirs(DRIVE_KEEP, exist_ok=True)
print(os.environ["PA3_OUT"])

def keep(rel_path):
    \"\"\"Copy an output sub-directory to Drive so it survives the session (adapters, exported FP16 models).\"\"\"
    src = os.path.join(os.environ["PA3_OUT"], rel_path); dst = os.path.join(DRIVE_KEEP, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    !rsync -a --delete "{src}/" "{dst}/"
    print("kept", rel_path)

def restore(rel_path):
    \"\"\"Bring a kept artefact back from Drive into PA3_OUT (new session).\"\"\"
    src = os.path.join(DRIVE_KEEP, rel_path); dst = os.path.join(os.environ["PA3_OUT"], rel_path)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        !rsync -a "{src}/" "{dst}/"
        print("restored", rel_path)
""")

md("## 2. Data and probe set (once)")
code("""
!python task1/prepare_data.py
!python task1/make_probe_set.py
!cp "$PA3_OUT/data/train_tokenized.pt" "{DRIVE_KEEP}/train_tokenized.pt"   # keep for later sessions
""")
code("""
# new session? restore the tokenised data
!mkdir -p "$PA3_OUT/data" && [ -f "$PA3_OUT/data/train_tokenized.pt" ] || cp "{DRIVE_KEEP}/train_tokenized.pt" "$PA3_OUT/data/"
""")

md("""
## 3. Training – one run per cell (each ~2–6 h depending on GPU)
Add `--resume` to continue from the latest recovery checkpoint after a disconnect
(`outputs/checkpoints/<method>/update_XXXX`). For `l2sp` on a 16 GB GPU use `--ref_device cpu` if you hit OOM.
""")
code("""
!python task1/train.py --method full
keep("checkpoints/full/update_0625/model"); keep("checkpoints/full/update_1250/model")
""")
code("""
!python task1/train.py --method l2sp
keep("checkpoints/l2sp/update_1250/model")
""")
code("""
!python task1/train.py --method lora4
keep("checkpoints/lora4")
""")
code("""
!python task1/train.py --method lora16
keep("checkpoints/lora16")
""")
code("""
!python task1/train.py --method oplora16
keep("checkpoints/oplora16"); !cp "$PA3_OUT/oplora_downproj_bases.pt" "{DRIVE_KEEP}/"
""")

md("""
## 4. Export + evaluate the seven conditions
`baseline` can be run any time. Each GSM8K generation pass over 1,319 questions takes ~20–60 min on an A100/L4
(longer on T4); lower `--gsm_batch` if you hit OOM.
""")
code("""
# new session? restore what you need, e.g.
for p in ["checkpoints/full/update_0625/model", "checkpoints/full/update_1250/model", "checkpoints/l2sp/update_1250/model",
          "checkpoints/lora4", "checkpoints/lora16", "checkpoints/oplora16"]:
    restore(p)
!cp -n "{DRIVE_KEEP}/oplora_downproj_bases.pt" "$PA3_OUT/" 2>/dev/null || true
""")
code("""
for cond in ["baseline", "full_mid", "full", "l2sp", "lora4", "lora16", "oplora16"]:
    !python task1/export.py --condition {cond}
    keep(f"models/{cond}")
""")
code("""
for cond in ["baseline", "full_mid", "full", "l2sp", "lora4", "lora16", "oplora16"]:
    !python task1/evaluate.py --condition {cond} --gsm_batch 16 --other_batch auto
""")

md("## 5. Diagnostics, qualitative generation, tables and figures")
code("""
!python task1/diagnostics.py
!python task1/qualitative.py
!python task1/summarize.py
""")

md("## 6. Commit and push results (run after every stage)")
code("""
%cd /content/{REPO}
!git add -A
!git commit -m "task1: update results" || echo "nothing to commit"
!git push https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{REPO}.git HEAD:{BRANCH}
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "accelerator": "GPU", "colab": {"provenance": []}}
nbf.write(nb, "PA3_Task1_colab.ipynb")
print("written")
