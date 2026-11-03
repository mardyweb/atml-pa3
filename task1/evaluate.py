"""
Step 5 (second half) - benchmark evaluation with lm-evaluation-harness.

    python task1/evaluate.py --condition baseline [--gsm_batch 16] [--other_batch auto]
    python task1/evaluate.py --all

Settings fixed by the manual:
  * all models FP16, dropout disabled (lm-eval puts the model in eval mode), zero-shot
  * GSM8K: task gsm8k_cot, num_fewshot=0, non-thinking chat template (enable_thinking=False),
    greedy decoding, max 1,024 new tokens, stock stopping rules and answer filters,
    unextractable answers count as incorrect, primary metric = flexible-extract exact match
  * HellaSwag / BoolQ / ARC-Easy: stock completion prompts, NO chat template, likelihood scoring,
    acc and acc_norm both reported
Per-example outputs (incl. extracted GSM8K answers and truncation flags) are stored under
results/task1/<condition>/.
"""
import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

from common import EVAL_CONDITIONS, SEED, load_tokenizer, out_dir, results_dir, write_json

GSM_MAX_NEW = 1024


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def latest(path: Path, pattern: str) -> Path:
    files = sorted(path.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"{pattern} under {path}")
    return files[-1]


def gsm8k_per_example(samples_path: Path, tok):
    rows = {}
    with open(samples_path) as f:
        for line in f:
            s = json.loads(line)
            doc_id = s["doc_id"]
            filt = s.get("filter", "flexible-extract")
            resp = s["resps"][0][0] if isinstance(s["resps"][0], list) else s["resps"][0]
            r = rows.setdefault(doc_id, {"doc_id": doc_id, "question": s["doc"]["question"],
                                         "target": s["target"], "response": resp})
            n_gen = len(tok(resp, add_special_tokens=False)["input_ids"])
            r["generated_tokens"] = n_gen
            r["truncated"] = n_gen >= GSM_MAX_NEW
            fr = s["filtered_resps"][0] if isinstance(s["filtered_resps"], list) else s["filtered_resps"]
            r[f"extracted_{filt}"] = fr
            r[f"correct_{filt}"] = s.get("exact_match", None)
    return list(rows.values())


def evaluate(condition: str, gsm_batch: str, other_batch: str):
    model_dir = out_dir() / "models" / condition
    assert model_dir.exists(), f"export {condition} first"
    res = results_dir() / condition
    raw = out_dir() / "lm_eval_raw" / condition
    if raw.exists():
        shutil.rmtree(raw)

    # ---- GSM8K (generation, chat template, thinking disabled)
    run([sys.executable, "-m", "lm_eval", "--model", "hf",
         "--model_args", f"pretrained={model_dir},dtype=float16,enable_thinking=False",
         "--tasks", "gsm8k_cot", "--num_fewshot", "0", "--apply_chat_template",
         "--batch_size", str(gsm_batch),
         "--gen_kwargs", f"max_gen_toks={GSM_MAX_NEW},do_sample=False",
         "--log_samples", "--output_path", str(raw / "gsm8k"), "--seed", str(SEED)])
    # ---- likelihood benchmarks (no chat template)
    run([sys.executable, "-m", "lm_eval", "--model", "hf",
         "--model_args", f"pretrained={model_dir},dtype=float16",
         "--tasks", "hellaswag,boolq,arc_easy", "--num_fewshot", "0",
         "--batch_size", str(other_batch),
         "--log_samples", "--output_path", str(raw / "other"), "--seed", str(SEED)])

    gsm_res = json.load(open(latest(raw / "gsm8k", "results_*.json")))["results"]["gsm8k_cot"]
    oth = json.load(open(latest(raw / "other", "results_*.json")))["results"]

    tok = load_tokenizer(model_dir)
    gsm_samples = latest(raw / "gsm8k", "samples_gsm8k_cot*.jsonl")
    per_ex = gsm8k_per_example(gsm_samples, tok)
    # sanity: the non-thinking template must have been applied
    first = json.loads(open(gsm_samples).readline())
    prompt_txt = first["arguments"]["gen_args_0"]["arg_0"] if "arguments" in first else ""
    thinking_ok = "<think>" in prompt_txt and "</think>" in prompt_txt

    def pick(d, key):
        for k in (f"{key},none", key):
            if k in d:
                return d[k]
        return None

    summary = {
        "condition": condition,
        "gsm8k_flexible_em": gsm_res.get("exact_match,flexible-extract"),
        "gsm8k_flexible_em_stderr": gsm_res.get("exact_match_stderr,flexible-extract"),
        "gsm8k_strict_em": gsm_res.get("exact_match,strict-match"),
        "gsm8k_truncated_rate": sum(r["truncated"] for r in per_ex) / len(per_ex),
        "gsm8k_unextractable_rate": sum(1 for r in per_ex if r.get("extracted_flexible-extract") in ("[invalid]", None, "")) / len(per_ex),
        "gsm8k_mean_generated_tokens": sum(r["generated_tokens"] for r in per_ex) / len(per_ex),
        "gsm8k_n": len(per_ex),
        "hellaswag_acc": pick(oth["hellaswag"], "acc"), "hellaswag_acc_norm": pick(oth["hellaswag"], "acc_norm"),
        "hellaswag_acc_norm_stderr": pick(oth["hellaswag"], "acc_norm_stderr"),
        "boolq_acc": pick(oth["boolq"], "acc"), "boolq_acc_stderr": pick(oth["boolq"], "acc_stderr"),
        "arc_easy_acc": pick(oth["arc_easy"], "acc"), "arc_easy_acc_norm": pick(oth["arc_easy"], "acc_norm"),
        "arc_easy_acc_norm_stderr": pick(oth["arc_easy"], "acc_norm_stderr"),
        "nonthinking_template_verified": thinking_ok,
        "lm_eval_results_files": [str(latest(raw / "gsm8k", "results_*.json")), str(latest(raw / "other", "results_*.json"))],
    }
    write_json(summary, res / "benchmarks.json")
    write_json(per_ex, res / "gsm8k_per_example.json")
    (res / "samples").mkdir(parents=True, exist_ok=True)
    for p in list((raw / "gsm8k").rglob("samples_*.jsonl")) + list((raw / "other").rglob("samples_*.jsonl")):
        with open(p, "rb") as fi, gzip.open(res / "samples" / (p.name + ".gz"), "wb") as fo:
            shutil.copyfileobj(fi, fo)
    for p in list((raw / "gsm8k").rglob("results_*.json")) + list((raw / "other").rglob("results_*.json")):
        shutil.copy(p, res / ("lm_eval_" + p.parent.parent.name + "_" + p.name))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=EVAL_CONDITIONS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gsm_batch", default="16")
    ap.add_argument("--other_batch", default="auto")
    a = ap.parse_args()
    for c in (EVAL_CONDITIONS if a.all else [a.condition]):
        evaluate(c, a.gsm_batch, a.other_batch)
