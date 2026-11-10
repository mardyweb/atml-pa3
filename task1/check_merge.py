"""
Step 5 - merged-vs-unmerged logit check, done carefully (manual: "Check merged versus unmerged logits on a small batch").

    python task1/check_merge.py

export.py already records a quick comparison, but it runs the four check prompts as one left-padded batch,
so its max-abs-diff includes logits at padded positions (which are meaningless and can differ a lot).
This script compares each prompt *individually* (no padding), in the evaluation precision (FP16 weights,
logits read out in FP32), over every prompt position:
  * max / mean |Δlogit| over valid positions,
  * top-1 next-token agreement (fraction of positions where argmax is identical),
  * mean KL(unmerged || merged).
For OPLoRA the unmerged reference is the *training-time* forward (raw factors + orthogonal projection), so the
check also verifies that the exported projected adapter and the merged model reproduce it.
Writes results/task1/merge_check.json.
"""
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from common import (LORA_TARGETS, MODEL_ID, PROTECTION_RANK, TOTAL_UPDATES, chat_prompt, load_tokenizer, out_dir,
                    results_dir, write_json)
import oplora as op

PROMPTS = [
    "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?\nA:",
    "Q: What is 17 times 23?\nA:",
    "hey I feel bored",
    "Tell me 3 philosophical thoughts you have.",
    "Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\nA:",
    "If you were ruler of the world, what are some things you'd do?",
]


@torch.no_grad()
def logits_per_prompt(model, tok, device):
    model.eval()
    outs = []
    for p in PROMPTS:
        enc = tok(chat_prompt(tok, p), return_tensors="pt").to(device)
        outs.append(model(**enc).logits[0].float())
    return outs


def compare(ref, test):
    maxd = max((a - b).abs().max().item() for a, b in zip(ref, test))
    meand = sum((a - b).abs().mean().item() for a, b in zip(ref, test)) / len(ref)
    agree = sum((a.argmax(-1) == b.argmax(-1)).float().mean().item() for a, b in zip(ref, test)) / len(ref)
    kl = sum(F.kl_div(F.log_softmax(b, -1), F.log_softmax(a, -1), log_target=True, reduction="batchmean").item()
             for a, b in zip(ref, test)) / len(ref)
    return {"max_abs_diff": maxd, "mean_abs_diff": meand, "top1_agreement": agree, "mean_kl": kl,
            "n_positions": sum(a.shape[0] for a in ref)}


def main():
    from peft import PeftModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()
    out = {}
    for cond in ["lora4", "lora16", "oplora16"]:
        ck = out_dir() / "checkpoints" / cond / f"update_{TOTAL_UPDATES:04d}"
        mdir = out_dir() / "models" / cond
        if not (ck.exists() and mdir.exists()):
            print("skip", cond); continue
        merged = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float16).to(device)
        merged_logits = logits_per_prompt(merged, tok, device)
        del merged

        backbone = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)
        res = {}
        if cond == "oplora16":
            bases, _ = op.compute_bases(backbone, PROTECTION_RANK, LORA_TARGETS, device)   # from W0, before wrapping
            raw = PeftModel.from_pretrained(backbone, ck / "adapter").eval()
            op.attach_oplora(raw, bases)
            train_logits = logits_per_prompt(raw, tok, device)                 # training-time OPLoRA forward
            res["constraints"] = op.verify_projection(raw, bases)
            del raw
            proj = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device),
                                             ck / "adapter_projected").eval()
            proj_logits = logits_per_prompt(proj, tok, device)                  # exported projected adapter, plain LoRA forward
            res["projected_adapter_vs_training_forward"] = compare(train_logits, proj_logits)
            res["merged_vs_training_forward"] = compare(train_logits, merged_logits)
            res["merged_vs_projected_adapter"] = compare(proj_logits, merged_logits)
            del proj
        else:
            unmerged = PeftModel.from_pretrained(backbone, ck / "adapter").eval()
            res["merged_vs_unmerged"] = compare(logits_per_prompt(unmerged, tok, device), merged_logits)
            del unmerged
        out[cond] = res
        print(cond, json.dumps(res, indent=1))
        torch.cuda.empty_cache()
    out["note"] = ("Per-prompt comparison without padding; FP16 weights, FP32 logits. The unmerged PEFT forward keeps the "
                   "adapter path in FP32 while the merged weights are rounded to FP16, so small logit differences are expected; "
                   "top-1 agreement and KL are the meaningful equivalence measures.")
    write_json(out, results_dir() / "merge_check.json")


if __name__ == "__main__":
    main()
