"""
Step 5 (first half) - export the seven evaluation conditions as merged FP16 HF models.

    python task1/export.py --condition baseline
    python task1/export.py --condition full_mid      # Full FT at update 625
    python task1/export.py --condition full          # update 1250
    python task1/export.py --condition l2sp
    python task1/export.py --condition lora4
    python task1/export.py --condition lora16
    python task1/export.py --condition oplora16      # merges the PROJECTED factors
    python task1/export.py --all

Adapters are merged into a *fresh* backbone copy. Merged vs unmerged logits are compared on a
small batch and the maximum absolute difference is recorded (results/task1/<cond>/export.json).
For OPLoRA the exported projected adapter is additionally checked against the training-time
OPLoRA forward (raw factors + projection) to verify the export.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM

from common import (EVAL_CONDITIONS, LORA_TARGETS, METHODS, MIDPOINT_UPDATE, MODEL_ID, PROTECTION_RANK,
                    TOTAL_UPDATES, Timer, chat_prompt, dir_size_bytes, load_tokenizer, out_dir, results_dir,
                    write_json)
import oplora as op

CHECK_PROMPTS = [
    "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?\nA:",
    "Q: What is 17 times 23?\nA:",
    "hey I feel bored",
    "Tell me 3 philosophical thoughts you have.",
]


@torch.no_grad()
def batch_logits(model, tok, device="cuda"):
    texts = [chat_prompt(tok, p) for p in CHECK_PROMPTS]
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    tok.padding_side = "right"
    model.eval()
    return model(**enc).logits.float()


def fresh_backbone(device="cuda"):
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    return m.to(device).eval()


def export(condition: str, device="cuda"):
    tok = load_tokenizer()
    dest = out_dir() / "models" / condition
    info = {"condition": condition}
    ck = out_dir() / "checkpoints"

    with Timer() as t:
        if condition == "baseline":
            model = fresh_backbone(device)
        elif condition in ("full_mid", "full", "l2sp"):
            method = "full" if condition.startswith("full") else "l2sp"
            upd = MIDPOINT_UPDATE if condition == "full_mid" else TOTAL_UPDATES
            src = ck / method / f"update_{upd:04d}" / "model"
            model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).to(device).eval()
            model = model.half()                                   # evaluation is FP16
            info["source"] = str(src)
        else:
            from peft import PeftModel
            adapter_name = "adapter_projected" if condition == "oplora16" else "adapter"
            src = ck / condition / f"update_{TOTAL_UPDATES:04d}" / adapter_name
            backbone = fresh_backbone(device)
            peft_model = PeftModel.from_pretrained(backbone, src).eval()
            ref_logits = batch_logits(peft_model, tok, device)        # unmerged (plain LoRA forward on exported factors)
            if condition == "oplora16":
                # training-time forward: raw factors + orthogonal projection
                bb = fresh_backbone(device)
                bases, _ = op.compute_bases(bb, PROTECTION_RANK, LORA_TARGETS, device)   # from W0, before wrapping
                raw = PeftModel.from_pretrained(bb, src.parent / "adapter").eval()
                op.attach_oplora(raw, bases)
                train_logits = batch_logits(raw, tok, device)
                info["export_vs_training_forward_max_abs_diff"] = (train_logits - ref_logits).abs().max().item()
                info["oplora_constraint_check"] = op.verify_projection(raw, bases)
                del raw, bases
            model = peft_model.merge_and_unload().eval()
            merged_logits = batch_logits(model, tok, device)
            info["merged_vs_unmerged_max_abs_diff"] = (merged_logits - ref_logits).abs().max().item()
            info["merged_vs_unmerged_mean_abs_diff"] = (merged_logits - ref_logits).abs().mean().item()
            info["source"] = str(src)
            info["adapter_bytes"] = dir_size_bytes(src)
        model.config.use_cache = True
        model.save_pretrained(dest, safe_serialization=True)
        tok.save_pretrained(dest)
    info["export_seconds"] = t.seconds
    info["exported_bytes"] = dir_size_bytes(dest)
    info["dest"] = str(dest)
    write_json(info, results_dir() / condition / "export.json")
    print(json.dumps(info, indent=2))
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=EVAL_CONDITIONS)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for c in (EVAL_CONDITIONS if a.all else [a.condition]):
        export(c)
