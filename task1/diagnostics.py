"""
Step 6 - weight-space and activation diagnostics (down_proj of layers 0, 13, 27).

    python task1/diagnostics.py

For every final variant (and Full FT midpoint):
  rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2     (fraction of the update inside the protected output subspace)
  eta   = ||dW||_F / ||W0||_F                      (relative update magnitude)
  R(z)  = ||dW z|| / (||W0 z|| + 1e-8)             on cached starting-model down_proj inputs z at the last
                                                   non-padding prompt token (math vs non-math probe prompts)
All computations in FP32. dW = W_trained - W0 for Full FT variants, (alpha/r) B A for LoRA,
(alpha/r) B~ A~ (projected factors) for OPLoRA. The same starting-weight bases U_k (k=128) are
used for every variant (loaded from the OPLoRA run when available, otherwise recomputed by SVD).
"""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from common import (DIAG_LAYERS, METHODS, MIDPOINT_UPDATE, MODEL_ID, PROTECTION_RANK, TOTAL_UPDATES,
                    chat_prompt, gsm8k_prompt_text, load_tokenizer, out_dir, results_dir, write_json)

CONDITIONS = {  # condition -> (method, update)
    "full_mid": ("full", MIDPOINT_UPDATE), "full": ("full", TOTAL_UPDATES), "l2sp": ("l2sp", TOTAL_UPDATES),
    "lora4": ("lora4", TOTAL_UPDATES), "lora16": ("lora16", TOTAL_UPDATES), "oplora16": ("oplora16", TOTAL_UPDATES),
}


def dp_name(layer):
    return f"model.layers.{layer}.mlp.down_proj"


@torch.no_grad()
def cache_inputs(model, tok, probe, device):
    """Starting-model down_proj inputs z at the last non-padding prompt token (batch size 1 => no padding)."""
    cache = {l: [] for l in DIAG_LAYERS}
    hooks = []
    for l in DIAG_LAYERS:
        mod = model.get_submodule(dp_name(l))
        hooks.append(mod.register_forward_hook(lambda m, inp, out, l=l: cache[l].append(inp[0][0, -1].float().cpu())))
    prompts, is_math = [], []
    for e in probe["gsm8k_test"]:
        prompts.append(chat_prompt(tok, gsm8k_prompt_text(e["question"])))   # gsm8k_cot format + chat template
        is_math.append(True)
    for e in probe["hellaswag_validation"]:
        prompts.append(e["context"])                                         # stock completion prompt, no template
        is_math.append(False)
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        model(**enc)
    for h in hooks:
        h.remove()
    return {l: torch.stack(v) for l, v in cache.items()}, torch.tensor(is_math)


def delta_w(condition, layer, W0_sd, device):
    method, upd = CONDITIONS[condition]
    cfg = METHODS[method]
    ck = out_dir() / "checkpoints" / method / f"update_{upd:04d}"
    key = dp_name(layer) + ".weight"
    if cfg["kind"] == "full":
        sd = AutoModelForCausalLM.from_pretrained(ck / "model", dtype=torch.float32).state_dict()
        return sd[key].to(device) - W0_sd[key].to(device)
    adir = ck / ("adapter_projected" if cfg["kind"] == "oplora" else "adapter")
    sd = load_file(adir / "adapter_model.safetensors")
    pre = f"base_model.model.{dp_name(layer)}"
    A = sd[f"{pre}.lora_A.weight"].float().to(device)
    B = sd[f"{pre}.lora_B.weight"].float().to(device)
    return (cfg["alpha"] / cfg["r"]) * (B @ A)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()
    probe = json.load(open(results_dir() / "probe_set.json"))
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()
    W0_sd = {dp_name(l) + ".weight": base.get_submodule(dp_name(l)).weight.detach().clone() for l in DIAG_LAYERS}
    z, is_math = cache_inputs(base, tok, probe, device)
    torch.save({"z": z, "is_math": is_math}, out_dir() / "probe_activations.pt")
    del base; torch.cuda.empty_cache()

    bases_file = out_dir() / "oplora_downproj_bases.pt"
    saved = torch.load(bases_file) if bases_file.exists() else {}
    Uk = {}
    for l in DIAG_LAYERS:
        if dp_name(l) in saved:
            Uk[l] = saved[dp_name(l)][0].to(device)
        else:
            U, _, _ = torch.linalg.svd(W0_sd[dp_name(l) + ".weight"].to(device), full_matrices=False)
            Uk[l] = U[:, :PROTECTION_RANK].contiguous()
    out = {"k": PROTECTION_RANK, "layers": DIAG_LAYERS, "bases_source": str(bases_file) if saved else "recomputed_svd",
           "conditions": {}}
    for cond in CONDITIONS:
        ck_dir = out_dir() / "checkpoints" / CONDITIONS[cond][0] / f"update_{CONDITIONS[cond][1]:04d}"
        if not ck_dir.exists():
            print("skip (no checkpoint):", cond); continue
        out["conditions"][cond] = {}
        for l in DIAG_LAYERS:
            W0 = W0_sd[dp_name(l) + ".weight"].to(device)
            dW = delta_w(cond, l, W0_sd, device)
            dW_f2 = dW.pow(2).sum().item()
            proj = Uk[l] @ (Uk[l].T @ dW)
            rho = proj.pow(2).sum().item() / dW_f2 if dW_f2 > 0 else float("nan")
            eta = (dW_f2 ** 0.5) / W0.norm().item()
            Z = z[l].to(device)                                   # (64, din)
            R = (Z @ dW.T).norm(dim=1) / ((Z @ W0.T).norm(dim=1) + 1e-8)
            m = is_math.to(device)
            out["conditions"][cond][str(l)] = {
                "rho_k": rho, "eta": eta, "dW_fro": dW_f2 ** 0.5, "W0_fro": W0.norm().item(),
                "R_math_mean": R[m].mean().item(), "R_nonmath_mean": R[~m].mean().item(),
                "R_math_median": R[m].median().item(), "R_nonmath_median": R[~m].median().item(),
                "R_per_prompt": R.tolist(),
            }
            print(cond, l, {k: round(v, 5) for k, v in out["conditions"][cond][str(l)].items() if isinstance(v, float)})
    out["is_math"] = is_math.tolist()
    write_json(out, results_dir() / "diagnostics.json")


if __name__ == "__main__":
    main()
