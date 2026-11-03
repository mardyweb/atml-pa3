"""
Step 6 - tables and figures for the report (reads only results/task1/*).

    python task1/summarize.py

Produces results/task1/tables/*.md|csv and results/task1/figures/*.png:
  benchmarks.md        7 conditions x 4 benchmarks, primary score + change from baseline (pp)
  resources.md         trainable params, train time, peak memory, checkpoint sizes, overheads
  diagnostics.md       rho_k, eta, R(z) math / non-math per layer and variant
  sft_loss.png         combined SFT-loss curves (5 runs) + learning rate
  l2sp_penalty.png     Omega, lambda*Omega, SFT and total loss for the L2-SP run
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import EVAL_CONDITIONS, METHODS, results_dir

R = results_dir()
TAB, FIG = R / "tables", R / "figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

PRIMARY = [("GSM8K (flex EM)", "gsm8k_flexible_em"), ("HellaSwag (acc_norm)", "hellaswag_acc_norm"),
           ("BoolQ (acc)", "boolq_acc"), ("ARC-Easy (acc_norm)", "arc_easy_acc_norm")]
SECONDARY = [("GSM8K strict EM", "gsm8k_strict_em"), ("GSM8K truncated", "gsm8k_truncated_rate"),
             ("GSM8K unextractable", "gsm8k_unextractable_rate"), ("HellaSwag acc", "hellaswag_acc"),
             ("ARC-Easy acc", "arc_easy_acc")]


def load_bench():
    rows = {}
    for c in EVAL_CONDITIONS:
        p = R / c / "benchmarks.json"
        if p.exists():
            rows[c] = json.load(open(p))
    return rows


def benchmark_table(b):
    if "baseline" not in b:
        print("no baseline benchmarks yet"); return
    base = b["baseline"]
    recs = []
    for c, d in b.items():
        row = {"condition": c}
        for name, key in PRIMARY:
            v = d.get(key)
            row[name] = f"{100*v:.1f}" if v is not None else "-"
            row[name + " Δpp"] = f"{100*(v-base[key]):+.1f}" if (v is not None and c != "baseline") else ("0.0" if c == "baseline" else "-")
        for name, key in SECONDARY:
            v = d.get(key)
            row[name] = f"{100*v:.1f}" if v is not None else "-"
        recs.append(row)
    df = pd.DataFrame(recs)
    df.to_csv(TAB / "benchmarks.csv", index=False)
    (TAB / "benchmarks.md").write_text(df.to_markdown(index=False))
    print(df.to_markdown(index=False))


def resources_table():
    recs = []
    for m in METHODS:
        p = R / m / "summary.json"
        if not p.exists():
            continue
        s = json.load(open(p))
        ck = s["checkpoints"]
        final = [c for c in ck if c["dir"].endswith("1250")]
        ex = R / m / "export.json"
        ex = json.load(open(ex)) if ex.exists() else {}
        oh = s["overhead"]
        recs.append({
            "method": m, "trainable params": f"{s['trainable_params']:,}",
            "train time (min)": f"{s['train_seconds']/60:.1f}",
            "peak mem alloc (GB)": f"{s['peak_mem_allocated_bytes']/1e9:.2f}",
            "peak mem reserved (GB)": f"{s['peak_mem_reserved_bytes']/1e9:.2f}",
            "recovery ckpt (GB)": f"{final[0]['bytes']/1e9:.2f}" if final else "-",
            "adapter/model export (MB)": f"{ex.get('adapter_bytes', ex.get('exported_bytes', 0))/1e6:.1f}",
            "skipped updates": len(s["skipped_updates"]),
            "displacement ||θ-θ0|| (mid / final)": " / ".join(f"{v:.3f}" for v in s["displacement_l2"].values()) or "-",
            "overhead": (f"ref {oh['reference_bytes']/1e9:.2f} GB, penalty {oh['penalty_compute_seconds']:.0f}s" if m == "l2sp"
                         else f"SVD {oh['oplora_preprocess']['preprocess_seconds']:.1f}s, bases {oh['oplora_preprocess']['basis_bytes']/1e6:.0f} MB, export {sum((oh.get('oplora_export_seconds') or {}).values()):.1f}s" if m == "oplora16"
                         else "-"),
        })
    df = pd.DataFrame(recs)
    df.to_csv(TAB / "resources.csv", index=False)
    (TAB / "resources.md").write_text(df.to_markdown(index=False))
    print(df.to_markdown(index=False))


def loss_plots():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for m in METHODS:
        p = R / m / "train_log.jsonl"
        if not p.exists():
            continue
        df = pd.read_json(p, lines=True)
        ax[0].plot(df["update"], df["sft_loss"].rolling(10, min_periods=1).mean(), label=m, lw=1)
        ax[1].plot(df["update"], df["lr"], label=m, lw=1)
    ax[0].set(xlabel="optimizer update", ylabel="SFT loss (10-update moving avg)", title="SFT loss")
    ax[1].set(xlabel="optimizer update", ylabel="learning rate", title="LR schedule", yscale="log")
    ax[0].legend(); ax[1].legend(); fig.tight_layout(); fig.savefig(FIG / "sft_loss.png", dpi=160)

    p = R / "l2sp" / "train_log.jsonl"
    if p.exists():
        df = pd.read_json(p, lines=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].plot(df["update"], df["omega"], label="Ω = ½‖θ−θ₀‖²"); ax[0].set(xlabel="update", yscale="log", title="L2-SP proximity term"); ax[0].legend()
        ax[1].plot(df["update"], df["sft_loss"], label="SFT loss", lw=1)
        ax[1].plot(df["update"], df["lam_omega"], label="λΩ", lw=1)
        ax[1].plot(df["update"], df["total_loss"], label="total", lw=1, alpha=0.7)
        ax[1].set(xlabel="update", title="L2-SP objective components", yscale="log"); ax[1].legend()
        fig.tight_layout(); fig.savefig(FIG / "l2sp_penalty.png", dpi=160)


def diagnostics_table():
    p = R / "diagnostics.json"
    if not p.exists():
        return
    d = json.load(open(p))
    recs = []
    for cond, layers in d["conditions"].items():
        for l, v in layers.items():
            recs.append({"condition": cond, "layer": int(l), "rho_k": round(v["rho_k"], 4), "eta": round(v["eta"], 4),
                         "R math (mean)": round(v["R_math_mean"], 4), "R non-math (mean)": round(v["R_nonmath_mean"], 4),
                         "R math/non-math": round(v["R_math_mean"] / max(v["R_nonmath_mean"], 1e-12), 3)})
    df = pd.DataFrame(recs)
    df.to_csv(TAB / "diagnostics.csv", index=False)
    (TAB / "diagnostics.md").write_text(df.to_markdown(index=False))
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    benchmark_table(load_bench())
    resources_table()
    loss_plots()
    diagnostics_table()
