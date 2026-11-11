"""
Task 2 - figures, tables and qualitative examples for the report (manual Step 6 / Required Evidence).

    python task2/analyze_vlm.py

results/task2/figures/learning_curves.png   training loss, dev caption loss, answer NLL, swap penalties (stage boundaries marked)
results/task2/tables/final_<suite>.md|csv   four conditions x accuracy (overall / by type), pair diagnostics, NLL, time, CIs
results/task2/tables/advantages.md          correct-image advantages with bootstrap CIs
results/task2/tables/resources.md           trainable params / runtime / peak memory by stage, feature caching, generation time
results/task2/qualitative/examples.md       aligned complementary-pair examples (stated selection rule) + copied images
"""
import json
import shutil
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import common2 as C

R = C.results_dir()
FIG, TAB, QUAL = R / "figures", R / "tables", R / "qualitative"
for d in (FIG, TAB, QUAL):
    d.mkdir(parents=True, exist_ok=True)


def learning_curves():
    log = pd.read_json(R / "train_log.jsonl", lines=True)
    dev = pd.read_json(R / "dev_metrics.jsonl", lines=True)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    a = ax[0, 0]; a.plot(log["update"], log["loss"], lw=.4, alpha=.4); a.plot(log["update"], log["loss"].rolling(25, min_periods=1).mean(), lw=1.2)
    a.set(title="training loss (example-weighted CE)", xlabel="update", yscale="log")
    a = ax[0, 1]; a.plot(dev["update"], dev["caption_loss"], marker="o"); a.set(title="dev caption reference loss", xlabel="update")
    a = ax[1, 0]
    for suite, ls in [("ground_dev", "-"), ("vqa_dev", "--")]:
        for cond in ("correct", "complementary", "noimage"):
            a.plot(dev["update"], dev[f"{suite}_nll_{cond}"], ls, marker=".", label=f"{suite} {cond}")
    a.set(title="dev reference-answer NLL", xlabel="update"); a.legend(fontsize=7)
    a = ax[1, 1]
    for suite in ("ground_dev", "vqa_dev"):
        a.plot(dev["update"], dev[f"{suite}_swap_penalty"], marker="o", label=suite)
    a.axhline(0, color="k", lw=.5); a.set(title="image-swap penalty  ΔNLL = NLL(partner) − NLL(correct)", xlabel="update"); a.legend()
    for a in ax.flat:
        for e in C.STAGE_ENDS[:-1]:
            a.axvline(e, color="gray", ls=":", lw=.8)
    fig.tight_layout(); fig.savefig(FIG / "learning_curves.png", dpi=160)


def final_tables():
    m = C.read_json(R / "final" / "metrics.json")
    for suite in m:
        recs = []
        for cond in C.CONDITIONS:
            d = m[suite][cond]; b = m[suite]["bootstrap"]["conditions"][cond]
            row = {"condition": cond, "accuracy": f"{100*d['accuracy']:.1f}",
                   "acc CI95": f"[{100*b['accuracy_ci95'][0]:.1f}, {100*b['accuracy_ci95'][1]:.1f}]"}
            for t, v in d["accuracy_by_type"].items():
                row[f"acc {t} (n={d['n_by_type'][t]})"] = f"{100*v:.1f}"
            row.update({"pair acc": f"{100*d['pair_accuracy']:.1f}", "identical": f"{100*d['identical_answer_rate']:.1f}",
                        "empty": f"{100*d['empty_answer_rate']:.1f}", "ref NLL": f"{d['mean_reference_nll']:.3f}",
                        "NLL CI95": f"[{b['nll_ci95'][0]:.3f}, {b['nll_ci95'][1]:.3f}]",
                        "gen s/answer": f"{d['generation_seconds_per_answer']:.3f}"})
            recs.append(row)
        df = pd.DataFrame(recs); df.to_csv(TAB / f"final_{suite}.csv", index=False); (TAB / f"final_{suite}.md").write_text(df.to_markdown(index=False))
        print(f"\n{suite}\n" + df.to_markdown(index=False))
    recs = []
    for suite in m:
        for cond, d in m[suite]["bootstrap"]["advantage_of_correct_over"].items():
            recs.append({"suite": suite, "correct vs": cond, "Δ accuracy (pp)": f"{100*d['accuracy_diff']:+.1f}",
                         "CI95": f"[{100*d['accuracy_diff_ci95'][0]:+.1f}, {100*d['accuracy_diff_ci95'][1]:+.1f}]",
                         "Δ NLL (other − correct)": f"{d['nll_diff']:+.3f}",
                         "NLL CI95": f"[{d['nll_diff_ci95'][0]:+.3f}, {d['nll_diff_ci95'][1]:+.3f}]"})
    df = pd.DataFrame(recs); df.to_csv(TAB / "advantages.csv", index=False); (TAB / "advantages.md").write_text(df.to_markdown(index=False))
    print("\n" + df.to_markdown(index=False))


def resources_table():
    r = C.read_json(R / "resources_train.json"); fc = C.read_json(C.repo_root() / "results" / "task2" / "feature_cache.json")
    m = C.read_json(R / "final" / "metrics.json")
    recs = [{"stage": k, "trainable params": f"{v['trainable_params']:,}", "frozen LM params": f"{v['frozen_lm_params']:,}",
             "runtime (min)": f"{v['seconds']/60:.1f}", "peak GPU mem (GB)": f"{v['peak_mem_bytes']/1e9:.2f}",
             "saved weights (MB)": f"{v.get('weights_bytes', 0)/1e6:.1f}"} for k, v in r["stages"].items()]
    recs.append({"stage": "feature caching (SigLIP)", "trainable params": "0", "frozen LM params": "-",
                 "runtime (min)": f"{fc['total_seconds']/60:.1f}", "peak GPU mem (GB)": f"{fc['peak_gpu_mem_bytes']/1e9:.2f}",
                 "saved weights (MB)": f"{sum(s['bytes'] for s in fc['splits'].values())/1e6:.0f} (features)"})
    recs.append({"stage": "dev evaluations (total)", "trainable params": "-", "frozen LM params": "-",
                 "runtime (min)": f"{r['dev_eval_seconds']/60:.1f}", "peak GPU mem (GB)": "-", "saved weights (MB)": "-"})
    gt = {s: {c: m[s][c]["generation_seconds_per_answer"] for c in C.CONDITIONS} for s in m}
    recs.append({"stage": "generation s/answer (correct | noimage)", "trainable params": "-", "frozen LM params": "-",
                 "runtime (min)": " / ".join(f"{s}: {gt[s]['correct']:.3f} | {gt[s]['noimage']:.3f}" for s in gt),
                 "peak GPU mem (GB)": "-", "saved weights (MB)": "-"})
    df = pd.DataFrame(recs); df.to_csv(TAB / "resources.csv", index=False); (TAB / "resources.md").write_text(df.to_markdown(index=False))
    print("\n" + df.to_markdown(index=False))


def qualitative(n_sensitive=3, n_insensitive=2):
    """
    Selection rule (stated in the report): within each suite, in pair-ID order, take the first `n_sensitive` pairs whose
    correct-image predictions are both right AND change under the complementary swap (image-sensitive), and the first
    `n_insensitive` pairs whose two correct-image predictions are identical (image-insensitive).
    """
    gens = C.read_json(R / "final" / "generations.json")
    idx = C.read_json(C.data_dir() / "image_index.json"); fname = {e["image_id"]: (e["split"], e["file_name"]) for e in idx}
    by = defaultdict(dict)
    for g in gens:
        by[(g["suite"], g["pair_id"], g["k"])][g["condition"]] = g
    lines = ["# Qualitative complementary-pair examples\n", qualitative.__doc__.strip() + "\n"]
    (QUAL / "images").mkdir(exist_ok=True)
    for suite in ("ground_test", "vqa_test"):
        pair_ids = sorted({k[1] for k in by if k[0] == suite})
        sens, insens = [], []
        for pid in pair_ids:
            a, b = by[(suite, pid, 0)], by[(suite, pid, 1)]
            both_right = a["correct"]["matches_canonical"] and b["correct"]["matches_canonical"]
            changes = (a["correct"]["prediction_norm"] != a["complementary"]["prediction_norm"]) and (b["correct"]["prediction_norm"] != b["complementary"]["prediction_norm"])
            identical = a["correct"]["prediction_norm"] == b["correct"]["prediction_norm"]
            if both_right and changes and len(sens) < n_sensitive:
                sens.append(pid)
            elif identical and len(insens) < n_insensitive:
                insens.append(pid)
        lines.append(f"\n## {suite}\n")
        for label, ids in [("Image-sensitive and correct", sens), ("Image-insensitive (identical predictions across the pair)", insens)]:
            lines.append(f"\n### {label}\n")
            for pid in ids:
                a = by[(suite, pid, 0)]["correct"]
                lines.append(f"\n**{pid}** — question: *{a['question']}*  (type: {a['type']})\n")
                lines.append("| image | reference(s) | correct | complementary | mismatched | no image |\n|---|---|---|---|---|---|")
                for k in (0, 1):
                    g = by[(suite, pid, k)]
                    im = g["correct"]["image_used"]; split, fn = fname[im]
                    src = C.out_dir() / "images" / split / fn
                    if src.exists():
                        shutil.copy(src, QUAL / "images" / fn)
                    refs = g["correct"]["reference"] if suite == "ground_test" else f"{g['correct']['canonical']} ({', '.join(g['correct']['answers'])})"
                    cells = [f"`{g[c]['prediction']}`" + (" ✓" if g[c]["matches_canonical"] else "") for c in C.CONDITIONS]
                    lines.append(f"| {fn} | {refs} | " + " | ".join(cells) + " |")
    (QUAL / "examples.md").write_text("\n".join(lines))
    print("qualitative examples written to", QUAL / "examples.md")


if __name__ == "__main__":
    learning_curves()
    final_tables()
    resources_table()
    qualitative()
