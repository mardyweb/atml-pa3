"""
Step 6 - figures, tables and qualitative examples for the report.

    python task3/analyze3.py

results/task3/figures/  accuracy_coverage.png, accuracy_vs_cost.png, groups_failures.png, trajectories.png
results/task3/tables/   methods_<ds>.md, failures_<ds>.md, signals.md, fusion_mathvista.md, live.md, adaptive_<ds>.md
results/task3/qualitative/examples.md  (selection disagreement, revision, premature stopping; stated selection rules)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common3 as C

R = C.results_dir(); FIG, TAB, QUAL = R / "figures", R / "tables", R / "qualitative"
for d in (FIG, TAB, QUAL):
    d.mkdir(parents=True, exist_ok=True)
LABEL = {"first": "FirstAns", "sc": "Self-consistency", "best_L": "Best mean log-prob", "best_K": "Best self-certainty",
         "rwc_K": "Self-certainty + RWC", "shortest": "Shortest rollout", "best_V": "Best vision-only", "rwc_V": "Vision-only + RWC",
         "best_Z": "Best VisConf", "rwc_Z": "VisConf + RWC", "best_Ztok": "Best tokenwise VisConf", "rwc_Ztok": "Tokenwise VisConf + RWC"}


def curves(ds):
    p = R / "offline" / f"{ds}_curves.csv"
    return pd.read_csv(p) if p.exists() else None


def fig_accuracy_coverage(dss):
    fig, axes = plt.subplots(1, len(dss), figsize=(6 * len(dss), 4.2), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        c = curves(ds); c = c[c["split"] == "heldout"]
        cov = c[c["method"] == "sc"].sort_values("N")
        ax.plot(cov["N"], 100 * cov["coverage"], "k--", label="Oracle coverage")
        for m in [m for m in LABEL if m in set(c["method"]) and m not in ("shortest", "best_V", "rwc_V")]:
            s = c[c["method"] == m].sort_values("N"); ax.plot(s["N"], 100 * s["accuracy"], marker="o", ms=3, label=LABEL[m])
        ax.set(xscale="log", xticks=C.BUDGETS, xticklabels=C.BUDGETS, xlabel="candidates N", ylabel="accuracy / coverage (%)", title=f"{ds} (held-out)")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "accuracy_coverage.png", dpi=160)


def fig_accuracy_vs_cost(dss):
    fig, axes = plt.subplots(1, len(dss), figsize=(6 * len(dss), 4.2), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        c = curves(ds); c = c[c["split"] == "heldout"]
        for m in ("first", "sc", "rwc_K", "best_K") + (("rwc_Z",) if (c["method"] == "rwc_Z").any() else ()):
            s = c[c["method"] == m].sort_values("N"); ax.plot(s["cost"], 100 * s["accuracy"], marker="o", ms=3, label=LABEL[m])
        p = R / "adaptive" / f"{ds}_heldout.json"
        if p.exists():
            h = C.read_json(p)
            for fam, mk in (("agreement", "s"), ("quality", "^")):
                ax.scatter(h["adaptive"][fam]["mean_cost"], 100 * h["adaptive"][fam]["accuracy"], marker=mk, s=70, zorder=5, label=f"adaptive ({fam})")
        ax.set(xscale="log", xlabel="generated tokens per question (mean)", ylabel="accuracy (%)", title=f"{ds} (held-out)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "accuracy_vs_cost.png", dpi=160)


def fig_groups_failures(dss):
    fig, axes = plt.subplots(2, len(dss), figsize=(6 * len(dss), 7), squeeze=False)
    for j, ds in enumerate(dss):
        c = curves(ds); c = c[c["split"] == "heldout"]
        s = c[c["method"] == "sc"].sort_values("N")
        axes[0, j].plot(s["N"], s["n_groups"], marker="o", label="distinct answer groups"); axes[0, j].plot(s["N"], s["n_no_answer"], marker="s", label="NO_ANSWER candidates")
        axes[0, j].set(xscale="log", xticks=C.BUDGETS, xticklabels=C.BUDGETS, xlabel="N", title=f"{ds}: answer diversity"); axes[0, j].legend(fontsize=8)
        ax = axes[1, j]; w = 0.35
        for k, m in enumerate(("sc", "rwc_K")):
            f = c[c["method"] == m].sort_values("N"); x = np.arange(len(f)) + (k - 0.5) * w
            ax.bar(x, 100 * f["cov_fail"], w, label=f"{LABEL[m]}: no correct candidate", color=["#4c72b0", "#dd8452"][k], alpha=.6)
            ax.bar(x, 100 * f["sel_fail"], w, bottom=100 * f["cov_fail"], label=f"{LABEL[m]}: correct available, missed", color=["#4c72b0", "#dd8452"][k], hatch="//")
        ax.set(xticks=np.arange(len(C.BUDGETS)), xticklabels=C.BUDGETS, xlabel="N", ylabel="% of questions wrong", title=f"{ds}: failure decomposition"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "groups_failures.png", dpi=160)


def tables_methods(dss):
    for ds in dss:
        c = curves(ds); c = c[c["split"] == "heldout"]
        boot = C.read_json(R / "offline" / f"{ds}_bootstrap.json")
        piv = c.pivot(index="method", columns="N", values="accuracy")
        rows = []
        for m in [m for m in LABEL if m in piv.index]:
            row = {"method": LABEL[m]}
            for N in C.BUDGETS:
                ci = boot[str(N)]["accuracy"][m]["ci95"]; row[f"N={N}"] = f"{100*piv.loc[m, N]:.1f} [{100*ci[0]:.1f},{100*ci[1]:.1f}]"
            d = boot["32"]["diff_vs_sc"].get(m); row["Δ vs SC @32 (pp)"] = f"{100*d['mean']:+.1f} [{100*d['ci95'][0]:+.1f},{100*d['ci95'][1]:+.1f}]" if d else "—"
            row["cost @32"] = f"{c[(c.method == m) & (c.N == 32)]['cost'].iloc[0]:.0f}"
            rows.append(row)
        cov = c[c["method"] == "sc"].sort_values("N")
        rows.append({"method": "Oracle coverage", **{f"N={N}": f"{100*v:.1f}" for N, v in zip(cov["N"], cov["coverage"])}, "Δ vs SC @32 (pp)": "—", "cost @32": "—"})
        df = pd.DataFrame(rows); (TAB / f"methods_{ds}.md").write_text(df.to_markdown(index=False)); df.to_csv(TAB / f"methods_{ds}.csv", index=False)
        f = pd.read_csv(R / "offline" / f"{ds}_failures.csv"); f = f[f["method"].isin(["first", "sc", "rwc_K", "rwc_Z"])]
        f["method"] = f["method"].map(LABEL); f[["accuracy", "coverage", "cov_fail", "sel_fail"]] = (100 * f[["accuracy", "coverage", "cov_fail", "sel_fail"]]).round(1)
        (TAB / f"failures_{ds}.md").write_text(f.to_markdown(index=False))
        print(f"\n{ds}\n" + df.to_markdown(index=False))


def tables_signals(dss):
    rows = []
    for ds in dss:
        p = R / "signals" / f"{ds}_quartiles.json"
        if not p.exists(): continue
        s = C.read_json(p)
        for split in ("heldout", "mixed_heldout"):
            for k, v in s[split].items():
                if isinstance(v, dict):
                    rows.append({"dataset": ds, "questions": split, "signal": k, "top−bottom quartile correctness (pp)": f"{100*v['mean']:+.1f} [{100*v['ci95'][0]:+.1f},{100*v['ci95'][1]:+.1f}]"})
    if rows:
        (TAB / "signals.md").write_text(pd.DataFrame(rows).to_markdown(index=False))
    p = R / "signals" / "mathvista_confidence_vision.json"
    if p.exists():
        cv = C.read_json(p); c = curves("mathvista"); c = c[c["split"] == "heldout"]
        fus = c[c["method"].isin(["best_Z", "rwc_Z", "best_Ztok", "rwc_Ztok"])].pivot(index="method", columns="N", values="accuracy")
        fus.index = [LABEL[m] for m in fus.index]
        txt = ("Per-rollout Pearson corr(c_t, v_t): mean %.3f, median %.3f, %.0f%% positive; by correctness %s\n"
               "Cov_t(c,v) = Z_tok − Z: mean %.4f, %.0f%% positive; by correctness %s\n"
               "Top-1 agreement between Z and Z_tok rankings (held-out questions): %.1f%%\n\n" %
               (cv["per_rollout_corr"]["mean"], cv["per_rollout_corr"]["median"], 100 * cv["per_rollout_corr"]["frac_positive"], cv["corr_by_correctness"],
                cv["cov_c_v"]["mean"], 100 * cv["cov_c_v"]["frac_positive"], cv["cov_c_v"]["by_correctness"], 100 * cv["rank_agreement_Z_vs_Ztok"]))
        (TAB / "fusion_mathvista.md").write_text(txt + (100 * fus).round(1).to_markdown())
    p = R / "signals" / "mathvista_trajectories.json"
    if p.exists():
        tr = C.read_json(p); names = [k for k in tr if k != "rule"]
        fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.2 * len(names)), sharex=False)
        for ax, n in zip(np.atleast_1d(axes), names):
            t = tr[n]; ax.plot(t["c"], label="calibrated confidence c_t", lw=.9); ax.plot(t["v"], label="visual engagement v_t", lw=.9)
            ax.set_title(f"{n}: q={t['sample_id']} r={t['rollout_id']} correct={t['correct']} Z={t['Z']:.3f} Z_tok={t['Ztok']:.3f} corr={t['corr']:.2f}", fontsize=8); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / "trajectories.png", dpi=160)


def table_live():
    p = R / "live" / "summary.json"
    if not p.exists(): return
    s = C.read_json(p); rows = []
    for k, v in s["strategies"].items():
        rows.append({"strategy": k, "accuracy": f"{100*v['accuracy']:.1f} [{100*v['accuracy_ci95'][0]:.0f},{100*v['accuracy_ci95'][1]:.0f}]",
                     "output tok/question": f"{v['mean_output_tokens']:.0f}", "input tok/question": f"{v['mean_input_tokens']:.0f}",
                     "allowance used": f"{100*v['allowance_used']:.0f}%", "runtime s/question": f"{v['mean_seconds']:.1f}",
                     "truncation": f"{100*v['truncation_rate']:.0f}%", "extraction": f"{100*v['extraction_rate']:.0f}%"})
    df = pd.DataFrame(rows); t = s["revision_transitions"]
    txt = df.to_markdown(index=False) + (f"\n\nRevision transitions (relative to the shared initial answer, acc {100*t['initial_accuracy']:.0f}%): "
          f"wrong→correct {t['wrong_to_correct']}, correct→wrong {t['correct_to_wrong']}, correct→correct {t['correct_to_correct']}, wrong→wrong {t['wrong_to_wrong']}, "
          f"answer changed {t['answer_changed']}/{s['n_questions']}. Parallel oracle (any of 3 correct): {100*s['parallel_oracle_any_correct']:.0f}%. "
          f"GPU {s['gpu']}, peak memory {s['peak_gpu_mem_bytes']/1e9:.1f} GB.")
    (TAB / "live.md").write_text(txt); print("\n" + txt)


def tables_adaptive(dss):
    for ds in dss:
        p = R / "adaptive" / f"{ds}_heldout.json"
        if not p.exists(): continue
        h, sel = C.read_json(p), C.read_json(R / "adaptive" / f"{ds}_selected.json")
        rows = []
        for n, v in h["fixed_sc_heldout"].items():
            rows.append({"policy": f"fixed-{n} self-consistency", "setting": "—", "accuracy": f"{100*v['accuracy']:.1f}", "mean tokens": f"{v['mean_cost']:.0f}", "mean rollouts": n, "stops @4/8/16/32": "—"})
        for fam, v in h["adaptive"].items():
            st = sel["selected"][fam]
            rows.append({"policy": f"adaptive ({fam})", "setting": f"τ={st['tau']}" + (f", γ={st['gamma']}" if st["gamma"] is not None else ""),
                         "accuracy": f"{100*v['accuracy']:.1f} [{100*v['accuracy_ci95'][0]:.1f},{100*v['accuracy_ci95'][1]:.1f}]",
                         "mean tokens": f"{v['mean_cost']:.0f} ({100*v['cost_ratio_vs_fixed32']:.0f}% of fixed-32)", "mean rollouts": f"{v['mean_rollouts']:.1f}",
                         "stops @4/8/16/32": "/".join(f"{100*v['stop_distribution'][str(n)]:.0f}%" for n in C.STOP_CHECKS + [32])})
        df = pd.DataFrame(rows)
        extra = "\n\n" + "\n".join(f"{fam}: early stops {100*v['early_stops']['fraction']:.0f}% of runs; wrong early stops {100*v['early_stops']['wrong_fraction_of_early']:.0f}% of early stops, "
                                   f"of which fixed-32 would have been correct in {v['early_stops']['wrong_but_fixed32_correct']} runs; pool fraction correct by stop count {v['pool_frac_correct_by_stop']}"
                                   for fam, v in h["adaptive"].items())
        (TAB / f"adaptive_{ds}.md").write_text(df.to_markdown(index=False) + extra); print(f"\n{ds} adaptive\n" + df.to_markdown(index=False))


def qualitative(dss):
    lines = ["# Qualitative examples (Task 3)\n"]
    # 1. selection disagreement: held-out, order 0, N=32, SC vs self-certainty+RWC disagree and exactly one is correct; first 3 by sorted ID
    for ds in dss:
        p = C.out_dir() / f"offline_{ds}_full.parquet"
        if not p.exists(): continue
        full = pd.read_parquet(p); splits = C.read_json(R / "splits.json")[ds]
        text = pd.read_parquet(C.out_dir() / "cache" / f"{ds}_text.parquet")
        sub = full[(full["order"] == 0) & (full["N"] == 32) & (full["sample_id"].isin(splits["heldout"]))]
        sc = sub[sub.method == "sc"].set_index("sample_id"); rw = sub[sub.method == "rwc_K"].set_index("sample_id")
        cand = [q for q in sorted(sc.index) if sc.loc[q, "group"] != rw.loc[q, "group"] and (sc.loc[q, "correct"] != rw.loc[q, "correct"])][:3]
        lines.append(f"\n## {ds}: selection disagreement (rule: held-out, order 0, N=32, SC and self-certainty+RWC pick different answers and exactly one is correct; first 3 by sorted question ID)\n")
        qtab = pd.read_csv(R / f"{ds}_questions.csv", dtype={"sample_id": str}).set_index("sample_id")
        for q in cand:
            t = text[text.sample_id == q]
            lines.append(f"\n**{q}** — gold `{t['gold_answer'].iloc[0]}` — groups {qtab.loc[q, 'groups'][:300]}\n")
            lines.append(f"- question: {str(t['question_text'].iloc[0])[:400]}")
            lines.append(f"- SC picked rollout {int(sc.loc[q, 'rollout_id'])} → `{t[t.rollout_id == sc.loc[q, 'rollout_id']]['extracted_answer'].iloc[0]}` correct={bool(sc.loc[q, 'correct'])}")
            lines.append(f"- self-certainty+RWC picked rollout {int(rw.loc[q, 'rollout_id'])} → `{t[t.rollout_id == rw.loc[q, 'rollout_id']]['extracted_answer'].iloc[0]}` correct={bool(rw.loc[q, 'correct'])}")
    # 2. revision transitions from the live run
    p = R / "live" / "calls.jsonl"
    if p.exists():
        calls = [json.loads(l) for l in open(p)]; byq = {}
        for c in calls: byq.setdefault(c["sample_id"], {})[c["strategy"] + str(c["k"])] = c
        lines.append("\n## Live refinement (rule: first wrong→correct and first correct→wrong revision in live-question order)\n")
        for want, label in [((False, True), "wrong→correct"), ((True, False), "correct→wrong")]:
            for q in C.read_json(R / "splits.json")["math"]["live"]:
                r = byq.get(q, {})
                if "parallel0" in r and "revision0" in r and (r["parallel0"]["correct"], r["revision0"]["correct"]) == want:
                    lines.append(f"\n**{label}: {q}** gold `{r['parallel0']['gold']}`; initial `{r['parallel0']['extracted']}` ({r['parallel0']['output_tokens']} tok, truncated={r['parallel0']['truncated']}) → revised `{r['revision0']['extracted']}` ({r['revision0']['output_tokens']} tok)\n")
                    lines.append("```\n" + r["revision0"]["response"][-700:] + "\n```")
                    break
    # 3. premature stopping (quality controller, held-out, order 0): first 2 early stops that are wrong while fixed-32 SC is right
    for ds in dss:
        p = R / "adaptive" / f"{ds}_per_question.csv.gz"
        if not p.exists(): continue
        pq = pd.read_csv(p, dtype={"sample_id": str}); pre = pd.read_parquet(C.out_dir() / f"adaptive_{ds}_precompute.parquet")
        q32 = pre[(pre.n == 32) & (pre.order == 0)].set_index("sample_id")
        sub = pq[(pq.controller == "quality") & (pq.order == 0) & (pq.stop_n < 32) & (~pq.sc_correct)]
        picks = [q for q in sorted(sub.sample_id) if q32.loc[q, "sc_correct"]][:2]
        lines.append(f"\n## {ds}: premature stopping (rule: quality-aware controller, held-out, order 0, stopped before 32 with a wrong answer while fixed-32 SC was right; first 2 by sorted ID)\n")
        for q in picks:
            r = sub[sub.sample_id == q].iloc[0]; at = pre[(pre.sample_id == q) & (pre.order == 0)].set_index("n")
            lines.append(f"- **{q}**: stopped at n={int(r.stop_n)} (support {at.loc[int(r.stop_n), 'support']:.2f}, RWC margin {at.loc[int(r.stop_n), 'rwc_margin']:.2f}); "
                         f"fixed-32 support {at.loc[32, 'support']:.2f}; fraction of correct rollouts in the full pool {at.loc[32, 'pool_frac_correct']:.2f}")
    (QUAL / "examples.md").write_text("\n".join(lines)); print("qualitative ->", QUAL / "examples.md")


def main():
    dss = [ds for ds in ("math", "mathvista") if (R / "offline" / f"{ds}_curves.csv").exists()]
    fig_accuracy_coverage(dss); fig_accuracy_vs_cost(dss); fig_groups_failures(dss)
    tables_methods(dss); tables_signals(dss); table_live(); tables_adaptive(dss); qualitative(dss)


if __name__ == "__main__":
    main()
