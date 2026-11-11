"""
Step 3 - what the signals capture (N = 32, full pools; order-independent).

    python task3/signals.py

* Within-question association between each signal and correctness: mean correctness in the top vs bottom quartile
  (8 rollouts each) of the signal, difference averaged over questions (all / held-out / only "mixed" questions).
  Signals: length T, mean log-prob L, self-certainty K, and for MathVista V (visual engagement), C (calibrated
  confidence, calibrated on the full 32-candidate set) and Z = V*C. Shortest-rollout selection is in the offline sweep.
* MathVista: per-rollout Pearson correlation between token confidence c_t and visual engagement v_t, summarised across
  questions; Cov_t(c, v) = Z_tok - Z per rollout; tokenwise-vs-trajectory fusion agreement.
* A fixed set of illustrative trajectories (stated rule) with their per-token series.
Outputs results/task3/signals/*.json|csv
"""
import json

import numpy as np
import pandas as pd

import common3 as C
from engine import Data, calibrated_tokens, visconf

SIGNALS_COMMON = {"length": lambda d, gl: d.T[gl].astype(float), "mean_logp": lambda d, gl: d.L[gl], "self_certainty": lambda d, gl: d.K[gl]}


def quartile_analysis(data: Data, out):
    rows = []
    for qi, q in enumerate(data.qids):
        gl = data.gidx(qi, np.arange(C.N_ROLLOUTS)); corr = data.correct[gl].astype(float)
        sig = {k: f(data, gl) for k, f in SIGNALS_COMMON.items()}
        if data.has_vision:
            Cm, Z, Ztok = visconf(data, gl)
            sig.update({"visual_engagement": data.V[gl], "calibrated_conf": Cm, "visconf": Z, "visconf_tokenwise": Ztok})
        rec = {"sample_id": q, "n_correct": int(corr.sum())}
        for k, s in sig.items():
            s = np.where(np.isnan(s), -np.inf, s)
            order = np.argsort(-s, kind="stable"); top, bot = order[:8], order[-8:]
            rec[f"{k}_top_minus_bottom"] = float(corr[top].mean() - corr[bot].mean())
        rows.append(rec)
    df = pd.DataFrame(rows)
    held = set(data.splits["heldout"])
    summary = {}
    for name, sub in [("all", df), ("heldout", df[df["sample_id"].isin(held)]), ("mixed_heldout", df[df["sample_id"].isin(held) & (df["n_correct"] > 0) & (df["n_correct"] < 32)])]:
        summary[name] = {"n_questions": len(sub)}
        for col in [c for c in df.columns if c.endswith("_top_minus_bottom")]:
            summary[name][col.replace("_top_minus_bottom", "")] = C.paired_bootstrap(sub[col].to_numpy())
    df.to_csv(out / f"{data.ds}_quartiles_per_question.csv", index=False)
    C.write_json(summary, out / f"{data.ds}_quartiles.json")
    print(data.ds, "top-bottom quartile correctness gap (heldout):", {k: round(v["mean"], 4) for k, v in summary["heldout"].items() if isinstance(v, dict)})


def vision_analysis(data: Data, out):
    rows = []
    for qi, q in enumerate(data.qids):
        gl = data.gidx(qi, np.arange(C.N_ROLLOUTS))
        Cm, Z, Ztok = visconf(data, gl)
        for j, g in enumerate(gl):
            c, v = calibrated_tokens(data, gl, g)
            r = float(np.corrcoef(c, v)[0, 1]) if len(c) > 2 and c.std() > 0 and v.std() > 0 else np.nan
            rows.append({"sample_id": q, "rollout_id": int(j), "corr_c_v": r, "cov_c_v": float(Ztok[j] - Z[j]), "Z": float(Z[j]),
                         "Ztok": float(Ztok[j]), "V": float(data.V[g]), "C": float(Cm[j]), "correct": bool(data.correct[g]), "T": int(data.T[g])})
    df = pd.DataFrame(rows); df.to_csv(out / "mathvista_token_signals.csv.gz", index=False, compression="gzip")
    per_q = df.groupby("sample_id")["corr_c_v"].mean()
    held = set(data.splits["heldout"])
    summ = {"per_rollout_corr": {"mean": float(df["corr_c_v"].mean()), "median": float(df["corr_c_v"].median()),
                                 "frac_positive": float((df["corr_c_v"] > 0).mean()), "n_defined": int(df["corr_c_v"].notna().sum())},
            "per_question_mean_corr": C.paired_bootstrap(per_q.dropna().to_numpy()),
            "corr_by_correctness": {str(k): float(v) for k, v in df.groupby("correct")["corr_c_v"].mean().items()},
            "cov_c_v": {"mean": float(df["cov_c_v"].mean()), "frac_positive": float((df["cov_c_v"] > 0).mean()),
                        "by_correctness": {str(k): float(v) for k, v in df.groupby("correct")["cov_c_v"].mean().items()}},
            "hist_corr": np.histogram(df["corr_c_v"].dropna(), bins=20, range=(-1, 1))[0].tolist(),
            "rank_agreement_Z_vs_Ztok": float(np.mean([
                np.argmax(df[df.sample_id == q]["Z"].to_numpy()) == np.argmax(df[df.sample_id == q]["Ztok"].to_numpy()) for q in data.qids if q in held]))}
    C.write_json(summ, out / "mathvista_confidence_vision.json")
    print("confidence-vision:", json.dumps({k: v for k, v in summ.items() if k != "hist_corr"}, indent=1)[:800])
    # illustrative trajectories: dev-set rollouts with >= 20 tokens; largest +Cov, largest -Cov, highest-Z correct, highest-Z incorrect
    dev = df[df["sample_id"].isin(set(data.splits["dev"])) & (df["T"] >= 20)]
    picks = {"largest_positive_cov": dev.loc[dev["cov_c_v"].idxmax()], "largest_negative_cov": dev.loc[dev["cov_c_v"].idxmin()],
             "highest_Z_correct": dev[dev["correct"]].loc[dev[dev["correct"]]["Z"].idxmax()],
             "highest_Z_incorrect": dev[~dev["correct"]].loc[dev[~dev["correct"]]["Z"].idxmax()]}
    traj = {}
    for name, row in picks.items():
        qi = data.qidx[row["sample_id"]]; gl = data.gidx(qi, np.arange(C.N_ROLLOUTS)); g = gl[int(row["rollout_id"])]
        c, v = calibrated_tokens(data, gl, g)
        traj[name] = {"sample_id": row["sample_id"], "rollout_id": int(row["rollout_id"]), "correct": bool(row["correct"]),
                      "Z": float(row["Z"]), "Ztok": float(row["Ztok"]), "corr": float(row["corr_c_v"]), "c": c.tolist(), "v": v.tolist()}
    traj["rule"] = "dev-set MathVista rollouts with >= 20 tokens: largest positive / negative Cov_t(c,v); highest-Z correct; highest-Z incorrect"
    C.write_json(traj, out / "mathvista_trajectories.json")


def main():
    out = C.results_dir() / "signals"; out.mkdir(parents=True, exist_ok=True)
    for ds in ("math", "mathvista"):
        try:
            data = Data(ds)
        except FileNotFoundError:
            print("skip", ds); continue
        quartile_analysis(data, out)
        if data.has_vision:
            vision_analysis(data, out)


if __name__ == "__main__":
    main()
