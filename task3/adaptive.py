"""
Step 5 - adaptive sampling (offline simulation over the saved rollout orders).

    python task3/adaptive.py

Rollouts are revealed in the saved random orders; stopping is checked after n in {4, 8, 16}, otherwise at 32.
Both controllers return the current self-consistency answer.
  agreement-based : stop if a unique, non-NO_ANSWER leading answer is supported by >= tau of the n candidates
  quality-aware   : agreement condition AND RWC winner == SC winner AND RWC margin >= gamma
                    (scores: self-certainty K for MATH, VisConf Z for MathVista; scores, calibration and ranks are
                    recomputed on the revealed candidates only; margin = (W1 - W2) / sum W, 0 with one group)
Development set: tau in {.5,.625,.75,.875,1}, gamma in {0,.1,.25,.5}, plus the always-32 fallback. Per family select the
lowest mean token cost whose dev accuracy is >= fixed-32 self-consistency - 1 pp (averaged over the ten orders);
ties -> higher accuracy, then smaller tau, then smaller gamma. Frozen settings are then evaluated on the held-out set.
Outputs results/task3/adaptive/<ds>_{dev_grid.csv, selected.json, heldout.json, per_question.csv.gz, stop_analysis.json}
"""
import itertools
import json

import numpy as np
import pandas as pd

import common3 as C
from engine import Data, method_scores, rwc_margin, rwc_select, sc_select

CHECKS = C.STOP_CHECKS + [C.N_ROLLOUTS]


def precompute(data: Data):
    """Per (question, order, n): SC / RWC decisions on the first n revealed candidates."""
    score_key = "Z" if data.has_vision else "K"
    rows = []
    for qi, q in enumerate(data.qids):
        full = data.gidx(qi, np.arange(C.N_ROLLOUTS))
        pool_frac_correct = float(data.correct[full].mean())
        for o in range(C.N_ORDERS):
            order = data.orders[qi, o]
            for n in CHECKS:
                cand = order[:n]; gl = data.gidx(qi, cand); groups = data.group[gl]
                sc_pos, votes = sc_select(groups)
                lead = max(votes.values()); leaders = [g for g, v in votes.items() if v == lead]
                sc_group = int(groups[sc_pos])
                unique_lead = len(leaders) == 1 and leaders[0] != -1
                support = lead / n
                sc = method_scores(data, gl)[score_key]
                rwc_pos, W = rwc_select(groups, sc)
                rows.append({"sample_id": q, "order": o, "n": n, "sc_correct": bool(data.correct[gl[sc_pos]]), "sc_group": sc_group,
                             "unique_lead": unique_lead, "support": support, "rwc_group": int(groups[rwc_pos]),
                             "rwc_margin": rwc_margin(W), "cost": int(data.T[gl].sum()), "pool_frac_correct": pool_frac_correct})
        if qi % 100 == 0:
            print(f"  {data.ds}: {qi}/{len(data.qids)}", flush=True)
    return pd.DataFrame(rows)


def simulate(pre: pd.DataFrame, tau, gamma, quality: bool):
    """Returns per (question, order) stop n, correctness, cost for one controller setting. tau=None -> always 32."""
    piv = {n: pre[pre["n"] == n].set_index(["sample_id", "order"]).sort_index() for n in CHECKS}
    idx = piv[32].index
    stop_n = np.full(len(idx), 32); done = np.zeros(len(idx), bool)
    if tau is not None:
        for n in C.STOP_CHECKS:
            d = piv[n].loc[idx]
            cond = d["unique_lead"].to_numpy() & (d["support"].to_numpy() >= tau - 1e-12)
            if quality:
                cond &= (d["rwc_group"].to_numpy() == d["sc_group"].to_numpy()) & (d["rwc_margin"].to_numpy() >= gamma - 1e-12)
            new = cond & ~done
            stop_n[new] = n; done |= new
    res = pd.DataFrame({"sample_id": [i[0] for i in idx], "order": [i[1] for i in idx], "stop_n": stop_n})
    for col in ("sc_correct", "cost"):
        res[col] = [piv[n].loc[(q, o), col] for q, o, n in zip(res["sample_id"], res["order"], stop_n)]
    return res


def metrics(res, qset):
    r = res[res["sample_id"].isin(qset)]
    pq = r.groupby("sample_id").agg(acc=("sc_correct", "mean"), cost=("cost", "mean"), n=("stop_n", "mean"))
    dist = r["stop_n"].value_counts(normalize=True).reindex(CHECKS, fill_value=0.0)
    return {"accuracy": float(pq["acc"].mean()), "mean_cost": float(pq["cost"].mean()), "mean_rollouts": float(pq["n"].mean()),
            "stop_distribution": {str(n): float(dist[n]) for n in CHECKS}, "n_questions": len(pq)}, pq


def main():
    out = C.results_dir() / "adaptive"; out.mkdir(parents=True, exist_ok=True)
    for ds in ("math", "mathvista"):
        try:
            data = Data(ds)
        except FileNotFoundError:
            print("skip", ds); continue
        pre = precompute(data)
        pre.to_parquet(C.out_dir() / f"adaptive_{ds}_precompute.parquet", index=False)
        dev, held = set(data.splits["dev"]), set(data.splits["heldout"])

        # fixed-N self-consistency baselines (same orders)
        fixed = {}
        for n in CHECKS:
            r = pre[pre["n"] == n].rename(columns={"n": "stop_n"})
            fixed[n] = {"dev": metrics(r, dev)[0], "heldout": metrics(r, held)[0]}
        base_dev = fixed[32]["dev"]["accuracy"]

        grid, sims = [], {}
        settings = [("agreement", None, None)] + [("agreement", t, None) for t in C.TAUS] + \
                   [("quality", None, None)] + [("quality", t, g) for t, g in itertools.product(C.TAUS, C.GAMMAS)]
        for fam, tau, gamma in settings:
            res = simulate(pre, tau, gamma, fam == "quality"); sims[(fam, tau, gamma)] = res
            m, _ = metrics(res, dev)
            grid.append({"family": fam, "tau": tau, "gamma": gamma, **{k: v for k, v in m.items() if k != "stop_distribution"},
                         **{f"stop_{n}": m["stop_distribution"][str(n)] for n in CHECKS}, "eligible": m["accuracy"] >= base_dev - C.TOLERANCE_PP / 100})
        grid = pd.DataFrame(grid); grid.to_csv(out / f"{ds}_dev_grid.csv", index=False)

        selected, held_res, per_q = {}, {}, []
        for fam in ("agreement", "quality"):
            g = grid[(grid["family"] == fam) & grid["eligible"]].copy()
            g["tau_s"] = g["tau"].fillna(99); g["gamma_s"] = g["gamma"].fillna(99)
            g = g.sort_values(["mean_cost", "accuracy", "tau_s", "gamma_s"], ascending=[True, False, True, True])
            best = g.iloc[0]
            tau = None if pd.isna(best["tau"]) else float(best["tau"]); gamma = None if pd.isna(best["gamma"]) else float(best["gamma"])
            selected[fam] = {"tau": tau, "gamma": gamma, "dev": {k: float(best[k]) for k in ("accuracy", "mean_cost", "mean_rollouts")}}
            res = sims[(fam, tau, gamma)]
            m, pq = metrics(res, held)
            # paired comparison with fixed-32 SC on held-out
            r32 = pre[pre["n"] == 32].rename(columns={"n": "stop_n"}); _, pq32 = metrics(r32, held)
            m["accuracy_ci95"] = C.paired_bootstrap(pq["acc"].to_numpy())["ci95"]
            m["diff_vs_fixed32"] = C.paired_bootstrap(pq["acc"].to_numpy(), pq32.loc[pq.index, "acc"].to_numpy())
            m["cost_ratio_vs_fixed32"] = m["mean_cost"] / fixed[32]["heldout"]["mean_cost"]
            # incorrect early stops
            h = res[res["sample_id"].isin(held)].merge(r32[["sample_id", "order", "sc_correct", "pool_frac_correct"]].rename(columns={"sc_correct": "sc32_correct"}), on=["sample_id", "order"])
            early = h[h["stop_n"] < 32]
            m["early_stops"] = {"fraction": float(len(early) / len(h)), "wrong_fraction_of_early": float((~early["sc_correct"]).mean()) if len(early) else 0.0,
                                "wrong_but_fixed32_correct": int(((~early["sc_correct"]) & early["sc32_correct"]).sum()),
                                "wrong_and_fixed32_wrong": int(((~early["sc_correct"]) & ~early["sc32_correct"]).sum()),
                                "correct_early_stops": int(early["sc_correct"].sum())}
            m["pool_frac_correct_by_stop"] = {str(n): float(h[h["stop_n"] == n]["pool_frac_correct"].mean()) if (h["stop_n"] == n).any() else None for n in CHECKS}
            held_res[fam] = m
            res2 = res.copy(); res2["controller"] = fam; per_q.append(res2)
        C.write_json({"selected": selected, "fixed_sc": {str(k): v for k, v in fixed.items()}, "dev_fixed32_accuracy": base_dev}, out / f"{ds}_selected.json")
        C.write_json({"adaptive": held_res, "fixed_sc_heldout": {str(k): v["heldout"] for k, v in fixed.items()}}, out / f"{ds}_heldout.json")
        pd.concat(per_q).to_csv(out / f"{ds}_per_question.csv.gz", index=False, compression="gzip")
        print(ds, "selected:", json.dumps(selected), "\nheld-out:", json.dumps({f: {k: held_res[f][k] for k in ("accuracy", "mean_cost", "mean_rollouts")} for f in held_res}),
              "\nfixed32 held-out:", fixed[32]["heldout"]["accuracy"], fixed[32]["heldout"]["mean_cost"])


if __name__ == "__main__":
    main()
