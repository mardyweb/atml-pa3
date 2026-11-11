"""
Steps 1-2 - offline budget sweep (manual: "Offline evaluation").

    python task3/offline.py [--datasets math mathvista]

For every question, each of the ten saved rollout orders and every budget N in {1,2,4,8,16,32}: run all selection
methods on the nested prefix, record the selected rollout, its correctness, the generated-token cost (FirstAns pays
only for the first candidate; every other method pays for all N), oracle coverage, #distinct answer groups.
Outcomes are averaged over orders within a question, then over questions. 95% CIs from 1,000 paired question-level
bootstrap resamples (held-out split). Failure decomposition: wrong because no correct candidate existed vs wrong
although a correct candidate was available.
Outputs results/task3/offline/<ds>_{curves.csv, per_question.csv.gz, bootstrap.json, failures.csv}
and outputs/task3/offline_<ds>_full.parquet (every question x order x N x method row).
"""
import argparse

import numpy as np
import pandas as pd

import common3 as C
from engine import Data, METHODS_COMMON, METHODS_VISION, method_scores, select_all


def sweep(data: Data):
    methods = METHODS_COMMON + (METHODS_VISION if data.has_vision else [])
    rows = []
    for qi, q in enumerate(data.qids):
        for o in range(C.N_ORDERS):
            order = data.orders[qi, o]
            for N in C.BUDGETS:
                cand = order[:N]; gl = data.gidx(qi, cand)
                groups = data.group[gl]; corr = data.correct[gl]; T = data.T[gl]
                sel = select_all(data, gl, groups, method_scores(data, gl))
                base = {"sample_id": q, "order": o, "N": N, "coverage": bool(corr.any()),
                        "n_groups": int(len(set(groups[groups >= 0].tolist()))), "n_no_answer": int((groups < 0).sum()),
                        "cost_all": int(T.sum())}
                for m, pos in sel.items():
                    rows.append({**base, "method": m, "pos": pos, "rollout_id": int(cand[pos]), "correct": bool(corr[pos]),
                                 "cost": int(T[pos]) if m == "first" else int(T.sum()), "group": int(groups[pos])})
        if qi % 100 == 0:
            print(f"  {data.ds}: {qi}/{len(data.qids)} questions", flush=True)
    return pd.DataFrame(rows), methods


def summarise(df, data: Data, methods, out):
    split_of = {q: "dev" for q in data.splits["dev"]}; split_of.update({q: "heldout" for q in data.splits["heldout"]})
    df["split"] = df["sample_id"].map(split_of)
    # average over orders within a question
    pq = df.groupby(["split", "sample_id", "N", "method"], as_index=False).agg(
        accuracy=("correct", "mean"), cost=("cost", "mean"), coverage=("coverage", "mean"),
        n_groups=("n_groups", "mean"), n_no_answer=("n_no_answer", "mean"),
        sel_fail=("correct", lambda s: 0.0), cov_fail=("correct", lambda s: 0.0))
    # failure decomposition per row then averaged
    df["cov_fail"] = (~df["correct"]) & (~df["coverage"])
    df["sel_fail"] = (~df["correct"]) & (df["coverage"])
    fail = df.groupby(["split", "sample_id", "N", "method"], as_index=False)[["cov_fail", "sel_fail"]].mean()
    pq = pq.drop(columns=["sel_fail", "cov_fail"]).merge(fail, on=["split", "sample_id", "N", "method"])
    pq.to_csv(out / f"{data.ds}_per_question.csv.gz", index=False, compression="gzip")

    curves = []
    for split in ("heldout", "dev", "all"):
        sub = pq if split == "all" else pq[pq["split"] == split]
        g = sub.groupby(["N", "method"], as_index=False).agg(accuracy=("accuracy", "mean"), cost=("cost", "mean"),
                                                             coverage=("coverage", "mean"), n_groups=("n_groups", "mean"),
                                                             n_no_answer=("n_no_answer", "mean"), cov_fail=("cov_fail", "mean"),
                                                             sel_fail=("sel_fail", "mean"), n_questions=("sample_id", "nunique"))
        g.insert(0, "split", split); curves.append(g)
    curves = pd.concat(curves); curves.to_csv(out / f"{data.ds}_curves.csv", index=False)

    # bootstrap on held-out, per N: CI of each method's accuracy and of differences vs SC and vs FirstAns
    held = pq[pq["split"] == "heldout"]
    boot = {}
    for N in C.BUDGETS:
        acc = {m: held[(held["N"] == N) & (held["method"] == m)].sort_values("sample_id")["accuracy"].to_numpy() for m in methods}
        cov = held[(held["N"] == N) & (held["method"] == "sc")].sort_values("sample_id")["coverage"].to_numpy()
        boot[str(N)] = {"coverage": C.paired_bootstrap(cov),
                        "accuracy": {m: C.paired_bootstrap(acc[m]) for m in methods},
                        "diff_vs_sc": {m: C.paired_bootstrap(acc[m], acc["sc"]) for m in methods if m != "sc"},
                        "diff_vs_first": {m: C.paired_bootstrap(acc[m], acc["first"]) for m in methods if m != "first"}}
    C.write_json(boot, out / f"{data.ds}_bootstrap.json")
    fails = curves[curves["split"] == "heldout"][["N", "method", "accuracy", "coverage", "cov_fail", "sel_fail"]]
    fails.to_csv(out / f"{data.ds}_failures.csv", index=False)
    print(curves[(curves["split"] == "heldout")].pivot(index="N", columns="method", values="accuracy").round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["math", "mathvista"])
    args = ap.parse_args()
    out = C.results_dir() / "offline"; out.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        data = Data(ds)
        df, methods = sweep(data)
        df.to_parquet(C.out_dir() / f"offline_{ds}_full.parquet", index=False)
        summarise(df, data, methods, out)


if __name__ == "__main__":
    main()
