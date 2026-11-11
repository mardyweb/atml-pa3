"""
Step 1 (part a) - load + join the supplied rollout tables, verify token-list alignment, build answer groups and
correctness labels, compute per-rollout scores, save compact caches and the question splits.

    python task3/load_data.py [--datasets math mathvista]

Outputs
  outputs/task3/cache/<ds>.npz      per-rollout arrays (T, L, K, V, correct, group) + ragged per-token kl / v (MathVista)
  results/task3/splits.json          dev / held-out / live question IDs (sorted IDs permuted with default_rng(6304))
  results/task3/scorer.json          scorer versions, extraction statistics
  results/task3/<ds>_questions.csv   per-question: gold, #distinct groups, #NO_ANSWER, #correct rollouts
"""
import argparse
import json
import urllib.request

import numpy as np
import pandas as pd

import common3 as C
from answers import MathScorer, MathVistaScorer, load_mathvista_meta

KEYS = ["dataset", "sample_id", "rollout_id"]


def _arr(x):
    return np.asarray(x, dtype=np.float32)


def load_dataset(ds):
    folder = C.data_root() / C.DATASETS[ds]
    gen = pd.read_parquet(folder / "generations.parquet")
    prob = pd.read_parquet(folder / "probability_metrics.parquet")
    gen["sample_id"] = gen["sample_id"].astype(str); prob["sample_id"] = prob["sample_id"].astype(str)
    df = gen.merge(prob, on=KEYS, how="inner", suffixes=("", "_p"))
    assert len(df) == len(gen) == len(prob), "join lost rows"
    if ds == "mathvista":
        att = pd.read_parquet(folder / "attention_metrics.parquet")
        att["sample_id"] = att["sample_id"].astype(str)
        df = df.merge(att, on=KEYS, how="inner", suffixes=("", "_a"))
        assert len(df) == len(gen), "attention join lost rows"
    df = df.sort_values(["sample_id", "rollout_id"]).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["math", "mathvista"])
    args = ap.parse_args()
    cache = C.out_dir() / "cache"; cache.mkdir(parents=True, exist_ok=True)
    R = C.results_dir()
    scorer_info = {}
    splits_all = {}
    for ds in args.datasets:
        print(f"=== {ds}", flush=True)
        df = load_dataset(ds)
        print("columns:", list(df.columns)); print("first row (truncated):", {k: (str(v)[:80]) for k, v in df.iloc[0].to_dict().items()}, flush=True)
        qids = sorted(df["sample_id"].unique().tolist())
        assert len(qids) == C.N_QUESTIONS and len(df) == C.N_QUESTIONS * C.N_ROLLOUTS, (len(qids), len(df))
        assert (df.groupby("sample_id")["rollout_id"].apply(lambda s: sorted(s.tolist()) == list(range(C.N_ROLLOUTS)))).all()

        # ---- alignment checks (per-token lists vs retained length)
        n_tok = df["n_generated_text_tokens"].to_numpy().astype(np.int64)
        len_ids = df["generated_token_ids"].apply(len).to_numpy()
        len_logp = df["logp"].apply(len).to_numpy(); len_kl = df["kl_u_p"].apply(len).to_numpy()
        align = {"ids_vs_n": int((len_ids != n_tok).sum()), "logp_vs_n": int((len_logp != n_tok).sum()), "kl_vs_n": int((len_kl != n_tok).sum())}
        if ds == "mathvista":
            for col in ("img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"):
                align[f"{col}_vs_n"] = int((df[col].apply(len).to_numpy() != n_tok).sum())
        print("alignment mismatches:", align, flush=True)
        assert all(v == 0 for v in align.values()), align

        # ---- scores
        T = n_tok
        L = df["logp"].apply(lambda x: float(np.mean(_arr(x))) if len(x) else np.nan).to_numpy()
        K = df["kl_u_p"].apply(lambda x: float(np.mean(_arr(x))) if len(x) else np.nan).to_numpy()
        out = {"T": T, "L": L, "K": K}
        if ds == "mathvista":
            v_list = []
            for a, b, c in zip(df["img_attn_total"], df["prompt_text_attn_total"], df["generated_text_attn_total"]):
                a, b, c = _arr(a), _arr(b), _arr(c)
                v_list.append(a / np.maximum(a + b + c, 1e-12))
            out["V"] = np.array([float(v.mean()) if len(v) else np.nan for v in v_list])
            offsets = np.concatenate([[0], np.cumsum(T)]).astype(np.int64)
            out["tok_offsets"] = offsets
            out["tok_kl"] = np.concatenate([_arr(x) for x in df["kl_u_p"]]) if T.sum() else np.zeros(0, np.float32)
            out["tok_v"] = np.concatenate(v_list) if T.sum() else np.zeros(0, np.float32)
            assert len(out["tok_kl"]) == len(out["tok_v"]) == offsets[-1]

        # ---- answer groups + correctness (gold used only for correctness)
        if ds == "math":
            scorer = MathScorer()
            gold_of = {q: str(g) for q, g in zip(df["sample_id"], df["gold_answer"])}
        else:
            meta_path = C.out_dir() / "testmini.json"
            if not meta_path.exists():
                urllib.request.urlretrieve(C.MATHVISTA_META_URL, meta_path)
            scorer = MathVistaScorer(load_mathvista_meta(meta_path))
            gold_of = {q: scorer.gold(q) for q in qids}
        group = np.full(len(df), -1, np.int64); correct = np.zeros(len(df), bool)
        reps_by_q, q_rows = {}, []
        exts = [None if (e is None or (isinstance(e, float) and np.isnan(e)) or str(e).strip() == "") else str(e) for e in df["extracted_answer"].tolist()]
        for qi, q in enumerate(qids):
            sl = slice(qi * C.N_ROLLOUTS, (qi + 1) * C.N_ROLLOUTS)
            gids, reps = scorer.group(q, exts[sl])
            group[sl] = gids; reps_by_q[q] = reps
            corr = [scorer.is_correct(q, gold_of[q], e) for e in exts[sl]]
            correct[sl] = corr
            q_rows.append({"sample_id": q, "gold": gold_of[q], "n_groups": len(reps), "n_no_answer": int(sum(g == -1 for g in gids)),
                           "n_correct": int(sum(corr)), "n_capped": int(df["hit_max_new_tokens"].iloc[sl].sum()), "groups": json.dumps(reps)[:2000]})
            if qi % 100 == 0:
                print(f"  grouped {qi}/{len(qids)}", flush=True)
        out.update({"group": group, "correct": correct, "capped": df["hit_max_new_tokens"].to_numpy().astype(bool)})
        np.savez_compressed(cache / f"{ds}.npz", qids=np.array(qids), **out)
        pd.DataFrame(q_rows).to_csv(R / f"{ds}_questions.csv", index=False)
        # text needed later (live subtask + qualitative): questions, gold, extracted answers
        df[["sample_id", "rollout_id", "question_text", "gold_answer", "extracted_answer", "hit_max_new_tokens", "n_generated_text_tokens"]] \
            .to_parquet(cache / f"{ds}_text.parquet", index=False)
        df[["sample_id", "rollout_id", "generated_text"]].to_parquet(cache / f"{ds}_generations.parquet", index=False)

        splits_all[ds] = C.split_questions(qids)
        scorer_info[ds] = {"scorer": scorer.name, "version": scorer.version, "n_rollouts": len(df),
                           "null_extractions": int(sum(e is None for e in exts)), "capped": int(out["capped"].sum()),
                           "rollout_accuracy": float(correct.mean()), "questions_with_any_correct": int(sum(r["n_correct"] > 0 for r in q_rows)),
                           "mean_groups_per_question": float(np.mean([r["n_groups"] for r in q_rows])), "alignment_mismatches": align}
        print(json.dumps(scorer_info[ds], indent=1), flush=True)
    C.write_json(splits_all, R / "splits.json")
    C.write_json(scorer_info, R / "scorer.json")
    np.save(C.out_dir() / "cache" / "orders.npy", C.rollout_orders(C.N_QUESTIONS))
    np.save(R / "rollout_orders.npy", C.rollout_orders(C.N_QUESTIONS))


if __name__ == "__main__":
    main()
