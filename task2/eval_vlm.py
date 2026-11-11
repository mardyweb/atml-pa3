"""
Task 2 - final evaluation (manual: "Final evaluation" + Step 5).

    python task2/eval_vlm.py [--weights outputs/task2/stage3]

Both final-test suites (64 grounding pairs / 128 questions, 96 VQAv2 pairs / 192 questions), greedy decoding,
thinking disabled, 16-new-token cap, batch size 8, under four image conditions:
  correct | complementary (partner image) | mismatched (fixed mapping from splits.json) | noimage (visual span omitted).
Scores: grounding = normalised exact match (presence / counting); VQAv2 = local soft consensus accuracy over the ten
answers (yes/no, number, other). Also pair accuracy, identical-answer rate, empty-answer rate, mean reference-answer
NLL (teacher-forced, termination excluded) and generation time per answer. Correct-image advantages over the other
conditions with 95% CIs from 4,000 bootstrap resamples of complete pairs (same resampled pair IDs across conditions).
Writes results/task2/final/{generations.json, metrics.json}.
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch

import common2 as C
from vlm import VLM, Connector, FeatureStore, build_lm


def load_model(weights_dir, device):
    cfg = C.read_json(C.results_dir() / "config.json")
    lm, tok = build_lm(device)
    store = FeatureStore(C.out_dir() / "features")
    connector = Connector(store.dim, 1024, lm.config.hidden_size).to(device)
    connector.load_state_dict(torch.load(weights_dir / "connector.pt", map_location=device))
    vlm = VLM(lm, tok, connector, cfg["calibration"]["scale"], device)
    vlm.load_adapter_state(torch.load(weights_dir / "adapter.pt", map_location="cpu"))
    vlm.lm.eval(); vlm.connector.eval()
    return vlm, store


def suite_questions(suite, pairs, mismatch):
    """One record per question with the image to use under every condition."""
    by_id = {p["pair_id"]: p for p in pairs}
    qs = []
    for p in pairs:
        partner = by_id[mismatch[p["pair_id"]]]
        for k, q in enumerate(p["q"]):
            other = p["q"][1 - k]
            rec = {"suite": suite, "pair_id": p["pair_id"], "k": k, "question": q["question"],
                   "type": p.get("type") or p.get("answer_type"),
                   "images": {"correct": q["image_id"], "complementary": other["image_id"],
                              "mismatched": partner["q"][k]["image_id"], "noimage": None}}
            if suite == "ground_test":
                rec.update({"reference": q["answer"], "canonical": C.normalize_answer(q["answer"]), "answers_norm": None})
            else:
                rec.update({"qid": q["qid"], "reference": q["target"], "canonical": q["consensus"],
                            "answers": q["answers"], "answers_norm": [C.normalize_answer(a) for a in q["answers"]]})
            rec["item"] = {"text": f"{q['question']}\n{C.ANSWER_SUFFIX}", "target": rec["reference"], "kind": "short"}
            qs.append(rec)
    return qs


def score(rec, pred):
    pn = C.normalize_answer(pred)
    if rec["suite"] == "ground_test":
        s = float(pn == rec["canonical"])
    else:
        s = C.vqa_soft_accuracy(pn, rec["answers_norm"])
    return pn, s, pn == rec["canonical"]


@torch.no_grad()
def run_condition(vlm, store, qs, cond):
    preds, nlls, gen_time = [None] * len(qs), [None] * len(qs), 0.0
    B = C.GEN_BATCH
    for s in range(0, len(qs), B):
        chunk = qs[s:s + B]
        ims = [q["images"][cond] for q in chunk]
        feats = None if cond == "noimage" else store.batch(ims, vlm.device)
        enc = [vlm.encode({**q["item"], "target": None}, with_image=cond != "noimage") for q in chunk]
        b = vlm.collate(enc, feats, left_pad=True)
        t = time.perf_counter()
        out = vlm.generate(b)
        if vlm.device == "cuda":
            torch.cuda.synchronize()
        gen_time += time.perf_counter() - t
        enc_t = [vlm.encode(q["item"], with_image=cond != "noimage") for q in chunk]
        bt = vlm.collate(enc_t, None if feats is None else feats)
        nl = vlm.example_losses(bt, exclude_eos=True).tolist()
        for i, q in enumerate(chunk):
            preds[s + i], nlls[s + i] = out[i], nl[i]
    return preds, nlls, gen_time / len(qs)


def pair_metrics(qs, preds, scores, canon_ok, nlls):
    pairs = defaultdict(dict)
    for q, p, sc, ok, nl in zip(qs, preds, scores, canon_ok, nlls):
        pairs[q["pair_id"]][q["k"]] = (C.normalize_answer(p), sc, ok, nl)
    pa = [float(v[0][2] and v[1][2]) for v in pairs.values()]
    ident = [float(v[0][0] == v[1][0]) for v in pairs.values()]
    return {"pair_accuracy": float(np.mean(pa)), "identical_answer_rate": float(np.mean(ident))}


def bootstrap(qs, per_cond, n_boot=C.N_BOOT):
    """Resample complete pairs (same resampled IDs across conditions); CIs for accuracy / NLL and correct-minus-other differences."""
    pair_ids = sorted({q["pair_id"] for q in qs})
    idx_of = {p: i for i, p in enumerate(pair_ids)}
    n = len(pair_ids)
    acc = {c: np.zeros((n, 2)) for c in per_cond}; nll = {c: np.zeros((n, 2)) for c in per_cond}
    for c, (preds, scores, ok, nls) in per_cond.items():
        for q, sc, nl in zip(qs, scores, nls):
            acc[c][idx_of[q["pair_id"]], q["k"]] = sc; nll[c][idx_of[q["pair_id"]], q["k"]] = nl
    rng = np.random.default_rng(C.SEED)
    samples = rng.integers(0, n, size=(n_boot, n))
    out = {"n_pairs": n, "n_boot": n_boot, "conditions": {}, "advantage_of_correct_over": {}}
    acc_b = {c: acc[c][samples].mean(axis=(1, 2)) for c in per_cond}
    nll_b = {c: nll[c][samples].mean(axis=(1, 2)) for c in per_cond}
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    for c in per_cond:
        out["conditions"][c] = {"accuracy_ci95": ci(acc_b[c]), "nll_ci95": ci(nll_b[c])}
    for c in per_cond:
        if c == "correct":
            continue
        da, dn = acc_b["correct"] - acc_b[c], nll_b[c] - nll_b["correct"]
        out["advantage_of_correct_over"][c] = {
            "accuracy_diff": float(acc["correct"].mean() - acc[c].mean()), "accuracy_diff_ci95": ci(da),
            "nll_diff": float(nll[c].mean() - nll["correct"].mean()), "nll_diff_ci95": ci(dn)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out, R, D = C.out_dir(), C.results_dir(), C.data_dir()
    weights = C.repo_root() / args.weights if args.weights else out / ("stage3" + C.tag())
    vlm, store = load_model(weights, device)
    splits = C.read_json(D / "splits.json")
    gens, metrics = [], {}
    for suite, fname in [("ground_test", "ground_test"), ("vqa_test", "vqa_test")]:
        qs = suite_questions(suite, C.read_json(D / f"{fname}.json"), splits["mismatch"][suite])
        per_cond, m = {}, {}
        for cond in C.CONDITIONS:
            preds, nlls, gt = run_condition(vlm, store, qs, cond)
            scored = [score(q, p) for q, p in zip(qs, preds)]
            pn, sc, ok = zip(*scored)
            per_cond[cond] = (preds, list(sc), list(ok), nlls)
            by_type = defaultdict(list)
            for q, s_ in zip(qs, sc):
                by_type[q["type"]].append(s_)
            m[cond] = {"accuracy": float(np.mean(sc)), "accuracy_by_type": {t: float(np.mean(v)) for t, v in by_type.items()},
                       "n_by_type": {t: len(v) for t, v in by_type.items()},
                       **pair_metrics(qs, preds, sc, ok, nlls),
                       "empty_answer_rate": float(np.mean([p == "" for p in pn])),
                       "mean_reference_nll": float(np.mean(nlls)), "generation_seconds_per_answer": gt, "n_questions": len(qs)}
            for q, p, n_, s_, o_, nl in zip(qs, preds, pn, sc, ok, nlls):
                gens.append({k: v for k, v in q.items() if k != "item"} | {"condition": cond, "image_used": q["images"][cond],
                            "prediction": p, "prediction_norm": n_, "score": s_, "matches_canonical": bool(o_), "reference_nll": nl})
            print(suite, cond, json.dumps({k: v for k, v in m[cond].items() if not isinstance(v, dict)}), flush=True)
        m["bootstrap"] = bootstrap(qs, per_cond)
        metrics[suite] = m
    C.write_json(gens, R / "final" / "generations.json")
    C.write_json(metrics, R / "final" / "metrics.json")
    print(json.dumps({s: {c: round(metrics[s][c]["accuracy"], 4) for c in C.CONDITIONS} for s in metrics}, indent=1))


if __name__ == "__main__":
    main()
