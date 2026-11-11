"""
Step 4 - live comparison of parallel sampling vs sequential refinement (50 MATH questions, 3,072-token allowance).

    python task3/live.py            (resumable: finished questions are skipped)

Model: Qwen/Qwen2.5-3B-Instruct at the pinned revision, FP16, no quantization, batch size 1,
temperature 0.7 / top-p 0.8 / top-k 20 / repetition penalty 1.0, native chat template, system "You are a helpful
assistant.", user = problem + "\n\n" + instruction. Every call is independently seeded.
  single      : one response, 3,072 new tokens
  parallel    : three responses, 1,024 each; self-consistency picks the answer (ties -> earliest)
  sequential  : parallel response #1 (shared) appended as an assistant turn + review instruction; one
                critique-and-revision response capped at 2,048; the revised answer is final
Saves every prompt/response/seed/token counts/truncation flag/wall time to results/task3/live/calls.jsonl and a
summary (accuracy, actual input/output tokens, runtime, truncation + extraction rates, revision transitions) to
results/task3/live/summary.json. Grading uses math-verify against the gold answers of the supplied export.
"""
import json
import time

import pandas as pd
import torch

import common3 as C
from answers import MathScorer
from engine import sc_select


def build_messages(problem):
    return [{"role": "system", "content": C.SYSTEM_MSG}, {"role": "user", "content": f"{problem}\n\n{C.INSTRUCTION}"}]


@torch.no_grad()
def generate(model, tok, messages, cap, seed, device):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(device)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=cap, pad_token_id=tok.pad_token_id or tok.eos_token_id, **C.GEN_KW)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    gen = out[0, enc["input_ids"].shape[1]:]
    n_out = int(gen.numel())
    eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    truncated = n_out >= cap and int(gen[-1]) not in eos_ids
    resp = tok.decode(gen, skip_special_tokens=True)
    return {"prompt": text, "response": resp, "input_tokens": int(enc["input_ids"].shape[1]), "output_tokens": n_out,
            "truncated": bool(truncated), "seconds": dt, "seed": seed, "cap": cap}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    out = C.results_dir() / "live"; out.mkdir(parents=True, exist_ok=True)
    calls_path = out / "calls.jsonl"
    done = {}
    if calls_path.exists():
        for l in open(calls_path):
            r = json.loads(l); done.setdefault(r["sample_id"], []).append(r)
    splits = C.read_json(C.results_dir() / "splits.json")["math"]
    live_ids = splits["live"]
    text = pd.read_parquet(C.out_dir() / "cache" / "math_text.parquet").drop_duplicates("sample_id").set_index("sample_id")
    scorer = MathScorer()

    t_load = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(C.LIVE_MODEL, revision=C.LIVE_REVISION)
    model = AutoModelForCausalLM.from_pretrained(C.LIVE_MODEL, revision=C.LIVE_REVISION, dtype=torch.float16).to(device).eval()
    load_seconds = time.perf_counter() - t_load
    torch.cuda.reset_peak_memory_stats()

    f = open(calls_path, "a")
    for qi, q in enumerate(live_ids):
        if len(done.get(q, [])) >= 5:
            continue
        problem, gold = text.loc[q, "question_text"], str(text.loc[q, "gold_answer"])
        base = C.SEED * 1000 + qi * 10
        msgs = build_messages(problem)
        recs = []
        recs.append({"strategy": "single", "k": 0, **generate(model, tok, msgs, C.CAP_SINGLE, base + 0, device)})
        par = [generate(model, tok, msgs, C.CAP_PARALLEL, base + 1 + k, device) for k in range(3)]
        recs += [{"strategy": "parallel", "k": k, **p} for k, p in enumerate(par)]
        rev_msgs = msgs + [{"role": "assistant", "content": par[0]["response"]}, {"role": "user", "content": C.REVIEW_INSTRUCTION}]
        recs.append({"strategy": "revision", "k": 0, **generate(model, tok, rev_msgs, C.CAP_REVISION, base + 4, device)})
        for r in recs:
            r.update({"sample_id": q, "gold": gold, "extracted": C.extract_boxed(r["response"])})
            r["correct"] = scorer.is_correct(q, gold, r["extracted"])
            f.write(json.dumps(r) + "\n")
        f.flush()
        done[q] = recs
        print(f"[{qi + 1}/{len(live_ids)}] {q}: single={recs[0]['correct']} par={[r['correct'] for r in recs[1:4]]} rev={recs[4]['correct']} "
              f"tokens={[r['output_tokens'] for r in recs]}", flush=True)
    f.close()
    summarise(done, live_ids, scorer, out, {"gpu": torch.cuda.get_device_name(0), "load_seconds": load_seconds,
                                             "peak_gpu_mem_bytes": torch.cuda.max_memory_allocated()})


def summarise(done, live_ids, scorer, out, env):
    def groups_sc(exts):
        gids, _ = scorer.group(None, exts)
        return sc_select(gids)[0]
    summ = {"model": C.LIVE_MODEL, "revision": C.LIVE_REVISION, "dtype": "float16", "n_questions": len(live_ids),
            "scorer": scorer.name, "strategies": {}, **env}


    per_q = []
    for q in live_ids:
        r = {x["strategy"] + str(x["k"]): x for x in done[q]}
        single, p = r["single0"], [r["parallel0"], r["parallel1"], r["parallel2"]]
        rev = r["revision0"]
        pos = groups_sc([x["extracted"] for x in p])
        per_q.append({
            "sample_id": q,
            "single_correct": single["correct"], "single_out": single["output_tokens"], "single_in": single["input_tokens"], "single_sec": single["seconds"],
            "single_trunc": single["truncated"], "single_extracted": single["extracted"] is not None,
            "parallel_correct": p[pos]["correct"], "parallel_out": sum(x["output_tokens"] for x in p), "parallel_in": sum(x["input_tokens"] for x in p),
            "parallel_sec": sum(x["seconds"] for x in p), "parallel_trunc": sum(x["truncated"] for x in p) / 3, "parallel_extracted": sum(x["extracted"] is not None for x in p) / 3,
            "parallel_selected_k": pos, "parallel_any_correct": any(x["correct"] for x in p),
            "initial_correct": p[0]["correct"], "revised_correct": rev["correct"],
            "sequential_out": p[0]["output_tokens"] + rev["output_tokens"], "sequential_in": p[0]["input_tokens"] + rev["input_tokens"],
            "sequential_sec": p[0]["seconds"] + rev["seconds"], "sequential_trunc": (p[0]["truncated"] + rev["truncated"]) / 2,
            "sequential_extracted": (int(p[0]["extracted"] is not None) + int(rev["extracted"] is not None)) / 2,
            "revision_changed_answer": not scorer.equivalent(str(p[0]["extracted"]), str(rev["extracted"])) if (p[0]["extracted"] and rev["extracted"]) else (p[0]["extracted"] is not None) != (rev["extracted"] is not None),
        })
    pq = pd.DataFrame(per_q); pq.to_csv(out / "per_question.csv", index=False)
    for s, acc_col in [("single", "single_correct"), ("parallel", "parallel_correct"), ("sequential", "revised_correct")]:
        summ["strategies"][s] = {
            "accuracy": float(pq[acc_col].mean()), "accuracy_ci95": C.paired_bootstrap(pq[acc_col].to_numpy().astype(float))["ci95"],
            "mean_output_tokens": float(pq[f"{s}_out"].mean()), "mean_input_tokens": float(pq[f"{s}_in"].mean()),
            "total_output_tokens": int(pq[f"{s}_out"].sum()), "mean_seconds": float(pq[f"{s}_sec"].mean()), "total_seconds": float(pq[f"{s}_sec"].sum()),
            "truncation_rate": float(pq[f"{s}_trunc"].mean()), "extraction_rate": float(pq[f"{s}_extracted"].mean()),
            "allowance_used": float(pq[f"{s}_out"].mean() / C.TOTAL_ALLOWANCE)}
    summ["parallel_oracle_any_correct"] = float(pq["parallel_any_correct"].mean())
    summ["revision_transitions"] = {"wrong_to_correct": int((~pq["initial_correct"] & pq["revised_correct"]).sum()),
                                    "correct_to_wrong": int((pq["initial_correct"] & ~pq["revised_correct"]).sum()),
                                    "correct_to_correct": int((pq["initial_correct"] & pq["revised_correct"]).sum()),
                                    "wrong_to_wrong": int((~pq["initial_correct"] & ~pq["revised_correct"]).sum()),
                                    "answer_changed": int(pq["revision_changed_answer"].sum()), "initial_accuracy": float(pq["initial_correct"].mean())}
    summ["paired_diffs"] = {"parallel_minus_single": C.paired_bootstrap(pq["parallel_correct"].astype(float), pq["single_correct"].astype(float)),
                            "sequential_minus_single": C.paired_bootstrap(pq["revised_correct"].astype(float), pq["single_correct"].astype(float)),
                            "sequential_minus_parallel": C.paired_bootstrap(pq["revised_correct"].astype(float), pq["parallel_correct"].astype(float))}
    C.write_json(summ, out / "summary.json")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
