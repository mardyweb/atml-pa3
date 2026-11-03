"""
Step 0 - Training data (manual: "Dataset, Models, and Experimental Setup / Training data").

Select the first 40,000 ELIGIBLE examples from meta-math/MetaMathQA in source order.
An example is excluded if
  * query or response is empty,
  * its chat-formatted length (prompt + response + <|im_end|>) exceeds 2,048 tokens,
  * its `query` or `original_question` matches a GSM8K *test* question after text normalisation.
Retained examples are neither truncated nor shuffled.

Outputs
  outputs/task1/data/train_tokenized.pt   list of {"input_ids", "labels", "src_idx"}  (large, not committed)
  results/task1/data_selection.json       source indices kept + exclusion statistics      (committed)
"""
import argparse
from collections import Counter

import torch
from datasets import load_dataset
from tqdm import tqdm

from common import (MAX_SEQ_LEN, N_TRAIN_EXAMPLES, encode_sft_example, load_tokenizer,
                    normalize_text, out_dir, results_dir, write_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=N_TRAIN_EXAMPLES)
    args = ap.parse_args()

    tok = load_tokenizer()

    print("Loading GSM8K test questions for contamination filtering ...")
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")
    gsm_norm = {normalize_text(q) for q in gsm_test["question"]}

    print("Streaming MetaMathQA in source order ...")
    mm = load_dataset("meta-math/MetaMathQA", split="train")

    kept, kept_idx, reasons = [], [], Counter()
    n_tokens_total = 0
    pbar = tqdm(total=args.limit)
    for i, ex in enumerate(mm):
        q, r = ex["query"], ex["response"]
        if not q or not q.strip() or not r or not r.strip():
            reasons["empty"] += 1
            continue
        oq = ex.get("original_question") or ""
        if normalize_text(q) in gsm_norm or (oq and normalize_text(oq) in gsm_norm):
            reasons["gsm8k_test_overlap"] += 1
            continue
        ids, labels = encode_sft_example(tok, q, r)
        if len(ids) > MAX_SEQ_LEN:
            reasons["too_long"] += 1
            continue
        kept.append({"input_ids": ids, "labels": labels, "src_idx": i, "type": ex["type"]})
        kept_idx.append(i)
        n_tokens_total += len(ids)
        pbar.update(1)
        if len(kept) >= args.limit:
            break
    pbar.close()

    data_dir = out_dir() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save(kept, data_dir / "train_tokenized.pt")

    lens = [len(k["input_ids"]) for k in kept]
    resp_lens = [sum(1 for l in k["labels"] if l != -100) for k in kept]
    stats = {
        "n_kept": len(kept),
        "last_source_index_scanned": kept_idx[-1] if kept_idx else None,
        "exclusions": dict(reasons),
        "type_counts": dict(Counter(k["type"] for k in kept)),
        "total_tokens": n_tokens_total,
        "mean_len": sum(lens) / len(lens),
        "max_len": max(lens),
        "mean_response_tokens": sum(resp_lens) / len(resp_lens),
        "kept_source_indices": kept_idx,
    }
    write_json(stats, results_dir() / "data_selection.json")
    print({k: v for k, v in stats.items() if k != "kept_source_indices"})


if __name__ == "__main__":
    main()
