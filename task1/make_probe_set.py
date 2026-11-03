"""
Activation-probe set (manual: "Diagnostic and qualitative inputs").
Sample 32 GSM8K test questions and 32 HellaSwag validation contexts with seed 6304,
BEFORE inspecting any model results. IDs are saved and reused for every model.

    python task1/make_probe_set.py
"""
import numpy as np
from datasets import load_dataset

from common import SEED, hellaswag_context, results_dir, write_json


def main():
    rng = np.random.default_rng(SEED)
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = load_dataset("Rowan/hellaswag", split="validation")
    gsm_idx = sorted(rng.choice(len(gsm), size=32, replace=False).tolist())
    hs_idx = sorted(rng.choice(len(hs), size=32, replace=False).tolist())
    probe = {
        "seed": SEED,
        "gsm8k_test": [{"index": i, "question": gsm[i]["question"]} for i in gsm_idx],
        "hellaswag_validation": [{"index": i, "ind": hs[i]["ind"], "context": hellaswag_context(hs[i])} for i in hs_idx],
    }
    write_json(probe, results_dir() / "probe_set.json")
    print("saved", results_dir() / "probe_set.json")


if __name__ == "__main__":
    main()
