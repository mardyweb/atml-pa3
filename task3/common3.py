"""Shared constants / helpers for ATML PA3 Task 3 (test-time compute)."""
import json
import os
from pathlib import Path

import numpy as np

SEED = 6304
DATASETS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}      # key -> folder name in ATML-PA3-Data
N_ROLLOUTS = 32
N_ORDERS = 10
BUDGETS = [1, 2, 4, 8, 16, 32]
N_QUESTIONS = int(os.environ.get("PA3_N_QUESTIONS", 1000))   # env overrides are for the synthetic test only
N_DEV, N_LIVE = int(os.environ.get("PA3_N_DEV", 200)), int(os.environ.get("PA3_N_LIVE", 50))
N_BOOT = 1000
NO_ANSWER = "NO_ANSWER"

# adaptive sampling
STOP_CHECKS = [4, 8, 16]
TAUS = [0.5, 0.625, 0.75, 0.875, 1.0]
GAMMAS = [0.0, 0.1, 0.25, 0.5]
TOLERANCE_PP = 1.0

# live generation (manual Step 4 + README provenance)
LIVE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LIVE_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SYSTEM_MSG = "You are a helpful assistant."
INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW_INSTRUCTION = ("Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
                      "Give your final answer inside \\boxed{}.")
GEN_KW = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)
CAP_SINGLE, CAP_PARALLEL, CAP_REVISION, TOTAL_ALLOWANCE = 3072, 1024, 2048, 3072

MATHVISTA_META_URL = "https://raw.githubusercontent.com/lupantech/MathVista/main/data/testmini.json"


def repo_root() -> Path:
    return Path(os.environ.get("PA3_ROOT", Path(__file__).resolve().parents[1]))


def out_dir() -> Path:
    p = Path(os.environ.get("PA3_OUT3", repo_root() / "outputs" / "task3")); p.mkdir(parents=True, exist_ok=True); return p


def results_dir() -> Path:
    p = repo_root() / "results" / "task3"; p.mkdir(parents=True, exist_ok=True); return p


def data_root() -> Path:
    """Folder holding qwen25-math/ and qwen25vl-mathvista/ (copied from the shared Drive folder)."""
    return Path(os.environ.get("PA3_DATA3", out_dir() / "data"))


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_default)


def _default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def split_questions(qids):
    """Sort lexicographically, permute with default_rng(6304): first 200 dev, rest held-out; first 50 held-out = live."""
    qids = sorted(qids)
    perm = np.random.default_rng(SEED).permutation(len(qids))
    ordered = [qids[i] for i in perm]
    return {"dev": ordered[:N_DEV], "heldout": ordered[N_DEV:], "live": ordered[N_DEV:N_DEV + N_LIVE], "perm_order": ordered}


def rollout_orders(n_questions):
    """Ten seeded permutations of the 32 rollout IDs per question (questions in sorted-ID order)."""
    rng = np.random.default_rng(SEED)
    return np.stack([np.stack([rng.permutation(N_ROLLOUTS) for _ in range(N_ORDERS)]) for _ in range(n_questions)]).astype(np.int8)


def paired_bootstrap(a, b=None, n_boot=N_BOOT, seed=SEED):
    """95% percentile CI of mean(a) (or of mean(a-b) when b given) over question-level resamples."""
    a = np.asarray(a, float); d = a if b is None else a - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}


def extract_boxed(text: str):
    """README rule: last literal '\\boxed{', balanced-brace scan; absent / empty / unclosed -> None."""
    if text is None:
        return None
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = i + len("\\boxed{"); depth = 1; out = []
    while j < len(text):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                s = "".join(out).strip()
                return s if s else None
        out.append(ch); j += 1
    return None
