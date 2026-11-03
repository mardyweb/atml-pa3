"""
Shared constants and helpers for ATML PA3 Task 1 (LLM fine-tuning).

Everything that the manual fixes (seed, model id, prompt formatting, LoRA targets,
diagnostic layers, ...) lives here so that every script uses exactly the same values.
"""
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------- fixed by the manual
SEED = 6304
MODEL_ID = os.environ.get("PA3_MODEL_ID", "Qwen/Qwen3-0.6B")   # env override is for smoke tests only
MAX_SEQ_LEN = 2048
N_TRAIN_EXAMPLES = 40_000
MICRO_BATCH = 2
GRAD_ACCUM = 16
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM          # 32
TOTAL_UPDATES = N_TRAIN_EXAMPLES // EFFECTIVE_BATCH  # 1250
WARMUP_UPDATES = 38
CHECKPOINT_UPDATES = [312, 625, 937, 1250]           # 9,984 / 20,000 / 29,984 / 40,000 examples
MIDPOINT_UPDATE = 625

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DIAG_LAYERS = [0, 13, 27]        # zero-based layers whose down_proj we analyse
PROTECTION_RANK = 128            # OPLoRA k (also used for rho_k on every variant)

# Method table: name -> hyper-parameters fixed by the manual
METHODS = {
    "full":    dict(kind="full",  lr=2e-5, lam=0.0,  r=None, alpha=None, k=None),
    "l2sp":    dict(kind="full",  lr=2e-5, lam=1e-3, r=None, alpha=None, k=None),
    "lora4":   dict(kind="lora",  lr=2e-4, lam=0.0,  r=4,    alpha=8,    k=None),
    "lora16":  dict(kind="lora",  lr=2e-4, lam=0.0,  r=16,   alpha=32,   k=None),
    "oplora16": dict(kind="oplora", lr=2e-4, lam=0.0, r=16,  alpha=32,   k=PROTECTION_RANK),
}

# Seven evaluation conditions (name -> directory of an exported, merged HF model)
EVAL_CONDITIONS = ["baseline", "full_mid", "full", "l2sp", "lora4", "lora16", "oplora16"]


# ---------------------------------------------------------------- paths
def repo_root() -> Path:
    return Path(os.environ.get("PA3_ROOT", Path(__file__).resolve().parents[1]))


def out_dir() -> Path:
    """Large outputs (checkpoints, exported models) - NOT committed to git."""
    p = Path(os.environ.get("PA3_OUT", repo_root() / "outputs" / "task1"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir() -> Path:
    """Small machine-readable results - committed to git."""
    p = repo_root() / "results" / "task1"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- seeding
def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------- tokenizer / prompts
def load_tokenizer(path_or_id: str = MODEL_ID):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path_or_id)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def chat_prompt(tok, user_text: str, system_text: str | None = None) -> str:
    """Non-thinking Qwen3 chat prompt (enable_thinking=False), ready for the assistant turn."""
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def im_end_id(tok) -> int:
    """<|im_end|> is Qwen's end-of-turn token (also its eos for chat)."""
    return tok.convert_tokens_to_ids("<|im_end|>")


def encode_sft_example(tok, question: str, response: str):
    """
    Returns (input_ids, labels). Loss is computed only on assistant-response tokens
    and the end-of-turn token; prompt tokens are masked with -100.
    Prompt and response are tokenised separately and concatenated so that the prompt
    boundary is identical to the boundary used at generation time.
    """
    prompt_ids = tok(chat_prompt(tok, question), add_special_tokens=False)["input_ids"]
    resp_ids = tok(response, add_special_tokens=False)["input_ids"] + [im_end_id(tok)]
    input_ids = prompt_ids + resp_ids
    labels = [-100] * len(prompt_ids) + resp_ids
    return input_ids, labels


_norm_re = re.compile(r"[^a-z0-9]+")


def normalize_text(s: str) -> str:
    """Text normalisation used for GSM8K-test contamination matching."""
    return _norm_re.sub(" ", s.lower()).strip()


# ---------------------------------------------------------------- misc utils
def gsm8k_prompt_text(question: str) -> str:
    """lm-eval gsm8k_cot zero-shot doc_to_text (before chat template)."""
    return f"Q: {question}\nA:"


def hellaswag_context(doc: dict) -> str:
    """lm-eval hellaswag preprocessing of the context (doc_to_text)."""
    def preprocess(text):
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        text = text.replace("  ", " ")
        return text
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return preprocess(doc["activity_label"] + ": " + ctx)


def dir_size_bytes(path) -> int:
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.t0


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
