"""
Shared constants and helpers for ATML PA3 Task 2 (SigLIP-Qwen VLM).
Everything fixed by the manual lives here; scripts import this module as `C` and read attributes
at run time (so a test harness can shrink the dataset sizes).
"""
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch

SEED = 6304
LM_ID = os.environ.get("PA3_MODEL_ID", "Qwen/Qwen3-0.6B")
VISION_ID = os.environ.get("PA3_VISION_ID", "google/siglip-base-patch16-224")
N_VIS = int(os.environ.get("PA3_VISUAL_TOKENS", "196"))   # 196, or 49 for the optional 2x2-pooled variant
GRID = 14

# ---- shared training settings (manual table)
MICRO_BATCH = 8
GRAD_ACCUM = 2
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM          # 16
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0
WARMUP_FRAC = 0.05
FINAL_LR_FRAC = 0.10
Q_CAP, CAPTION_CAP, ANSWER_CAP = 80, 64, 16

# ---- curriculum: (name, updates, connector peak LR, LoRA peak LR, mixture probabilities)
STAGES = [
    dict(name="caption", updates=2250, conn_lr=1e-3,  lora_lr=None, mix=None),
    dict(name="vit",     updates=1500, conn_lr=1e-4,  lora_lr=2e-5, mix={"presence": .50, "counting": .25, "caption": .25}),
    dict(name="vqa",     updates=750,  conn_lr=5e-5,  lora_lr=1e-5, mix={"vqa": .70, "presence": .10, "counting": .10, "caption": .10}),
]
STAGE_ENDS = [2250, 3750, 4500]
TOTAL_UPDATES = 4500
CKPT_EVERY = 250
DEV_EVAL_AT = sorted({0, *range(500, 4501, 500), *STAGE_ENDS})
LORA = dict(r=16, alpha=32, dropout=0.05, targets=["q_proj", "k_proj", "v_proj", "o_proj"])

# ---- data sizes (manual tables)
N_CAPTION_IMAGES, CAPTIONS_PER_IMAGE = 6000, 3
N_CAPTION_DEV = 96
VQA_TRAIN = {"yes/no": 640, "number": 256, "other": 640}
VQA_DEV = {"yes/no": 12, "number": 8, "other": 12}
VQA_TEST = {"yes/no": 36, "number": 24, "other": 36}
GROUND_DEV_PAIRS = {"presence": 12, "counting": 12}     # 24 pairs / 48 questions
GROUND_TEST_PAIRS = {"presence": 32, "counting": 32}    # 64 pairs / 128 questions
GROUND_POOL_PAIRS = {"presence": 4000, "counting": 4000}
COUNT_RANGE = (1, 4)

# ---- prompts
CAPTION_PROMPT = "Describe this image in one sentence."
ANSWER_SUFFIX = "Answer with a single word or short phrase."
VIS_START, VIS_END, IMG_PAD = "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"

# ---- evaluation
GEN_MAX_NEW, GEN_BATCH = 16, 8
N_BOOT = 4000
CONDITIONS = ["correct", "complementary", "mismatched", "noimage"]

# ---- download sources
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
VQA_BASE = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
VQA_FILES = {
    "q_train": ("v2_Questions_Train_mscoco.zip", "v2_OpenEnded_mscoco_train2014_questions.json"),
    "q_val": ("v2_Questions_Val_mscoco.zip", "v2_OpenEnded_mscoco_val2014_questions.json"),
    "a_train": ("v2_Annotations_Train_mscoco.zip", "v2_mscoco_train2014_annotations.json"),
    "a_val": ("v2_Annotations_Val_mscoco.zip", "v2_mscoco_val2014_annotations.json"),
    "p_train": ("v2_Complementary_Pairs_Train_mscoco.zip", "v2_mscoco_train2014_complementary_pairs.json"),
    "p_val": ("v2_Complementary_Pairs_Val_mscoco.zip", "v2_mscoco_val2014_complementary_pairs.json"),
}


# ---------------------------------------------------------------- paths
def repo_root() -> Path:
    return Path(os.environ.get("PA3_ROOT", Path(__file__).resolve().parents[1]))


def out_dir() -> Path:
    p = Path(os.environ.get("PA3_OUT2", repo_root() / "outputs" / "task2"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir() -> Path:
    p = repo_root() / "results" / ("task2" if N_VIS == 196 else f"task2_{N_VIS}tok")
    p.mkdir(parents=True, exist_ok=True)
    return p


def tag() -> str:
    """Suffix that keeps the optional 49-token run's checkpoints/weights apart from the main run."""
    return "" if N_VIS == 196 else f"_{N_VIS}tok"


def data_dir() -> Path:
    """Compact, git-tracked dataset files (selected captions / pairs / questions with answers)."""
    p = repo_root() / "results" / "task2" / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def read_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- VQA answer normalisation
# Re-implementation of the official VQA evaluation normalisation (processPunctuation + processDigitArticle).
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd",
    "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
    "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
    "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}
_MANUAL_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
               "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def _process_punctuation(text):
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or (re.search(_COMMA_STRIP, text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return _PERIOD_STRIP.sub("", out, re.UNICODE)


def _process_digit_article(text):
    words = []
    for w in text.lower().split():
        w = _MANUAL_MAP.get(w, w)
        if w not in _ARTICLES:
            words.append(w)
    return " ".join(_CONTRACTIONS.get(w, w) for w in words)


def normalize_answer(ans: str) -> str:
    ans = ans.replace("\n", " ").replace("\t", " ").strip()
    return _process_digit_article(_process_punctuation(ans)).strip()


def vqa_soft_accuracy(pred_norm: str, ref_norm: list) -> float:
    """s(a) = 1/10 sum_i min(1, 1/3 sum_{j != i} 1[a == a_j]) over the ten reference answers."""
    n = len(ref_norm)
    tot = 0.0
    for i in range(n):
        m = sum(1 for j in range(n) if j != i and ref_norm[j] == pred_norm)
        tot += min(1.0, m / 3.0)
    return tot / n


def consensus(ref_norm: list):
    """(most common normalised answer, its count)."""
    from collections import Counter
    a, c = Counter(ref_norm).most_common(1)[0]
    return a, c


# ---------------------------------------------------------------- misc
_PLURAL_IRREGULAR = {"person": "people", "bus": "buses", "knife": "knives", "scissors": "scissors", "skis": "skis",
                     "sheep": "sheep", "mouse": "mice", "bench": "benches", "couch": "couches", "sandwich": "sandwiches",
                     "wine glass": "wine glasses", "hair drier": "hair driers", "toothbrush": "toothbrushes",
                     "hot dog": "hot dogs", "potted plant": "potted plants", "sports ball": "sports balls",
                     "teddy bear": "teddy bears", "traffic light": "traffic lights", "fire hydrant": "fire hydrants",
                     "stop sign": "stop signs", "parking meter": "parking meters", "baseball bat": "baseball bats",
                     "baseball glove": "baseball gloves", "tennis racket": "tennis rackets", "cell phone": "cell phones",
                     "dining table": "dining tables"}


def plural(name: str) -> str:
    if name in _PLURAL_IRREGULAR:
        return _PLURAL_IRREGULAR[name]
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    return name + "s"


def with_article(name: str) -> str:
    return ("an " if name[0] in "aeiou" else "a ") + name


def presence_question(name):
    return f"Does this image contain {with_article(name)}?"


def counting_question(name):
    return f"How many {plural(name)} can you see?"


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        import time
        self._t = time; self.t0 = time.perf_counter(); return self

    def __exit__(self, *a):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = self._t.perf_counter() - self.t0
