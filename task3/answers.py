"""
Answer equivalence, grouping and grading (manual: "Answers and correctness").

MATH      : math-verify equivalence (version recorded). Candidates are grouped by pairwise equivalence (union-find over
            distinct strings, cheap canonical pre-grouping first). Grading uses the same `verify` against the gold answer.
MathVista : official normalisation rules from the released question metadata (question_type, answer_type, precision,
            choices); multiple-choice answers are represented as option letters. Grading = same normalisation of the gold.
Missing / unparseable extractions -> NO_ANSWER group, always incorrect, kept in voting and cost accounting.
Gold answers are used only for grading, never for grouping.
"""
import re
from functools import lru_cache

import common3 as C

# ----------------------------------------------------------------------------- MATH (math-verify)
_SIMPLE = [(r"\\left", ""), (r"\\right", ""), (r"\\dfrac", r"\\frac"), (r"\\tfrac", r"\\frac"), (r"\\!", ""), (r"\\,", ""),
           (r"\\;", ""), (r"\\ ", ""), (r"\s+", ""), (r"\^\{\\circ\}", ""), (r"\^\\circ", ""), (r"\\%", ""), (r"\$", ""),
           (r"\\text\{([^}]*)\}", r"\1"), (r"\\mathrm\{([^}]*)\}", r"\1"), (r"\.$", "")]


def _canon_math(s: str) -> str:
    s = s.strip()
    for pat, rep in _SIMPLE:
        s = re.sub(pat, rep, s)
    return s


class MathScorer:
    def __init__(self):
        import math_verify
        from math_verify import parse, verify
        self._parse, self._verify = parse, verify
        try:
            from importlib.metadata import version
            self.version = version("math-verify")
        except Exception:
            self.version = getattr(math_verify, "__version__", "unknown")
        self.name = f"math-verify {self.version}"

    @lru_cache(maxsize=None)
    def parse(self, s: str):
        if s is None:
            return ()
        t = s if ("$" in s or "\\boxed" in s) else f"${s}$"
        try:
            return tuple(self._parse(t))
        except Exception:
            return ()

    @lru_cache(maxsize=None)
    def equivalent(self, a: str, b: str) -> bool:
        if a == b or _canon_math(a) == _canon_math(b):
            return True
        pa, pb = self.parse(a), self.parse(b)
        if not pa or not pb:
            return False
        try:
            return bool(self._verify(list(pa), list(pb)) or self._verify(list(pb), list(pa)))
        except Exception:
            return False

    def is_correct(self, qid, gold: str, ext) -> bool:
        if ext is None or gold is None:
            return False
        return self.equivalent(str(gold), str(ext))

    def group(self, qid, exts):
        """Group ids per rollout (-1 = NO_ANSWER) and a representative string per group, without gold."""
        reps, gids = [], []
        for e in exts:
            if e is None or str(e).strip() == "":
                gids.append(-1); continue
            e = str(e); g = None
            for k, r in enumerate(reps):
                if self.equivalent(r, e):
                    g = k; break
            if g is None:
                reps.append(e); g = len(reps) - 1
            gids.append(g)
        return gids, reps


# ----------------------------------------------------------------------------- MathVista (metadata rules)
class MathVistaScorer:
    def __init__(self, meta: dict):
        """meta: pid(str) -> {question_type, answer_type, precision, choices, answer}"""
        self.meta = meta
        self.name = "MathVista testmini metadata rules (answer_type / choices / precision), letters for multiple choice"
        self.version = "reimplementation of MathVista normalize_extracted_answer + safe_equal"

    def normalize(self, qid, ext):
        if ext is None:
            return None
        m = self.meta[str(qid)]
        s = str(ext).strip()
        if s == "":
            return None
        qt, at = m.get("question_type"), m.get("answer_type")
        if qt == "multi_choice":
            choices = m.get("choices") or []
            letters = [chr(65 + i) for i in range(len(choices))]
            if len(s) == 1 and s.upper() in letters:
                return s.upper()
            mm = re.fullmatch(r"\(?([A-Za-z])\)?[.:]?", s)
            if mm and mm.group(1).upper() in letters:
                return mm.group(1).upper()
            for i, c in enumerate(choices):
                if s.lower() == str(c).strip().lower():
                    return letters[i]
            return "RAW:" + s.lower()                       # not an option -> own group, graded incorrect
        try:
            if at == "integer":
                return str(int(float(s.replace(",", ""))))
            if at == "float":
                prec = m.get("precision")
                prec = int(prec) if prec is not None else 2
                return str(round(float(s.replace(",", "")), prec))
        except Exception:
            return None
        return s.lower().rstrip(".")                         # text / list

    def gold(self, qid):
        m = self.meta[str(qid)]
        return self.normalize(qid, m["answer"])

    def is_correct(self, qid, gold_unused, ext) -> bool:
        n = self.normalize(qid, ext)
        return n is not None and n == self.gold(qid)

    def group(self, qid, exts):
        reps, gids = [], []
        for e in exts:
            n = self.normalize(qid, e)
            if n is None:
                gids.append(-1); continue
            if n in reps:
                gids.append(reps.index(n))
            else:
                reps.append(n); gids.append(len(reps) - 1)
        return gids, reps


def load_mathvista_meta(path):
    raw = C.read_json(path)
    items = raw.values() if isinstance(raw, dict) else raw
    meta = {}
    for it in items:
        meta[str(it["pid"])] = {"question_type": it.get("question_type"), "answer_type": it.get("answer_type"),
                                "precision": it.get("precision"), "choices": it.get("choices"), "answer": it.get("answer")}
    return meta
