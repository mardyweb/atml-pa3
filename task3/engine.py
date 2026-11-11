"""
Selection engine shared by the offline sweep, the signal analysis and the adaptive controllers.

Candidates are always handled *in the current random order*; all tie-breaks follow the manual:
  score / vote ties -> earliest supporting rollout in the current order; RWC uses average ranks for tied scores;
  at N = 1 every rule returns the only candidate.
"""
import numpy as np
from scipy.stats import rankdata

import common3 as C


class Data:
    """Per-rollout arrays of one dataset, indexed by global rollout index = q_index * 32 + rollout_id."""
    def __init__(self, ds):
        z = np.load(C.out_dir() / "cache" / f"{ds}.npz", allow_pickle=True)
        self.ds = ds
        self.qids = [str(q) for q in z["qids"]]
        self.qidx = {q: i for i, q in enumerate(self.qids)}
        self.T, self.L, self.K = z["T"].astype(np.int64), z["L"].astype(np.float64), z["K"].astype(np.float64)
        self.group, self.correct, self.capped = z["group"].astype(np.int64), z["correct"].astype(bool), z["capped"].astype(bool)
        self.has_vision = "V" in z.files
        if self.has_vision:
            self.V = z["V"].astype(np.float64)
            self.tok_offsets, self.tok_kl, self.tok_v = z["tok_offsets"], z["tok_kl"].astype(np.float64), z["tok_v"].astype(np.float64)
        self.orders = np.load(C.out_dir() / "cache" / "orders.npy").astype(np.int64)   # (Q, 10, 32)
        self.splits = C.read_json(C.results_dir() / "splits.json")[ds]

    def gidx(self, qi, rollouts):
        return qi * C.N_ROLLOUTS + np.asarray(rollouts, dtype=np.int64)

    def tokens(self, g):
        s, e = self.tok_offsets[g], self.tok_offsets[g + 1]
        return self.tok_kl[s:e], self.tok_v[s:e]


# ----------------------------------------------------------------------------- VisConf calibration on a candidate set
def visconf(data: Data, gl):
    """
    For the candidate set gl (global indices, in order): pool token KLs, calibrate with the 5th / 95th percentiles
    (linear interpolation), return per-candidate C_i, Z_i = V_i C_i, Z_tok_i = mean_t(c_t v_t), Cov_t(c, v) = Z_tok - Z.
    """
    kls, vs, seg = [], [], []
    for g in gl:
        k, v = data.tokens(g); kls.append(k); vs.append(v); seg.append(len(k))
    kl = np.concatenate(kls) if kls else np.zeros(0); v = np.concatenate(vs) if vs else np.zeros(0)
    if len(kl) == 0:
        n = len(gl); return np.zeros(n), np.zeros(n), np.zeros(n)
    q05, q95 = np.percentile(kl, [5, 95])
    c = np.clip((kl - q05) / max(q95 - q05, 1e-8), 0.0, 1.0)
    bounds = np.concatenate([[0], np.cumsum(seg)])
    seg = np.asarray(seg)
    Cm = np.array([c[bounds[i]:bounds[i + 1]].mean() if seg[i] else 0.0 for i in range(len(gl))])
    Ztok = np.array([(c[bounds[i]:bounds[i + 1]] * v[bounds[i]:bounds[i + 1]]).mean() if seg[i] else 0.0 for i in range(len(gl))])
    Z = data.V[gl] * Cm
    return Cm, Z, Ztok


def calibrated_tokens(data: Data, gl, g):
    """Per-token (c_t, v_t) of rollout g calibrated on candidate set gl (for correlations / trajectories)."""
    kl = np.concatenate([data.tokens(x)[0] for x in gl])
    q05, q95 = np.percentile(kl, [5, 95])
    k, v = data.tokens(g)
    return np.clip((k - q05) / max(q95 - q05, 1e-8), 0, 1), v


# ----------------------------------------------------------------------------- selection rules (positions in order)
def _earliest_of_group(groups, g):
    return int(np.flatnonzero(groups == g)[0])


def sc_select(groups):
    """Self-consistency: one vote per rollout (NO_ANSWER = -1 votes as a group). Returns (position, votes_by_group)."""
    groups = np.asarray(groups)
    if len(groups) == 1:
        return 0, {int(groups[0]): 1}
    ids, counts = np.unique(groups, return_counts=True)
    best = ids[counts == counts.max()]
    pos = min(_earliest_of_group(groups, g) for g in best)
    return pos, dict(zip(ids.tolist(), counts.tolist()))


def best_select(scores):
    s = np.asarray(scores, float)
    if len(s) == 1:
        return 0
    s = np.where(np.isnan(s), -np.inf, s)
    return int(np.argmax(s))                                 # argmax returns the earliest maximum


def rwc_weights(scores):
    s = np.asarray(scores, float); N = len(s)
    if N == 1:
        return np.ones(1)
    s = np.where(np.isnan(s), -np.inf, s)
    r = rankdata(-s, method="average")                       # rank 1 = best, average ranks for ties
    return (N - r) / (N - 1)


def rwc_select(groups, scores):
    """Rank-weighted consensus. Returns (position, W_by_group)."""
    groups = np.asarray(groups)
    if len(groups) == 1:
        return 0, {int(groups[0]): 1.0}
    w = rwc_weights(scores)
    W = {}
    for g, wi in zip(groups.tolist(), w):
        W[g] = W.get(g, 0.0) + float(wi)
    top = max(W.values())
    best = [g for g, v in W.items() if v == top]
    return min(_earliest_of_group(groups, g) for g in best), W


def rwc_margin(W):
    """(largest - second largest) / total vote weight; 0 when only one answer group exists."""
    vals = sorted(W.values(), reverse=True)
    tot = sum(vals)
    if len(vals) < 2 or tot <= 0:
        return 0.0
    return (vals[0] - vals[1]) / tot


def method_scores(data: Data, gl):
    """All quality scores for candidate set gl (global indices in order). VisConf-type scores are set-calibrated."""
    s = {"L": data.L[gl], "K": data.K[gl], "neglen": -data.T[gl].astype(float)}
    if data.has_vision:
        Cm, Z, Ztok = visconf(data, gl)
        s.update({"V": data.V[gl], "C": Cm, "Z": Z, "Ztok": Ztok})
    return s


METHODS_COMMON = ["first", "sc", "best_L", "best_K", "rwc_K", "shortest"]
METHODS_VISION = ["best_V", "rwc_V", "best_Z", "rwc_Z", "best_Ztok", "rwc_Ztok"]


def select_all(data: Data, gl, groups, scores):
    """Position chosen by every method for the candidate set (gl in order)."""
    out = {"first": 0, "sc": sc_select(groups)[0], "best_L": best_select(scores["L"]), "best_K": best_select(scores["K"]),
           "rwc_K": rwc_select(groups, scores["K"])[0], "shortest": best_select(scores["neglen"])}
    if data.has_vision:
        for k in ("V", "Z", "Ztok"):
            out[f"best_{k}"] = best_select(scores[k]); out[f"rwc_{k}"] = rwc_select(groups, scores[k])[0]
    return out
