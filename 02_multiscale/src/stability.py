"""Ranking-stability metrics (C5/C6). All functions take the long score table written by c4_scores.py."""
import numpy as np, pandas as pd
from scipy import stats
from cfg import *
import multiscale as M

def ranks_of(long, scenario):
    """dict scale -> rank array in the common facility order."""
    ids = long[long.scale == REF].facility_id.to_numpy()
    out = {}
    for s in SCALES:
        t = long[long.scale == s]
        assert (t.facility_id.to_numpy() == ids).all()
        out[s] = t[f"echo_dc_rank_{scenario}"].to_numpy()
    return out

def raws_of(long, scenario):
    return {s: long.loc[long.scale == s, f"echo_dc_score_raw_{scenario}"].to_numpy() for s in SCALES}

def spearman(a, b):
    return float(stats.spearmanr(a, b)[0])

def kendall(a, b):
    return float(stats.kendalltau(a, b)[0])          # tau-b

def global_metrics(ranks, k):
    sp = M.pairwise(ranks, spearman)
    kt = M.pairwise(ranks, kendall)
    top = {s: M.top_set(ranks[s], k) for s in SCALES}
    t20 = {s: M.top_set(ranks[s], 20) for s in SCALES}
    ov = M.pairwise(ranks, lambda a, b: len(M.top_set(a, k) & M.top_set(b, k)))
    ov20 = M.pairwise(ranks, lambda a, b: len(M.top_set(a, 20) & M.top_set(b, 20)))
    jac = M.pairwise(ranks, lambda a, b: len(M.top_set(a, k) & M.top_set(b, k)) / len(M.top_set(a, k) | M.top_set(b, k)))
    all4 = set.intersection(*top.values()); any4 = set.union(*top.values())
    return {"spearman": sp, "kendall": kt, "top_overlap": ov, "top_jaccard": jac, "top20_overlap": ov20,
            "n_top_all_four": len(all4), "n_top_any_scale": len(any4), "k": k}

def boot_spearman(raw_a, raw_b, n_boot=1000, seed=20260707):
    """Percentile bootstrap (resampling facilities) of Spearman rho between two score vectors."""
    rng = np.random.default_rng(seed)
    n = len(raw_a); out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        out[i] = stats.spearmanr(raw_a[idx], raw_b[idx])[0]
    return out
