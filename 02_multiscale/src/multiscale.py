"""ECHO-DC V2 stage 02: multiscale scoring library.

The scoring equations are NOT re-implemented: stage 01's `scoring.py` (verbatim port of the original notebooks) is
called as-is. A radius is selected by presenting that radius's inputs to it under the 2.5 km column names it expects
("column adapter"). Only the protocol-B (fixed-reference) transform and the stability metrics are new code.
"""
import math
import numpy as np, pandas as pd
from cfg import *
import scoring as S                                    # stage 01, read-only

# --- scale classification of the nine score inputs ------------------------------------------------------------------
# model column (name used by scoring.py)  ->  source-column template at radius s
SCALE_DEPENDENT = {
    "hotmaps_heat_sum_2p5km": "hotmaps_heat_sum_{s}",                 # extensive: cumulative total, grows ~ area
    "pop_density_2p5km": "pop_density_{s}",                           # intensive: people per km2
    "anchor_count_2p5km": "anchor_count_{s}",                         # extensive: count in catchment
    "anchor_type_diversity_2p5km": "anchor_type_diversity_{s}",       # bounded count (0-4) of categories present
    "built_pct_2p5km": "built_pct_{s}",                               # intensive: % GHSL built surface
    "dw_built_pct_2p5km": "dw_built_pct_{s}",                         # intensive: % Dynamic World built
    "major_road_density_2p5km_km_per_km2": "major_road_density_{s}_km_per_km2",   # intensive: km per km2
}
SCALE_INVARIANT = ["hdd_2024", "nearest_anchor_distance_m", "nearest_major_road_distance_m"]
GROUPS = {   # attribution groups (single-factor substitution, C5-D)
    "heat_demand": ["hotmaps_heat_sum_2p5km"],
    "anchors": ["anchor_count_2p5km", "anchor_type_diversity_2p5km"],
    "urban_compactness": ["built_pct_2p5km", "dw_built_pct_2p5km"],
    "road_density": ["major_road_density_2p5km_km_per_km2"],
    "population_density": ["pop_density_2p5km"],
}
GENERIC = {c: c.replace("_2p5km", "") for c in S.ALL_IND + ["norm_ghsl_built_pct_2p5km", "norm_dw_built_pct_2p5km"]}
N_TOP = None   # set per population: ceil(10%) -> 122 for n=1216 (same rule as stage 01 b5_validate)

def n_top(n):
    return int(math.ceil(n * 0.10))

# --- data ------------------------------------------------------------------------------------------------------------
def load_table():
    return pd.read_csv(TABLE105)

def scale_inputs(df, s, only=None):
    """Copy of df whose 2.5 km-named score inputs hold radius-`s` values (all, or only the listed model columns)."""
    d = df.copy()
    for model_col, tmpl in SCALE_DEPENDENT.items():
        if only is None or model_col in only:
            d[model_col] = df[tmpl.format(s=s)].to_numpy()
    return d

# --- Protocol A (scale-relative) ------------------------------------------------------------------------------------
def score_scale(df, s, only=None):
    """Original normalise + 3 scenarios at radius s. Entropy weights are re-fitted on the radius-s normalised data."""
    n = S.normalise(scale_inputs(df, s, only))
    out, w = S.score_all(n)
    return out, w

def score_with_entropy_held(out, w_ref):
    """Add scenario 'entropy_ref': entropy weights fitted at the 2.5 km reference, applied to this radius' indicators."""
    return S.compute_echo_scores(out, "entropy_ref", w_ref["entropy"][0], w_ref["entropy"][1])

def generic(out, name):
    """Strip the misleading '_2p5km' suffix from a scored frame -> tidy columns for export."""
    o = out.rename(columns=GENERIC)
    return o

# --- Protocol B (fixed 2.5 km reference): only for intensive / invariant indicators ----------------------------------
# (indicator, log1p?, reversed?) ; the fit is on the 2.5 km column exactly like S.robust_minmax*
B_ELIGIBLE = {
    "norm_ghsl_built_pct": ("built_pct", False),
    "norm_dw_built_pct": ("dw_built_pct", False),
    "norm_major_road_density": ("major_road_density", False),
    "norm_pop_density": ("pop_density", True),
}

def fit_ref(series, log, lo_q=0.05, hi_q=0.95):
    x = np.log1p(series) if log else series
    return float(x.quantile(lo_q)), float(x.quantile(hi_q))

def apply_ref(series, lo, hi, log):
    x = np.log1p(series) if log else series
    return ((x - lo) / (hi - lo)).clip(0, 1)

def feasibility_protocol_b(df, s, w_ref_entropy_feas=None):
    """Feasibility under a FIXED 2.5 km normalisation (Protocol B). Feasibility's four inputs are all intensive
    (built %, km/km2, people/km2) or radius-invariant (nearest road), so a common reference is well defined.
    Returns DataFrame with feasibility_B_{equal,literature_informed[,entropy_ref]} and the B-normalised inputs."""
    d = pd.DataFrame({"facility_id": df.facility_id})
    ref = {}
    for k, (stem, log) in B_ELIGIBLE.items():
        col = {"built_pct": "built_pct_{s}", "dw_built_pct": "dw_built_pct_{s}",
               "major_road_density": "major_road_density_{s}_km_per_km2", "pop_density": "pop_density_{s}"}[stem]
        lo, hi = fit_ref(df[col.format(s=REF)], log)
        d[k] = apply_ref(df[col.format(s=s)], lo, hi, log).to_numpy()
        ref[k] = (lo, hi)
    # nearest-road distance is a facility-level constant -> identical to the 2.5 km normalisation
    d["norm_nearest_major_road_distance_m"] = S.reverse_score(S.robust_minmax_log(df["nearest_major_road_distance_m"])).to_numpy()
    d["norm_urban_compactness"] = (d["norm_ghsl_built_pct"] + d["norm_dw_built_pct"]) / 2
    d["norm_major_road_density_"] = d["norm_major_road_density"]
    feas_ind = ["norm_urban_compactness", "norm_major_road_density", "norm_nearest_major_road_distance_m", "norm_pop_density"]
    d[feas_ind] = d[feas_ind].replace([np.inf, -np.inf], np.nan).fillna(0)
    d["feasibility_B_equal"] = sum(d[c] * 0.25 for c in feas_ind)
    lit = dict(zip(feas_ind, S.LIT_FEAS.values()))
    d["feasibility_B_literature_informed"] = sum(d[c] * w for c, w in lit.items())
    if w_ref_entropy_feas is not None:
        d["feasibility_B_entropy_ref"] = sum(d[c] * w for c, w in zip(feas_ind, w_ref_entropy_feas.values()))
    return d.drop(columns=["norm_major_road_density_"]), ref

# --- rank / overlap metrics -------------------------------------------------------------------------------------------
def rank_from_raw(raw):
    """Same rule as scoring.compute_echo_scores: descending raw score, ties by row order."""
    return pd.Series(raw).rank(ascending=False, method="first").astype(int).to_numpy()

def top_set(rank, k):
    return set(np.flatnonzero(np.asarray(rank) <= k))

def pairwise(ranks, fn):
    """ranks: dict scale -> array. Returns DataFrame (scale x scale) of fn(rank_a, rank_b)."""
    m = pd.DataFrame(index=list(ranks), columns=list(ranks), dtype=float)
    for a in ranks:
        for b in ranks:
            m.loc[a, b] = fn(ranks[a], ranks[b])
    return m

def sensitivity_table(ids, country, ranks, k):
    """Per-facility rank statistics across scales. ranks: dict scale -> array (SCALES order)."""
    R = np.column_stack([ranks[s] for s in SCALES])
    t = pd.DataFrame(R, columns=[f"rank_{s}" for s in SCALES])
    t.insert(0, "country", country)
    t.insert(0, "facility_id", ids)
    t["best_rank"], t["worst_rank"] = R.min(1), R.max(1)
    t["rank_range"] = t.worst_rank - t.best_rank
    t["rank_sd"] = R.std(1, ddof=1)
    top = R <= k
    t["n_top_decile"] = top.sum(1)
    ref_i = SCALES.index(REF)
    for j, s in enumerate(SCALES):
        if s != REF:
            t[f"top_status_{s}_vs_ref"] = np.where(top[:, j] & ~top[:, ref_i], "enters",
                                          np.where(~top[:, j] & top[:, ref_i], "exits",
                                          np.where(top[:, j], "stays_top", "stays_out")))
    t["top_pattern"] = np.where(top.all(1), "all_four", np.where(top.any(1), "some", "none"))
    return t
