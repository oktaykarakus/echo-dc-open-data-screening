"""ECHO-DC V1 scoring: verbatim port of normalisation_weighting_and_score.ipynb (cells 2-17) and
score construction.ipynb (cells 15-20). No formula, weight or rule is changed here."""
import numpy as np, pandas as pd

# ---- cell 2: normalisation helpers --------------------------------------------------------------
def robust_minmax(series, lower_q=0.05, upper_q=0.95):
    lower, upper = series.quantile(lower_q), series.quantile(upper_q)
    return ((series - lower) / (upper - lower)).clip(0, 1)

def robust_minmax_log(series, lower_q=0.05, upper_q=0.95):
    return robust_minmax(np.log1p(series), lower_q, upper_q)

def reverse_score(s):
    return 1 - s

OPP = ["norm_heat_sum_2p5km", "norm_pop_density_2p5km", "norm_hdd_2024", "norm_anchor_count_2p5km",
       "norm_anchor_diversity_2p5km", "norm_nearest_anchor_distance_m"]
FEAS = ["norm_urban_compactness_2p5km", "norm_major_road_density_2p5km", "norm_nearest_major_road_distance_m",
        "norm_pop_density_2p5km"]
ALL_IND = list(dict.fromkeys(OPP + FEAS))
LIT_OPP = dict(zip(OPP, [0.35, 0.20, 0.15, 0.15, 0.10, 0.05]))
LIT_FEAS = dict(zip(FEAS, [0.35, 0.30, 0.20, 0.15]))
EPS = 1e-6

# ---- cell 3-4: normalised indicators ---------------------------------------------------------------
def normalise(df):
    """Returns a copy of df with norm_* columns. NaN/inf -> 0 only for the 9 score indicators (cell 4)."""
    df = df.copy()
    df["norm_heat_sum_2p5km"] = robust_minmax_log(df["hotmaps_heat_sum_2p5km"])
    df["norm_pop_density_2p5km"] = robust_minmax_log(df["pop_density_2p5km"])
    df["norm_hdd_2024"] = robust_minmax(df["hdd_2024"])
    df["norm_anchor_count_2p5km"] = robust_minmax_log(df["anchor_count_2p5km"])
    df["norm_anchor_diversity_2p5km"] = robust_minmax(df["anchor_type_diversity_2p5km"])
    df["norm_nearest_anchor_distance_m"] = reverse_score(robust_minmax_log(df["nearest_anchor_distance_m"]))
    df["norm_major_road_density_2p5km"] = robust_minmax(df["major_road_density_2p5km_km_per_km2"])
    df["norm_nearest_major_road_distance_m"] = reverse_score(robust_minmax_log(df["nearest_major_road_distance_m"]))
    df["norm_ghsl_built_pct_2p5km"] = robust_minmax(df["built_pct_2p5km"])
    df["norm_dw_built_pct_2p5km"] = robust_minmax(df["dw_built_pct_2p5km"])
    df["norm_urban_compactness_2p5km"] = (df["norm_ghsl_built_pct_2p5km"] + df["norm_dw_built_pct_2p5km"]) / 2
    df[ALL_IND] = df[ALL_IND].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

# ---- cell 6: weights & scores --------------------------------------------------------------------------
def make_equal_weights(ind):
    return {c: 1 / len(ind) for c in ind}

def entropy_weights(data, indicators, eps=1e-12):
    X = data[indicators].copy().replace([np.inf, -np.inf], np.nan).fillna(0) + eps
    P = X.div(X.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    n = len(X)
    if n <= 1:
        return make_equal_weights(indicators)
    entropy = -(P * np.log(P + eps)).sum(axis=0) / np.log(n)
    div = 1 - entropy
    if div.sum() == 0:
        return make_equal_weights(indicators)
    return (div / div.sum()).to_dict()

def weighted_sum(data, weights):
    s = np.zeros(len(data))
    for c, w in weights.items():
        s += data[c].to_numpy() * w
    return s

def compute_echo_scores(data, name, w_opp, w_feas):
    out = data.copy()
    out[f"opportunity_score_{name}"] = weighted_sum(out, w_opp)
    out[f"feasibility_score_{name}"] = weighted_sum(out, w_feas)
    raw = np.sqrt((out[f"opportunity_score_{name}"] + EPS) * (out[f"feasibility_score_{name}"] + EPS))
    out[f"echo_dc_score_raw_{name}"] = raw
    out[f"echo_dc_score_{name}"] = raw.round(4)
    # method="first": ties broken by row order of the input table (= master CSV order)
    out[f"echo_dc_rank_{name}"] = raw.rank(ascending=False, method="first").astype(int)
    return out

def score_all(df_norm):
    """Equal / literature / entropy scenarios exactly as in the final notebook (cells 8-15)."""
    w = {
        "equal": (make_equal_weights(OPP), make_equal_weights(FEAS)),
        "literature_informed": (LIT_OPP, LIT_FEAS),
        "entropy": (entropy_weights(df_norm, OPP), entropy_weights(df_norm, FEAS)),
    }
    out = df_norm.copy()
    for k, (wo, wf) in w.items():
        out = compute_echo_scores(out, k, wo, wf)
    return out, w

def early_literature_score(df_norm):
    """Variant used by the EARLIER draft (score construction.ipynb cells 15-20): Opp/Feas rounded to 4 dp
    before the geometric mean; reported score rounded to 4 dp; rank by raw score, method='first'.
    Used only to compare like-for-like with the recovered folium-map scores."""
    opp = pd.Series(weighted_sum(df_norm, LIT_OPP)).round(4)
    feas = pd.Series(weighted_sum(df_norm, LIT_FEAS)).round(4)
    raw = np.sqrt((opp + EPS) * (feas + EPS))
    return pd.DataFrame({"early_opp": opp, "early_feas": feas, "early_score_raw": raw,
                         "early_score": raw.round(4), "early_rank": raw.rank(ascending=False, method="first").astype(int)})
