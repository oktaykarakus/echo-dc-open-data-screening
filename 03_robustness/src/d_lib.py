"""ECHO-DC V2 stage 03: shared library.

stage 01 `scoring.py` (verbatim V1 port) and stage 02 `multiscale.py` are imported READ-ONLY and are never modified.
Everything new here is: weight manipulation (drop / renormalise / multiply), a parameterised normalisation window,
rank-comparison metrics and a protected-file manifest. All temp/cache paths are forced inside the project workspace.
"""
import os, sys, math, json, hashlib, tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PD = HERE.parent                                   # .../03_robustness
V2 = PD.parent                                     # .../ECHO_DC_V2
_TMP = PD / ".tmp"
for _k, _sub in {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "XDG_CACHE_HOME": "cache", "MPLCONFIGDIR": "mpl"}.items():
    (_TMP / _sub).mkdir(parents=True, exist_ok=True)
    os.environ[_k] = str(_TMP / _sub)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
tempfile.tempdir = str(_TMP / "tmp")

PB = V2 / "01_baseline_scoring"
PC = V2 / "02_multiscale"
sys.path[:0] = [str(PC / "src"), str(PB / "src")]

import numpy as np, pandas as pd
from scipy import stats
import scoring as S                                # stage 01 (read only)
import multiscale as M                             # stage 02 (read only)

RESULTS = Path(os.environ.get("ECHO_D_RESULTS", PD / "results"))
FIGURES = Path(os.environ.get("ECHO_D_FIGURES", PD / "figures"))
RESULTS.mkdir(parents=True, exist_ok=True)

SCALES = ["1km", "2p5km", "5km", "10km"]
REF = "2p5km"
SCENARIOS = ["equal", "literature_informed", "entropy"]
TOL = 1e-12

# ---- indicator groups (names follow scoring.py's 2.5 km convention; M.score_scale feeds other radii under these names)
POP = "norm_pop_density_2p5km"
FAMILIES = {   # D3 leave-one-family-out. Population density is handled by D2, not D3. No family is a whole component.
    "heat_demand": ["norm_heat_sum_2p5km", "norm_hdd_2024"],
    "anchors": ["norm_anchor_count_2p5km", "norm_anchor_diversity_2p5km", "norm_nearest_anchor_distance_m"],
    "urban_compactness": ["norm_urban_compactness_2p5km"],
    "road_accessibility": ["norm_major_road_density_2p5km", "norm_nearest_major_road_distance_m"],
}
SUPP_FAMILIES = {"hotmaps_only": ["norm_heat_sum_2p5km"], "hdd_only": ["norm_hdd_2024"]}   # sub-ablations of heat_demand
D2_CONFIGS = {"D2-A": [], "D2-B": "opp", "D2-C": "feas"}                                       # where population is REMOVED


# ---- data -------------------------------------------------------------------------------------------------------------
def load_table():
    return M.load_table()

def n_top(n):
    return int(math.ceil(n * 0.10))

def baseline(df, scale=REF):
    """Normalised + scored frame at `scale` (Protocol A), all three weighting scenarios, plus the fitted weights."""
    out, w = M.score_scale(df, scale)
    return out, w

def full_weights(scenario, norm):
    if scenario == "equal":
        return S.make_equal_weights(S.OPP), S.make_equal_weights(S.FEAS)
    if scenario == "literature_informed":
        return dict(S.LIT_OPP), dict(S.LIT_FEAS)
    if scenario == "entropy":
        return S.entropy_weights(norm, S.OPP), S.entropy_weights(norm, S.FEAS)
    raise KeyError(scenario)


# ---- weight manipulation ------------------------------------------------------------------------------------------------
def adjust(w, drop=(), mult=None):
    """Drop indicators and/or multiply some weights, then renormalise to sum 1 within the component.
    Surviving weights keep their relative proportions (equal -> equal on the remainder)."""
    mult = mult or {}
    unknown = (set(drop) | set(mult)) - set(S.OPP) - set(S.FEAS)
    if unknown:
        raise KeyError(unknown)
    w2 = {k: v * mult.get(k, 1.0) for k, v in w.items() if k not in drop}
    if not w2:
        raise ValueError("a component would be left empty")
    tot = sum(w2.values())
    return {k: v / tot for k, v in w2.items()}

def adjust_pair(w_opp, w_feas, drop=(), mult=None):
    d, m = set(drop), mult or {}
    return (adjust({k: v for k, v in w_opp.items()}, [x for x in d if x in w_opp], {k: v for k, v in m.items() if k in w_opp}),
            adjust({k: v for k, v in w_feas.items()}, [x for x in d if x in w_feas], {k: v for k, v in m.items() if k in w_feas}))

def d2_weights(cfg, w_opp, w_feas):
    """D2-A original; D2-B population removed from Opportunity only; D2-C removed from Feasibility only."""
    where = D2_CONFIGS[cfg]
    if where == "opp":
        return adjust(w_opp, [POP]), dict(w_feas)
    if where == "feas":
        return dict(w_opp), adjust(w_feas, [POP])
    return dict(w_opp), dict(w_feas)


# ---- scoring with arbitrary weights (original geometric aggregation, original tie rule) ---------------------------------
def score(norm, w_opp, w_feas):
    """Returns DataFrame(opp, feas, raw, rank). Calls stage 01's compute_echo_scores unchanged."""
    o = S.compute_echo_scores(norm, "x", w_opp, w_feas)
    return pd.DataFrame({"opp": o["opportunity_score_x"].to_numpy(), "feas": o["feasibility_score_x"].to_numpy(),
                         "raw": o["echo_dc_score_raw_x"].to_numpy(), "rank": o["echo_dc_rank_x"].to_numpy()})


# ---- parameterised normalisation window (D4-A) ------------------------------------------------------------------------------
def normalise_q(df, lo=0.05, hi=0.95):
    """Identical to scoring.normalise except for the clipping window (lo, hi). (0.05, 0.95) reproduces it exactly.
    lo=0, hi=1 -> quantiles are min/max, clipping is inactive, indicator = (x - min)/(max - min) after the same log1p /
    direction steps."""
    d = df.copy()
    rm = lambda s: S.robust_minmax(s, lo, hi)
    rml = lambda s: S.robust_minmax_log(s, lo, hi)
    d["norm_heat_sum_2p5km"] = rml(df["hotmaps_heat_sum_2p5km"])
    d["norm_pop_density_2p5km"] = rml(df["pop_density_2p5km"])
    d["norm_hdd_2024"] = rm(df["hdd_2024"])
    d["norm_anchor_count_2p5km"] = rml(df["anchor_count_2p5km"])
    d["norm_anchor_diversity_2p5km"] = rm(df["anchor_type_diversity_2p5km"])
    d["norm_nearest_anchor_distance_m"] = S.reverse_score(rml(df["nearest_anchor_distance_m"]))
    d["norm_major_road_density_2p5km"] = rm(df["major_road_density_2p5km_km_per_km2"])
    d["norm_nearest_major_road_distance_m"] = S.reverse_score(rml(df["nearest_major_road_distance_m"]))
    d["norm_ghsl_built_pct_2p5km"] = rm(df["built_pct_2p5km"])
    d["norm_dw_built_pct_2p5km"] = rm(df["dw_built_pct_2p5km"])
    d["norm_urban_compactness_2p5km"] = (d["norm_ghsl_built_pct_2p5km"] + d["norm_dw_built_pct_2p5km"]) / 2
    d[S.ALL_IND] = d[S.ALL_IND].replace([np.inf, -np.inf], np.nan).fillna(0)
    return d

def swap_indicators(base_norm, alt_norm, cols):
    """base_norm with the listed score-indicator columns taken from alt_norm (one indicator at a time in D4-A)."""
    o = base_norm.copy()
    o[cols] = alt_norm[cols]
    return o


# ---- ranking helpers & comparison metrics --------------------------------------------------------------------------------
def rank_desc(primary, tiebreak=None):
    """Rank 1 = largest. Ties on `primary` are broken by `tiebreak` (larger first), then by row order (stage 01 rule)."""
    primary = np.asarray(primary, float)
    n = len(primary)
    keys = (np.arange(n), -np.asarray(tiebreak, float), -primary) if tiebreak is not None else (np.arange(n), -primary)
    order = np.lexsort(keys)
    r = np.empty(n, int)
    r[order] = np.arange(1, n + 1)
    return r

def compare(base_rank, alt_rank, k=None):
    """Rank-agreement metrics of alt vs base on one facility set (ranks are 1..n on that set)."""
    b, a = np.asarray(base_rank), np.asarray(alt_rank)
    n = len(b)
    k = k or n_top(n)
    d = a - b
    ad = np.abs(d)
    tb, ta = b <= k, a <= k
    t20b, t20a = b <= 20, a <= 20
    return {"n": n, "k_top_decile": k,
            "spearman": float(stats.spearmanr(b, a)[0]), "kendall_tau_b": float(stats.kendalltau(b, a)[0]),
            "top_decile_overlap": int((tb & ta).sum()), "top_decile_overlap_share": float((tb & ta).sum() / k),
            "top20_overlap": int((t20b & t20a).sum()),
            "median_abs_drank": float(np.median(ad)), "mean_abs_drank": float(ad.mean()), "p90_abs_drank": float(np.percentile(ad, 90)),
            "p99_abs_drank": float(np.percentile(ad, 99)), "max_abs_drank": int(ad.max()),
            "share_abs_drank_ge100": float((ad >= 100).mean()), "share_abs_drank_ge200": float((ad >= 200).mean()),
            "n_enter_top_decile": int((ta & ~tb).sum()), "n_leave_top_decile": int((tb & ~ta).sum())}

def simple_rankings(o, tiebreak=True):
    """D1 comparators from a scored frame `o` (baseline(); works at any radius because M.score_scale presents radius-r
    inputs under the 2.5 km names). Heat and population are ranked on their ORIGINAL normalised indicator; because 5th/95th
    clipping creates ties (about 5% of sites at 1.0), ties are broken by the same indicator before clipping
    (log1p raw value), then by row order. tiebreak=False uses row order only (stage 01 rule)."""
    th = np.log1p(o.hotmaps_heat_sum_2p5km) if tiebreak else None
    tp = np.log1p(o.pop_density_2p5km) if tiebreak else None
    return {"heat_hotmaps": rank_desc(o.norm_heat_sum_2p5km, th), "population_density": rank_desc(o.norm_pop_density_2p5km, tp),
            "opportunity": rank_desc(o.opportunity_score_equal), "feasibility": rank_desc(o.feasibility_score_equal)}

def write_csv(df, name, index=False, digits=6):
    p = RESULTS / name
    df.round(digits).to_csv(p, index=index)
    return p

def write_json(obj, name):
    p = RESULTS / name
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    return p


# ---- protected-file manifest (Phase A/B/C must not change) -------------------------------------------------------------
PROTECTED_DIRS = ["01_baseline_scoring", "02_multiscale"]

def protected_manifest():
    out = {}
    for d in PROTECTED_DIRS:
        for p in sorted((V2 / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                out[str(p.relative_to(V2))] = h.hexdigest()
    return out
