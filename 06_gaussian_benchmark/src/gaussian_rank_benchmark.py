#!/usr/bin/env python3
"""G1-B / B1 — Gaussian-copula dependence benchmark for four-radius shortlist statistics.

DESCRIPTIVE benchmark only. Question: given the observed four-radius Spearman matrix, how many
facilities would be top-decile at all radii / have rank range >= 200 / etc. if the four rankings were
exchangeable draws from a 4-variate Gaussian dependence with that matrix?  It does NOT model geography,
clustering, country composition, heterogeneity across facilities, or heat-reuse suitability.
Nothing here changes any ECHO-DC score.  Inputs are read-only.

Run:  /usr/bin/python3 gaussian_rank_benchmark.py            (outputs -> ../results)
      ECHO_G1B_RESULTS=/some/dir /usr/bin/python3 gaussian_rank_benchmark.py   (determinism check)
"""
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
V2 = PHASE.parent
TMP = PHASE / ".tmp"
TMP.mkdir(exist_ok=True)
for k in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[k] = str(TMP)
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

# ----------------------------------------------------------------------------- configuration
CONFIG = {
    "seed": 20261107,
    "n_replicates": 2000,        # bounded; stability checked by split halves (see meta)
    "top_decile_k": 122,         # ceil(0.10 * 1216), as Phases B-E
    "top20_k": 20,
    "range_threshold": 200,      # stage 02 'substantially sensitive'
    "extra_range_thresholds": [50, 100, 400],
    "radii": ["1km", "2p5km", "5km", "10km"],
    "rank_rule": "rank 1 = highest latent value; ties (probability zero) broken by row order via stable sort",
}
OUT = Path(os.environ.get("ECHO_G1B_RESULTS", PHASE / "results"))
OUT.mkdir(parents=True, exist_ok=True)

C_LONG = V2 / "02_multiscale/results/multiscale_scores_long.csv"
C_FAC = V2 / "02_multiscale/results/c5_facility_sensitivity_equal.csv"
C_DEC = V2 / "02_multiscale/results/c5_range_by_rank_decile_2p5km.csv"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- 1. inputs and alignment
long = pd.read_csv(C_LONG)
fac = pd.read_csv(C_FAC)
n = fac.shape[0]
assert n == 1216 and fac.facility_id.is_unique, "expected 1,216 unique facility IDs"
piv = long.pivot(index="facility_id", columns="radius_km", values="echo_dc_rank_equal")
piv.columns = ["1km" if c == 1 else "2p5km" if c == 2.5 else "5km" if c == 5 else "10km" for c in piv.columns]
piv = piv.loc[fac.facility_id, CONFIG["radii"]]          # aligned to c5 order
c5 = fac.set_index("facility_id").loc[piv.index, [f"rank_{r}" for r in CONFIG["radii"]]]
c5.columns = CONFIG["radii"]
check = {
    "n_facilities": int(n),
    "ids_unique": bool(fac.facility_id.is_unique),
    "ids_identical_across_radii": bool(
        all(set(long.loc[long.radius_km == r, "facility_id"]) == set(fac.facility_id) for r in (1, 2.5, 5, 10))
    ),
    "rows_per_radius": {str(r): int((long.radius_km == r).sum()) for r in (1, 2.5, 5, 10)},
    "ranks_are_permutations_1_to_n": bool(all(sorted(piv[c]) == list(range(1, n + 1)) for c in piv)),
    "c5_ranks_equal_long_table_ranks": bool((piv.values == c5.values).all()),
}
assert all(v for k, v in check.items() if isinstance(v, bool)), check
R_obs = piv.values.astype(int)                           # (n, 4) observed ranks


def stats(R, k=CONFIG["top_decile_k"], k20=CONFIG["top20_k"]):
    top = R <= k
    rng_ = R.max(1) - R.min(1)
    out = {
        "top_decile_all4": int(top.all(1).sum()),
        "top_decile_at_least_once": int(top.any(1).sum()),
        "top20_all4": int((R <= k20).all(1).sum()),
        "range_ge_200": int((rng_ >= CONFIG["range_threshold"]).sum()),
    }
    for t in CONFIG["extra_range_thresholds"]:
        out[f"range_ge_{t}"] = int((rng_ >= t).sum())
    names = CONFIG["radii"]
    for i in range(4):
        for j in range(i + 1, 4):
            out[f"overlap_top_decile_{names[i]}_vs_{names[j]}"] = int((top[:, i] & top[:, j]).sum())
    out["rank_range_median"] = float(np.median(rng_))
    out["rank_range_p90"] = float(np.percentile(rng_, 90))
    out["rank_range_p99"] = float(np.percentile(rng_, 99))
    out["max_rank_displacement"] = int(rng_.max())
    return out


# ----------------------------------------------------------------------------- 2. dependence structure
rho = spearmanr(R_obs).correlation                       # 4x4 Spearman of observed ranks
assert np.allclose(rho, rho.T) and np.allclose(np.diag(rho), 1)
# For a bivariate Gaussian with Pearson r, Spearman rho_S = (6/pi) * arcsin(r/2)  =>  r = 2 sin(pi*rho_S/6).
r_latent = 2.0 * np.sin(np.pi * rho / 6.0)
np.fill_diagonal(r_latent, 1.0)
eig = np.linalg.eigvalsh(r_latent)
psd = bool(eig.min() > -1e-12)
pd_strict = bool(eig.min() > 1e-12)
assert psd, f"latent correlation matrix not PSD: min eigenvalue {eig.min()}"
L = np.linalg.cholesky(r_latent) if pd_strict else None
assert L is not None, "matrix PSD but singular; Cholesky not applicable (would need eigen-decomposition)"

# ----------------------------------------------------------------------------- 3. simulation
rng = np.random.default_rng(CONFIG["seed"])
R_rep = CONFIG["n_replicates"]
rows = []
for _ in range(R_rep):
    z = rng.standard_normal((n, 4)) @ L.T                # latent scores with correlation r_latent
    ranks = np.empty((n, 4), dtype=int)
    for c in range(4):
        order = np.argsort(-z[:, c], kind="stable")      # rank 1 = highest latent score
        ranks[order, c] = np.arange(1, n + 1)
    rows.append(stats(ranks))
sim = pd.DataFrame(rows)

# decile diagnostic: median rank range by decile of the 2.5 km rank (descriptive; not a different null)
dec_edges = np.linspace(0, n, 11)
j25 = CONFIG["radii"].index("2p5km")


def decile_median_range(R):
    rng_ = R.max(1) - R.min(1)
    d = np.minimum((R[:, j25] - 1) * 10 // n, 9)
    return np.array([np.median(rng_[d == q]) for q in range(10)])


rng2 = np.random.default_rng(CONFIG["seed"] + 1)
dec_rows = []
for _ in range(300):
    z = rng2.standard_normal((n, 4)) @ L.T
    ranks = np.empty((n, 4), dtype=int)
    for c in range(4):
        order = np.argsort(-z[:, c], kind="stable")
        ranks[order, c] = np.arange(1, n + 1)
    dec_rows.append(decile_median_range(ranks))
dec_rows = np.array(dec_rows)
dec_obs = decile_median_range(R_obs)
dec_tab = pd.DataFrame({
    "rank_decile_at_2p5km": np.arange(1, 11),
    "observed_median_range": dec_obs,
    "benchmark_mean_of_median_range": dec_rows.mean(0),
    "benchmark_p05": np.percentile(dec_rows, 5, axis=0),
    "benchmark_p95": np.percentile(dec_rows, 95, axis=0),
}).round(2)

# ----------------------------------------------------------------------------- 4. summary table
obs = stats(R_obs)
recs = []
for key, o in obs.items():
    s = sim[key].to_numpy(dtype=float)
    below, equal = (s < o).mean(), (s == o).mean()
    pct = 100 * (below + 0.5 * equal)                   # mid-rank percentile of observed within simulated values
    lo, hi = np.percentile(s, 5), np.percentile(s, 95)
    recs.append({
        "statistic": key, "observed": o,
        "benchmark_mean": round(s.mean(), 2), "benchmark_median": round(float(np.median(s)), 2),
        "benchmark_sd": round(s.std(ddof=1), 2),
        "central90_lo": round(float(lo), 2), "central90_hi": round(float(hi), 2),
        "observed_percentile_in_simulation": round(pct, 1),
        "observed_inside_central90": bool(lo <= o <= hi),
        "observed_minus_mean": round(o - s.mean(), 2),
    })
summary = pd.DataFrame(recs)

# MC stability: split halves of replicates + MC standard error of the mean
half = R_rep // 2
mc = {}
for key in ("top_decile_all4", "range_ge_200", "top20_all4", "top_decile_at_least_once"):
    a, b = sim[key].iloc[:half].mean(), sim[key].iloc[half:].mean()
    mc[key] = {"mean_first_half": round(float(a), 3), "mean_second_half": round(float(b), 3),
               "mc_standard_error_of_mean": round(float(sim[key].std(ddof=1) / np.sqrt(R_rep)), 3)}

meta = {
    "config": CONFIG,
    "software": {"python": platform.python_version(), "numpy": np.__version__,
                 "scipy": scipy.__version__, "pandas": pd.__version__, "platform": platform.platform()},
    "inputs_sha256": {str(p.relative_to(V2)): sha256(p) for p in (C_LONG, C_FAC, C_DEC)},
    "alignment_checks": check,
    "observed_spearman_matrix": np.round(rho, 6).tolist(),
    "latent_gaussian_correlation": np.round(r_latent, 6).tolist(),
    "transformation": "r = 2 sin(pi * rho_S / 6); exact for bivariate Gaussian (rho_S = (6/pi) arcsin(r/2))",
    "latent_min_eigenvalue": float(eig.min()), "psd": psd, "strictly_positive_definite": pd_strict,
    "mc_stability": mc,
    "limitations": [
        "Descriptive dependence benchmark, not a spatially explicit or causal null model.",
        "Facilities exchangeable: no geography, clustering, country composition, or per-facility heterogeneity.",
        "Gaussian copula fitted to pairwise Spearman only; real dependence may be non-elliptical or tail-dependent.",
        "Simulation intervals describe Monte Carlo variation of the benchmark, not confidence intervals for the real population.",
        "Says nothing about validity or heat-reuse suitability of any ranking.",
    ],
}
summary.to_csv(OUT / "benchmark_summary.csv", index=False)
dec_tab.to_csv(OUT / "benchmark_decile_diagnostic.csv", index=False)
(OUT / "benchmark_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
print(summary.to_string(index=False))
print(dec_tab.to_string(index=False))
print("latent min eig", eig.min(), "| MC", mc)
