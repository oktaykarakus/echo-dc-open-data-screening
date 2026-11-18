"""D0: verify the B0-QA baseline before any stage 03 analysis. Stops (non-zero exit) on any discrepancy.
Also freezes a SHA-256 manifest of Phase A/B/C files (written once; later runs compare against it)."""
import json, sys
from d_lib import *

def independent_norm(df):
    """Hand-written numpy re-implementation of the nine V1 indicators (does not call scoring.py)."""
    def q(x, p): return np.nanpercentile(x, p * 100)          # linear interpolation == pandas default
    def mm(x):
        lo, hi = q(x, .05), q(x, .95)
        return np.clip((x - lo) / (hi - lo), 0, 1)
    L = lambda c: np.log1p(df[c].to_numpy(float))
    A = lambda c: df[c].to_numpy(float)
    o = {"norm_heat_sum_2p5km": mm(L("hotmaps_heat_sum_2p5km")), "norm_pop_density_2p5km": mm(L("pop_density_2p5km")),
         "norm_hdd_2024": mm(A("hdd_2024")), "norm_anchor_count_2p5km": mm(L("anchor_count_2p5km")),
         "norm_anchor_diversity_2p5km": mm(A("anchor_type_diversity_2p5km")),
         "norm_nearest_anchor_distance_m": 1 - mm(L("nearest_anchor_distance_m")),
         "norm_major_road_density_2p5km": mm(A("major_road_density_2p5km_km_per_km2")),
         "norm_nearest_major_road_distance_m": 1 - mm(L("nearest_major_road_distance_m")),
         "norm_urban_compactness_2p5km": (mm(A("built_pct_2p5km")) + mm(A("dw_built_pct_2p5km"))) / 2}
    return {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in o.items()}

def main():
    fails, rep = [], {}
    df = load_table()
    ref = pd.read_csv(PB / "results" / "scores_B0QA.csv")
    ids0 = pd.read_csv(PB / "data" / "facility_ids_original.csv")
    long = pd.read_csv(PC / "results" / "multiscale_scores_long.csv")
    # --- identifiers
    rep["n_rows"] = len(df); rep["n_unique_ids"] = int(df.facility_id.nunique()); rep["n_countries"] = int(df.country.nunique())
    if not (len(df) == 1216 and df.facility_id.is_unique and df.country.nunique() == 20): fails.append("cohort size/uniqueness")
    if set(df.facility_id) != set(ids0.facility_id) or len(ids0) != 1216: fails.append("IDs differ from facility_ids_original.csv")
    if not (df.facility_id.to_numpy() == ref.facility_id.to_numpy()).all(): fails.append("row order differs from scores_B0QA.csv")
    # --- exact 2.5 km reproduction
    out, w = baseline(df)
    gate = {}
    for sc in SCENARIOS:
        for k in ["opportunity_score", "feasibility_score", "echo_dc_score_raw", "echo_dc_score"]:
            gate[f"{k}_{sc}"] = float(np.abs(out[f"{k}_{sc}"].to_numpy() - ref[f"{k}_{sc}"].to_numpy()).max())
        gate[f"rank_mismatch_{sc}"] = int((out[f"echo_dc_rank_{sc}"].to_numpy() != ref[f"echo_dc_rank_{sc}"].to_numpy()).sum())
    for c in S.ALL_IND:
        gate[c] = float(np.abs(out[c].to_numpy() - ref[c].to_numpy()).max())
    rep["b0qa_reproduction_max_abs_diff"] = max(v for k, v in gate.items() if not k.startswith("rank_mismatch"))
    rep["b0qa_rank_mismatches"] = {sc: gate[f"rank_mismatch_{sc}"] for sc in SCENARIOS}
    if rep["b0qa_reproduction_max_abs_diff"] > TOL or any(rep["b0qa_rank_mismatches"].values()): fails.append("B0-QA not reproduced")
    # --- vs the stage 02 long table at 2.5 km
    t = long[long.scale == REF].reset_index(drop=True)
    rep["phaseC_2p5km_rank_mismatch_equal"] = int((t.echo_dc_rank_equal.to_numpy() != out.echo_dc_rank_equal.to_numpy()).sum())
    rep["phaseC_2p5km_max_abs_raw_diff_equal"] = float(np.abs(t.echo_dc_score_raw_equal.to_numpy() - out.echo_dc_score_raw_equal.to_numpy()).max())
    if rep["phaseC_2p5km_rank_mismatch_equal"] or rep["phaseC_2p5km_max_abs_raw_diff_equal"] > TOL: fails.append("stage 02 2.5 km table differs")
    # --- independent re-implementation of transformations / directions
    ind = independent_norm(df)
    rep["independent_norm_max_abs_diff"] = {k: float(np.abs(v - out[k].to_numpy()).max()) for k, v in ind.items()}
    if max(rep["independent_norm_max_abs_diff"].values()) > 1e-12: fails.append("independent normalisation differs")
    raw = {"norm_heat_sum_2p5km": "hotmaps_heat_sum_2p5km", "norm_pop_density_2p5km": "pop_density_2p5km", "norm_hdd_2024": "hdd_2024",
           "norm_anchor_count_2p5km": "anchor_count_2p5km", "norm_anchor_diversity_2p5km": "anchor_type_diversity_2p5km",
           "norm_nearest_anchor_distance_m": "nearest_anchor_distance_m", "norm_major_road_density_2p5km": "major_road_density_2p5km_km_per_km2",
           "norm_nearest_major_road_distance_m": "nearest_major_road_distance_m", "norm_urban_compactness_2p5km": "built_pct_2p5km"}
    rep["direction_spearman_norm_vs_raw"] = {k: float(stats.spearmanr(out[k], df[v])[0]) for k, v in raw.items()}
    for k, r in rep["direction_spearman_norm_vs_raw"].items():
        rev = k in ("norm_nearest_anchor_distance_m", "norm_nearest_major_road_distance_m")
        if (rev and r > -0.9) or (not rev and r < 0.9): fails.append(f"direction {k}")
    # --- baseline structure
    rep["top_decile_k"] = n_top(len(df)); rep["rank_is_permutation"] = bool(sorted(out.echo_dc_rank_equal) == list(range(1, 1217)))
    rep["finite_scores"] = bool(np.isfinite(out[[c for c in out.columns if c.startswith(("echo_dc_score", "opportunity", "feasibility"))]].to_numpy(float)).all())
    if not (rep["top_decile_k"] == 122 and rep["rank_is_permutation"] and rep["finite_scores"]): fails.append("rank/score validity")
    # --- determinism of the baseline
    out2, _ = baseline(df)
    rep["deterministic_rerun_identical"] = bool(np.array_equal(out.echo_dc_score_raw_equal.to_numpy(), out2.echo_dc_score_raw_equal.to_numpy()))
    if not rep["deterministic_rerun_identical"]: fails.append("non-deterministic")
    # --- frozen manifest of protected files
    mp = RESULTS / "d0_protected_manifest.json"
    cur = protected_manifest()
    if mp.exists():
        old = json.load(open(mp))
        rep["protected_files_changed"] = sorted(k for k in set(old) | set(cur) if old.get(k) != cur.get(k))
        if rep["protected_files_changed"]: fails.append("protected Phase A/B/C files changed")
    else:
        json.dump(cur, open(mp, "w"), indent=0, sort_keys=True)
        rep["protected_files_changed"] = []
    rep["protected_files_n"] = len(cur)
    rep["status"] = "PASS" if not fails else "FAIL"; rep["failures"] = fails
    write_json(rep, "d0_baseline_verification.json")
    print(json.dumps({k: rep[k] for k in ("status", "failures", "n_rows", "b0qa_reproduction_max_abs_diff", "b0qa_rank_mismatches",
                                         "independent_norm_max_abs_diff", "protected_files_n", "protected_files_changed")}, indent=1))
    if fails: sys.exit(1)

if __name__ == "__main__":
    main()
