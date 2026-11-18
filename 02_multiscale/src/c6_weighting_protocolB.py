"""C6 (weighting-scenario interaction, concise) and the Protocol-B (fixed 2.5 km reference) summary for Feasibility.
Entropy is reported two ways: 'entropy' = weights RE-FITTED at each scale (Protocol A, as the original does at 2.5 km);
'entropy_ref' = weights HELD at the 2.5 km values, so only indicator values change between scales."""
import json
import numpy as np, pandas as pd
from scipy import stats
from cfg import *
import multiscale as M, stability as ST

SCEN = ["equal", "literature_informed", "entropy", "entropy_ref"]
NB = 1000

def main():
    long = pd.read_csv(RESULTS / "multiscale_scores_long.csv")
    df = M.load_table(); n = len(df); k = M.n_top(n)
    R = {sc: ST.ranks_of(long, sc) for sc in SCEN}; RAW = {sc: ST.raws_of(long, sc) for sc in SCEN}
    rows, boot = [], {}
    for sc in SCEN:
        g = ST.global_metrics(R[sc], k)
        sens = M.sensitivity_table(df.facility_id, df.country, R[sc], k)
        r = {"scenario": sc, "n_top_all_four": g["n_top_all_four"], "n_top_any_scale": g["n_top_any_scale"],
             "top20_all_four": len(set.intersection(*[M.top_set(R[sc][s], 20) for s in SCALES])),
             "n_range_ge100": int((sens.rank_range >= 100).sum()), "n_range_ge200": int((sens.rank_range >= 200).sum()),
             "median_rank_range": float(sens.rank_range.median())}
        for s in SCALES:
            if s == REF: continue
            b = ST.boot_spearman(RAW[sc][REF], RAW[sc][s], NB); boot[(sc, s)] = b
            r[f"rho_{s}"] = g["spearman"].loc[REF, s]; r[f"rho_{s}_ci_lo"], r[f"rho_{s}_ci_hi"] = np.percentile(b, [2.5, 97.5])
            r[f"tau_{s}"] = g["kendall"].loc[REF, s]
            r[f"top_overlap_{s}"] = int(g["top_overlap"].loc[REF, s]); r[f"top20_overlap_{s}"] = int(g["top20_overlap"].loc[REF, s])
            r[f"median_abs_dRank_{s}"] = float(np.median(np.abs(R[sc][s] - R[sc][REF])))
        r["rho_1km_vs_10km"] = g["spearman"].loc["1km", "10km"]
        rows.append(r)
    pd.DataFrame(rows).to_csv(RESULTS / "c6_weighting_scenarios_summary.csv", index=False)
    # paired bootstrap of rho differences vs equal (same resamples => paired)
    rng_rows = []
    for sc in SCEN[1:]:
        for s in SCALES:
            if s == REF: continue
            # boot_spearman uses the same default seed for every call -> identical resamples -> paired differences
            d = boot[(sc, s)] - boot[("equal", s)]
            rng_rows.append({"scenario": sc, "scale": s, "rho_diff_vs_equal": float(np.mean(d)), "ci_lo": float(np.percentile(d, 2.5)),
                             "ci_hi": float(np.percentile(d, 97.5)), "ci_excludes_0": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)})
    pd.DataFrame(rng_rows).to_csv(RESULTS / "c6_paired_bootstrap_rho_diff_vs_equal.csv", index=False)
    # agreement between scenarios, per scale
    cross = []
    for s in SCALES:
        for a, b in [("equal", "literature_informed"), ("equal", "entropy"), ("equal", "entropy_ref"), ("literature_informed", "entropy")]:
            cross.append({"scale": s, "pair": f"{a} vs {b}", "spearman": ST.spearman(R[a][s], R[b][s]),
                          "top_decile_overlap": len(M.top_set(R[a][s], k) & M.top_set(R[b][s], k))})
    pd.DataFrame(cross).to_csv(RESULTS / "c6_cross_scenario_agreement.csv", index=False)
    # jointly robust priority sites
    tops = [M.top_set(R[sc][s], k) for sc in SCEN[:3] for s in SCALES]
    robust12 = set.intersection(*tops); union12 = set.union(*tops)
    fam = pd.DataFrame({"facility_id": df.facility_id, "country": df.country})
    cnt = np.zeros(n, int)
    for t in tops:
        cnt[list(t)] += 1
    fam["n_of_12_scale_x_scenario_in_top_decile"] = cnt
    fam.to_csv(RESULTS / "c6_priority_site_robustness_12cells.csv", index=False)
    robust = {"n_top_decile_in_all_12_cells": len(robust12), "n_in_any_cell": len(union12),
              "countries_of_robust": df.country.iloc[sorted(robust12)].value_counts().to_dict()}
    # entropy weights by scale
    w = pd.read_csv(RESULTS / "weights_by_scale.csv"); we = w[w.scenario == "entropy"].set_index("scale")
    wcols = [c for c in we.columns if c.startswith(("opp:", "feas:"))]
    wsum = we[wcols].T.rename_axis("weight").reset_index(); wsum.to_csv(RESULTS / "c6_entropy_weights_by_scale.csv", index=False)
    wshift = {s: float(np.abs(we.loc[s, wcols].astype(float) - we.loc[REF, wcols].astype(float)).max()) for s in SCALES if s != REF}

    # ---------------- Protocol B (Feasibility only) ----------------
    bl = pd.read_csv(RESULTS / "protocolB_feasibility_long.csv")
    A = long.set_index(["scale"])
    brow, indrow = [], []
    for sc, acol in [("equal", "equal"), ("literature_informed", "literature_informed"), ("entropy_ref", "entropy_ref")]:
        fa = {s: long.loc[long.scale == s, f"feasibility_score_{acol}"].to_numpy() for s in SCALES}
        fb = {s: bl.loc[bl.scale == s, f"feasibility_B_{sc}"].to_numpy() for s in SCALES}
        for s in SCALES:
            if s == REF: continue
            brow.append({"scenario": sc, "scale": s,
                         "protoA_rho_feas_vs_2p5km": ST.spearman(fa[s], fa[REF]), "protoB_rho_feas_vs_2p5km": ST.spearman(fb[s], fb[REF]),
                         "rho_A_vs_B_same_scale": ST.spearman(fa[s], fb[s]),
                         "mean_feas_protoA": float(fa[s].mean()), "mean_feas_protoB": float(fb[s].mean()), "mean_feas_2p5km": float(fa[REF].mean()),
                         "protoB_top_decile_feas_overlap_vs_2p5km": len(M.top_set(stats.rankdata(-fb[s], method="ordinal"), k) & M.top_set(stats.rankdata(-fb[REF], method="ordinal"), k)),
                         "protoA_top_decile_feas_overlap_vs_2p5km": len(M.top_set(stats.rankdata(-fa[s], method="ordinal"), k) & M.top_set(stats.rankdata(-fa[REF], method="ordinal"), k))})
    pd.DataFrame(brow).to_csv(RESULTS / "c2_protocolB_feasibility_summary.csv", index=False)
    for ind in ["norm_ghsl_built_pct", "norm_dw_built_pct", "norm_major_road_density", "norm_pop_density"]:
        for s in SCALES:
            x = bl.loc[bl.scale == s, ind].to_numpy()
            indrow.append({"indicator": ind, "scale": s, "mean_norm_protoB": float(x.mean()), "share_clipped_at_1": float((x >= 1).mean()), "share_clipped_at_0": float((x <= 0).mean())})
    pd.DataFrame(indrow).to_csv(RESULTS / "c2_protocolB_indicator_levels.csv", index=False)
    json.dump({"robust_priority_sites": robust, "entropy_max_abs_weight_shift_vs_2p5km": wshift, "n_bootstrap": NB, "seed": 20260707},
              open(RESULTS / "c6_summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(pd.DataFrame(rows).round(3).T.to_string()); print(pd.DataFrame(rng_rows).round(4).to_string(index=False))
    print(pd.DataFrame(cross).round(3).to_string(index=False)); print(robust); print(wshift)
    print(pd.DataFrame(brow).round(3).to_string(index=False)); print(pd.DataFrame(indrow).round(3).to_string(index=False))
    print(wsum.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
