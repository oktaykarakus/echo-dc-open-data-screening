"""C5: ranking-stability analysis for the PRIMARY (equal-weight, Protocol A) scenario."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from cfg import *
import multiscale as M, stability as ST
import scoring as S

SC = "equal"
THR = 100          # country metric threshold as specified in the brief (8.2% of N=1216)
SUBST = 200        # facility-level "substantially scale-sensitive": rank range >= 200 (16% of N; > 3x the 99th pct of stage 01 reconstruction noise)

def main():
    long = pd.read_csv(RESULTS / "multiscale_scores_long.csv")
    df = M.load_table()
    ids = df.facility_id.to_numpy(); ctry = df.country.to_numpy(); n = len(df); k = M.n_top(n)
    ranks = ST.ranks_of(long, SC)
    # ---- A. global -------------------------------------------------------------------------------------------------
    g = ST.global_metrics(ranks, k)
    for name in ["spearman", "kendall", "top_overlap", "top_jaccard", "top20_overlap"]:
        g[name].round(4).to_csv(RESULTS / f"c5_global_{name}_equal.csv")
    # ---- B. facility level ------------------------------------------------------------------------------------------
    sens = M.sensitivity_table(ids, ctry, ranks, k)
    for s in SCALES:
        if s != REF:
            sens[f"dRank_{s}_vs_ref"] = sens[f"rank_{s}"] - sens[f"rank_{REF}"]     # >0 = worse (larger rank number)
    # ---- driver attribution (single-factor substitution at the facility level) --------------------------------------
    hyb = {}                                                                          # (group, s) -> rank array
    for gname, cols in M.GROUPS.items():
        for s in SCALES:
            if s == REF: continue
            out, _ = M.score_scale(df, s, only=cols)
            hyb[(gname, s)] = out[f"echo_dc_rank_{SC}"].to_numpy()
    drv, drvmag = [], []
    for i in range(n):
        best = max(((abs(hyb[(gn, s)][i] - ranks[REF][i]), gn) for gn in M.GROUPS for s in SCALES if s != REF))
        drvmag.append(best[0]); drv.append(best[1])
    sens["largest_single_factor_shift"] = drvmag; sens["largest_single_factor_driver"] = drv
    sens.insert(2, "name", df.name.to_numpy())
    sens.to_csv(RESULTS / "c5_facility_sensitivity_equal.csv", index=False)
    cnt = {f"range_ge_{t}": int((sens.rank_range >= t).sum()) for t in (50, 100, 200, 400)}
    sub = sens[sens.rank_range >= SUBST]
    # characteristics of substantially scale-sensitive facilities
    f = df.assign(sensitive=(sens.rank_range >= SUBST).to_numpy(), rank25=ranks[REF])
    def prof(x):
        return {"n": len(x), "median_rank_2p5km": float(x.rank25.median()), "median_pop_density_2p5km": float(x.pop_density_2p5km.median()),
                "median_anchor_count_2p5km": float(x.anchor_count_2p5km.median()),
                "share_zero_anchor_1km": float((x.anchor_count_1km == 0).mean()), "share_zero_road_1km": float((x.major_road_density_1km_km_per_km2 == 0).mean()),
                "share_hotmaps_gap": float(x.hotmaps_small_buffer_missing_originally.mean()),
                "median_heat_ratio_10km_over_1km": float((x.hotmaps_heat_sum_10km / x.hotmaps_heat_sum_1km.replace(0, np.nan)).median()),
                "median_nearest_anchor_m": float(x.nearest_anchor_distance_m.median()), "median_nearest_road_m": float(x.nearest_major_road_distance_m.median())}
    profile = pd.DataFrame({"substantially_sensitive": prof(f[f.sensitive]), "other": prof(f[~f.sensitive])})
    profile.to_csv(RESULTS / "c5_sensitive_profile.csv")
    # sensitivity by rank band at 2.5 km (magnitude context: mid-table ranks are densely packed)
    band = pd.cut(ranks[REF], [0, 122, 243, 365, 486, 608, 729, 851, 972, 1094, 1216], labels=[f"D{i}" for i in range(1, 11)])
    byband = sens.assign(band=band).groupby("band", observed=True).agg(n=("rank_range", "size"), median_range=("rank_range", "median"),
                                                                     share_range_ge100=("rank_range", lambda x: (x >= THR).mean()),
                                                                     share_range_ge200=("rank_range", lambda x: (x >= SUBST).mean()))
    byband.to_csv(RESULTS / "c5_range_by_rank_decile_2p5km.csv")
    bycountry_sens = sens.assign(sub=sens.rank_range >= SUBST).groupby("country").agg(n=("sub", "size"), n_sensitive=("sub", "sum"), share_sensitive=("sub", "mean"),
                                                                                   median_range=("rank_range", "median"))
    bycountry_sens.to_csv(RESULTS / "c5_sensitive_share_by_country.csv")
    drivers = sub.largest_single_factor_driver.value_counts().rename_axis("driver").reset_index(name="n_sensitive")
    drivers["share"] = drivers.n_sensitive / len(sub); drivers.to_csv(RESULTS / "c5_sensitive_drivers.csv", index=False)
    # ---- C. country level --------------------------------------------------------------------------------------------
    rows = []
    for c, idx in pd.Series(range(n)).groupby(ctry):
        idx = idx.to_numpy(); r = {"country": c, "n": len(idx), "n_top_2p5km": int((ranks[REF][idx] <= k).sum())}
        r["share_top_2p5km"] = r["n_top_2p5km"] / len(idx)
        for s in SCALES:
            if s == REF: continue
            d = ranks[s][idx] - ranks[REF][idx]
            q1, med, q3 = np.percentile(d, [25, 50, 75])
            r.update({f"{s}_median_dRank": med, f"{s}_IQR_dRank": q3 - q1, f"{s}_q25": q1, f"{s}_q75": q3,
                      f"{s}_pct_abs_ge100": 100 * float((np.abs(d) >= THR).mean()),
                      f"{s}_n_top": int((ranks[s][idx] <= k).sum())})
            r[f"{s}_dTop_vs_2p5km"] = r[f"{s}_n_top"] - r["n_top_2p5km"]
        r["small_n_flag"] = len(idx) < 15
        rows.append(r)
    ct = pd.DataFrame(rows).sort_values("n", ascending=False)
    ct.to_csv(RESULTS / "c5_country_patterns_equal.csv", index=False)
    kw = {}
    big = [c for c in ct.country[ct.n >= 15]]
    for s in SCALES:
        if s == REF: continue
        d = ranks[s] - ranks[REF]
        groups = [d[ctry == c] for c in big]
        H, p = stats.kruskal(*groups)
        nn = sum(len(x) for x in groups)
        kw[s] = {"H": float(H), "p": float(p), "epsilon_sq": float((H - len(big) + 1) / (nn - len(big))), "countries": len(big), "n": nn}
    # ---- D. component behaviour ----------------------------------------------------------------------------------------
    comp = []
    for s in SCALES:
        t = long[long.scale == s]; t0 = long[long.scale == REF]
        for part in ("opportunity", "feasibility"):
            a, b = t[f"{part}_score_{SC}"].to_numpy(), t0[f"{part}_score_{SC}"].to_numpy()
            ra, rb = stats.rankdata(-a, method="ordinal"), stats.rankdata(-b, method="ordinal")
            comp.append({"component": part, "scale": s, "spearman_vs_2p5km": ST.spearman(a, b), "mean_abs_dScore": float(np.abs(a - b).mean()),
                         "mean_score": float(a.mean()), "sd_score": float(a.std(ddof=1)), "mean_abs_dRank": float(np.abs(ra - rb).mean())})
    pd.DataFrame(comp).to_csv(RESULTS / "c5_component_scores_vs_2p5km.csv", index=False)
    indrows = []
    for col in ["norm_heat_sum", "norm_pop_density", "norm_anchor_count", "norm_anchor_diversity", "norm_urban_compactness", "norm_major_road_density"]:
        for s in SCALES:
            if s == REF: continue
            a, b = long.loc[long.scale == s, col].to_numpy(), long.loc[long.scale == REF, col].to_numpy()
            indrows.append({"indicator": col, "scale": s, "spearman_vs_2p5km": ST.spearman(a, b), "mean_abs_dNorm": float(np.abs(a - b).mean()),
                            "mean_norm": float(a.mean()), "share_at_1": float((a >= 1).mean()), "share_at_0": float((a <= 0).mean())})
    pd.DataFrame(indrows).to_csv(RESULTS / "c5_indicator_level_vs_2p5km.csv", index=False)
    attr = []
    for gname, cols in M.GROUPS.items():
        for s in SCALES:
            if s == REF: continue
            only = hyb[(gname, s)]                                                      # only this group at radius s
            rest, _ = M.score_scale(df, s, only=[c for c in S_ALL_SCALE_DEP if c not in cols])     # everything at s EXCEPT group (group kept at 2.5 km)
            rest = rest[f"echo_dc_rank_{SC}"].to_numpy()
            d1, d2 = np.abs(only - ranks[REF]), np.abs(rest - ranks[REF])
            attr.append({"group": gname, "scale": s, "only_group_at_s_rho": ST.spearman(only, ranks[REF]), "only_group_mean_abs_dRank": float(d1.mean()),
                         "only_group_n_ge100": int((d1 >= THR).sum()), "all_but_group_rho": ST.spearman(rest, ranks[REF]),
                         "all_but_group_mean_abs_dRank": float(d2.mean()), "all_at_s_mean_abs_dRank": float(np.abs(ranks[s] - ranks[REF]).mean()),
                         "all_at_s_rho": ST.spearman(ranks[s], ranks[REF])})
            attr[-1]["contribution_reduction_in_mean_abs_dRank"] = attr[-1]["all_at_s_mean_abs_dRank"] - attr[-1]["all_but_group_mean_abs_dRank"]
    pd.DataFrame(attr).to_csv(RESULTS / "c5_factor_attribution_equal.csv", index=False)
    # ---- Hotmaps-gap sensitivity ----------------------------------------------------------------------------------------
    keep = ~df.hotmaps_small_buffer_missing_originally.to_numpy()
    gapchk = {s: {"rho_all": ST.spearman(ranks[s], ranks[REF]), "rho_excl_15_gap_sites": ST.spearman(ranks[s][keep], ranks[REF][keep])} for s in SCALES if s != REF}
    # ---- summary json ------------------------------------------------------------------------------------------------------
    pb = Path(__file__).resolve().parents[2] / "01_baseline_scoring"
    rec = pd.read_csv(pb / "data" / "recovered_literature_scores.csv").merge(pd.read_csv(B0QA)[["facility_id", "echo_dc_rank_literature_informed"]], on="facility_id")
    nz = (rec.echo_dc_rank_literature_informed - rec.lit_rank_recovered).abs()
    noise = {"what": "stage 01 B0-QA literature rank vs recovered original-era literature rank", "median": float(nz.median()), "p90": float(nz.quantile(.9)),
             "p99": float(nz.quantile(.99)), "max": int(nz.max()), "share_ge100": float((nz >= 100).mean()), "spearman_phaseB": 0.9990}
    summ = {"reconstruction_noise_floor": noise, "k_top_decile": k, "n": n, "threshold_moderate": THR, "threshold_substantial": SUBST,
            "n_top_all_four": g["n_top_all_four"], "n_top_any_scale": g["n_top_any_scale"],
            "spearman_vs_2p5km": {s: g["spearman"].loc[REF, s] for s in SCALES if s != REF},
            "top20_all_four": len(set.intersection(*[M.top_set(ranks[s], 20) for s in SCALES])),
            "rank_range_counts": cnt, "n_sensitive": len(sub), "share_sensitive": len(sub) / n,
            "median_rank_range": float(sens.rank_range.median()), "median_abs_dRank": {s: float(np.median(np.abs(ranks[s] - ranks[REF]))) for s in SCALES if s != REF},
            "kruskal_country_effect_on_dRank": kw, "hotmaps_gap_check": gapchk,
            "top_status_counts": {s: sens[f"top_status_{s}_vs_ref"].value_counts().to_dict() for s in SCALES if s != REF},
            "top_pattern_counts": sens.top_pattern.value_counts().to_dict()}
    json.dump(summ, open(RESULTS / "c5_summary_equal.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))

S_ALL_SCALE_DEP = list(M.SCALE_DEPENDENT)

if __name__ == "__main__":
    main()
