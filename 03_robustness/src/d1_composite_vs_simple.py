"""D1: ECHO-DC (equal weights, 2.5 km) against single-indicator and single-component rankings.
Measures redundancy/complementarity of RANKINGS only; says nothing about external validity."""
from d_lib import *

def main():
    df = load_table(); o, _ = baseline(df); n = len(df); k = n_top(n)
    comp_rank = o.echo_dc_rank_equal.to_numpy(); comp_raw = o.echo_dc_score_raw_equal.to_numpy()
    simple = simple_rankings(o); simple_rowtie = simple_rankings(o, tiebreak=False)
    prim = {"heat_hotmaps": o.norm_heat_sum_2p5km.to_numpy(), "population_density": o.norm_pop_density_2p5km.to_numpy(),
            "opportunity": o.opportunity_score_equal.to_numpy(), "feasibility": o.feasibility_score_equal.to_numpy()}
    rows, xt, lists = [], [], []
    for name, r in simple.items():
        m = compare(comp_rank, r, k)
        m["comparator"] = name
        m["pearson_score_r"] = float(stats.pearsonr(comp_raw, prim[name])[0])
        v = pd.Series(prim[name])
        m["n_distinct_indicator_values"] = int(v.nunique()); m["share_at_max"] = float((v == v.max()).mean()); m["share_at_min"] = float((v == v.min()).mean())
        m["largest_tie_group"] = int(v.value_counts().iloc[0])
        mr = compare(comp_rank, simple_rowtie[name], k)                       # tie-handling sensitivity (row-order only)
        m["top_decile_overlap_rowtie"] = mr["top_decile_overlap"]; m["top20_overlap_rowtie"] = mr["top20_overlap"]; m["spearman_rowtie"] = mr["spearman"]
        tc, ts = comp_rank <= k, r <= k
        m["both_top_decile"] = int((tc & ts).sum()); m["composite_only_top_decile"] = int((tc & ~ts).sum())
        m["simple_only_top_decile"] = int((~tc & ts).sum()); m["neither"] = int((~tc & ~ts).sum())
        co = tc & ~ts
        m["composite_only_median_simple_rank"] = float(np.median(r[co])) if co.any() else np.nan
        m["composite_only_share_simple_rank_gt_2k"] = float((r[co] > 2 * k).mean()) if co.any() else np.nan   # clearly outside, not marginal
        m["composite_only_share_FR"] = float((o.country.to_numpy()[co] == "FR").mean()) if co.any() else np.nan
        rows.append(m)
        for i in np.flatnonzero(co):
            lists.append({"comparator": name, "facility_id": o.facility_id[i], "country": o.country[i], "name": o["name"][i],
                          "composite_rank": int(comp_rank[i]), "simple_rank": int(r[i]), "rank_gap": int(r[i] - comp_rank[i])})
    # union of the two simple demand/population screens, and a linear-reproduction check
    th, tp, tc = simple["heat_hotmaps"] <= k, simple["population_density"] <= k, comp_rank <= k
    X = np.column_stack([np.ones(n), prim["heat_hotmaps"], prim["population_density"]])
    beta, *_ = np.linalg.lstsq(X, comp_raw, rcond=None); r2 = 1 - ((comp_raw - X @ beta) ** 2).sum() / ((comp_raw - comp_raw.mean()) ** 2).sum()
    Xr = np.column_stack([np.ones(n), simple["heat_hotmaps"], simple["population_density"]])
    br, *_ = np.linalg.lstsq(Xr, comp_rank, rcond=None); r2r = 1 - ((comp_rank - Xr @ br) ** 2).sum() / ((comp_rank - comp_rank.mean()) ** 2).sum()
    # same two predictors plus both components' non-shared parts is circular; so also report how composite relates to its parts
    union = {"k": k, "composite_top_decile_in_union_heat_or_pop_top_decile": int((tc & (th | tp)).sum()),
             "composite_top_decile_in_neither": int((tc & ~(th | tp)).sum()), "union_size": int((th | tp).sum()),
             "composite_top_decile_in_heat_and_pop": int((tc & th & tp).sum()),
             "R2_composite_score_on_heat_pop_indicators": float(r2), "R2_composite_rank_on_heat_pop_ranks": float(r2r),
             "spearman_opportunity_vs_feasibility": float(stats.spearmanr(prim["opportunity"], prim["feasibility"])[0]),
             "spearman_heat_vs_population_indicator": float(stats.spearmanr(prim["heat_hotmaps"], prim["population_density"])[0])}
    write_csv(pd.DataFrame(rows).set_index("comparator").reset_index(), "d1_composite_vs_simple_summary.csv")
    write_csv(pd.DataFrame(lists), "d1_composite_only_sites.csv")
    fr = pd.DataFrame({"facility_id": o.facility_id, "country": o.country, "composite_rank": comp_rank,
                       **{f"rank_{k_}": v for k_, v in simple.items()}})
    for k_ in simple: fr[f"gap_{k_}"] = fr[f"rank_{k_}"] - fr.composite_rank
    write_csv(fr, "d1_facility_ranks.csv")
    # country composition of the composite top decile and of each simple top decile
    comp_tab = pd.DataFrame({"composite": pd.Series(o.country[comp_rank <= k]).value_counts(),
                             **{k_: pd.Series(o.country[r <= k]).value_counts() for k_, r in simple.items()}}).fillna(0).astype(int)
    comp_tab["n_facilities"] = pd.Series(o.country).value_counts()
    write_csv(comp_tab.sort_values("composite", ascending=False).rename_axis("country").reset_index(), "d1_top_decile_by_country.csv")
    write_json(union, "d1_union_and_linear_reproduction.json")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    s = pd.DataFrame(rows).set_index("comparator")
    print(s[["spearman", "kendall_tau_b", "pearson_score_r", "top_decile_overlap", "top20_overlap", "median_abs_drank", "p90_abs_drank", "max_abs_drank",
             "share_abs_drank_ge100", "both_top_decile", "composite_only_top_decile", "composite_only_median_simple_rank",
             "composite_only_share_simple_rank_gt_2k", "composite_only_share_FR"]].round(3).T)
    print(s[["n_distinct_indicator_values", "share_at_max", "share_at_min", "largest_tie_group", "top_decile_overlap_rowtie", "top20_overlap_rowtie", "spearman_rowtie"]].round(3).T)
    print(json.dumps(union, indent=1)); print(comp_tab.sort_values("composite", ascending=False).head(8))

if __name__ == "__main__":
    main()
