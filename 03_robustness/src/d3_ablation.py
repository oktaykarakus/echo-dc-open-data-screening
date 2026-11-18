"""D3: leave-one-family-out ablations (equal weights, 2.5 km) + diagnosis of stage 02's 'heat demand = 0%' driver share.
Dependence tests only: they show how far rankings move when a family is removed, not that a family is valid or useful."""
from d_lib import *

# stage 02 attribution groups (multiscale.GROUPS, defined on raw inputs) -> the normalised score indicators they feed
PHASEC_GROUP_TO_NORM = {"heat_demand": ["norm_heat_sum_2p5km"], "anchors": ["norm_anchor_count_2p5km", "norm_anchor_diversity_2p5km"],
                        "urban_compactness": ["norm_urban_compactness_2p5km"], "road_density": ["norm_major_road_density_2p5km"],
                        "population_density": [POP]}

def main():
    df = load_table(); o, _ = baseline(df); n = len(df); k = n_top(n)
    wo, wf = full_weights("equal", o)
    base = o.echo_dc_rank_equal.to_numpy(); ctry = o.country.to_numpy()
    nominal = {i: 0.5 * wo.get(i, 0) + 0.5 * wf.get(i, 0) for i in set(S.OPP) | set(S.FEAS)}      # linearised weight in the composite
    allfam = {**FAMILIES, **{f"[sub] {k_}": v for k_, v in SUPP_FAMILIES.items()}}
    rows, ranks, ent_lists = [], {}, []
    for fam, cols in allfam.items():
        a, b = adjust_pair(wo, wf, drop=cols)
        assert abs(sum(a.values()) - 1) < 1e-12 and abs(sum(b.values()) - 1) < 1e-12 and len(a) >= 2 and len(b) >= 2
        r = score(o, a, b); ranks[fam] = r["rank"].to_numpy()
        m = compare(base, ranks[fam], k)
        m.update({"family": fam, "indicators_removed": "|".join(c.replace("norm_", "").replace("_2p5km", "") for c in cols), "n_removed": len(cols),
                  "nominal_weight_removed": float(sum(nominal[c] for c in cols)), "n_opp_left": len(a), "n_feas_left": len(b),
                  "spearman_opportunity_vs_full": float(stats.spearmanr(r.opp, o.opportunity_score_equal)[0]),
                  "spearman_feasibility_vs_full": float(stats.spearmanr(r.feas, o.feasibility_score_equal)[0])})
        ent = (ranks[fam] <= k) & (base > k); lea = (ranks[fam] > k) & (base <= k)
        m["entering_by_country"] = ";".join(f"{c}:{v}" for c, v in pd.Series(ctry[ent]).value_counts().items())
        m["leaving_by_country"] = ";".join(f"{c}:{v}" for c, v in pd.Series(ctry[lea]).value_counts().items())
        rows.append(m)
        for i in np.flatnonzero(ent | lea):
            ent_lists.append({"family": fam, "status": "enters" if ent[i] else "leaves", "facility_id": o.facility_id[i], "country": ctry[i],
                              "rank_full": int(base[i]), "rank_ablated": int(ranks[fam][i])})
    summ = pd.DataFrame(rows)
    primary = summ[~summ.family.str.startswith("[sub]")]
    most = primary.sort_values(["top_decile_overlap", "spearman"]).iloc[0].family               # prespecified criterion (see report)
    summ["most_influential_primary"] = summ.family == most
    cols = ["family", "indicators_removed", "n_removed", "nominal_weight_removed", "n_opp_left", "n_feas_left"] + [c for c in summ.columns if c not in
            ("family", "indicators_removed", "n_removed", "nominal_weight_removed", "n_opp_left", "n_feas_left")]
    write_csv(summ[cols], "d3_ablation_summary.csv"); write_csv(pd.DataFrame(ent_lists), "d3_top_decile_entering_leaving.csv")
    write_csv(pd.DataFrame({"facility_id": o.facility_id, "country": ctry, "rank_full": base, **{f"rank_wo_{k_}": v for k_, v in ranks.items()}}), "d3_facility_ranks.csv")

    # ---- Heat-demand 0% diagnosis -------------------------------------------------------------------------------------
    G = M.GROUPS                                     # stage 02 attribution groups (heat = Hotmaps only; anchors = count + diversity)
    shift = {}                                       # (group, scale) -> |dRank| vs 2.5 km under single-factor substitution
    for g, gc in G.items():
        for s in SCALES:
            if s == REF: continue
            out, _ = M.score_scale(df, s, only=gc)
            shift[(g, s)] = np.abs(out["echo_dc_rank_equal"].to_numpy() - base)
    # (1) replicate stage 02's label exactly (max over group x scale of (|dRank|, name))
    lab = [max((shift[(g, s)][i], g) for g in G for s in SCALES if s != REF)[1] for i in range(n)]
    c5 = pd.read_csv(PC / "results" / "c5_facility_sensitivity_equal.csv")
    assert (c5.facility_id.to_numpy() == o.facility_id.to_numpy()).all()
    replicated = bool((np.array(lab) == c5.largest_single_factor_driver.to_numpy()).all())
    sens = (c5.rank_range >= 200).to_numpy()
    lab = np.array(lab)
    d_all = pd.Series(lab).value_counts().to_dict(); d_sens = pd.Series(lab[sens]).value_counts().to_dict()
    # (2) per-scale strict wins and ties
    perscale = {}
    for s in SCALES:
        if s == REF: continue
        M_ = np.column_stack([shift[(g, s)] for g in G]); mx = M_.max(1); gi = list(G).index("heat_demand")
        perscale[s] = {"heat_strict_wins": int(((M_[:, gi] == mx) & ((M_ == mx[:, None]).sum(1) == 1) & (mx > 0)).sum()),
                       "heat_shift_ge10": int((M_[:, gi] >= 10).sum()), "heat_shift_ge25": int((M_[:, gi] >= 25).sum()),
                       "heat_shift_ge50": int((M_[:, gi] >= 50).sum()), "heat_shift_ge100": int((M_[:, gi] >= 100).sum()),
                       "heat_mean_abs": float(M_[:, gi].mean()), "heat_p99_abs": float(np.percentile(M_[:, gi], 99)), "heat_max_abs": int(M_[:, gi].max()),
                       "heat_mean_share_of_summed_single_factor_shifts": float((M_[:, gi][M_.sum(1) > 0] / M_.sum(1)[M_.sum(1) > 0]).mean()),
                       "heat_mean_share_in_sensitive_sites": float((M_[sens, gi] / M_[sens].sum(1)).mean())}
    # (3) stage 02's revert-based attribution and (4) level-influence by ablation at 2.5 km vs mean scale shift, same five groups
    att = pd.read_csv(PC / "results" / "c5_factor_attribution_equal.csv")
    att["share_of_total_mean_abs_dRank"] = att.contribution_reduction_in_mean_abs_dRank / att.all_at_s_mean_abs_dRank
    lvl = []
    for g, gc in G.items():
        wo_, wf_ = adjust_pair(wo, wf, drop=PHASEC_GROUP_TO_NORM[g])
        rk = score(o, wo_, wf_)["rank"].to_numpy()
        d_ = np.abs(rk - base)
        lvl.append({"group": g, "ablation_mean_abs_dRank_at_2p5km": float(d_.mean()), "ablation_spearman": float(stats.spearmanr(base, rk)[0]),
                    "ablation_top_decile_overlap": int(((rk <= k) & (base <= k)).sum()),
                    "mean_abs_dRank_single_factor_scale_shift_avg_over_radii": float(np.mean([shift[(g, s)].mean() for s in SCALES if s != REF])),
                    "revert_share_of_total_mean_abs_dRank_avg": float(att[att.group == g].share_of_total_mean_abs_dRank.mean()),
                    "argmax_driver_count_all_1216": int(d_all.get(g, 0)), "argmax_driver_count_sensitive_520": int(d_sens.get(g, 0))})
    lvl = pd.DataFrame(lvl)
    lvl["note"] = np.where(lvl.group == "population_density", "ablation removes population from BOTH components (context only, not a D2 configuration)", "")
    write_csv(lvl, "d3_heat_diagnosis_by_group.csv")
    heat = {"phaseC_label_replicated_exactly": replicated, "n_sensitive_sites": int(sens.sum()),
            "argmax_label_counts_all_1216": d_all, "argmax_label_counts_sensitive": d_sens,
            "argmax_rule": "label = max over 5 groups x 3 radii of (|dRank|, group name): winner-take-all, ties resolved alphabetically",
            "per_scale_heat_diagnostics": perscale,
            "phaseC_revert_share_of_total_mean_abs_dRank_heat": {r.scale: float(r.share_of_total_mean_abs_dRank) for r in att[att.group == "heat_demand"].itertuples()},
            "norm_heat_spearman_vs_2p5km": {s: float(stats.spearmanr(M.score_scale(df, s)[0].norm_heat_sum_2p5km, o.norm_heat_sum_2p5km)[0]) for s in SCALES if s != REF},
            "most_influential_primary_family_prespecified_criterion": most,
            "criterion": "lowest top-decile overlap with the full index among the four primary families (ties: lowest Spearman)"}
    write_json(heat, "d3_heat_diagnosis.json")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print(summ[["family", "n_removed", "nominal_weight_removed", "spearman", "kendall_tau_b", "top_decile_overlap", "top20_overlap", "median_abs_drank", "p90_abs_drank",
                "p99_abs_drank", "max_abs_drank", "n_enter_top_decile", "spearman_opportunity_vs_full", "spearman_feasibility_vs_full"]].round(4).to_string())
    print(summ[["family", "entering_by_country", "leaving_by_country"]].to_string())
    print(lvl.drop(columns="note").round(3).to_string()); print(json.dumps(heat, indent=1, default=float)); print("most influential:", most)

if __name__ == "__main__":
    main()
