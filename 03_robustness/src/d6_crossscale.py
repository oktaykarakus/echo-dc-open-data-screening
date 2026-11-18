"""D6: targeted cross-scale checks (equal weights, Protocol A, radii 1/2.5/5/10 km). Not a factorial design:
 (a) composite vs simple indicators at every radius; (b) effect of each tested alternative (D2-B, D2-C, most influential D3
 ablation, D4-A windows) at every radius; (c) whether stage 02's multiscale stability metrics survive each alternative."""
from d_lib import *
from d4_normalisation import WINDOWS

def config_table(most):
    fam = FAMILIES[most]
    return {"base": ("base_5_95", lambda wo, wf: (wo, wf)),
            "D2-B (no pop in Opp)": ("base_5_95", lambda wo, wf: d2_weights("D2-B", wo, wf)),
            "D2-C (no pop in Feas)": ("base_5_95", lambda wo, wf: d2_weights("D2-C", wo, wf)),
            f"D3 w/o {most}": ("base_5_95", lambda wo, wf: adjust_pair(wo, wf, drop=fam)),
            "D4-A no clipping": ("none_min_max", lambda wo, wf: (wo, wf)),
            "D4-A 2.5-97.5": ("modest_2p5_97p5", lambda wo, wf: (wo, wf))}

def ranks_for(df, s, window, wfun):
    sc = M.scale_inputs(df, s)
    lo, hi = WINDOWS[window]
    nm = S.normalise(sc) if window == "base_5_95" else normalise_q(sc, lo, hi)
    wo, wf = wfun(S.make_equal_weights(S.OPP), S.make_equal_weights(S.FEAS))
    return score(nm, wo, wf)["rank"].to_numpy()

def main():
    df = load_table(); n = len(df); k = n_top(n)
    d3 = pd.read_csv(RESULTS / "d3_ablation_summary.csv"); most = d3.loc[d3.most_influential_primary, "family"].iloc[0]
    cfg = config_table(most)
    R = {c: {s: ranks_for(df, s, w, f) for s in SCALES} for c, (w, f) in cfg.items()}
    # --- stage 02 reproduction check for the baseline ----------------------------------------------------------------------
    pc_sp = pd.read_csv(PC / "results" / "c5_global_spearman_equal.csv", index_col=0); pc_ov = pd.read_csv(PC / "results" / "c5_global_top_overlap_equal.csv", index_col=0)
    chk = {}
    for s in SCALES:
        if s == REF: continue
        chk[s] = {"rho": float(stats.spearmanr(R["base"][s], R["base"][REF])[0]), "rho_phaseC": float(pc_sp.loc[REF, s]),
                  "overlap": int((np.isin(np.flatnonzero(R["base"][s] <= k), np.flatnonzero(R["base"][REF] <= k))).sum()), "overlap_phaseC": int(pc_ov.loc[REF, s])}
        assert abs(chk[s]["rho"] - chk[s]["rho_phaseC"]) < 6e-5 and chk[s]["overlap"] == chk[s]["overlap_phaseC"], chk[s]
    # --- (a) composite vs simple indicators by radius ----------------------------------------------------------------------
    a_rows = []
    for s in SCALES:
        o, _ = baseline(df, s); simp = simple_rankings(o)
        assert np.array_equal(R["base"][s], o.echo_dc_rank_equal.to_numpy())
        for nm, r in simp.items():
            m = compare(R["base"][s], r, k); m.update({"scale": s, "comparator": nm}); a_rows.append(m)
        # heat-vs-population redundancy at each radius
        a_rows[-1]["spearman_heat_vs_pop_indicator"] = float(stats.spearmanr(o.norm_heat_sum_2p5km, o.norm_pop_density_2p5km)[0])
    A = pd.DataFrame(a_rows); write_csv(A[["scale", "comparator", "spearman", "kendall_tau_b", "top_decile_overlap", "top20_overlap", "median_abs_drank", "p90_abs_drank", "max_abs_drank",
                                           "share_abs_drank_ge100"]], "d6_composite_vs_simple_by_radius.csv")
    heatpop = {s: float(stats.spearmanr(*[baseline(df, s)[0][c] for c in ("norm_heat_sum_2p5km", "norm_pop_density_2p5km")])[0]) for s in SCALES}
    # --- (b) effect of each alternative at each radius ---------------------------------------------------------------------
    b_rows = []
    for c in cfg:
        if c == "base": continue
        for s in SCALES:
            m = compare(R["base"][s], R[c][s], k); m.update({"config": c, "scale": s}); b_rows.append(m)
    B = pd.DataFrame(b_rows); write_csv(B[["config", "scale", "spearman", "kendall_tau_b", "top_decile_overlap", "top20_overlap", "median_abs_drank", "p90_abs_drank", "p99_abs_drank",
                                           "max_abs_drank", "n_enter_top_decile"]], "d6_alternative_effects_by_radius.csv")
    # --- (c) multiscale stability within each configuration -----------------------------------------------------------------
    c_rows = []; base_all4 = set.intersection(*[set(np.flatnonzero(R["base"][s] <= k)) for s in SCALES])
    for c in cfg:
        top = {s: set(np.flatnonzero(R[c][s] <= k)) for s in SCALES}
        all4, any4 = set.intersection(*top.values()), set.union(*top.values())
        row = {"config": c, "n_top_decile_all_four_radii": len(all4), "n_top_decile_any_radius": len(any4),
               "baseline_stable_set_retained": len(all4 & base_all4), "baseline_stable_set_size": len(base_all4),
               "jaccard_stable_set_vs_baseline": len(all4 & base_all4) / len(all4 | base_all4),
               "top20_all_four": len(set.intersection(*[set(np.flatnonzero(R[c][s] <= 20)) for s in SCALES])),
               "spearman_1km_vs_10km": float(stats.spearmanr(R[c]["1km"], R[c]["10km"])[0])}
        for s in SCALES:
            if s == REF: continue
            row[f"rho_{s}_vs_2p5km"] = float(stats.spearmanr(R[c][s], R[c][REF])[0]); row[f"top_overlap_{s}_vs_2p5km"] = len(top[s] & top[REF])
        row["share_rank_range_ge200"] = float((np.column_stack([R[c][s] for s in SCALES]).max(1) - np.column_stack([R[c][s] for s in SCALES]).min(1) >= 200).mean())
        c_rows.append(row)
    C = pd.DataFrame(c_rows); write_csv(C, "d6_multiscale_stability_by_config.csv")
    write_json({"most_influential_family": most, "phaseC_reproduction_check_baseline": chk, "spearman_heat_vs_pop_indicator_by_radius": heatpop}, "d6_summary.json")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print("heat-vs-pop indicator rho by radius", {s: round(v, 3) for s, v in heatpop.items()}); print("most influential:", most)
    print(A.pivot(index="comparator", columns="scale", values="spearman").round(3)[SCALES].to_string())
    print(A.pivot(index="comparator", columns="scale", values="top_decile_overlap")[SCALES].to_string())
    print(A.pivot(index="comparator", columns="scale", values="top20_overlap")[SCALES].to_string())
    print(B.pivot(index="config", columns="scale", values="spearman").round(4)[SCALES].to_string()); print(B.pivot(index="config", columns="scale", values="top_decile_overlap")[SCALES].to_string())
    print(B.pivot(index="config", columns="scale", values="p90_abs_drank")[SCALES].to_string())
    print(C.round(4).T.to_string())

if __name__ == "__main__":
    main()
