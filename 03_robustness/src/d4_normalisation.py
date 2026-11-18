"""D4: (A) percentile-clipping window, 2.5 km equal weights; (B) Hotmaps missingness at 1 km (complete-case).
Normalisation (all variants): t(x) = log1p(x) for heat, population, anchor count, nearest distances; identity otherwise;
x' = clip((t(x) - Q_lo) / (Q_hi - Q_lo), 0, 1), reversed (1 - x') for the two nearest-distance indicators.
Variants change only (lo, hi): base (5th, 95th) | none (min, max; clip inactive) | modest (2.5th, 97.5th)."""
from d_lib import *

WINDOWS = {"base_5_95": (0.05, 0.95), "none_min_max": (0.0, 1.0), "modest_2p5_97p5": (0.025, 0.975)}
IND_RAW = {"norm_heat_sum_2p5km": ("hotmaps_heat_sum_2p5km", True), "norm_pop_density_2p5km": ("pop_density_2p5km", True),
           "norm_hdd_2024": ("hdd_2024", False), "norm_anchor_count_2p5km": ("anchor_count_2p5km", True),
           "norm_anchor_diversity_2p5km": ("anchor_type_diversity_2p5km", False), "norm_nearest_anchor_distance_m": ("nearest_anchor_distance_m", True),
           "norm_major_road_density_2p5km": ("major_road_density_2p5km_km_per_km2", False), "norm_nearest_major_road_distance_m": ("nearest_major_road_distance_m", True),
           "norm_ghsl_built_pct_2p5km": ("built_pct_2p5km", False), "norm_dw_built_pct_2p5km": ("dw_built_pct_2p5km", False)}

def equal_score(norm):
    wo, wf = S.make_equal_weights(S.OPP), S.make_equal_weights(S.FEAS)
    return score(norm, wo, wf)

def part_a(df):
    base_norm = S.normalise(df); n = len(df); k = n_top(n)
    b = equal_score(base_norm); br = b["rank"].to_numpy()
    rows, indrows, winrows, swaprows = [], [], [], []
    alts = {}
    for name, (lo, hi) in WINDOWS.items():
        alt = normalise_q(df, lo, hi); alts[name] = alt
        if name == "base_5_95":
            assert all(np.array_equal(alt[c].to_numpy(), base_norm[c].to_numpy()) for c in S.ALL_IND), "window (0.05,0.95) must reproduce V1 exactly"
            continue
        r = equal_score(alt); m = compare(br, r["rank"].to_numpy(), k); m["variant"] = name
        m["pearson_score_r"] = float(stats.pearsonr(b.raw, r.raw)[0]); rows.append(m)
        for ind in S.ALL_IND:
            indrows.append({"variant": name, "indicator": ind, "spearman_vs_base": float(stats.spearmanr(base_norm[ind], alt[ind])[0]),
                            "mean_abs_diff": float(np.abs(base_norm[ind] - alt[ind]).mean()), "sd_base": float(base_norm[ind].std()), "sd_alt": float(alt[ind].std()),
                            "share_at_0_base": float((base_norm[ind] <= 0).mean()), "share_at_1_base": float((base_norm[ind] >= 1).mean()),
                            "share_at_0_alt": float((alt[ind] <= 0).mean()), "share_at_1_alt": float((alt[ind] >= 1).mean())})
            # one-indicator-at-a-time swap: which indicator's window change moves the ranking?
            sw = equal_score(swap_indicators(base_norm, alt, [ind])); ms = compare(br, sw["rank"].to_numpy(), k)
            swaprows.append({"variant": name, "indicator_swapped": ind, "spearman": ms["spearman"], "top_decile_overlap": ms["top_decile_overlap"],
                             "mean_abs_drank": ms["mean_abs_drank"], "max_abs_drank": ms["max_abs_drank"]})
    for name, (lo, hi) in WINDOWS.items():
        for ind, (col, lg) in IND_RAW.items():
            t = np.log1p(df[col]) if lg else df[col]
            winrows.append({"variant": name, "indicator": ind, "transform": "log1p" if lg else "identity",
                            "lower_value": float(t.quantile(lo)), "upper_value": float(t.quantile(hi)),
                            "share_clipped_low": float((t < t.quantile(lo)).mean()), "share_clipped_high": float((t > t.quantile(hi)).mean())})
    return pd.DataFrame(rows), pd.DataFrame(indrows), pd.DataFrame(swaprows), pd.DataFrame(winrows), b, alts

def complete_case(df, drop_mask, scale, label):
    """Original (zero-filled, n=1216) vs complete-case (drop_mask facilities excluded, ALL quantiles refitted on the remainder).
    Both re-ranked on the SAME common set; top decile = ceil(10% of common set) in both."""
    s_all = M.scale_inputs(df, scale)
    keep = ~drop_mask
    n_c = int(keep.sum()); k_c = n_top(n_c)
    t0 = equal_score(S.normalise(s_all)); t0_full_rank = t0["rank"].to_numpy()
    t1 = equal_score(S.normalise(s_all[keep].reset_index(drop=True)))
    r0c = rank_desc(t0.raw.to_numpy()[keep]); r1 = t1["rank"].to_numpy()               # re-rank original scores on common set (row order preserved)
    assert sorted(r0c) == list(range(1, n_c + 1)) and sorted(r1) == list(range(1, n_c + 1))
    m = compare(r0c, r1, k_c); m.update({"label": label, "scale": scale, "n_all": len(df), "n_excluded": int(drop_mask.sum()), "n_common": n_c})
    # position of the excluded sites in the ORIGINAL ranking
    ex = t0_full_rank[drop_mask]
    m.update({"excluded_median_original_rank": float(np.median(ex)), "excluded_min_original_rank": int(ex.min()), "excluded_max_original_rank": int(ex.max()),
              "excluded_in_original_top_decile": int((ex <= n_top(len(df))).sum()), "original_top_decile_size": n_top(len(df))})
    h0, h1 = np.log1p(s_all.hotmaps_heat_sum_2p5km), np.log1p(s_all.hotmaps_heat_sum_2p5km[keep])
    m.update({"heat_q05_original": float(h0.quantile(.05)), "heat_q05_complete_case": float(h1.quantile(.05)),
              "heat_q95_original": float(h0.quantile(.95)), "heat_q95_complete_case": float(h1.quantile(.95)),
              "excluded_norm_heat_original_max": float(S.normalise(s_all).norm_heat_sum_2p5km[drop_mask].max())})
    # does the exclusion change the 1 km vs 2.5 km agreement reported in stage 02?
    ref0 = equal_score(S.normalise(df)); ref1 = equal_score(S.normalise(df[keep].reset_index(drop=True)))
    m["rho_vs_2p5km_original_common"] = float(stats.spearmanr(r0c, rank_desc(ref0.raw.to_numpy()[keep]))[0])
    m["rho_vs_2p5km_complete_case"] = float(stats.spearmanr(r1, ref1["rank"])[0])
    # which common-set sites change top-decile status
    ids = df.facility_id.to_numpy()[keep]; ent = (r1 <= k_c) & (r0c > k_c); lea = (r1 > k_c) & (r0c <= k_c)
    ch = pd.DataFrame({"scale": scale, "facility_id": ids[ent | lea], "country": df.country.to_numpy()[keep][ent | lea],
                       "status": np.where(ent[ent | lea], "enters", "leaves"), "rank_original_common": r0c[ent | lea], "rank_complete_case": r1[ent | lea]})
    return m, ch

def main():
    df = load_table(); n = len(df)
    A, I, SW, W, base, alts = part_a(df)
    write_csv(A, "d4a_clipping_summary.csv"); write_csv(I, "d4a_indicator_level.csv"); write_csv(SW, "d4a_swap_one_indicator.csv"); write_csv(W, "d4a_windows.csv")
    # ---------- D4-B -----------------------------------------------------------------------------------------------
    flag = df.hotmaps_small_buffer_missing_originally.to_numpy()
    ev = {"n_flagged_1km_gap": int(flag.sum()), "n_zero_heat_1km": int((df.hotmaps_heat_sum_1km == 0).sum()),
          "n_zero_heat_1km_not_flagged (observed zeros)": int(((df.hotmaps_heat_sum_1km == 0) & ~flag).sum()),
          "n_zero_heat_mean_1km": int((df.hotmaps_heat_mean_1km == 0).sum()),
          "zero_heat_sum_by_radius": {s: int((df[f"hotmaps_heat_sum_{s}"] == 0).sum()) for s in SCALES},
          "zero_heat_sum_2p5km_ids": df.facility_id[df.hotmaps_heat_sum_2p5km == 0].tolist(),
          "zero_2p5km_ids_all_within_flagged_15": bool(flag[(df.hotmaps_heat_sum_2p5km == 0).to_numpy()].all()),
          "flagged_with_zero_at_2p5km": int((flag & (df.hotmaps_heat_sum_2p5km == 0).to_numpy()).sum()),
          "flagged_median_pop_density_1km": float(df.pop_density_1km[flag].median()), "all_median_pop_density_1km": float(df.pop_density_1km.median())}
    m1, c1 = complete_case(df, flag, "1km", "D4-B primary: 15 flagged 1 km gaps excluded, 1 km")
    z25 = (df.hotmaps_heat_sum_2p5km == 0).to_numpy()
    m2, c2 = complete_case(df, z25, "2p5km", "evidence-based check: 2 sites with exactly 0 heat at 2.5 km excluded, 2.5 km")
    write_csv(pd.DataFrame([m1, m2]).set_index("label").reset_index(), "d4b_hotmaps_missingness.csv")
    write_csv(pd.concat([c1, c2]), "d4b_top_decile_changes.csv"); write_json(ev, "d4b_missingness_evidence.json")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(A[["variant", "spearman", "kendall_tau_b", "pearson_score_r", "top_decile_overlap", "top20_overlap", "median_abs_drank", "p90_abs_drank", "p99_abs_drank", "max_abs_drank",
             "n_enter_top_decile"]].round(4).to_string())
    print(I.pivot(index="indicator", columns="variant", values="spearman_vs_base").round(3).to_string())
    print(W[W.variant != "base_5_95"].assign(k=lambda d: d.variant).pivot(index="indicator", columns="variant", values="share_clipped_high").round(3).to_string())
    print(SW.pivot(index="indicator_swapped", columns="variant", values="top_decile_overlap").to_string())
    print(SW.pivot(index="indicator_swapped", columns="variant", values="spearman").round(4).to_string())
    print(json.dumps(ev, indent=1)); print(pd.DataFrame([m1, m2]).T.to_string()); print(pd.concat([c1, c2]).to_string())

if __name__ == "__main__":
    main()
