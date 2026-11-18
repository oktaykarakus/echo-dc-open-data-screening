"""D2: population density appears in Opportunity (weight 1/6) and Feasibility (weight 1/4). Equal weights, 2.5 km.
D2-A original | D2-B pop removed from Opportunity only | D2-C pop removed from Feasibility only.
Remaining weights renormalised within the affected component; geometric aggregation and all other indicators unchanged."""
from d_lib import *

def main():
    df = load_table(); o, _ = baseline(df); n = len(df); k = n_top(n)
    wo, wf = full_weights("equal", o)
    res, cfgw = {}, {}
    for c in D2_CONFIGS:
        a, b = d2_weights(c, wo, wf)
        res[c] = score(o, a, b); cfgw[c] = {"opp": a, "feas": b}
    assert np.array_equal(res["D2-A"]["rank"].to_numpy(), o.echo_dc_rank_equal.to_numpy())          # D2-A == baseline exactly
    assert np.allclose(res["D2-A"]["raw"], o.echo_dc_score_raw_equal, atol=0, rtol=0)
    pairs = [("D2-A", "D2-B"), ("D2-A", "D2-C"), ("D2-B", "D2-C")]
    rows = []
    for x, y in pairs:
        m = compare(res[x]["rank"].to_numpy(), res[y]["rank"].to_numpy(), k)
        m.update({"base": x, "alt": y, "pearson_score_r": float(stats.pearsonr(res[x].raw, res[y].raw)[0]),
                  "spearman_score_raw": float(stats.spearmanr(res[x].raw, res[y].raw)[0])})
        rows.append(m)
    cmp_ = pd.DataFrame(rows); cmp_.insert(0, "alt", cmp_.pop("alt")); cmp_.insert(0, "base", cmp_.pop("base"))
    write_csv(cmp_, "d2_population_comparisons.csv")
    # component relationships
    pop = o[POP].to_numpy(); rel = []
    for c, r in res.items():
        rel.append({"config": c, "spearman_opp_feas": float(stats.spearmanr(r.opp, r.feas)[0]), "pearson_opp_feas": float(stats.pearsonr(r.opp, r.feas)[0]),
                    "spearman_opp_pop": float(stats.spearmanr(r.opp, pop)[0]), "spearman_feas_pop": float(stats.spearmanr(r.feas, pop)[0]),
                    "spearman_composite_pop": float(stats.spearmanr(r.raw, pop)[0]), "mean_opp": float(r.opp.mean()), "mean_feas": float(r.feas.mean()),
                    "n_top_decile_FR": int((o.country.to_numpy()[r["rank"].to_numpy() <= k] == "FR").sum()),
                    "n_top_decile_GB": int((o.country.to_numpy()[r["rank"].to_numpy() <= k] == "GB").sum()),
                    "n_top_decile_DE": int((o.country.to_numpy()[r["rank"].to_numpy() <= k] == "DE").sum())})
    rel = pd.DataFrame(rel); write_csv(rel, "d2_component_relationships.csv")
    # nominal (linearised at Opp = Feas) weight each indicator carries in the composite: sqrt(O*F) ~ (O + F)/2
    nom = {}
    for c in D2_CONFIGS:
        t = {}
        for ind, wv in cfgw[c]["opp"].items(): t[ind] = t.get(ind, 0) + 0.5 * wv
        for ind, wv in cfgw[c]["feas"].items(): t[ind] = t.get(ind, 0) + 0.5 * wv
        nom[c] = t
    write_csv(pd.DataFrame(nom).rename_axis("indicator").reset_index(), "d2_nominal_linearised_weights.csv")
    # largest rank changes (B and C vs A), ten each direction
    big = []
    base = res["D2-A"]["rank"].to_numpy()
    for c in ("D2-B", "D2-C"):
        d = res[c]["rank"].to_numpy() - base
        for lab, idx in (("improves", np.argsort(d)[:10]), ("worsens", np.argsort(-d)[:10])):
            for i in idx:
                big.append({"config": c, "direction": lab, "facility_id": o.facility_id[i], "country": o.country[i], "name": o["name"][i],
                            "rank_A": int(base[i]), "rank_alt": int(res[c]["rank"].iloc[i]), "delta_rank": int(d[i]), "pop_indicator": round(float(pop[i]), 3)})
    write_csv(pd.DataFrame(big), "d2_largest_rank_changes.csv")
    # where do the changes sit? |delta rank| by baseline decile, and which sites differ by pop level
    dec = pd.cut(base, [0, 122, 243, 365, 486, 608, 729, 851, 972, 1094, 1216], labels=[f"D{i}" for i in range(1, 11)])
    byd = pd.DataFrame({"decile": dec, "abs_B": np.abs(res["D2-B"]["rank"].to_numpy() - base), "abs_C": np.abs(res["D2-C"]["rank"].to_numpy() - base)}
                       ).groupby("decile", observed=True).agg(median_abs_B=("abs_B", "median"), median_abs_C=("abs_C", "median"),
                                                              p90_abs_B=("abs_B", lambda x: np.percentile(x, 90)), p90_abs_C=("abs_C", lambda x: np.percentile(x, 90)))
    write_csv(byd.reset_index(), "d2_abs_rank_change_by_baseline_decile.csv")
    # direction of change vs pop level: Spearman(pop, delta rank)
    dirn = {c: float(stats.spearmanr(pop, res[c]["rank"].to_numpy() - base)[0]) for c in ("D2-B", "D2-C")}
    write_json({"spearman_pop_indicator_vs_delta_rank": dirn, "interpretation": "negative = higher-population sites move up (rank number falls) relative to A"}, "d2_direction.json")
    # facility-level ranks (compact) for figures
    write_csv(pd.DataFrame({"facility_id": o.facility_id, "country": o.country, "rank_A": base, "rank_B": res["D2-B"]["rank"], "rank_C": res["D2-C"]["rank"]}), "d2_facility_ranks.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(cmp_[["base", "alt", "spearman", "kendall_tau_b", "pearson_score_r", "top_decile_overlap", "top20_overlap", "median_abs_drank", "p90_abs_drank",
                "p99_abs_drank", "max_abs_drank", "share_abs_drank_ge100", "n_enter_top_decile"]].round(4).to_string())
    print(rel.round(3).to_string()); print(pd.DataFrame(nom).round(4).to_string()); print(byd.round(1).to_string()); print(dirn)
    print(pd.DataFrame(big).query("config=='D2-B'").head(5).to_string()); print(pd.DataFrame(big).query("config=='D2-C'").head(5).to_string())

if __name__ == "__main__":
    main()
