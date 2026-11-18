"""D5: do the stage 03 conclusions depend on the weighting scenario? (2.5 km)
Part 1: re-run D1/D2/D3 under the three EXISTING scenarios (equal, literature-informed, entropy). Ablated/removed indicators are
        handled by proportional renormalisation of that scenario's full-model weights within the component.
Part 2: prespecified structured perturbations around equal weights (no search, no optimisation):
        weight multiplier in {0 (= ablation), 0.5, 2.0} applied to five indicator groups, renormalised within component.
        Question not answered by stage 01/C: is the ablation result an extreme corner, or does ranking respond smoothly near the baseline?"""
from d_lib import *

PERTURB_GROUPS = {**FAMILIES, "population_density": [POP]}      # population multiplier applies in BOTH components
MULTS = [0.0, 0.5, 2.0]                                         # prespecified

def main():
    df = load_table(); o, _ = baseline(df); n = len(df); k = n_top(n)
    rows = []
    def add(analysis, config, scn, base_rank, rank, extra=None):
        m = compare(base_rank, rank, k); m.update({"analysis": analysis, "config": config, "scenario": scn}); m.update(extra or {}); rows.append(m)
    for scn in SCENARIOS:
        wo, wf = full_weights(scn, o)
        full = score(o, wo, wf); base = full["rank"].to_numpy()
        assert np.array_equal(base, o[f"echo_dc_rank_{scn}"].to_numpy())                       # scenario baseline reproduces B0-QA
        # D1 composites vs simple (components use the scenario's own weights)
        simple = {"heat_hotmaps": rank_desc(o.norm_heat_sum_2p5km, np.log1p(o.hotmaps_heat_sum_2p5km)),
                  "population_density": rank_desc(o.norm_pop_density_2p5km, np.log1p(o.pop_density_2p5km)),
                  "opportunity": rank_desc(full.opp), "feasibility": rank_desc(full.feas)}
        for nm, r in simple.items(): add("D1", nm, scn, base, r)
        for c in ("D2-B", "D2-C"):
            a, b = d2_weights(c, wo, wf); add("D2", c, scn, base, score(o, a, b)["rank"].to_numpy())
        for fam, cols in FAMILIES.items():
            a, b = adjust_pair(wo, wf, drop=cols); add("D3", fam, scn, base, score(o, a, b)["rank"].to_numpy())
    S1 = pd.DataFrame(rows); write_csv(S1[["analysis", "config", "scenario", "spearman", "kendall_tau_b", "top_decile_overlap", "top20_overlap", "median_abs_drank",
                                         "p90_abs_drank", "p99_abs_drank", "max_abs_drank", "n_enter_top_decile"]], "d5_scenario_robustness.csv")
    # consistency of the ordering of family influence across scenarios (by Spearman with the scenario's full index; by overlap)
    d3 = S1[S1.analysis == "D3"].pivot(index="config", columns="scenario", values="spearman")
    d3o = S1[S1.analysis == "D3"].pivot(index="config", columns="scenario", values="top_decile_overlap")
    order = {s: d3[s].sort_values().index.tolist() for s in SCENARIOS}                           # most influential first
    tau = {f"{a}_vs_{b}": float(stats.kendalltau(d3[a], d3[b])[0]) for i, a in enumerate(SCENARIOS) for b in SCENARIOS[i + 1:]}
    d1o = S1[S1.analysis == "D1"].pivot(index="config", columns="scenario", values="spearman")
    # Part 2
    wo, wf = full_weights("equal", o); base = o.echo_dc_rank_equal.to_numpy(); prow = []
    for g, cols in PERTURB_GROUPS.items():
        for mlt in MULTS:
            a, b = adjust_pair(wo, wf, mult={c: mlt for c in cols})
            assert abs(sum(a.values()) - 1) < 1e-12 and abs(sum(b.values()) - 1) < 1e-12
            m = compare(base, score(o, a, b)["rank"].to_numpy(), k); m.update({"group": g, "multiplier": mlt}); prow.append(m)
    P = pd.DataFrame(prow); write_csv(P[["group", "multiplier", "spearman", "kendall_tau_b", "top_decile_overlap", "top20_overlap", "median_abs_drank", "p90_abs_drank",
                                       "max_abs_drank", "n_enter_top_decile"]], "d5_weight_perturbations.csv")
    write_json({"family_influence_order_by_scenario_most_influential_first": order, "kendall_tau_of_family_spearman_between_scenarios": tau,
                "multipliers": MULTS, "groups": {g: [c.replace("norm_", "").replace("_2p5km", "") for c in v] for g, v in PERTURB_GROUPS.items()}}, "d5_summary.json")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    for a in ("D1", "D2", "D3"):
        t = S1[S1.analysis == a]
        print(a, "\n", t.pivot(index="config", columns="scenario", values="spearman").round(4).to_string(), "\n", t.pivot(index="config", columns="scenario", values="top_decile_overlap").to_string(),
              "\n", t.pivot(index="config", columns="scenario", values="top20_overlap").to_string())
    print(json.dumps({"order": order, "tau": tau}, indent=1)); print(P[["group", "multiplier", "spearman", "top_decile_overlap", "top20_overlap", "median_abs_drank", "max_abs_drank"]].round(4).to_string())

if __name__ == "__main__":
    main()
