"""F2-B step 2 (Gate 3): frozen five-arm evaluation on the labelled French records.
Precondition: results/f2b_labels.csv exists and its SHA-256 equals the hash recorded BEFORE scores were loaded.
Run:  source src/env.sh && /usr/bin/python3 src/f2b_02_evaluate.py"""
import os, sys, json, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f2b_lib as L

PH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.environ.get("ECHO_F2B_RESULTS", os.path.join(PH, "results"))
REPS = int(os.environ.get("ECHO_F2B_REPS", "2000"))
SEED = 20261107
ARMS_ALL = L.ARMS + L.DESCR


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    lp = os.path.join(RES, "f2b_labels.csv")
    rec = open(os.path.join(RES, "f2b_labels.sha256")).read().split()[0]
    assert sha(lp) == rec, "label table changed after hashing"
    audit = json.load(open(os.path.join(RES, "f2b_outcome_audit.json")))
    lab = pd.read_csv(lp)
    df, vals, pos = L.load_arms()
    ii = lab.facility_id.map({f: i for i, f in enumerate(df.facility_id)}).values
    assert not np.isnan(ii.astype(float)).any()
    ii = ii.astype(int)

    def sub(mask, status_col="status_2500", indet_as_neg=False, units=False):
        """Return evaluation structure (record- or unit-level) for the records in mask."""
        l = lab[mask].copy()
        l["inv"] = ii[mask.values] if hasattr(mask, "values") else ii[mask]
        if indet_as_neg:
            l["y"] = (l[status_col] == "positive").astype(int); l = l.copy()
        else:
            l = l[l[status_col] != "indeterminate"].copy(); l["y"] = (l[status_col] == "positive").astype(int)
        if not units:
            idx = l.inv.values
            return {"y": l.y.values, "vals": {a: vals[a][idx] for a in ARMS_ALL}, "pos": {a: pos[a][idx] for a in ARMS_ALL},
                    "block": {km: l[f"block{km}"].values for km in (25, 50, 100)}, "n": len(l), "ids": l.facility_id.values}
        # unit level: score = best member per arm; label = any member positive; block = block of first member (master order)
        l = l.sort_values("inv")
        g = l.groupby("unit_id", sort=False)
        y = g.y.max().values
        v = {a: g.inv.apply(lambda s, a=a: vals[a][s.values].max()).values for a in ARMS_ALL}
        p = {a: g.inv.apply(lambda s, a=a: pos[a][s.values].min()).values for a in ARMS_ALL}
        blk = {km: g[f"block{km}"].first().values for km in (25, 50, 100)}
        return {"y": y, "vals": v, "pos": p, "block": blk, "n": len(y), "ids": g.facility_id.first().values}

    allm = pd.Series(True, index=lab.index)
    big = lab[lab.status_2500 != "indeterminate"].block50.value_counts().index[0]
    variants = {
        "primary_2500m": sub(allm),
        "S3_threshold_1000m": sub(allm, "status_1000"),
        "S1_campus_units": sub(allm, units=True),
        "S2_exclude_13_salle_technique": sub(~lab.salle_technique),
        "S2b_exclude_4_broadcaster (descriptive)": sub(~lab.four_broadcaster),
        "S4a_indeterminate_scored_negative": sub(allm, indet_as_neg=True),
        "S4b_exclude_largest_50km_block": sub(lab.block50 != big),
        "S5_tier2_traces_as_existing": sub(allm, "status_t2exist_2500"),
    }

    def run(v, km=50, reps=REPS, record_level=False, tag=0):
        return L.block_bootstrap(v["y"], v["vals"], v["pos"], v["block"][km], ARMS_ALL, reps,
                                 seed=int(np.random.SeedSequence([SEED, km, tag, int(record_level)]).generate_state(1)[0]), record_level=record_level)

    out = {}
    # ---- primary + block-size sensitivity + record-level benchmark ----
    P = variants["primary_2500m"]
    prim = {}
    for km in (50, 25, 100):
        prim[km] = run(P, km, tag=1)
    rec_lvl = run(P, 50, record_level=True, tag=1)
    pt = prim[50][0]
    hw = {km: (prim[km][1]["dauc|A"][1] - prim[km][1]["dauc|A"][0]) / 2 for km in prim}
    ratio = max(hw.values()) / min(hw.values())
    widest = max(hw, key=hw.get) if ratio > 1.5 else 50
    ci_primary = prim[widest][1]["dauc|A"]
    deff = (prim[50][2]["sd"]["dauc|A"] / rec_lvl[2]["sd"]["dauc|A"]) ** 2

    rows = []
    for a in ARMS_ALL:
        r = {"arm": a, "AUC": pt[f"auc|{a}"]}
        for km in (50, 25, 100):
            r[f"AUC_ci90_lo_{km}km"], r[f"AUC_ci90_hi_{km}km"] = prim[km][1][f"auc|{a}"]
        if a != "P":
            r["dAUC_vs_P"] = pt[f"dauc|{a}"]
            for km in (50, 25, 100):
                r[f"dAUC_ci90_lo_{km}km"], r[f"dAUC_ci90_hi_{km}km"] = prim[km][1][f"dauc|{a}"]
        rows.append(r)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(RES, "f2b_five_arm_auc.csv"), index=False)

    # ---- fixed budgets ----
    rows = []
    for f in (10, 20):
        for a in ARMS_ALL:
            r = {"budget_pct": f, "arm": a, "k_full_sample": int(np.ceil(f / 100 * P["n"])),
                 "TP": int(pt[f"tp@{f}|{a}"]), "precision": pt[f"precision@{f}|{a}"], "recall": pt[f"recall@{f}|{a}"], "enrichment": pt[f"enrichment@{f}|{a}"]}
            r["precision_ci90_lo"], r["precision_ci90_hi"] = prim[50][1][f"precision@{f}|{a}"]
            if a != "P":
                r["dTP_vs_P"] = int(pt[f"dtp@{f}|{a}"]); r["dPrecision_vs_P"] = pt[f"dprecision@{f}|{a}"]
                r["dPrecision_ci90_lo"], r["dPrecision_ci90_hi"] = prim[50][1][f"dprecision@{f}|{a}"]
            rows.append(r)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(RES, "f2b_fixed_budget.csv"), index=False)

    # ---- sensitivities (block bootstrap at 50 km) ----
    sens, sres = [], {}
    for name, v in variants.items():
        if name == "primary_2500m":
            p_, c_, i_ = prim[50]
        else:
            p_, c_, i_ = run(v, 50, tag=hash(name) % 1000 if False else (list(variants).index(name) + 10))
        sres[name] = (p_, c_, i_)
        r = {"variant": name, "n_evaluated": v["n"], "n_positive": int(v["y"].sum()), "prevalence": round(float(v["y"].mean()), 4),
             "AUC_A": p_["auc|A"], "AUC_P": p_["auc|P"]}
        for a in ("A", "H", "B*", "C"):
            r[f"dAUC_{a}"] = p_[f"dauc|{a}"]; r[f"dAUC_{a}_lo"], r[f"dAUC_{a}_hi"] = c_[f"dauc|{a}"]
        r["dPrec10_A"] = p_["dprecision@10|A"]; r["dPrec20_A"] = p_["dprecision@20|A"]
        sens.append(r)
    pd.DataFrame(sens).round(4).to_csv(os.path.join(RES, "f2b_sensitivity.csv"), index=False)

    # ---- descriptive checks ----
    from scipy.stats import spearmanr
    idxP = np.arange(P["n"])
    desc = {"n_eligible": P["n"], "n_positive": int(P["y"].sum()), "prevalence": float(P["y"].mean()),
            "spearman_A_vs_P_on_eligible": float(spearmanr(P["vals"]["A"], P["vals"]["P"]).correlation),
            "spearman_A_vs_H_on_eligible": float(spearmanr(P["vals"]["A"], P["vals"]["H"]).correlation),
            "spearman_B*_vs_A_on_eligible": float(spearmanr(P["vals"]["B*"], P["vals"]["A"]).correlation),
            "block_counts_50km": {"nonempty": int(len(set(P["block"][50]))), "bootstrap_groups": prim[50][2]["groups"]},
            "bootstrap": {"reps": REPS, "seed_base": SEED, "skipped_reps_50km": prim[50][2]["skipped"], "skipped_25km": prim[25][2]["skipped"],
                          "skipped_100km": prim[100][2]["skipped"]},
            "dAUC_A_halfwidth_by_block_km": {str(k): round(v, 4) for k, v in hw.items()}, "halfwidth_ratio_max_over_min": round(ratio, 3),
            "interval_used_as_primary_km": widest, "record_level_benchmark_dAUC_A_ci90": rec_lvl[1]["dauc|A"],
            "approx_design_effect_dAUC_A_(var_block/var_record)": round(deff, 2)}
    # access context (descriptive): partial Spearman with the label given D, S, F, over eligible records
    sel = ii[(lab.status_2500 != "indeterminate").values]
    Dd, Ss, Ff = df.D.values[sel], df.S.values[sel], df.F.values[sel]
    yv = P["y"]
    parts = {}
    for nm, col in (("road_density", df.road_density.values[sel]), ("nearest_road_distance", df.road_dist.values[sel])):
        parts[nm] = L.partial_spearman(yv, col, [Dd, Ss, Ff])
    rng = np.random.default_rng(SEED + 5)
    blk = P["block"][50]; groups = [np.where(blk == b)[0] for b in np.unique(blk)]
    boots = {k: [] for k in parts}
    cols = {"road_density": df.road_density.values[sel], "nearest_road_distance": df.road_dist.values[sel]}
    for _ in range(REPS):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        if yv[idx].sum() in (0, len(idx)): continue
        for k in parts:
            boots[k].append(L.partial_spearman(yv[idx], cols[k][idx], [Dd[idx], Ss[idx], Ff[idx]]))
    desc["access_partial_spearman_given_DSF"] = {k: {"estimate": round(parts[k], 4), "ci90": [round(float(np.quantile(boots[k], .05)), 4), round(float(np.quantile(boots[k], .95)), 4)]} for k in parts}
    json.dump(desc, open(os.path.join(RES, "f2b_descriptive.json"), "w"), indent=1)

    # ---- decision rule (frozen) ----
    g = audit["gates_t2500"]; g_ok = g["G2_eligible_ge_250"] and g["G2_each_class_ge_40"] and g["G3_prevalence_15_85"]
    g4 = g["G4_blocks_ge_20"] and g["G4_largest_le_35pct"]
    dec = {}
    for arm in ("A", "H", "B*", "C"):
        d_pt = pt[f"dauc|{arm}"]; d_ci = prim[widest][1][f"dauc|{arm}"]
        hwid = (d_ci[1] - d_ci[0]) / 2
        prec_ok = all(pt[f"precision@{f}|{arm}"] >= pt[f"precision@{f}|P"] for f in (10, 20))
        s1, s2, s3 = sres["S1_campus_units"], sres["S2_exclude_13_salle_technique"], sres["S3_threshold_1000m"]
        s4 = (sres["S4a_indeterminate_scored_negative"], sres["S4b_exclude_largest_50km_block"])
        signs = [bool(s1[0][f"dauc|{arm}"] > 0), bool(s2[0][f"dauc|{arm}"] > 0), bool(s3[0][f"dauc|{arm}"] > 0), bool(all(x[0][f"dauc|{arm}"] > 0 for x in s4))]
        prec_ok = bool(prec_ok)
        below = any(sres[k][1][f"dauc|{arm}"][1] < 0 for k in sres if k not in ("primary_2500m", "S2b_exclude_4_broadcaster (descriptive)"))
        tier = L.decide(g_ok, d_pt, d_ci, pt["auc|" + arm], pt["auc|P"], hwid, g4, prec_ok, sum(signs) >= 3, below)
        dec[arm] = {"dAUC": round(d_pt, 4), "ci90": [round(d_ci[0], 4), round(d_ci[1], 4)], "halfwidth": round(hwid, 4), "precision_ok_both_budgets": prec_ok,
                    "sensitivity_signs_positive[S1,S2,S3,S4]": signs, "any_sensitivity_ci_entirely_below_0": bool(below), "tier": tier}
    json.dump({"gates_ok": bool(g_ok), "g4_ok": bool(g4), "tiers_vs_P": dec,
               "note": "Tier for A is the confirmatory result; H, B*, C are exploratory (same rules)."}, open(os.path.join(RES, "f2b_decision.json"), "w"), indent=1)
    print(pd.read_csv(os.path.join(RES, "f2b_five_arm_auc.csv")).iloc[:, :9].to_string(index=False))
    print(json.dumps(desc, indent=1)); print(json.dumps(dec, indent=1))
    print(pd.read_csv(os.path.join(RES, "f2b_sensitivity.csv")).to_string(index=False))
    print(pd.read_csv(os.path.join(RES, "f2b_fixed_budget.csv")).to_string(index=False))


if __name__ == "__main__":
    main()
