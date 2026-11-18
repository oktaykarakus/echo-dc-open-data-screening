"""E6: campus-level sensitivity view (separate from the 1,216-record analysis; the record table is never altered).

Convention (justified in the report): each campus group is ONE independent screening unit, represented by its highest-scoring member
AT THAT SCALE (equal weights). No scores are summed and no ranks are averaged: a campus's screening value is the best siting
candidate it contains. Units are re-ranked among units only (n_units), so denominators are never mixed with the 1,216 record view.
Brackets: 'minimal' (only coincident points, T1), 'primary' (T1 + same-brand T2 edges, chain-guarded), 'proximity-only' (every
pair <= 250 m, regardless of identity; an UPPER BOUND on merging, explicitly not a campus identification).
Sensitivity of the representative rule: medoid member (closest to the group's centroid, independent of score).
"""
from e_lib import *
from e2_campus_groups import build as build_groups, components
from scipy import stats

REC_TOL = 2          # stage 02/D gate values reproduced within this tolerance (see test)


def tops(n):
    return math.ceil(0.10 * n)


def views():
    t, pt, xy = build_groups()
    pos = {f: i for i, f in enumerate(t.facility_id)}
    out = {}
    out["record"] = np.arange(len(t))
    t1 = pt[(pt.tier == "T1_coincident") & (pt.country_a == pt.country_b)]
    out["minimal_T1"] = components(len(t), t1.id_a.map(pos).values, t1.id_b.map(pos).values)
    out["primary"] = pd.factorize(t.unit_id)[0]
    al = pt[pt.country_a == pt.country_b]
    out["proximity_only_250m"] = components(len(t), al.id_a.map(pos).values, al.id_b.map(pos).values)
    return t, xy, out


def unit_scores(ms, t, lab, xy, scale, rule="best"):
    """one row per unit: representative record index (row in master order) and its raw equal-weight score at `scale`."""
    m = ms[ms.scale == scale].set_index("facility_id").loc[t.facility_id]
    sc = m.echo_dc_score_raw_equal.values
    df = pd.DataFrame({"unit": lab, "i": np.arange(len(t)), "score": sc})
    if rule == "best":
        rep = df.sort_values(["unit", "score", "i"], ascending=[True, False, True]).groupby("unit").head(1)
    else:   # medoid
        rows = []
        for u, d in df.groupby("unit"):
            if len(d) == 1:
                rows.append(d.iloc[0]); continue
            p = xy[d.i.values]; c = p.mean(axis=0)
            rows.append(d.iloc[int(np.argmin(np.hypot(*(p - c).T)))])
        rep = pd.DataFrame(rows)
    rep = rep.sort_values("i").reset_index(drop=True)
    rep["rank"] = rep.sort_values(["score", "i"], ascending=[False, True]).reset_index().reset_index().set_index("index")["level_0"].reindex(rep.index).values + 1
    return rep


def metrics(ms, t, lab, xy, rule="best"):
    reps = {s: unit_scores(ms, t, lab, xy, s, rule) for s in SCALES}
    n = len(reps["2p5km"]); k = tops(n)
    base = reps["2p5km"].set_index("unit")
    R = pd.DataFrame({s: reps[s].set_index("unit")["rank"] for s in SCALES})
    out = dict(n_units=n, n_records=len(t), top_decile_k=k)
    for s in ["1km", "5km", "10km"]:
        out[f"spearman_{s}_vs_2p5km"] = round(float(stats.spearmanr(R["2p5km"], R[s])[0]), 4)
        out[f"top_decile_overlap_{s}_vs_2p5km"] = int(((R["2p5km"] <= k) & (R[s] <= k)).sum())
    ins = (R <= k)
    out["top_decile_all4_scales"] = int(ins.all(axis=1).sum()); out["top_decile_ge1_scale"] = int(ins.any(axis=1).sum())
    thr = 200 * n / len(t)                        # stage 02's '>= 200 places of 1,216', scaled to n units
    out["rank_range_threshold_scaled"] = round(thr, 1)
    out["share_rank_range_ge_scaled_threshold"] = round(float(((R.max(axis=1) - R.min(axis=1)) >= thr).mean()), 3)
    return out, R, reps


def country_table(t, lab, R, reps):
    k = tops(len(R))
    rep_i = reps["2p5km"].set_index("unit")["i"]
    top_units = R.index[R["2p5km"] <= k]
    cc = t.country.values[rep_i.loc[top_units].values]
    allc = t.country.values[rep_i.values]
    d = pd.DataFrame({"n_all": pd.Series(allc).value_counts(), "n_top": pd.Series(cc).value_counts()}).fillna(0).astype(int)
    d["share_all_pct"] = 100 * d.n_all / d.n_all.sum(); d["share_top_pct"] = 100 * d.n_top / d.n_top.sum()
    d["top_over_all_ratio"] = d.share_top_pct / d.share_all_pct
    return d.sort_index()


def main():
    ms = load_multiscale()
    t, xy, V = views()
    ms["_o"] = 0
    res, details = [], {}
    for name, lab in V.items():
        m, R, reps = metrics(ms, t, lab, xy, "best")
        m["view"] = name; m["rep_rule"] = "best-scoring member"
        res.append(m); details[name] = (R, reps, lab)
    m, R, reps = metrics(ms, t, V["primary"], xy, "medoid"); m["view"] = "primary"; m["rep_rule"] = "medoid member (convention sensitivity)"; res.append(m)
    pd.DataFrame(res).to_csv(RESULTS / "e4_campus_view_metrics.csv", index=False)

    # record-level top decile composition vs campus units (primary)
    R, reps, lab = details["primary"]
    rec2 = ms[ms.scale == "2p5km"].set_index("facility_id").loc[t.facility_id]
    rec_rank = rec2.echo_dc_rank_equal.values
    in_top_rec = rec_rank <= TOP10
    unit_of = pd.Series(lab)
    units_in_rec_top = unit_of[in_top_rec]
    k = tops(len(R))
    top_units = set(R.index[R["2p5km"] <= k])
    comp = dict(record_top_decile_records=int(in_top_rec.sum()), record_top_decile_distinct_units=int(units_in_rec_top.nunique()),
                record_top_decile_records_in_multi_record_groups=int(t.campus_group[in_top_rec].notna().sum()),
                multi_record_groups_touching_record_top_decile=int(t.campus_group[in_top_rec].nunique()),
                unit_top_decile_k=k, unit_top_decile_units_also_containing_a_record_top_decile_member=int(len(top_units & set(units_in_rec_top))),
                unit_top_decile_units_new_vs_record_top_decile=int(len(top_units - set(units_in_rec_top))),
                record_top_decile_units_leaving=int(len(set(units_in_rec_top) - top_units)))
    # country composition: record view vs unit view (each own denominator)
    rc = pd.DataFrame({"n_all": t.country.value_counts(), "n_top": t.country[in_top_rec].value_counts()}).fillna(0).astype(int)
    rc["share_all_pct"] = 100 * rc.n_all / rc.n_all.sum(); rc["share_top_pct"] = 100 * rc.n_top / rc.n_top.sum(); rc["top_over_all_ratio"] = rc.share_top_pct / rc.share_all_pct
    uc = country_table(t, lab, R, reps)
    ct = rc.join(uc, lsuffix="_record", rsuffix="_unit", how="outer").fillna(0).round(2)
    ct["delta_top_share_pp"] = (ct.share_top_pct_unit - ct.share_top_pct_record).round(2)
    ct.to_csv(RESULTS / "e4_country_composition_record_vs_campus.csv")
    comp["country_total_variation_distance_pp"] = round(float(ct.delta_top_share_pp.abs().sum() / 2), 2)
    comp["max_abs_country_share_change_pp"] = round(float(ct.delta_top_share_pp.abs().max()), 2)

    # groupwise stable sets: unit top decile at all four scales; also record-level re-computation (reproduces stage 02)
    sub = ms.pivot(index="facility_id", columns="scale", values="echo_dc_rank_equal").loc[t.facility_id]
    comp["record_level_top_decile_all4"] = int((sub[SCALES] <= TOP10).all(axis=1).sum()); comp["record_level_top_decile_ge1"] = int((sub[SCALES] <= TOP10).any(axis=1).sum())
    comp["record_level_share_rank_range_ge200"] = round(float(((sub[SCALES].max(axis=1) - sub[SCALES].min(axis=1)) >= 200).mean()), 3)

    # --- external-case effects under the primary and bracketing views
    cases = pd.read_csv(RESULTS / "e3_case_record_ranks_with_groups.csv")
    ev = pd.read_csv(RESULTS / "e1_external_evidence_table.csv").rename(columns={"facility_name": "facility", "status_evidence_date": "status_date"})
    prim = ev.loc[ev.in_primary_operational_set, "case_id"].tolist()
    rows = []
    for name in ["minimal_T1", "primary", "proximity_only_250m"]:
        R, reps, lab = details[name]
        pos = pd.Series(lab, index=t.facility_id)
        for cid in prim + ["E09", "E10", "E12", "E13"]:
            d = cases[cases.case_id == cid]
            r = dict(view=name, case_id=cid, facility=ev.set_index("case_id").loc[cid, "facility"], n_records=len(d), n_units_spanned=int(pos.loc[d.facility_id].nunique()))
            for s in SCALES:
                us = pos.loc[d.facility_id].values
                r[f"unit_rank_{s}"] = float(np.median(R.loc[us, s].values))
            r["n_units"] = len(R)
            r["unit_rank_2p5km_top_share_pct"] = round(100 * r["unit_rank_2p5km"] / len(R), 1)
            idx = [int(t.index[t.facility_id == f][0]) for f in d.facility_id]
            rep_of_unit = reps["2p5km"].set_index("unit")["i"]
            r["n_records_that_are_unit_representative_2p5km"] = int(sum(rep_of_unit.loc[lab[i]] == i for i in idx))
            r["group_size_of_record(s)"] = ",".join(map(str, sorted(set(int((lab == lab[i]).sum()) for i in idx))))
            rows.append(r)
    ce = pd.DataFrame(rows)
    ce.to_csv(RESULTS / "e4_external_cases_under_campus_views.csv", index=False)
    # primary-set summary under each view (units thresholds)
    sm = []
    for name in ["record", "minimal_T1", "primary", "proximity_only_250m"]:
        if name == "record":
            d = pd.read_csv(RESULTS / "e3_project_positions.csv"); n = N
            ranks = {s: d[f"rank_{s}"].values for s in SCALES}
        else:
            d = ce[(ce.view == name) & ce.case_id.isin(prim)]; n = int(d.n_units.iloc[0])
            ranks = {s: d[f"unit_rank_{s}"].values for s in SCALES}
        k10, k20, k50 = tops(n), math.ceil(.2 * n), math.ceil(.5 * n)
        r = ranks["2p5km"]
        sm.append(dict(view=name, denominator=n, n_projects=len(r), median_rank_2p5km=float(np.median(r)), median_top_share_pct_2p5km=round(100 * float(np.median(r)) / n, 1),
                       top10=int((r <= k10).sum()), top20=int((r <= k20).sum()), top50=int((r <= k50).sum()),
                       top10_all_scales=" / ".join(str(int((ranks[s] <= k10).sum())) for s in SCALES), thresholds=f"{k10}/{k20}/{k50}"))
    pd.DataFrame(sm).to_csv(RESULTS / "e4_primary_set_summary_by_view.csv", index=False)
    json.dump(comp, open(RESULTS / "e4_top_decile_composition.json", "w"), indent=1)
    # unit-level facility table for the primary view (compact: representative + group at 2.5 km)
    reps25 = details["primary"][1]["2p5km"]
    ut = t.iloc[reps25.i.values][["facility_id", "country", "name", "operator", "campus_group", "group_size"]].reset_index(drop=True)
    ut["unit_id"] = t.unit_id.values[reps25.i.values]
    for s in SCALES:
        ut[f"unit_rank_{s}"] = details["primary"][0][s].loc[details["primary"][2][reps25.i.values]].values
    ut.to_csv(RESULTS / "e4_campus_units_primary.csv", index=False)
    return pd.DataFrame(res), comp, ce, pd.DataFrame(sm), ct


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    res, comp, ce, sm, ct = main()
    print(res.T.to_string()); print(json.dumps(comp, indent=1)); print(sm.to_string(index=False))
    print(ce[ce.view == "primary"][["case_id", "facility", "n_records", "n_units_spanned", "unit_rank_1km", "unit_rank_2p5km", "unit_rank_5km", "unit_rank_10km", "unit_rank_2p5km_top_share_pct", "group_size_of_record(s)"]].to_string(index=False))
    print(ct[["n_all_record", "n_top_record", "share_top_pct_record", "n_all_unit", "n_top_unit", "share_top_pct_unit", "delta_top_share_pp"]].to_string())
