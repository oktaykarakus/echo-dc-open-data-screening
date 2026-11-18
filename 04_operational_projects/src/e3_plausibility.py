"""E4: external plausibility analysis (descriptive; NOT predictive validation).

Matching convention (stated in the report):
  * one documented project = one analytical unit. Records of the same project are NOT independent cases.
  * Project position at a scale = median of the ranks of its linked records (primary); best-record and worst-record positions are sensitivities.
  * Primary set = status Operational AND match_confidence High (E1 rules). Everything else is reported separately.
  * 'Top x %' position = rank / 1,216; thresholds are rank <= 122 (10 %), 244 (20 %), 608 (50 %) - stage 02/D convention for the 10 % cut.
No classification metrics are computed: undocumented data centres are not negatives.
"""
from math import comb
from e_lib import *
from e1_evidence import build as build_evidence

IND = ["norm_heat_sum", "norm_pop_density", "norm_hdd_2024", "norm_anchor_count", "norm_anchor_diversity", "norm_nearest_anchor_distance_m",
       "norm_urban_compactness", "norm_major_road_density", "norm_nearest_major_road_distance_m"]
ROAD = ["norm_major_road_density", "norm_nearest_major_road_distance_m"]


def binom_tail(n, k, p):
    """P(X >= k), X ~ Bin(n, p): descriptive reference for 'k of n in the top share p if placement were random'."""
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def project_positions(rec, ev_ids, how="median"):
    f = {"median": np.median, "best": np.min, "worst": np.max}[how]
    rows = []
    for cid, d in rec[rec.case_id.isin(ev_ids)].groupby("case_id", sort=True):
        r = {"case_id": cid, "n_records": len(d)}
        for s in SCALES:
            r[f"rank_{s}"] = float(f(d[f"rank_{s}"]))
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(pos, label):
    out = {"set": label, "n_projects": int(len(pos))}
    r = pos["rank_2p5km"].values
    top = 100 * r / N
    out.update(median_rank_2p5km=float(np.median(r)), min_rank_2p5km=float(r.min()), max_rank_2p5km=float(r.max()),
               median_top_share_pct_2p5km=round(float(np.median(top)), 1), range_top_share_pct_2p5km=f"{top.min():.1f}-{top.max():.1f}",
               median_standing_pct_2p5km=round(float(np.median(100 * (N - r) / (N - 1))), 1))
    for s in SCALES:
        rs = pos[f"rank_{s}"].values
        out[f"top10_{s}"] = int((rs <= TOP10).sum()); out[f"top20_{s}"] = int((rs <= TOP20).sum()); out[f"top50_{s}"] = int((rs <= TOP50).sum())
        out[f"median_rank_{s}"] = float(np.median(rs))
    n = len(pos)
    out["expected_top10_if_random"] = round(.10 * n, 2); out["expected_top20_if_random"] = round(.20 * n, 2); out["expected_top50_if_random"] = round(.50 * n, 2)
    k10, k20, k50 = out["top10_2p5km"], out["top20_2p5km"], out["top50_2p5km"]
    out["binom_tail_P_top10_2p5km"] = round(binom_tail(n, k10, .10), 3)
    out["binom_tail_P_top20_2p5km"] = round(binom_tail(n, k20, .20), 3)
    out["binom_tail_P_top50_2p5km"] = round(binom_tail(n, k50, .50), 3)
    out["n_bottom_half_at_all_4_scales"] = int((pos[[f"rank_{s}" for s in SCALES]] > TOP50).all(axis=1).sum())
    out["n_top_half_at_all_4_scales"] = int((pos[[f"rank_{s}" for s in SCALES]] <= TOP50).all(axis=1).sum())
    return out


def main():
    ev, rec = build_evidence()
    ev = ev.rename(columns={"facility_name": "facility", "status_evidence_date": "status_date"})
    groups = pd.read_csv(RESULTS / "e2_record_campus_groups.csv")[["facility_id", "campus_group", "group_size", "group_status"]]
    rec = rec.merge(groups, on="facility_id", how="left")
    prim_ids = ev.loc[ev.in_primary_operational_set, "case_id"].tolist()
    ms = load_multiscale()
    # --- population ranks of components per scale (descending score, ties = min)
    for s in SCALES:
        m = ms[ms.scale == s]
        ro = m.opportunity_score_equal.rank(ascending=False, method="min").astype(int); rf = m.feasibility_score_equal.rank(ascending=False, method="min").astype(int)
        key = pd.DataFrame({"facility_id": m.facility_id.values, f"opp_rank_{s}": ro.values, f"feas_rank_{s}": rf.values})
        rec = rec.merge(key, on="facility_id", how="left")

    # ---- project positions (median convention) + sensitivities
    pos = project_positions(rec, prim_ids)
    pos = ev[["case_id", "facility", "country", "operator", "status_date", "match_confidence"]].merge(pos, on="case_id")
    for s in SCALES:
        pos[f"topshare_{s}"] = (100 * pos[f"rank_{s}"] / N).round(1)
    pos["scales_in_top10"] = (pos[[f"rank_{s}"for s in SCALES]] <= TOP10).sum(axis=1)
    pos["rank_range_across_scales"] = pos[[f"rank_{s}" for s in SCALES]].max(axis=1) - pos[[f"rank_{s}" for s in SCALES]].min(axis=1)
    pos["campus_groups_of_records"] = pos.case_id.map(rec.groupby("case_id").campus_group.agg(lambda x: ",".join(sorted(set(x.dropna())))))
    pos["n_records"] = pos.case_id.map(rec.groupby("case_id").size())
    pos.to_csv(RESULTS / "e3_project_positions.csv", index=False)

    summ = [summarise(pos, "PRIMARY: operational + high-confidence, median-of-records")]
    summ.append(summarise(project_positions(rec, prim_ids, "best"), "sensitivity: best-ranked linked record"))
    summ.append(summarise(project_positions(rec, prim_ids, "worst"), "sensitivity: worst-ranked linked record"))
    late = ev[(ev.case_id.isin(prim_ids)) & (ev.status_date >= "2023-01-01")].case_id.tolist()
    summ.append(summarise(project_positions(rec, late), "sensitivity: evidence dated >= 2023 only (%s)" % ",".join(late)))
    summ.append(summarise(project_positions(rec, prim_ids + ["E09"]), "sensitivity: + Tallaght (ambiguous match; median of 5 AWS records)"))
    # one-at-a-time exclusion of each primary project (leave-one-project-out)
    for c in prim_ids:
        summ.append(summarise(project_positions(rec, [x for x in prim_ids if x != c]), f"leave-out {c}"))
    pd.DataFrame(summ).to_csv(RESULTS / "e3_plausibility_summary.csv", index=False)

    # ---- non-primary cases reported separately (never pooled with the operational set)
    npos = project_positions(rec, [c for c in ev.case_id if c not in prim_ids and ev.set_index("case_id").loc[c, "n_matched_records"] > 0])
    npos = ev[["case_id", "facility", "status", "match_confidence"]].merge(npos, on="case_id")
    best = project_positions(rec, npos.case_id.tolist(), "best"); worst = project_positions(rec, npos.case_id.tolist(), "worst")
    npos["best_rank_2p5km"] = npos.case_id.map(best.set_index("case_id").rank_2p5km); npos["worst_rank_2p5km"] = npos.case_id.map(worst.set_index("case_id").rank_2p5km)
    npos.to_csv(RESULTS / "e3_nonprimary_positions.csv", index=False)

    # ---- components and road accessibility (2.5 km and all scales; median over linked records)
    rows = []
    d3 = pd.read_csv(PD / "results" / "d3_facility_ranks.csv").set_index("facility_id")
    for cid in prim_ids:
        d = rec[rec.case_id == cid]
        r = {"case_id": cid, "facility": ev.set_index("case_id").loc[cid, "facility"], "n_records": len(d)}
        for s in SCALES:
            r[f"opp_rank_{s}"] = float(np.median(d[f"opp_rank_{s}"])); r[f"feas_rank_{s}"] = float(np.median(d[f"feas_rank_{s}"]))
            r[f"opp_minus_feas_rank_{s}"] = r[f"opp_rank_{s}"] - r[f"feas_rank_{s}"]
        r["story_2p5km"] = ("Opportunity higher than Feasibility" if r["opp_rank_2p5km"] + 244 < r["feas_rank_2p5km"] else
                            "Feasibility higher than Opportunity" if r["feas_rank_2p5km"] + 244 < r["opp_rank_2p5km"] else "similar (<=244 places apart)")
        r["both_components_top20_2p5km"] = bool(r["opp_rank_2p5km"] <= TOP20 and r["feas_rank_2p5km"] <= TOP20)
        r["rank_2p5km_full"] = float(np.median(d3.loc[d.facility_id, "rank_full"])); r["rank_2p5km_without_roads"] = float(np.median(d3.loc[d.facility_id, "rank_wo_road_accessibility"]))
        r["road_removal_rank_shift"] = r["rank_2p5km_full"] - r["rank_2p5km_without_roads"]          # + means better without roads
        m = ms[(ms.scale == "2p5km")]
        for ind in IND:
            allv = m[ind].rank(pct=True, method="average") * 100
            r[f"pct_{ind}"] = round(float(allv[m.facility_id.isin(d.facility_id)].mean()), 0)
        pcts = {i: r[f"pct_{i}"] for i in IND}
        low = sorted(pcts, key=pcts.get)[:2]
        r["two_lowest_indicators_2p5km"] = "; ".join(f"{i.replace('norm_', '')} ({pcts[i]:.0f})" for i in low)
        r["road_indicators_pct_mean"] = round(float(np.mean([pcts[i] for i in ROAD])), 0)
        rows.append(r)
    comp = pd.DataFrame(rows)
    comp.to_csv(RESULTS / "e3_components_and_roads.csv", index=False)

    # ---- the four original cases: original vs B0-QA vs campus
    four = pd.read_csv(PB / "results" / "four_cases_vs_original_B0QA.csv")
    f = []
    cmap = {"Equinix PA10": "E01", "Telia HDC": "E02", "Meta Odense Data center": "E03", "NorthC Datacenter Aalsmeer": "E04"}
    for _, r in four.iterrows():
        cid = cmap[r["case"]]
        d = rec[rec.case_id == cid]
        f.append(dict(case=r["case"], case_id=cid, original_facility_id=r.facility_id, original_rank=int(r.orig_rank), b0qa_rank=int(r.new_rank),
                      project_records=len(d), project_ranks_2p5km="/".join(map(str, d.rank_2p5km)), project_ranks_1_5_10km=" ; ".join(f"{s}:{'/'.join(map(str, d['rank_'+s]))}" for s in ["1km", "5km", "10km"]),
                      original_record_is_in_E1_match=bool(r.facility_id in set(d.facility_id)), campus_groups=",".join(sorted(set(d.campus_group.dropna()))),
                      original_status_label={"E01": "Operational: Yes", "E02": "Operational: Yes", "E03": "Operational: Yes", "E04": "Operational: Not confirmed"}[cid],
                      phase_e_status=ev.set_index("case_id").loc[cid, "status"], phase_e_match=ev.set_index("case_id").loc[cid, "match_confidence"]))
    pd.DataFrame(f).to_csv(RESULTS / "e3_original_four_verification.csv", index=False)
    rec.to_csv(RESULTS / "e3_case_record_ranks_with_groups.csv", index=False)
    return ev, rec, pos, pd.DataFrame(summ), comp, npos


if __name__ == "__main__":
    ev, rec, pos, summ, comp, npos = main()
    pd.set_option("display.width", 250)
    print(pos[["case_id", "facility", "n_records", "rank_1km", "rank_2p5km", "rank_5km", "rank_10km", "topshare_2p5km", "scales_in_top10"]].to_string(index=False))
    print(summ.iloc[:5].T.to_string())
    print(comp[["case_id", "opp_rank_2p5km", "feas_rank_2p5km", "story_2p5km", "rank_2p5km_full", "rank_2p5km_without_roads", "road_indicators_pct_mean", "two_lowest_indicators_2p5km"]].to_string(index=False))
    print(npos.to_string(index=False))
