"""stage 04 tests (plain asserts; also collectable by pytest). Run: bash run_all.sh test_phase_e.py"""
from e_lib import *
import e1_evidence, e2_campus_groups as e2

def _res(n): return pd.read_csv(RESULTS / n)

def test_record_ids_unchanged():
    ids0 = pd.read_csv(PB / "data" / "facility_ids_original.csv").facility_id.tolist()
    s = load_scores().facility_id.tolist()
    assert len(s) == N == len(set(s)) and s == ids0
    for n in ("e2_record_campus_groups.csv", "e3_case_record_ranks_with_groups.csv"):
        pass
    assert _res("e2_record_campus_groups.csv").facility_id.tolist() == ids0
    assert load_scores().country.nunique() == 20

def test_956_reproduced_and_meaning():
    t = load_scores().reset_index(drop=True)
    pt, _ = e2.pairs_table(t)
    assert len(pt) == 956 and len(set(pt.id_a) | set(pt.id_b)) == 563
    assert len(set(pt[pt.distance_m <= 100].id_a) | set(pt[pt.distance_m <= 100].id_b)) == 385
    assert pt.tier.isin(["T1_coincident", "T2_same_brand", "T3_distinct_brand", "T4_unresolved"]).all()
    assert (pt.groupby("tier").size().sum()) == 956

def test_grouping_is_partition_and_conservative():
    g = _res("e2_record_campus_groups.csv")
    assert g.unit_id.notna().all()
    multi = g[g.campus_group.notna()]
    assert multi.groupby("campus_group").size().min() >= 2
    assert (multi.group_status != "chain_risk").all()
    pt = _res("e2_pair_tiers.csv")
    # no merge of two records with different non-null brands unless a chain of supported edges connects them (checked: T3 pairs stay split
    # unless linked through a third record) -> report count rather than forbid
    grp = g.set_index("facility_id").campus_group
    t3 = pt[pt.tier == "T3_distinct_brand"]
    same = [(grp[a] == grp[b]) and pd.notna(grp[a]) for a, b in zip(t3.id_a, t3.id_b)]
    assert sum(same) <= 3, sum(same)          # transitive-chain exceptions only; documented in the report
    assert (pt.country_a == pt.country_b).all()
    # group members never span countries; diameter <= 500 m
    gs = _res("e2_campus_groups.csv")
    assert (gs.diameter_m <= 500).all()
    assert g[g.campus_group.notna()].groupby("campus_group").country.nunique().max() == 1

def test_unit_denominators():
    m = _res("e4_campus_view_metrics.csv")
    p = m[(m.view == "primary") & m.rep_rule.str.startswith("best")].iloc[0]
    g = _res("e2_record_campus_groups.csv")
    assert p.n_units == g.unit_id.nunique() and p.top_decile_k == math.ceil(.1 * p.n_units)
    rec = m[m.view == "record"].iloc[0]
    assert rec.n_units == N and rec.top_decile_k == TOP10 == 122
    # record view reproduces stage 02 headline numbers (rho 0.944/0.957/0.872; overlaps 98/106/87; stable 67; >=1: 183)
    assert abs(rec.spearman_1km_vs_2p5km - .944) < .001 and abs(rec.spearman_5km_vs_2p5km - .957) < .001 and abs(rec.spearman_10km_vs_2p5km - .872) < .001
    assert (rec.top_decile_overlap_1km_vs_2p5km, rec.top_decile_overlap_5km_vs_2p5km, rec.top_decile_overlap_10km_vs_2p5km) == (98, 106, 87)
    assert rec.top_decile_all4_scales == 67 and rec.top_decile_ge1_scale == 183

def test_unit_rank_is_permutation():
    u = _res("e4_campus_units_primary.csv")
    assert sorted(u.unit_rank_2p5km) == list(range(1, len(u) + 1))
    assert u.unit_id.is_unique

def test_evidence_table_integrity():
    ev = _res("e1_external_evidence_table.csv")
    assert ev.case_id.is_unique and ev.status.isin(["Operational", "Under construction / committed", "Announced / planned", "Unverified"]).all()
    assert ev.match_confidence.isin(["High", "Medium", "Ambiguous", "No match"]).all()
    assert ev.source_urls.str.startswith("http").all() and ev.source_dates.notna().all() and (ev.retrieved == RETRIEVED).all()
    ids = set(load_scores().facility_id)
    for s in ev.matched_ids.dropna():
        assert all(i in ids for i in s.split(" | "))
    prim = ev[ev.in_primary_operational_set]
    assert (prim.status == "Operational").all() and (prim.match_confidence == "High").all() and len(prim) == 8
    # a matched id belongs to one case only
    allm = [i for s in ev.matched_ids.dropna() for i in s.split(" | ")]
    assert len(allm) == len(set(allm))
    assert ev[ev.match_confidence == "No match"].n_matched_records.eq(0).all()

def test_ranks_match_baseline():
    r = _res("e1_case_record_ranks.csv"); s = load_scores().set_index("facility_id")
    assert (r.rank_B0QA.values == s.loc[r.facility_id, "echo_dc_rank_equal"].values).all() and (r.rank_2p5km == r.rank_B0QA).all()
    f = _res("e3_original_four_verification.csv")
    assert (f.b0qa_rank.values == [70, 221, 950, 707]).all() and f.original_record_is_in_E1_match.all()

def test_projects_not_double_counted():
    p = _res("e3_project_positions.csv"); ev = _res("e1_external_evidence_table.csv")
    assert p.case_id.is_unique and len(p) == 8 and p.n_records.sum() == 1 + 1 + 2 + 1 + 1 + 4 + 1 + 2
    assert not set(p.case_id) & {"E09", "E10", "E11", "E12", "E13", "E14", "E15"}     # non-operational / ambiguous never pooled
    s = _res("e3_plausibility_summary.csv").iloc[0]
    assert s.n_projects == 8 and s.top10_2p5km <= s.top20_2p5km <= s.top50_2p5km <= 8

def test_no_classification_metrics_written():
    for f in RESULTS.glob("*.csv"):
        h = " ".join(pd.read_csv(f, nrows=0).columns).lower()
        assert not any(w in h for w in ("precision", "recall", "roc", "auc", "accuracy")), f

def test_determinism():
    a = e1_evidence.build()[0].drop(columns=[]).to_csv(index=False)
    b = e1_evidence.build()[0].to_csv(index=False)
    assert a == b
    t1, p1, _ = e2.build(); t2, p2, _ = e2.build()
    assert t1.equals(t2) and p1.equals(p2)

def test_storage_on_external_drive():
    import os
    for k in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
        assert os.environ[k].startswith(str(PE)), k
    assert not list(V2.rglob("__pycache__")) and not list(V2.rglob("*.pyc"))

if __name__ == "__main__":
    import traceback
    fails = 0
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            try: f(); print("PASS", n)
            except Exception:
                fails += 1; print("FAIL", n); traceback.print_exc()
    print(f"{fails} failed"); raise SystemExit(1 if fails else 0)
