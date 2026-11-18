"""stage 03 tests. Plain asserts; runs without pytest:  python test_phase_d.py   (pytest also collects the test_* functions)."""
import sys, traceback
from d_lib import *
from d0_verify import independent_norm
from d4_normalisation import complete_case, equal_score, WINDOWS
from d6_crossscale import config_table, ranks_for

DF = load_table(); O, _ = baseline(DF); N = len(DF); K = n_top(N)
WO, WF = full_weights("equal", O)

def _valid(r, n=N):
    assert sorted(r["rank"]) == list(range(1, n + 1))
    assert np.isfinite(r[["opp", "feas", "raw"]].to_numpy()).all()
    assert r[["opp", "feas"]].to_numpy().min() >= 0 and r[["opp", "feas"]].to_numpy().max() <= 1 + 1e-12

def test_facility_id_integrity():
    assert N == 1216 and DF.facility_id.is_unique and DF.country.nunique() == 20
    ids0 = pd.read_csv(PB / "data" / "facility_ids_original.csv")
    assert set(DF.facility_id) == set(ids0.facility_id)
    for f in ("d1_facility_ranks.csv", "d2_facility_ranks.csv", "d3_facility_ranks.csv"):
        t = pd.read_csv(RESULTS / f); assert len(t) == N and (t.facility_id.to_numpy() == DF.facility_id.to_numpy()).all(), f

def test_exact_baseline_reproduction():
    ref = pd.read_csv(PB / "results" / "scores_B0QA.csv")
    for sc in SCENARIOS:
        assert np.abs(O[f"echo_dc_score_raw_{sc}"] - ref[f"echo_dc_score_raw_{sc}"]).max() <= TOL
        assert (O[f"echo_dc_rank_{sc}"].to_numpy() == ref[f"echo_dc_rank_{sc}"].to_numpy()).all()
    for c in S.ALL_IND: assert np.abs(O[c] - ref[c]).max() <= TOL
    # generic scorer with full weights == baseline, all three scenarios
    for sc in SCENARIOS:
        wo, wf = full_weights(sc, O); r = score(O, wo, wf)
        assert np.array_equal(r["rank"].to_numpy(), O[f"echo_dc_rank_{sc}"].to_numpy()) and np.abs(r.raw - O[f"echo_dc_score_raw_{sc}"]).max() <= TOL

def test_independent_transformations():
    for k, v in independent_norm(DF).items(): assert np.abs(v - O[k].to_numpy()).max() <= TOL, k

def test_indicator_directions():
    rev = {"norm_nearest_anchor_distance_m": "nearest_anchor_distance_m", "norm_nearest_major_road_distance_m": "nearest_major_road_distance_m"}
    for k, raw in rev.items(): assert stats.spearmanr(O[k], DF[raw])[0] < -0.9
    for k, raw in {"norm_heat_sum_2p5km": "hotmaps_heat_sum_2p5km", "norm_pop_density_2p5km": "pop_density_2p5km", "norm_hdd_2024": "hdd_2024",
                   "norm_anchor_count_2p5km": "anchor_count_2p5km", "norm_major_road_density_2p5km": "major_road_density_2p5km_km_per_km2"}.items():
        assert stats.spearmanr(O[k], DF[raw])[0] > 0.9
    for alt in ("none_min_max", "modest_2p5_97p5"):                       # alternative windows keep every direction
        A = normalise_q(DF, *WINDOWS[alt])
        for k in rev: assert stats.spearmanr(A[k], DF[rev[k]])[0] < -0.9
        for c in S.ALL_IND: assert A[c].min() >= 0 and A[c].max() <= 1

def test_normalise_q_reproduces_v1():
    A = normalise_q(DF, 0.05, 0.95)
    for c in S.ALL_IND: assert np.array_equal(A[c].to_numpy(), O[c].to_numpy())
    N0 = normalise_q(DF, 0.0, 1.0)                                         # no clipping: pure (t - min)/(max - min)
    x = np.log1p(DF.pop_density_2p5km); assert np.allclose(N0.norm_pop_density_2p5km, (x - x.min()) / (x.max() - x.min()), atol=1e-12)

def test_weight_renormalisation():
    for sc in SCENARIOS:
        wo, wf = full_weights(sc, O)
        for fam, cols in FAMILIES.items():
            a, b = adjust_pair(wo, wf, drop=cols)
            assert abs(sum(a.values()) - 1) < 1e-12 and abs(sum(b.values()) - 1) < 1e-12 and not (set(a) | set(b)) & set(cols)
            for kk in a: assert abs(a[kk] / a[list(a)[0]] - wo[kk] / wo[list(a)[0]]) < 1e-9           # relative proportions preserved
    a, b = adjust_pair(WO, WF, drop=FAMILIES["anchors"]); assert all(abs(v - 1 / 3) < 1e-12 for v in a.values())     # equal -> equal on remainder
    assert adjust(WO, [], {c: 1.0 for c in WO}) == WO or all(abs(adjust(WO, [], {c: 1.0 for c in WO})[c] - WO[c]) < 1e-15 for c in WO)
    for bad in (lambda: adjust(WO, list(WO)), lambda: adjust(WO, ["not_an_indicator"])):
        try: bad(); raise AssertionError("expected failure")
        except (ValueError, KeyError): pass

def test_families_and_d2_structure():
    fam_all = [c for v in FAMILIES.values() for c in v]
    assert len(fam_all) == len(set(fam_all)) and POP not in fam_all and set(fam_all) <= set(S.OPP) | set(S.FEAS)
    for fam, cols in FAMILIES.items():                                                    # never an entire component
        assert len(set(S.OPP) - set(cols)) >= 2 and len(set(S.FEAS) - set(cols)) >= 2, fam
    a, b = d2_weights("D2-B", WO, WF); assert POP not in a and POP in b and len(a) == 5 and abs(sum(a.values()) - 1) < 1e-12 and b == WF
    a, b = d2_weights("D2-C", WO, WF); assert POP in a and POP not in b and len(b) == 3 and abs(sum(b.values()) - 1) < 1e-12 and a == WO
    a, b = d2_weights("D2-A", WO, WF); assert a == WO and b == WF and np.array_equal(score(O, a, b)["rank"], O.echo_dc_rank_equal)
    a, b = adjust_pair(WO, WF, mult={c: 1.0 for c in list(WO) + list(WF)}); assert np.array_equal(score(O, a, b)["rank"], O.echo_dc_rank_equal)

def test_valid_scores_and_ranks():
    for c in D2_CONFIGS: _valid(score(O, *d2_weights(c, WO, WF)))
    for cols in list(FAMILIES.values()) + list(SUPP_FAMILIES.values()): _valid(score(O, *adjust_pair(WO, WF, drop=cols)))
    for w in WINDOWS.values(): _valid(equal_score(S.normalise(DF) if w == (0.05, 0.95) else normalise_q(DF, *w)))
    for s in SCALES: _valid(score(baseline(DF, s)[0], WO, WF))
    for g in (POP,): _valid(score(O, *adjust_pair(WO, WF, mult={g: 2.0})))

def test_top_decile_definitions():
    assert n_top(1216) == 122 and n_top(1201) == 121 and n_top(1214) == 122
    r = O.echo_dc_rank_equal.to_numpy(); m = compare(r, r)
    assert m["top_decile_overlap"] == K and m["spearman"] == 1.0 and m["max_abs_drank"] == 0 and m["n_enter_top_decile"] == 0
    rng = np.random.default_rng(0); r2 = rng.permutation(N) + 1; m = compare(r, r2); m2 = compare(r2, r)
    assert m["top_decile_overlap"] == m2["top_decile_overlap"] and m["n_enter_top_decile"] == m["n_leave_top_decile"] == K - m["top_decile_overlap"]
    assert int((r <= K).sum()) == K and 0 <= m["top20_overlap"] <= 20
    # rank_desc: ties broken by secondary then row order
    assert rank_desc([1, 1, 1, 0], [0, 5, 5, 9]).tolist() == [3, 1, 2, 4]
    assert rank_desc([2, 3, 3]).tolist() == [3, 1, 2]

def test_common_set_comparison():
    flag = DF.hotmaps_small_buffer_missing_originally.to_numpy()
    m, ch = complete_case(DF, flag, "1km", "t")
    assert m["n_common"] == 1201 and m["n"] == 1201 and m["k_top_decile"] == 121 and m["n_excluded"] == 15
    assert m["n_enter_top_decile"] == m["n_leave_top_decile"] == len(ch) // 2 and not set(ch.facility_id) & set(DF.facility_id[flag])
    assert m["top_decile_overlap"] + m["n_enter_top_decile"] == 121 and m["excluded_in_original_top_decile"] == 0
    z = (DF.hotmaps_heat_sum_1km == 0).to_numpy(); assert (z == flag).all()          # every 1 km zero is a flagged gap (no observed zeros)

def test_determinism():
    a = ranks_for(DF, "1km", "none_min_max", lambda wo, wf: (wo, wf)); b = ranks_for(DF, "1km", "none_min_max", lambda wo, wf: (wo, wf))
    assert np.array_equal(a, b)
    c1 = compare(O.echo_dc_rank_equal, score(O, *d2_weights("D2-C", WO, WF))["rank"]); c2 = compare(O.echo_dc_rank_equal, score(O, *d2_weights("D2-C", WO, WF))["rank"])
    assert c1 == c2

def test_result_tables_consistent():
    need = ["d0_baseline_verification.json", "d1_composite_vs_simple_summary.csv", "d2_population_comparisons.csv", "d3_ablation_summary.csv", "d4a_clipping_summary.csv",
            "d4b_hotmaps_missingness.csv", "d5_scenario_robustness.csv", "d5_weight_perturbations.csv", "d6_alternative_effects_by_radius.csv", "d6_multiscale_stability_by_config.csv"]
    for f in need: assert (RESULTS / f).exists(), f
    d1 = pd.read_csv(RESULTS / "d1_composite_vs_simple_summary.csv"); assert len(d1) == 4 and (d1.both_top_decile + d1.composite_only_top_decile == K).all()
    assert len(pd.read_csv(RESULTS / "d3_ablation_summary.csv")) == 6 and len(pd.read_csv(RESULTS / "d5_weight_perturbations.csv")) == 15
    assert (pd.read_csv(RESULTS / "d3_ablation_summary.csv").n_enter_top_decile == pd.read_csv(RESULTS / "d3_ablation_summary.csv").n_leave_top_decile).all()
    assert json.load(open(RESULTS / "d3_heat_diagnosis.json"))["phaseC_label_replicated_exactly"] is True
    d0 = json.load(open(RESULTS / "d0_baseline_verification.json")); assert d0["status"] == "PASS" and d0["protected_files_changed"] == []
    base = pd.read_csv(RESULTS / "d6_multiscale_stability_by_config.csv").set_index("config").loc["base"]
    assert base.n_top_decile_all_four_radii == 67 and base.n_top_decile_any_radius == 183                                  # stage 02 headline numbers

def test_protected_files_unchanged():
    old = json.load(open(RESULTS / "d0_protected_manifest.json")); cur = protected_manifest()
    assert old == cur, sorted(k for k in set(old) | set(cur) if old.get(k) != cur.get(k))
    assert not [p for d in PROTECTED_DIRS for p in (V2 / d).rglob("__pycache__")]

if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try: fn(); print("PASS", name)
        except Exception: fails += 1; print("FAIL", name); traceback.print_exc()
    print(f"{fails} failed"); sys.exit(1 if fails else 0)
