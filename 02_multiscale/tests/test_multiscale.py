"""Focused tests for stage 02. Run: python -m pytest 02_multiscale/tests   (needs pandas, numpy, scipy; no geopandas).
Pipeline outputs must exist (run src/c4_scores.py first)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import pytest
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from cfg import *
import multiscale as M, stability as ST
import scoring as S

@pytest.fixture(scope="module")
def table():
    return M.load_table()

@pytest.fixture(scope="module")
def long():
    return pd.read_csv(RESULTS / "multiscale_scores_long.csv")

def test_reference_scale_reproduces_b0qa(table):
    """2.5 km through the column adapter == frozen stage 01 B0-QA (all scenarios, indicators, ranks)."""
    out, _ = M.score_scale(table, "2p5km"); ref = pd.read_csv(B0QA)
    assert (out.facility_id.values == ref.facility_id.values).all()
    for sc in SCENARIOS:
        for k in ["opportunity_score", "feasibility_score", "echo_dc_score_raw"]:
            assert np.abs(out[f"{k}_{sc}"] - ref[f"{k}_{sc}"]).max() < 1e-12
        assert (out[f"echo_dc_rank_{sc}"].values == ref[f"echo_dc_rank_{sc}"].values).all()
    for c in S.ALL_IND:
        assert np.abs(out[c] - ref[c]).max() < 1e-12

def test_adapter_uses_correct_radius_columns(table):
    d = M.scale_inputs(table, "10km")
    assert (d["anchor_count_2p5km"] == table["anchor_count_10km"]).all()
    assert (d["major_road_density_2p5km_km_per_km2"] == table["major_road_density_10km_km_per_km2"]).all()
    assert (d["hotmaps_heat_sum_2p5km"] == table["hotmaps_heat_sum_10km"]).all()
    for c in M.SCALE_INVARIANT:                      # invariants untouched
        assert (d[c] == table[c]).all()
    assert (M.scale_inputs(table, "2p5km")[list(M.SCALE_DEPENDENT)].values == table[list(M.SCALE_DEPENDENT)].values).all()

def test_invariants_are_not_scale_dependent(table):
    # normalised invariant indicators are identical at every radius
    a, _ = M.score_scale(table, "1km"); b, _ = M.score_scale(table, "10km")
    for c in ["norm_hdd_2024", "norm_nearest_anchor_distance_m", "norm_nearest_major_road_distance_m"]:
        assert (a[c].values == b[c].values).all()

def test_long_table_integrity(long, table):
    n = len(table)
    assert len(long) == 4 * n and long.groupby("scale").facility_id.nunique().eq(n).all()
    assert set(long.scale) == set(SCALES) and long.country.nunique() == 20
    for s in SCALES:
        t = long[long.scale == s]
        assert (t.facility_id.values == table.facility_id.values).all()           # common ordering
        for sc in SCENARIOS + ["entropy_ref"]:
            assert sorted(t[f"echo_dc_rank_{sc}"]) == list(range(1, n + 1))
    ind = [c for c in long.columns if c.startswith("norm_")]
    assert long[ind].min().min() >= 0 and long[ind].max().max() <= 1 and np.isfinite(long[ind].to_numpy()).all()

def test_rank_direction_and_top_decile(long):
    for s in SCALES:
        t = long[long.scale == s]
        top = t.sort_values("echo_dc_score_raw_equal", ascending=False).head(5).echo_dc_rank_equal.tolist()
        assert top == [1, 2, 3, 4, 5]
    assert M.n_top(1216) == 122                      # same definition as stage 01 (ceil 10%)

def test_distance_indicators_direction(table):
    out, _ = M.score_scale(table, "5km")
    # nearer -> higher normalised score (reversed)
    i, j = table.nearest_major_road_distance_m.idxmin(), table.nearest_major_road_distance_m.idxmax()
    assert out.loc[i, "norm_nearest_major_road_distance_m"] > out.loc[j, "norm_nearest_major_road_distance_m"]

def test_road_density_matches_length_over_area(table):
    for s in SCALES:
        d = table[f"major_road_length_{s}_km"] / table[f"buffer_area_{s}_km2"]
        assert np.abs(d - table[f"major_road_density_{s}_km_per_km2"]).max() < 1e-4        # 4 dp rounding
    r = {s: np.sqrt(table[f"buffer_area_{s}_km2"].mean() / np.pi) for s in SCALES}          # km, buffers in EPSG:3035 metres
    for s in SCALES:
        assert abs(r[s] - SCALE_KM[s]) / SCALE_KM[s] < 0.005
    # lengths non-decreasing with radius
    L = table[[f"major_road_length_{s}_km" for s in SCALES]].to_numpy()
    assert (np.diff(L, axis=1) >= -1e-9).all()

def test_c3_recompute_identical():
    p = RESULTS / "c3_road_recompute_vs_phaseB.csv"
    if not p.exists():
        pytest.skip("run c3_road_recompute.py (needs geopandas + cached roads)")
    t = pd.read_csv(p)
    assert (t["n_diff_gt_1e-9"] == 0).all() and (t.nan_mismatch == 0).all()

def test_protocol_b_reference_equals_protocol_a(table):
    a, w = M.score_scale(table, REF)
    b, ref = M.feasibility_protocol_b(table, REF)
    assert np.abs(b.feasibility_B_equal - a.feasibility_score_equal).max() < 1e-12
    assert np.abs(b.feasibility_B_literature_informed - a.feasibility_score_literature_informed).max() < 1e-12
    for k in ("norm_pop_density", "norm_major_road_density"):          # fitted on the 2.5 km column
        assert ref[k][0] < ref[k][1]

def test_protocol_b_keeps_fixed_window_and_clips(table):
    b10, ref = M.feasibility_protocol_b(table, "10km")
    lo, hi = ref["norm_major_road_density"]
    x = table["major_road_density_10km_km_per_km2"]
    assert np.allclose(b10.norm_major_road_density, ((x - lo) / (hi - lo)).clip(0, 1))
    assert b10.norm_major_road_density.between(0, 1).all()

def test_metrics_known_values():
    r = {s: np.arange(1, 11) for s in SCALES}
    g = ST.global_metrics(r, 2)
    assert g["spearman"].to_numpy().min() == pytest.approx(1.0) and g["n_top_all_four"] == 2
    r2 = dict(r); r2["10km"] = np.arange(10, 0, -1)
    g2 = ST.global_metrics(r2, 2)
    assert g2["spearman"].loc["1km", "10km"] == pytest.approx(-1.0) and g2["n_top_all_four"] == 0
    assert g2["top_overlap"].loc["1km", "10km"] == 0

def test_sensitivity_table_logic():
    ids = ["a", "b", "c"]; ctry = ["X", "X", "Y"]
    ranks = {"1km": np.array([1, 5, 9]), "2p5km": np.array([2, 4, 9]), "5km": np.array([30, 3, 9]), "10km": np.array([1, 50, 9])}
    t = M.sensitivity_table(ids, ctry, ranks, 5).set_index("facility_id")
    assert t.loc["a", "rank_range"] == 29 and t.loc["a", "best_rank"] == 1 and t.loc["a", "worst_rank"] == 30
    assert t.loc["a", "n_top_decile"] == 3 and t.loc["a", "top_status_5km_vs_ref"] == "exits"
    assert t.loc["b", "top_status_10km_vs_ref"] == "exits" and t.loc["c", "top_pattern"] == "none"
    assert t.loc["a", "rank_sd"] == pytest.approx(np.std([1, 2, 30, 1], ddof=1))

def test_deterministic_rerun(table, long):
    out, _ = M.score_scale(table, "5km")
    t = long[long.scale == "5km"]
    assert (out.echo_dc_rank_equal.values == t.echo_dc_rank_equal.values).all()
    assert np.abs(out.echo_dc_score_raw_entropy.values - t.echo_dc_score_raw_entropy.values).max() < 1e-12
