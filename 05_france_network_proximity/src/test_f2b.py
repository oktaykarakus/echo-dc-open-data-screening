"""F2-B tests (plain asserts; run: source src/env.sh && /usr/bin/python3 src/test_f2b.py)."""
import os, sys, json, hashlib, subprocess, math
import numpy as np, pandas as pd, shapely, geopandas as gpd
from pyproj import Transformer
from scipy.stats import mannwhitneyu
HERE = os.path.dirname(os.path.abspath(__file__)); PH = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import f2b_lib as L, f2b_01_labels as LB
RES = os.path.join(PH, "results"); EXT = os.path.join(PH, "data", "extracted")
lab = pd.read_csv(os.path.join(RES, "f2b_labels.csv"))

def test_network_status_filtering():
    Lh = gpd.read_file(os.path.join(EXT, "reseaux_de_chaleur.shp"))
    assert len(Lh) == 1035 and Lh.Identifian.notna().sum() == 842 and Lh.Identifian.isna().sum() == 193
    assert (gpd.read_file(os.path.join(EXT, "reseaux_de_chaleur_sans_traces.shp")).geom_type == "Point").all()
    src = open(os.path.join(HERE, "f2b_01_labels.py")).read()
    for banned in ("reseaux_de_froid", "reseaux_en_construction", "pdp.", "zones_et_reseaux", ".geojson"):
        assert f'"{banned}' not in src.replace("'", '"'), banned          # cold / construction / PDP layers never read
    assert "annee_crea" not in src   # commissioning year not used (coverage < 95 %)

def test_transformations():
    t, b = Transformer.from_crs(4326, 2154, always_xy=True), Transformer.from_crs(2154, 4326, always_xy=True)
    x, y = t.transform(2.35, 48.85); lon, lat = b.transform(x, y)
    assert abs(lon - 2.35) < 1e-7 and abs(lat - 48.85) < 1e-7
    x2, y2 = t.transform(2.35, 48.85 + 0.009)
    assert 990 < math.hypot(x2 - x, y2 - y) < 1010          # 0.009 deg lat ~ 1.0 km
    assert 6.8e6 < y < 6.9e6 and 6e5 < x < 7e5               # Paris in Lambert-93

def test_distance_and_independent_check():
    ln = shapely.linestrings([[0, 0], [1000, 0]]); pts = np.array([[500, 300], [2000, 0], [-100, 0]], float)
    d = LB.nearest_dist(pts, np.array([ln]))
    assert np.allclose(d, [300, 1000, 100])
    # independent recomputation with geopandas (union of lines) for 40 French facilities
    Lh = gpd.read_file(os.path.join(EXT, "reseaux_de_chaleur.shp")); g1 = shapely.make_valid(Lh.geometry.values[Lh.Identifian.notna().values])
    u = shapely.union_all(g1)
    s = lab.sample(40, random_state=1)
    ind = np.array([u.distance(shapely.Point(a, b)) for a, b in zip(s.x2154, s.y2154)])
    assert np.allclose(ind, s.d_tier1_m.values, atol=0.01), np.abs(ind - s.d_tier1_m.values).max()

def test_thresholds_and_indeterminate_rules():
    d1 = np.array([2500.0, 2500.01, 3000, 3000, 3000]); d2 = np.array([9e9, 9e9, 2000, 9e9, 9e9]); dc = np.array([9e9, 9e9, 9e9, 5000, 5000.01])
    assert list(LB.classify(d1, d2, dc, 2500)) == ["positive", "mapped_negative", "indeterminate", "indeterminate", "mapped_negative"]
    assert list(LB.classify(d1, d2, dc, 2500, tier2_as_existing=True))[2] == "positive"
    assert set(lab.status_2500) == {"positive", "mapped_negative", "indeterminate"}

def test_sample_eligibility_and_unit_consistency():
    a = json.load(open(os.path.join(RES, "f2b_outcome_audit.json")))
    assert len(lab) == 331 == a["french_records"] and lab.facility_id.is_unique and (lab.country == "FR").all()
    c = lab.status_2500.value_counts()
    assert (c["positive"], c["mapped_negative"], c["indeterminate"]) == (247, 75, 9) and a["t2500"]["eligible"] == 322
    assert lab.unit_id.nunique() == 276
    assert lab.groupby("unit_id").status_2500.nunique().max() == 1
    m = pd.read_csv(LB.MASTER, usecols=["facility_id"]); assert set(lab.facility_id) <= set(m.facility_id) and len(m) == 1216

def test_ties_and_pareto():
    v = np.array([1.0, 3.0, 3.0, 2.0]); p = L.strict_position(v)
    assert list(p) == [3, 0, 1, 2]                      # tie -> earlier row first
    V = np.array([[1, 1, 1], [1, 1, 1], [2, 0, 1], [0, 0, 0], [2, 2, 2]], float)
    assert list(L.pareto_layers(V)) == [2, 2, 2, 3, 1]  # (2,2,2) dominates all; duplicates share a layer; (2,0,1) dominated by (2,2,2) only
    V2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float); assert list(L.pareto_layers(V2)) == [1, 1, 1]
    df, vals, pos = L.load_arms()
    assert sorted(pos["C"]) == list(range(1216)) and sorted(pos["A"]) == list(range(1216))
    order = np.argsort(pos["C"]); lay = df.pareto_layer.values[order]; assert (np.diff(lay) >= 0).all()   # layer ascending
    s = pd.read_csv(L.SCORES); assert (pos["A"] + 1 == s.echo_dc_rank_equal.values).all()                  # reproduces stage 01 rank

def test_metric_denominators():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0]); key = np.arange(10)
    m = L.topk_metrics(y, key, np.arange(10), 0.10)
    assert m["k"] == 1 and m["tp"] == 1 and m["precision"] == 1.0 and abs(m["recall"] - 1 / 3) < 1e-12 and abs(m["enrichment"] - 1 / 0.3) < 1e-12
    m = L.topk_metrics(y, key, np.arange(10), 0.25); assert m["k"] == 3 and m["tp"] == 2 and abs(m["precision"] - 2 / 3) < 1e-12
    idx = np.array([0, 0, 5, 9]); m = L.topk_metrics(y, key, idx, 0.5); assert m["k"] == 2 and m["tp"] == 2   # duplicates allowed, k from resample size
    # AUC equals Mann-Whitney U / (n1 n0), ties 0.5
    rng = np.random.default_rng(3); yy = rng.integers(0, 2, 80); vv = np.round(rng.normal(size=80) + yy, 1)
    u = mannwhitneyu(vv[yy == 1], vv[yy == 0]).statistic / ((yy == 1).sum() * (yy == 0).sum()); assert abs(L.auc(yy, vv) - u) < 1e-12

def test_paired_block_resampling():
    blocks = np.array(list("aaabbcccc")); groups = [np.where(blocks == b)[0] for b in "abc"]
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = L.draw(groups, rng)
        cnt = pd.Series(idx).value_counts()
        for g in groups:                                   # all records of a block appear with identical multiplicity (resampled together)
            assert len(set(cnt.reindex(g).fillna(0).astype(int))) == 1
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1]); v = np.arange(9.0)
    pt, ci, info = L.block_bootstrap(y, {"A": v, "P": v}, {"A": np.argsort(-v), "P": np.argsort(-v)}, blocks, ["A", "P"], 200, 1)
    assert ci["dauc|A"] == (0.0, 0.0)                      # identical arms on identical resamples -> paired difference exactly 0
    a = L.block_bootstrap(y, {"A": v, "P": -v}, {"A": np.argsort(-v), "P": np.argsort(v)}, blocks, ["A", "P"], 200, 7)[1]
    b = L.block_bootstrap(y, {"A": v, "P": -v}, {"A": np.argsort(-v), "P": np.argsort(v)}, blocks, ["A", "P"], 200, 7)[1]
    assert a == b                                          # deterministic given seed

def test_labels_built_before_scores_and_unchanged():
    rec = open(os.path.join(RES, "f2b_labels.sha256")).read().split()[0]
    assert hashlib.sha256(open(os.path.join(RES, "f2b_labels.csv"), "rb").read()).hexdigest() == rec
    src = open(os.path.join(HERE, "f2b_01_labels.py")).read()
    assert "scores_B0QA" not in src and "echo_dc_score" not in src and "pop_density" not in src
    assert "usecols=[\"facility_id\", \"country\", \"lat\", \"lon\", \"name\", \"operator\", \"osm_type\"]" in src   # no score/indicator column is read
    if os.path.exists(os.path.join(RES, "f2b_five_arm_auc.csv")):
        assert os.path.getmtime(os.path.join(RES, "f2b_labels.csv")) < os.path.getmtime(os.path.join(RES, "f2b_five_arm_auc.csv"))

def test_decision_rule():
    ok = dict(g_ok=True, g4_ok=True, prec_ok=True, sens_pos=True, sens_any_ci_below0=False)
    assert L.decide(d_point=0.05, d_ci=(0.01, 0.09), auc_a=.8, auc_p=.75, hw=.04, **ok) == "Strong support"
    assert L.decide(d_point=0.05, d_ci=(-0.01, 0.1), auc_a=.8, auc_p=.75, hw=.05, **ok) == "Limited support"
    assert L.decide(d_point=-0.05, d_ci=(-0.09, -0.01), auc_a=.8, auc_p=.85, hw=.04, **ok).startswith("No demonstrated")
    assert L.decide(d_point=0.0, d_ci=(-0.05, 0.07), auc_a=.8, auc_p=.8, hw=.06001, **ok).startswith("No demonstrated")
    assert L.decide(d_point=0, d_ci=(-.01, .01), auc_a=.5, auc_p=.5, hw=.01, **ok).startswith("No demonstrated")
    assert L.decide(d_point=0.05, d_ci=(0.01, 0.09), auc_a=.8, auc_p=.75, hw=.04, g_ok=False, g4_ok=True, prec_ok=True, sens_pos=True, sens_any_ci_below0=False) == "Criterion failure"

if __name__ == "__main__":
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fs:
        f(); print("PASS", f.__name__)
    print(len(fs), "tests passed")
