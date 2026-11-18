"""B6: automated reproducibility / quality checks. Exit code 1 if any check fails. Writes results/qa_checks.json
and results/candidate_duplicate_campuses.csv (candidates are RECORDED ONLY - nothing is merged in stage 01)."""
import json, sys, hashlib
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
from common import *
import scoring as S

checks = []
def chk(name, ok, detail=""):
    checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})

ids0 = pd.read_csv(DATA / "facility_ids_original.csv")
m = pd.read_csv(MASTER87)
lit = pd.read_csv(DATA / "recovered_literature_scores.csv")
t = pd.read_csv(DATA / "analytical_table_105col.csv")
road = pd.read_csv(DATA / "road_indicators_reconstructed_250101.csv")

# inventory / lineage
chk("1216 original facility IDs", len(ids0) == 1216 and ids0.facility_id.is_unique, len(ids0))
chk("recovered lit-score IDs == master IDs", set(lit.facility_id) == set(m.facility_id) and lit.facility_id.is_unique)
chk("analytical table IDs == original IDs (no additions/removals)", set(t.facility_id) == set(ids0.facility_id) and len(t) == 1216)
chk("no duplicated facility IDs after joins", t.facility_id.is_unique and road.facility_id.is_unique)
chk("row order preserved vs master (tie-break order)", (t.facility_id.values == m.facility_id.values).all())
chk("master columns unchanged in analytical table", all(np.array_equal(t[c].values, m[c].values) or np.allclose(t[c], m[c], equal_nan=True) for c in m.columns if m[c].dtype != object))
# validity
chk("coordinates valid (lat 34-72, lon -25-45)", t.lat.between(34, 72).all() and t.lon.between(-25, 45).all())
chk("20 distinct valid country codes", t.country.nunique() == 20 and set(t.country) == set(EXTRACTS), sorted(t.country.unique()))
chk("facility_id prefix matches country", (t.facility_id.str[:2] == t.country).all())
# road units
d = road.nearest_major_road_distance_m
chk("nearest road distance finite, >0, <= 50 km (metres)", d.notna().all() and np.isfinite(d).all() and (d > 0).all() and (d <= 50_000).all(), f"min {d.min():.1f} max {d.max():.1f}")
dens = road.major_road_density_2p5km_km_per_km2
chk("density 2.5km in plausible km/km2 range [0,10]", dens.between(0, 10).all(), f"max {dens.max():.3f}")
chk("density == length/area (km/km2) within rounding", np.allclose(dens, (road.major_road_length_2p5km_km / road.buffer_area_2p5km_km2), atol=1e-3))
chk("buffer area 2.5 km ~ pi*2.5^2 (+-0.5%)", np.allclose(road.buffer_area_2p5km_km2, np.pi * 2.5 ** 2, rtol=5e-3))
chk("road lengths monotonic in buffer size", ((road.major_road_length_1km_km <= road.major_road_length_2p5km_km) & (road.major_road_length_2p5km_km <= road.major_road_length_5km_km) & (road.major_road_length_5km_km <= road.major_road_length_10km_km)).all())
chk("all 2.5 km lengths <= pi r^2 * 10 km/km2 sanity", (road.major_road_length_2p5km_km <= 200).all())
chk("provenance column marks reconstructed roads", t.road_source.str.startswith("reconstructed").all())
# scoring
for tag in ["B0", "B0QA"]:
    sc = pd.read_csv(RESULTS / f"scores_{tag}.csv")
    chk(f"[{tag}] 1216 scored rows, same IDs", len(sc) == 1216 and set(sc.facility_id) == set(ids0.facility_id))
    chk(f"[{tag}] all 9 norm indicators finite in [0,1]", np.isfinite(sc[S.ALL_IND]).all().all() and sc[S.ALL_IND].min().min() >= 0 and sc[S.ALL_IND].max().max() <= 1)
    for sc_name in ["equal", "literature_informed", "entropy"]:
        r = sc[f"echo_dc_rank_{sc_name}"]
        chk(f"[{tag}] {sc_name} ranks are a permutation of 1..1216", sorted(r) == list(range(1, 1217)))
        raw = sc[f"echo_dc_score_raw_{sc_name}"]
        o, f = sc[f"opportunity_score_{sc_name}"], sc[f"feasibility_score_{sc_name}"]
        chk(f"[{tag}] {sc_name} score == sqrt((O+eps)(F+eps))", np.allclose(raw, np.sqrt((o + 1e-6) * (f + 1e-6)), atol=1e-12))
        chk(f"[{tag}] {sc_name} rank monotone in score", (sc.sort_values(f"echo_dc_rank_{sc_name}")[f"echo_dc_score_raw_{sc_name}"].diff().dropna() <= 1e-15).all())
    meta = json.load(open(RESULTS / "scoring_run_meta.json"))[tag]["weights"]
    for sc_name, w in meta.items():
        chk(f"[{tag}] {sc_name} weights sum to 1 (opp,feas)", abs(sum(w["opp"].values()) - 1) < 1e-9 and abs(sum(w["feas"].values()) - 1) < 1e-9)
    chk(f"[{tag}] literature weights == 0.35/.20/.15/.15/.10/.05 and .35/.30/.20/.15", np.allclose(list(meta["literature_informed"]["opp"].values()), [.35, .2, .15, .15, .1, .05]) and np.allclose(list(meta["literature_informed"]["feas"].values()), [.35, .3, .2, .15]))
    chk(f"[{tag}] equal weights: opp 1/6, feas 1/4", np.allclose(list(meta["equal"]["opp"].values()), 1 / 6) and np.allclose(list(meta["equal"]["feas"].values()), .25))
    # direction: closer anchors / roads -> higher normalised value; denser -> higher
    tb = pd.read_csv(DATA / "analytical_table_105col.csv")
    chk(f"[{tag}] direction: nearest-anchor norm decreases with distance", pd.Series(sc.norm_nearest_anchor_distance_m.values).corr(tb.nearest_anchor_distance_m, method="spearman") < 0)
    chk(f"[{tag}] direction: nearest-road norm decreases with distance", pd.Series(sc.norm_nearest_major_road_distance_m.values).corr(tb.nearest_major_road_distance_m, method="spearman") < 0)
    chk(f"[{tag}] direction: road-density norm increases with density", pd.Series(sc.norm_major_road_density_2p5km.values).corr(tb.major_road_density_2p5km_km_per_km2, method="spearman") > 0)
    chk(f"[{tag}] direction: heat/pop/HDD/anchor-count norms increase", all(pd.Series(sc[n].values).corr(tb[c], method="spearman") > 0 for n, c in [("norm_heat_sum_2p5km", "hotmaps_heat_sum_2p5km"), ("norm_pop_density_2p5km", "pop_density_2p5km"), ("norm_hdd_2024", "hdd_2024"), ("norm_anchor_count_2p5km", "anchor_count_2p5km")]))
    # determinism: re-score and compare
    n2 = S.normalise(pd.read_csv(DATA / "analytical_table_105col.csv") if tag == "B0QA" else None) if tag == "B0QA" else None
    if n2 is not None:
        s2, _ = S.score_all(n2)
        chk(f"[{tag}] deterministic re-run gives identical ranks", (s2.echo_dc_rank_equal.values == sc.echo_dc_rank_equal.values).all())
    ties = int(sc.echo_dc_score_raw_equal.duplicated(keep=False).sum())
    chk(f"[{tag}] n tied raw equal-weight scores (resolved by input order, method='first')", True, ties)
# B0 vs B0-QA differ only for merge-dropped facilities
a, b = pd.read_csv(RESULTS / "scores_B0.csv"), pd.read_csv(RESULTS / "scores_B0QA.csv")
meta = json.load(open(RESULTS / "scoring_run_meta.json"))
chg = a.loc[(a.nearest_major_road_distance_m.fillna(-1) != b.nearest_major_road_distance_m.fillna(-1)), "facility_id"].tolist()
chk("B0 vs B0-QA input differs only for merge-dropped IDs", set(chg) == set(meta["merge_dropped_ids"]), chg)

# candidate duplicate campuses (RECORDED ONLY)
g = gpd.GeoDataFrame(t[["facility_id", "country", "name", "operator", "osm_type"]], geometry=gpd.points_from_xy(t.lon, t.lat), crs="EPSG:4326").to_crs("EPSG:3035")
xy = np.c_[g.geometry.x, g.geometry.y]
pairs = cKDTree(xy).query_pairs(250, output_type="ndarray")
dd = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
cand = pd.DataFrame({"id_a": g.facility_id.values[pairs[:, 0]], "id_b": g.facility_id.values[pairs[:, 1]], "distance_m": dd.round(1),
                     "name_a": g.name.values[pairs[:, 0]], "name_b": g.name.values[pairs[:, 1]], "operator_a": g.operator.values[pairs[:, 0]], "operator_b": g.operator.values[pairs[:, 1]]})
cand["same_operator"] = (cand.operator_a.fillna("?") == cand.operator_b.fillna("?")) & cand.operator_a.notna()
cand["same_name"] = cand.name_a.fillna("?") == cand.name_b.fillna("?")
cand.sort_values("distance_m").to_csv(RESULTS / "candidate_duplicate_campuses.csv", index=False)
within100 = set(cand.loc[cand.distance_m <= 100, ["id_a", "id_b"]].values.ravel())
within250 = set(cand[["id_a", "id_b"]].values.ravel())
chk("candidate duplicate campuses recorded (not merged)", True, f"{len(cand)} pairs <=250 m; {len(within100)} records <=100 m of another; {len(within250)} records <=250 m")

ok = all(c["pass"] for c in checks)
json.dump({"all_pass": ok, "n_checks": len(checks), "n_failed": sum(not c["pass"] for c in checks), "checks": checks,
           "duplicates": {"pairs_le_250m": len(cand), "records_le_100m": len(within100), "records_le_250m": len(within250)}}, open(RESULTS / "qa_checks.json", "w"), indent=1)
print(f"{len(checks)} checks, {sum(not c['pass'] for c in checks)} failed")
for c in checks:
    if not c["pass"]:
        print("FAIL:", c["check"], c["detail"])
sys.exit(0 if ok else 1)
