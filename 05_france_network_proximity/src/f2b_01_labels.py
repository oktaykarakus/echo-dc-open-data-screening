"""F2-B step 1 (Gates 1-2): build the network-proximity outcome for the French records.
Uses ONLY facility id/country/coordinates/name/operator (master CSV), stage 04 campus units, and the FCU layers.
NO ECHO-DC score, rank or indicator column is read here (labels are built blind to scores).
Run from the phase folder:  source src/env.sh && /usr/bin/python3 src/f2b_01_labels.py
"""
import json, hashlib, os, sys
import numpy as np, pandas as pd, geopandas as gpd, shapely
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PH = os.path.abspath(os.path.join(HERE, ".."))
MASTER = os.path.join(ROOT, "data", "inputs", "inventory_indicator_master_87col.csv")
CAMPUS = os.path.join(ROOT, "04_operational_projects", "results", "e2_record_campus_groups.csv")
EXT = os.path.join(PH, "data", "extracted")
OUT = os.environ.get("ECHO_F2B_RESULTS", os.path.join(PH, "results"))
os.makedirs(OUT, exist_ok=True)

T_PRIMARY, T_SECONDARY = 2500.0, 1000.0   # frozen (F2-A protocol)
CENTROID_RADIUS = 5000.0                  # frozen indeterminacy radius for centroid-only networks
BLOCKS_KM = (25, 50, 100)
SALLE = "salle technique"
FOUR = ["FR_node_7902392228", "FR_node_9002117313", "FR_node_7902413338", "FR_node_7902421304"]


def nearest_dist(points_xy, geoms):
    tree = shapely.STRtree(geoms)
    pts = shapely.points(points_xy[:, 0], points_xy[:, 1])
    idx, d = tree.query_nearest(pts, return_distance=True, all_matches=False)
    out = np.full(len(pts), np.nan)
    out[idx[0]] = d
    return out


def classify(d1, d2, dc, t, tier2_as_existing=False, centroid_radius=CENTROID_RADIUS):
    """-> array of 'positive' / 'mapped_negative' / 'indeterminate'."""
    d_exist = np.minimum(d1, d2) if tier2_as_existing else d1
    pos = d_exist <= t
    ind = (~pos) & (((d2 <= t) & (not tier2_as_existing)) | (dc <= centroid_radius))
    s = np.where(pos, "positive", np.where(ind, "indeterminate", "mapped_negative"))
    return s


def block_ids(x, y, km):
    return (np.floor(x / (km * 1000)).astype(int).astype(str) + "_" + np.floor(y / (km * 1000)).astype(int).astype(str))


def main():
    m = pd.read_csv(MASTER, usecols=["facility_id", "country", "lat", "lon", "name", "operator", "osm_type"])
    assert m.facility_id.is_unique and len(m) == 1216
    fr = m[m.country == "FR"].copy().reset_index(drop=True)
    n_fr = len(fr)
    missing_xy = int(fr[["lat", "lon"]].isna().any(axis=1).sum())
    fr = fr.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    to2154 = Transformer.from_crs(4326, 2154, always_xy=True)
    to3035 = Transformer.from_crs(4326, 3035, always_xy=True)
    x, y = to2154.transform(fr.lon.values, fr.lat.values)
    X, Y = to3035.transform(fr.lon.values, fr.lat.values)
    fr["x2154"], fr["y2154"] = x, y
    xy = np.column_stack([x, y])

    # --- network layers (EPSG:2154) ---
    L = gpd.read_file(os.path.join(EXT, "reseaux_de_chaleur.shp"))
    S = gpd.read_file(os.path.join(EXT, "reseaux_de_chaleur_sans_traces.shp"))
    assert str(L.crs.to_epsg()) == "2154" and str(S.crs.to_epsg()) == "2154"
    assert not L.geometry.isna().any() and not L.is_empty.any()
    n_invalid = int((~L.is_valid).sum())
    geoms = shapely.make_valid(L.geometry.values)                      # repair 4 invalid geometries; no geometry is dropped
    tier1 = geoms[L.Identifian.notna().values]                          # operating networks with national survey id
    tier2 = geoms[L.Identifian.isna().values]                           # status-unverified traces (no survey id)
    cents = S.geometry.values
    fr["d_tier1_m"] = nearest_dist(xy, tier1)
    fr["d_tier2_m"] = nearest_dist(xy, tier2)
    fr["d_centroid_m"] = nearest_dist(xy, cents)

    d1, d2, dc = fr.d_tier1_m.values, fr.d_tier2_m.values, fr.d_centroid_m.values
    for name, t in (("2500", T_PRIMARY), ("1000", T_SECONDARY)):
        fr["status_" + name] = classify(d1, d2, dc, t)
        fr["status_t2exist_" + name] = classify(d1, d2, dc, t, tier2_as_existing=True)
    for km in BLOCKS_KM:
        fr[f"block{km}"] = block_ids(X, Y, km)

    cm = pd.read_csv(CAMPUS, usecols=["facility_id", "unit_id", "campus_group", "group_status"])
    fr = fr.merge(cm, on="facility_id", how="left", validate="1:1")
    assert fr.unit_id.notna().all()
    fr["salle_technique"] = fr.name.fillna("").str.lower().str.contains(SALLE)
    fr["four_broadcaster"] = fr.facility_id.isin(FOUR)

    cols = ["facility_id", "country", "name", "operator", "osm_type", "lat", "lon", "x2154", "y2154",
            "d_tier1_m", "d_tier2_m", "d_centroid_m", "status_2500", "status_1000", "status_t2exist_2500", "status_t2exist_1000",
            "block25", "block50", "block100", "unit_id", "campus_group", "group_status", "salle_technique", "four_broadcaster"]
    lab = fr[cols].copy()
    for c in ("x2154", "y2154", "d_tier1_m", "d_tier2_m", "d_centroid_m"):
        lab[c] = lab[c].round(3)
    path = os.path.join(OUT, "f2b_labels.csv")
    lab.to_csv(path, index=False)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    open(os.path.join(OUT, "f2b_labels.sha256"), "w").write(h + "  f2b_labels.csv\n")

    # --- outcome audit ---
    aud = {"french_records": n_fr, "missing_coordinates": missing_xy, "records_evaluated": len(fr),
           "layer_features": {"existing_heat_traces_total": int(len(L)), "tier1_survey_id": int(len(tier1)), "tier2_no_id": int(len(tier2)),
                              "invalid_geometries_repaired": n_invalid, "centroid_only_points": int(len(S))},
           "unit_count": int(fr.unit_id.nunique())}
    for name in ("2500", "1000"):
        s = lab["status_" + name]
        elig = lab[s != "indeterminate"]
        npos, nneg, nind = int((s == "positive").sum()), int((s == "mapped_negative").sum()), int((s == "indeterminate").sum())
        e = {"positive": npos, "mapped_negative": nneg, "indeterminate": nind, "eligible": npos + nneg,
             "prevalence_eligible": round(npos / max(npos + nneg, 1), 4), "prevalence_all_records": round(npos / len(lab), 4)}
        # reasons for indeterminacy
        ind = lab[s == "indeterminate"]
        t = float(name)
        e["indeterminate_reason"] = {"tier2_trace_within_t": int((ind.d_tier2_m <= t).sum()),
                                     "centroid_within_5km_only": int(((ind.d_tier2_m > t) & (ind.d_centroid_m <= CENTROID_RADIUS)).sum())}
        conc = {}
        for km in BLOCKS_KM:
            b = elig["block%d" % km].value_counts()
            pb = elig[elig["status_" + name] == "positive"]["block%d" % km].value_counts()
            conc[str(km)] = {"nonempty_blocks_eligible": int(len(b)), "largest_block_share_eligible": round(float(b.iloc[0] / len(elig)), 3),
                             "top3_share_eligible": round(float(b.iloc[:3].sum() / len(elig)), 3),
                             "blocks_with_positive": int(len(pb)), "largest_block_share_positives": round(float(pb.iloc[0] / max(pb.sum(), 1)), 3) if len(pb) else None}
        e["concentration"] = conc
        # campus effects
        g = lab.groupby("unit_id")["status_" + name].agg(lambda v: "positive" if (v == "positive").any() else ("indeterminate" if (v == "indeterminate").any() else "mapped_negative"))
        mixed = lab.groupby("unit_id")["status_" + name].nunique()
        e["campus_units"] = {"units": int(len(g)), "positive": int((g == "positive").sum()), "mapped_negative": int((g == "mapped_negative").sum()),
                             "indeterminate": int((g == "indeterminate").sum()), "units_with_mixed_member_status": int((mixed > 1).sum())}
        e["distance_quantiles_m_tier1"] = {str(q): round(float(lab.d_tier1_m.quantile(q)), 0) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
        aud["t" + name] = e
        # variant: tier-2 counted as existing
        s2 = lab["status_t2exist_" + name]
        aud["t" + name]["variant_tier2_as_existing"] = {"positive": int((s2 == "positive").sum()), "mapped_negative": int((s2 == "mapped_negative").sum()),
                                                        "indeterminate": int((s2 == "indeterminate").sum())}
    # gates (frozen F2-A thresholds)
    def gates(e):
        c50 = e["concentration"]["50"]
        return {"G2_eligible_ge_250": e["eligible"] >= 250, "G2_each_class_ge_40": min(e["positive"], e["mapped_negative"]) >= 40,
                "G3_prevalence_15_85": 0.15 <= e["prevalence_eligible"] <= 0.85,
                "G4_blocks_ge_20": c50["nonempty_blocks_eligible"] >= 20, "G4_largest_le_35pct": c50["largest_block_share_eligible"] <= 0.35}
    aud["gates_t2500"], aud["gates_t1000"] = gates(aud["t2500"]), gates(aud["t1000"])
    aud["labels_sha256"] = h
    json.dump(aud, open(os.path.join(OUT, "f2b_outcome_audit.json"), "w"), indent=1)
    print(json.dumps(aud, indent=1))
    return lab


if __name__ == "__main__":
    main()
