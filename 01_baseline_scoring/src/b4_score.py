"""B4: assemble the analytical table (87-col master + reconstructed road indicators) and score.

B0     = historical reconstruction. Reproduces the original merge behaviour in which a facility with NO road
         intersecting any of its buffers is absent from `road_length_wide`, so the left-merge of the nearest-road
         distance leaves it NaN (-> norm 0). (Europe_DC_Map.ipynb cells 209-211.)
B0-QA  = corrected baseline: identical, except that nearest_major_road_distance_m (which WAS computed) is retained
         for those facilities. This is the only change.
"""
import json
import numpy as np, pandas as pd
from common import *
import scoring as S

ROAD_COLS = None

def assemble(road_csv=DATA / "road_indicators_reconstructed_250101.csv"):
    m = pd.read_csv(MASTER87)
    r = pd.read_csv(road_csv)
    assert r.facility_id.is_unique and set(r.facility_id) == set(m.facility_id)
    t = m.merge(r, on="facility_id", how="left", validate="1:1")
    assert (t.facility_id.values == m.facility_id.values).all()          # order (= tie-break order) preserved
    t["road_source"] = f"reconstructed: Geofabrik {SNAPSHOT} (OSM data 2025-01-01)"
    return t

def historical_variant(t):
    """B0: apply the original merge-drop behaviour."""
    h = t.copy()
    dropped = h["major_road_length_10km_km"].fillna(0).eq(0)             # no road in own 10 km buffer -> absent from road_length_wide
    for c in ["nearest_major_road_distance_m", "nearest_major_road_id", "nearest_major_road_class",
              "nearest_major_road_name", "nearest_major_road_ref"]:
        h.loc[dropped, c] = np.nan
    return h, h.loc[dropped, "facility_id"].tolist()

def score_variant(t):
    n = S.normalise(t)
    out, w = S.score_all(n)
    early = S.early_literature_score(n)
    out = pd.concat([out.reset_index(drop=True), early], axis=1)
    return out, w

def main():
    t = assemble()
    t.to_csv(DATA / "analytical_table_105col.csv", index=False)
    h, dropped = historical_variant(t)
    res = {}
    for tag, tab in [("B0", h), ("B0QA", t)]:
        out, w = score_variant(tab)
        keep = ["facility_id", "country", "name", "operator", "lat", "lon", "road_source",
                "major_road_density_2p5km_km_per_km2", "nearest_major_road_distance_m"] + S.ALL_IND + \
               [f"{k}_{s}" for s in ["equal", "literature_informed", "entropy"] for k in ["opportunity_score", "feasibility_score", "echo_dc_score_raw", "echo_dc_score", "echo_dc_rank"]] + \
               ["early_opp", "early_feas", "early_score", "early_rank"]
        out[keep].to_csv(RESULTS / f"scores_{tag}.csv", index=False)
        res[tag] = {"weights": {k: {"opp": v[0], "feas": v[1]} for k, v in w.items()}}
    json.dump({"merge_dropped_ids": dropped, **res}, open(RESULTS / "scoring_run_meta.json", "w"), indent=1)
    print("merge-dropped facilities (B0 NaN nearest distance):", dropped)

if __name__ == "__main__":
    main()
