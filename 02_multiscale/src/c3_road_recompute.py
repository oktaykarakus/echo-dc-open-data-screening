"""C3: independent re-computation of the road indicators at ALL four radii from the cached, prepared Geofabrik-250101
road geometries (CACHE/roads/*_250101.gpkg), using stage 01's own `roads.build` (unmodified). No PBF is read and nothing
is downloaded. Output is compared with the stage 01 table (which is what B0-QA used) column by column.
`osmium` is only needed to parse raw PBFs; a stub is injected so the cached-geometry path can run where it is absent."""
import sys, types, json, time
try:
    import osmium  # noqa: F401
except ImportError:
    sys.modules["osmium"] = types.ModuleType("osmium")      # cached path never touches it
import numpy as np, pandas as pd
from cfg import *
import common as PBC
import roads as R


def colnames():
    cols = []
    for s in SCALES:
        cols += [f"major_road_length_{s}_km", f"buffer_area_{s}_km2", f"major_road_density_{s}_km_per_km2"]
    return cols + ["nearest_major_road_distance_m"]

def main():
    assert PBC.SNAPSHOT == "250101"
    t0 = time.time()
    ind, stats, roads, ndup = R.build(list(PBC.EXTRACTS), log=lambda *a: None)
    ind = ind.drop(columns=["index_right"], errors="ignore")
    ref = pd.read_csv(TABLE105)
    assert len(ind) == 1216 and ind.facility_id.is_unique and set(ind.facility_id) == set(ref.facility_id)
    ind = ind.set_index("facility_id").loc[ref.facility_id]
    rows = []
    for c in colnames():
        a, b = ind[c].to_numpy(float), ref[c].to_numpy(float)
        d = np.abs(a - b)
        rows.append({"column": c, "n": len(a), "max_abs_diff": float(np.nanmax(d)), "n_exact": int((d == 0).sum()),
                     "n_diff_gt_1e-9": int((d > 1e-9).sum()), "nan_mismatch": int((np.isnan(a) != np.isnan(b)).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "c3_road_recompute_vs_phaseB.csv", index=False)
    json.dump({"snapshot": PBC.SNAPSHOT, "roads_combined_after_dedup": int(len(roads)), "dup_removed": int(ndup),
               "seconds": round(time.time() - t0, 1), "all_columns_identical": bool((out.n_diff_gt_1e-9 if False else out["n_diff_gt_1e-9"]).sum() == 0 and out.nan_mismatch.sum() == 0)},
              open(RESULTS / "c3_road_recompute_meta.json", "w"), indent=1)
    print(out.to_string(index=False)); print("seconds", round(time.time() - t0))

if __name__ == "__main__":
    main()
