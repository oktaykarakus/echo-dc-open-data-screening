"""B3: full reconstruction for all 1,216 facilities / 20 countries. Requires the 20 Geofabrik 250101 PBFs in CACHE/pbf
(see data/geofabrik_250101_sources.json for URLs). Re-runnable; per-country extraction is cached."""
import json, time, gc
import pandas as pd
from common import *
import roads as R

if __name__ == "__main__":
    t0 = time.time()
    ccs = list(EXTRACTS)
    ind, stats, roads, ndup = R.build(ccs)
    ind = ind.drop(columns=["index_right"], errors="ignore")
    assert len(ind) == 1216 and ind.facility_id.is_unique
    ind.to_csv(DATA / "road_indicators_reconstructed_250101.csv", index=False)
    json.dump({"snapshot": SNAPSHOT, "per_country": stats, "roads_combined_after_dedup": int(len(roads)), "dup_removed": int(ndup),
               "total_seconds": round(time.time() - t0, 1), "historical_combined_roads": 491568}, open(RESULTS / "road_build_stats.json", "w"), indent=1)
    man = [R.pbf_info(cc) for cc in ccs]
    json.dump(man, open(DATA / "osm_snapshot_manifest.json", "w"), indent=1)
    print("done", round(time.time() - t0), "s;", len(roads), "combined roads;", ndup, "duplicates removed")
