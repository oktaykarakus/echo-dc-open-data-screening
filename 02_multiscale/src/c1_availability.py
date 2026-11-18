"""C1: indicator-availability and classification table (A scale-dependent / B scale-invariant / C missing or not directly comparable)."""
import numpy as np, pandas as pd
from cfg import *
import multiscale as M

# (indicator, components, transform, class, source template or column, protocolB, provenance / note)
SPEC = [
 ("Hotmaps heat-demand sum", "Opp", "log1p, robust-minmax", "A", "hotmaps_heat_sum_{s}", "excluded",
  "Notebook: zonal SUM of 100 m Hotmaps 2015 raster per buffer (cells 103-110). Extensive (cumulative) -> grows ~ area, so a 2.5 km reference cannot be applied. 15 sites have a 1 km gap zero-filled by the authors (flag hotmaps_small_buffer_missing_originally)."),
 ("Population density", "Opp + Feas", "log1p, robust-minmax", "A", "pop_density_{s}", "eligible",
  "GHS-POP 2025, 100 m, sum / buffer area. Intensive (people per km2); counted in BOTH components."),
 ("GHSL built-up %", "Feas (compactness)", "robust-minmax, averaged with DW", "A", "built_pct_{s}", "eligible",
  "GHS-BUILT-S 2025, 100 m; built m2 / buffer m2. Intensive."),
 ("Dynamic World built-up %", "Feas (compactness)", "robust-minmax, averaged with GHSL", "A", "dw_built_pct_{s}", "eligible",
  "Dynamic World 2024 mode composite, 100 m mean. Intensive."),
 ("Anchor count", "Opp", "log1p, robust-minmax", "A", "anchor_count_{s}", "excluded",
  "Notebook cell 182: groupby(facility_id, buffer_km).size() on the anchor-buffer join -> genuinely radius-specific; extensive (count)."),
 ("Anchor-type diversity", "Opp", "robust-minmax", "A", "anchor_type_diversity_{s}", "excluded",
  "Notebook cell 184: nunique(anchor_category) per (facility, buffer_km) -> genuinely radius-specific; bounded 0-4; saturates (median = p95 = 4 at >= 5 km)."),
 ("Major-road density", "Feas", "robust-minmax", "A", "major_road_density_{s}_km_per_km2", "eligible",
  "Geofabrik 250101; clipped road length (EPSG:3035) / buffer polygon area. stage 02 recomputed all four radii from cached geometries: identical to stage 01."),
 ("Heating degree days (2024)", "Opp", "robust-minmax", "B", "hdd_2024", "invariant",
  "ERA5-Land annual HDD, mean over a fixed 15 km buffer (notebook cells 125-130). Not a function of the analysis radius."),
 ("Nearest-anchor distance", "Opp", "log1p, reversed, robust-minmax", "B", "nearest_anchor_distance_m", "invariant",
  "Global sjoin_nearest over the whole anchor layer (cell 185), not capped by any radius. NB: may exceed the radius when the catchment holds 0 anchors."),
 ("Nearest-major-road distance", "Feas", "log1p, reversed, robust-minmax", "B", "nearest_major_road_distance_m", "invariant",
  "Global sjoin_nearest over the combined road set (stage 01 roads.py); not capped by radius. Kept constant as required."),
]

def main():
    d = M.load_table()
    rows = []
    for name, comp, tr, cls, tmpl, protB, note in SPEC:
        r = {"indicator": name, "component": comp, "transformation": tr, "class": cls, "protocol_B": protB}
        if "{s}" in tmpl:
            for s in SCALES:
                x = d[tmpl.format(s=s)]
                r[f"{s}_nonnull"] = int(x.notna().sum()); r[f"{s}_zero"] = int((x == 0).sum())
                r[f"{s}_p05"] = float(x.quantile(.05)); r[f"{s}_median"] = float(x.median()); r[f"{s}_p95"] = float(x.quantile(.95))
            r["source_columns"] = "; ".join(tmpl.format(s=s) for s in SCALES)
        else:
            x = d[tmpl]
            for s in SCALES:
                r[f"{s}_nonnull"] = int(x.notna().sum()); r[f"{s}_zero"] = int((x == 0).sum())
                r[f"{s}_p05"], r[f"{s}_median"], r[f"{s}_p95"] = float(x.quantile(.05)), float(x.median()), float(x.quantile(.95))
            r["source_columns"] = tmpl + " (single value)"
        r["note"] = note
        rows.append(r)
    t = pd.DataFrame(rows)
    # scale-dependent indicators: how much does the quantile window move? (why fixed-reference needs care)
    t["p95_ratio_10km_over_2p5km"] = [ (r["10km_p95"] / r["2p5km_p95"]) if r["class"] == "A" else 1.0 for r in rows]
    t.to_csv(RESULTS / "c1_indicator_availability.csv", index=False)
    # class-C style flags (data-quality caveats that need a methodological decision, none blocks the analysis)
    flags = []
    flags.append(("Hotmaps 1 km gap (zero-filled)", int(d.hotmaps_small_buffer_missing_originally.sum()), "1 km heat sum = 0 for these; original authors' fill; kept (no invention). Sensitivity run in C5."))
    flags.append(("Anchor count = 0 at 1 km", int((d.anchor_count_1km == 0).sum()), "26% of sites have no anchor within 1 km; log1p(0)=0, diversity 0."))
    flags.append(("Road density = 0 at 1 km", int((d.major_road_density_1km_km_per_km2 == 0).sum()), "No strategic road within 1 km; p05 of the 1 km distribution is 0."))
    flags.append(("Anchor diversity saturated (=4) at 5 km", int((d.anchor_type_diversity_5km == 4).sum()), "median = p95 = 4, so norm diversity is 1 for >50% of sites at >=5 km."))
    flags.append(("Anchor diversity saturated (=4) at 10 km", int((d.anchor_type_diversity_10km == 4).sum()), "as above"))
    flags.append(("Nearest-anchor dist > radius when 0 anchors in catchment (1 km)", int(((d.anchor_count_1km == 0) & (d.nearest_anchor_distance_m <= 1000)).sum()),
                  "consistency check: 1 case (GB_way_187251460, nearest anchor 999.96 m) is explained by the 16-segment polygon buffer (area 3.1365 km2 vs pi); counts are otherwise consistent with nearest distance at all four radii"))
    pd.DataFrame(flags, columns=["flag", "n_facilities", "comment"]).to_csv(RESULTS / "c1_data_flags.csv", index=False)
    print(t[["indicator", "class", "protocol_B", "1km_zero", "2p5km_zero", "5km_zero", "10km_zero", "p95_ratio_10km_over_2p5km"]].to_string(index=False))
    print(pd.DataFrame(flags, columns=["flag", "n", "c"]).to_string(index=False))

if __name__ == "__main__":
    main()
