"""B3: reconstruct the two OSM strategic-road indicators following the ORIGINAL pipeline
(Europe_DC_Map.ipynb cells 193-214), using a dated Geofabrik snapshot instead of 'latest'.

Original logic preserved:
  * highway in {motorway,trunk,primary} + their *_link  (ways; relations carry no line geometry)
  * per-country extract, keep roads intersecting the country's own 10 km DC buffers (EPSG:3035)
  * concatenate countries, de-duplicate on (highway,name,ref,wkb)
  * nearest distance: point -> nearest road, planar EPSG:3035, over the combined road set
  * length: road.intersection(buffer).length / 1000 in EPSG:3035 (rounded 3 dp), density = length / buffer area (km2)
"""
import hashlib, time, sys, json
import numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely import wkb as swkb
import osmium
from common import *

CRS_M = "EPSG:3035"

# ----------------------------------------------------------------------------- source handling
def pbf_path(cc):
    return CACHE / "pbf" / f"{EXTRACTS[cc]}-{SNAPSHOT}.osm.pbf"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

def pbf_info(cc):
    p = pbf_path(cc)
    hd = osmium.io.Reader(str(p), osmium.osm.osm_entity_bits.NOTHING).header()
    return {"country": cc, "file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p),
            "replication_timestamp": hd.get("osmosis_replication_timestamp"),
            "replication_base_url": hd.get("osmosis_replication_base_url"),
            "writing_program": hd.get("generator")}

# ----------------------------------------------------------------------------- extraction
def extract_strategic_roads(cc):
    """All ways with highway in ROAD_CLASSES as LineStrings (EPSG:4326). Mirrors `osmium tags-filter` +
    `osmium export` + the notebook's geometry/class filters. Closed ways tagged area=yes are polygons in
    osmium export and are dropped (counted)."""
    flt = osmium.filter.TagFilter(*[("highway", c) for c in ROAD_CLASSES])
    fp = osmium.FileProcessor(str(pbf_path(cc)), osmium.osm.NODE | osmium.osm.WAY).with_locations().with_filter(flt)
    fac = osmium.geom.WKBFactory()
    rows, bad, area_dropped = [], 0, 0
    for w in fp:
        if not w.is_way():
            continue
        tags = w.tags
        if w.is_closed() and tags.get("area") == "yes":
            area_dropped += 1
            continue
        try:
            geom = swkb.loads(fac.create_linestring(w), hex=True)
        except Exception:
            bad += 1
            continue
        rows.append((w.id, tags.get("highway"), str(tags.get("name", ""))[:250], str(tags.get("ref", ""))[:250], geom))
    gdf = gpd.GeoDataFrame(rows, columns=["osm_way_id", "highway", "name", "ref", "geometry"], crs="EPSG:4326")
    return gdf, {"lines": len(gdf), "bad_geometry": bad, "area_closed_dropped": area_dropped,
                 "by_class": gdf["highway"].value_counts().to_dict()}

# ----------------------------------------------------------------------------- facilities
def load_facilities():
    m = pd.read_csv(MASTER87, usecols=["facility_id", "country", "lat", "lon"])
    pts = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(m["lon"], m["lat"]), crs="EPSG:4326").to_crs(CRS_M)
    rows = []
    for km in BUFFER_KM:                              # identical to Europe_DC_Map.ipynb cell 16 (default 16 segments/quadrant)
        b = pts[["facility_id", "country", "geometry"]].copy()
        b["buffer_km"] = km
        b["geometry"] = b.geometry.buffer(km * 1000)
        rows.append(b)
    buffers = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), geometry="geometry", crs=CRS_M)
    return pts, buffers

# ----------------------------------------------------------------------------- near-DC selection (per country, as original)
def near_dc_roads(roads4326, buffers10_country):
    r3035 = roads4326.to_crs(CRS_M)
    hit = gpd.sjoin(r3035[["geometry"]], buffers10_country[["geometry"]], how="inner", predicate="intersects")
    keep = np.unique(hit.index.values)
    return roads4326.loc[keep].copy()

def combine_and_dedup(parts):
    """Concatenate per-country near-DC road sets and drop exact duplicates (original cell 203 key)."""
    roads = pd.concat(parts, ignore_index=True)
    key = (roads["highway"].fillna("") + "_" + roads["name"].fillna("") + "_" + roads["ref"].fillna("") + "_" +
           pd.Series(shapely.to_wkb(roads.geometry.values, hex=True), index=roads.index))
    n0 = len(roads)
    roads = roads.loc[~key.duplicated()].reset_index(drop=True)
    return gpd.GeoDataFrame(roads, geometry="geometry", crs="EPSG:4326"), n0 - len(roads)

# ----------------------------------------------------------------------------- indicators
def compute_indicators(roads4326, pts, buffers):
    roads = roads4326.to_crs(CRS_M).reset_index(drop=True)
    roads["road_id"] = roads.index
    # nearest distance (original cell 205)
    nn = gpd.sjoin_nearest(pts[["facility_id", "geometry"]], roads[["road_id", "highway", "name", "ref", "geometry"]],
                           how="left", distance_col="nearest_major_road_distance_m")
    nn = (nn.rename(columns={"road_id": "nearest_major_road_id", "highway": "nearest_major_road_class",
                             "name": "nearest_major_road_name", "ref": "nearest_major_road_ref"})
            .sort_values(["facility_id", "nearest_major_road_distance_m"]).drop_duplicates("facility_id").drop(columns="geometry"))
    # clipped lengths (cells 206-209)
    b = buffers.reset_index(drop=True)
    b["buffer_id"] = b.index
    j = gpd.sjoin(roads[["road_id", "geometry"]], b[["buffer_id", "facility_id", "buffer_km", "geometry"]],
                  how="inner", predicate="intersects")
    bgeom = b.geometry.values[j["buffer_id"].values]
    clipped = shapely.intersection(j.geometry.values, bgeom)
    j["len_km"] = shapely.length(clipped) / 1000.0
    L = j.groupby(["facility_id", "buffer_km"])["len_km"].sum().unstack("buffer_km")
    L.columns = [f"major_road_length_{str(float(c)).replace('.0','').replace('.','p')}km_km" for c in L.columns]
    L = L.reset_index()
    area = b.assign(a=b.geometry.area / 1e6).pivot_table(index="facility_id", columns="buffer_km", values="a", aggfunc="first")
    lab = {1.0: "1km", 2.5: "2p5km", 5.0: "5km", 10.0: "10km"}
    out = pts[["facility_id"]].drop_duplicates().merge(L, on="facility_id", how="left")
    for km, s in lab.items():
        lc = f"major_road_length_{s}_km"
        out[lc] = out[lc].fillna(0).round(3) if lc in out else 0.0
        out[f"buffer_area_{s}_km2"] = out["facility_id"].map(area[km])
        out[f"major_road_density_{s}_km_per_km2"] = (out[lc] / out[f"buffer_area_{s}_km2"]).round(4)
    out = out.merge(nn, on="facility_id", how="left")
    return out, roads


# ----------------------------------------------------------------------------- driver
def build(countries, log=print):
    """Run the full reconstruction for the facilities of `countries`. Per-country extraction is cached in
    CACHE/roads/{cc}.gpkg. Returns (indicator_table, per_country_stats, roads_combined_4326)."""
    (CACHE / "roads").mkdir(parents=True, exist_ok=True)
    pts_all, buf_all = load_facilities()
    parts, stats = [], {}
    for cc in countries:
        t0 = time.time()
        cache = CACHE / "roads" / f"{cc}_{SNAPSHOT}.gpkg"
        if cache.exists():
            raw = gpd.read_file(cache, engine="pyogrio")
            ext = json.load(open(str(cache) + ".json"))
        else:
            raw, ext = extract_strategic_roads(cc)
            raw.to_file(cache, driver="GPKG", engine="pyogrio")
            json.dump(ext, open(str(cache) + ".json", "w"))
        b10 = buf_all[(buf_all.country == cc) & (buf_all.buffer_km == 10)]
        near = near_dc_roads(raw, b10)
        near["source_country_code"] = cc
        parts.append(near)
        stats[cc] = {**ext, "dc_facilities": int(len(b10)), "near_dc_roads": int(len(near)), "seconds": round(time.time() - t0, 1)}
        log(f"{cc}: {ext['lines']} lines, {len(near)} near DC, {stats[cc]['seconds']} s")
    roads, ndup = combine_and_dedup(parts)
    sel = pts_all.country.isin(countries)
    pts, buf = pts_all[sel], buf_all[buf_all.country.isin(countries)]
    ind, _ = compute_indicators(roads, pts, buf)
    return ind, stats, roads, ndup
