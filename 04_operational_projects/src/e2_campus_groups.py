"""E5: what the '956' is, and deterministic campus grouping rules (no manual merging; nothing is merged in the data).

956 = number of record PAIRS whose points lie <= 250 m apart in EPSG:3035 (stage 01 b6_qa_checks.py: cKDTree.query_pairs(250)).
These involve 563 records (385 have a partner <= 100 m). A pair is a proximity candidate, not a duplicate.

Edge evidence tiers (per pair, <= 250 m):
  T1 coincident     : d <= 10 m. Representative points essentially identical -> probable duplicate representation of one physical
                      feature (e.g. a relation and its own way, node on a building). Footprints are unavailable (no OSM re-extraction), so
                      'confirmed duplicate' cannot be asserted; T1 is the strongest tier available.
  T2 same-brand     : d <= 250 m and both records carry the same normalised brand key (operator, else the first name token) -> likely common campus.
  T3 distinct       : both brand keys present and different -> nearby but genuinely distinct facilities (kept separate).
  T4 unresolved     : at least one brand key missing and not T1 -> proximity only; never merged.
Groups = connected components over T1 and T2 edges only. Chain guard: a component with diameter > 500 m is re-split using only
T1 and T2 edges <= 100 m; if still > 500 m it is flagged 'chain_risk' and excluded from the campus convention.
"""
import re, unicodedata
from e_lib import *
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

GENERIC = {"data", "datacenter", "datacentre", "center", "centre", "dc", "the", "de", "des", "la", "le", "rechenzentrum", "rz", "colocation", "hosting", "server", "servers", "it", "cloud", "network", "networks", "telecom", "group", "gmbh", "ltd", "limited", "sa", "sas", "bv", "ag", "inc", "llc", "srl", "spa", "ab", "as", "oy"}
ALIAS = {"interxion": "digitalrealty", "digital": "digitalrealty", "equinix,": "equinix", "facebook": "meta", "telehouse": "telehouse", "kddi": "telehouse", "nlcd": "northc", "nldc": "northc", "ntt": "ntt", "e-shelter": "ntt", "eshelter": "ntt", "global": "global"}


def brand(name, operator):
    """Normalised brand key: operator if present, else first non-generic token of the name. Missing -> None."""
    for s in (operator, name):
        if isinstance(s, str) and s.strip():
            s2 = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("&", " ")
            toks = [t for t in re.findall(r"[a-z0-9]+", s2) if t not in GENERIC]
            if toks:
                k = toks[0]
                if k == "digital" and len(toks) > 1 and toks[1] == "realty":
                    k = "digitalrealty"
                return ALIAS.get(k, k)
    return None


def pairs_table(t, radius=250):
    to = Transformer.from_crs(4326, 3035, always_xy=True)
    x, y = to.transform(t.lon.values, t.lat.values)
    xy = np.c_[x, y]
    pr = cKDTree(xy).query_pairs(radius, output_type="ndarray")
    pr = pr[np.lexsort((pr[:, 1], pr[:, 0]))]
    d = np.hypot(*(xy[pr[:, 0]] - xy[pr[:, 1]]).T)
    ba = [brand(t.name.iat[i], t.operator.iat[i]) for i in range(len(t))]
    rows = []
    for (i, j), dist in zip(pr, d):
        a, b = ba[i], ba[j]
        if dist <= 10:
            tier = "T1_coincident"
        elif a and b and a == b:
            tier = "T2_same_brand"
        elif a and b:
            tier = "T3_distinct_brand"
        else:
            tier = "T4_unresolved"
        rows.append((t.facility_id.iat[i], t.facility_id.iat[j], round(float(dist), 1), t.country.iat[i], t.country.iat[j], a, b, tier))
    return pd.DataFrame(rows, columns=["id_a", "id_b", "distance_m", "country_a", "country_b", "brand_a", "brand_b", "tier"]), xy


def components(n, ia, ib):
    if len(ia) == 0:
        return np.arange(n)
    m = coo_matrix((np.ones(len(ia)), (ia, ib)), shape=(n, n))
    return connected_components(m, directed=False)[1]


def diameter(xy, idx):
    p = xy[idx]
    return float(np.max(np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1]))) if len(idx) > 1 else 0.0


def build():
    t = load_scores()[["facility_id", "country", "name", "operator", "lat", "lon", "echo_dc_rank_equal"]].reset_index(drop=True)
    pt, xy = pairs_table(t)
    pos = {f: i for i, f in enumerate(t.facility_id)}
    supported = pt[pt.tier.isin(["T1_coincident", "T2_same_brand"]) & (pt.country_a == pt.country_b)]
    ia, ib = supported.id_a.map(pos).values, supported.id_b.map(pos).values
    lab = components(len(t), ia, ib)
    # chain guard
    lab = pd.Series(lab)
    status = {}
    next_label = lab.max() + 1
    for g, idx in lab.groupby(lab).groups.items():
        idx = np.array(idx)
        if len(idx) < 2:
            continue
        if diameter(xy, idx) <= 500:
            status[g] = "supported"
            continue
        # re-split at 100 m
        sub = supported[(supported.distance_m <= 100) & supported.id_a.map(pos).isin(idx) & supported.id_b.map(pos).isin(idx)]
        l2 = components(len(t), sub.id_a.map(pos).values, sub.id_b.map(pos).values)
        for k in np.unique(l2[idx]):
            members = idx[l2[idx] == k]
            if len(members) == 0:
                continue
            lab.iloc[members] = next_label
            status[next_label] = "supported_resplit100" if diameter(xy, members) <= 500 else "chain_risk"
            next_label += 1
    t["raw_label"] = lab.values
    sizes = t.groupby("raw_label").facility_id.transform("size")
    t["group_size"] = sizes
    t["group_status"] = t.raw_label.map(status).where(sizes > 1, "singleton")
    t.loc[(t.group_status == "chain_risk"), "group_status"] = "chain_risk"
    # stable group ids: order by smallest member row, only for size>1 supported groups
    sup = t[(t.group_size > 1) & t.group_status.isin(["supported", "supported_resplit100"])]
    order = sup.reset_index().groupby("raw_label")["index"].min().sort_values()
    gid = {lab_: f"G{str(k + 1).zfill(3)}" for k, lab_ in enumerate(order.index)}
    t["campus_group"] = t.raw_label.map(gid)
    t["unit_id"] = np.where(t.campus_group.notna(), t.campus_group, t.facility_id)
    return t.drop(columns=["raw_label"]), pt, xy


def group_summary(t, xy):
    g = t[t.campus_group.notna()]
    rows = []
    for gid, d in g.groupby("campus_group", sort=True):
        idx = d.index.values
        rows.append(dict(campus_group=gid, country=d.country.iat[0], n_records=len(d), diameter_m=round(diameter(xy, idx), 0), status=d.group_status.iat[0],
                         brands="; ".join(sorted(set(d.apply(lambda r: brand(r["name"], r["operator"]) or "?", axis=1)))),
                         names="; ".join(sorted(set(d.name.dropna().astype(str)))[:6]),
                         best_rank=int(d.echo_dc_rank_equal.min()), worst_rank=int(d.echo_dc_rank_equal.max()), members=" | ".join(d.facility_id)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t, pt, xy = build()
    ref = pd.read_csv(PB / "results" / "candidate_duplicate_campuses.csv")
    assert len(pt) == len(ref) == 956, (len(pt), len(ref))
    assert set(map(tuple, ref[["id_a", "id_b"]].values)) | set(map(tuple, ref[["id_b", "id_a"]].values)) >= set(map(tuple, pt[["id_a", "id_b"]].values))
    gs = group_summary(t, xy)
    pt.to_csv(RESULTS / "e2_pair_tiers.csv", index=False)
    t.drop(columns=[]).to_csv(RESULTS / "e2_record_campus_groups.csv", index=False)
    gs.to_csv(RESULTS / "e2_campus_groups.csv", index=False)
    inv = set(pt.id_a) | set(pt.id_b)
    n100 = set(pt[pt.distance_m <= 100].id_a) | set(pt[pt.distance_m <= 100].id_b)
    summ = dict(pairs_le_250m=len(pt), records_in_any_pair=len(inv), records_with_partner_le_100m=len(n100),
                pairs_by_tier=pt.tier.value_counts().sort_index().to_dict(),
                pairs_by_tier_le100=pt[pt.distance_m <= 100].tier.value_counts().sort_index().to_dict(),
                records_by_status=t.group_status.value_counts().to_dict(),
                n_supported_groups=int(gs.shape[0]), records_in_supported_groups=int(gs.n_records.sum()),
                n_units=int(t.unit_id.nunique()), group_size_distribution=gs.n_records.value_counts().sort_index().to_dict(),
                max_group_size=int(gs.n_records.max()), chain_risk_records=int((t.group_status == "chain_risk").sum()),
                pairs_same_country_false=int((pt.country_a != pt.country_b).sum()))
    json.dump(summ, open(RESULTS / "e2_campus_summary.json", "w"), indent=1, default=int)
    print(json.dumps(summ, indent=1, default=int))
    print(gs.sort_values("n_records", ascending=False).head(15)[["campus_group", "country", "n_records", "diameter_m", "brands", "best_rank", "worst_rank", "status"]].to_string(index=False))
