"""C4: multiscale scores (Protocol A primary; Protocol B for Feasibility only). Hard gate: the 2.5 km run must reproduce
the frozen B0-QA scores within numerical tolerance, otherwise the script stops before writing any multiscale result."""
import json
import numpy as np, pandas as pd
from cfg import *
import multiscale as M
import scoring as S

TOL = 1e-12

def gate_against_b0qa(out25, ref):
    """2.5 km (this pipeline) vs frozen B0-QA scores. Returns dict of max abs diffs; raises on failure."""
    res = {}
    assert (out25.facility_id.values == ref.facility_id.values).all(), "facility order differs from B0-QA"
    for sc in SCENARIOS:
        for k in ["opportunity_score", "feasibility_score", "echo_dc_score_raw", "echo_dc_score"]:
            c = f"{k}_{sc}"
            res[c] = float(np.abs(out25[c].to_numpy() - ref[c].to_numpy()).max())
        r = f"echo_dc_rank_{sc}"
        res[r] = int((out25[r].to_numpy() != ref[r].to_numpy()).sum())     # number of rank mismatches
    for c in S.ALL_IND:
        res[c] = float(np.abs(out25[c].to_numpy() - ref[c].to_numpy()).max())
    bad = {k: v for k, v in res.items() if v > TOL}
    if bad:
        raise SystemExit(f"STOP: 2.5 km does not reproduce B0-QA: {bad}")
    return res

def main():
    df = M.load_table()
    ref = pd.read_csv(B0QA)
    n = len(df); assert n == 1216 and df.facility_id.is_unique and df.country.nunique() == 20
    scored, weights = {}, {}
    for s in SCALES:
        out, w = M.score_scale(df, s)
        scored[s], weights[s] = out, w
    gate = gate_against_b0qa(scored[REF], ref)
    print("GATE OK: 2.5 km reproduces B0-QA; max |diff| =", max(v for v in gate.values()))
    w_ref = weights[REF]
    rows, wrows, ties = [], [], []
    for s in SCALES:
        out = M.score_with_entropy_held(scored[s], w_ref)
        scored[s] = out
        keep = ["facility_id", "country"] + list(M.GENERIC) 
        g = M.generic(out, s)
        cols = ["facility_id", "country"] + [M.GENERIC[c] for c in M.GENERIC] + [
            f"{k}_{sc}" for sc in SCENARIOS + ["entropy_ref"] for k in
            ["opportunity_score", "feasibility_score", "echo_dc_score_raw", "echo_dc_rank"]]
        t = g[cols].copy(); t.insert(2, "scale", s); t.insert(3, "radius_km", SCALE_KM[s])
        rows.append(t)
        for sc in SCENARIOS + ["entropy_ref"]:
            ties.append({"scale": s, "scenario": sc, "n_tied_raw_scores": int(n - out[f"echo_dc_score_raw_{sc}"].nunique())})
        for sc, (wo, wf) in {**weights[s], "entropy_ref": w_ref["entropy"]}.items():
            wrows.append({"scale": s, "scenario": sc, **{"opp:" + k.replace("_2p5km", ""): v for k, v in wo.items()}, **{"feas:" + k.replace("_2p5km", ""): v for k, v in wf.items()}})
    long = pd.concat(rows, ignore_index=True)
    # QA on the assembled long table
    assert len(long) == 4 * n and long.groupby("scale").facility_id.nunique().eq(n).all()
    for s in SCALES:
        assert (long[long.scale == s].facility_id.values == df.facility_id.values).all()          # common ordering
        for sc in SCENARIOS + ["entropy_ref"]:
            r = long.loc[long.scale == s, f"echo_dc_rank_{sc}"]
            assert sorted(r) == list(range(1, n + 1))                                                 # ranks are a permutation
    num = long.drop(columns=["facility_id", "country", "scale"]).to_numpy(float)
    assert np.isfinite(num).all()
    ind = [c for c in long.columns if c.startswith("norm_")]
    assert long[ind].min().min() >= 0 and long[ind].max().max() <= 1
    long.to_csv(RESULTS / "multiscale_scores_long.csv", index=False)

    # Protocol B (Feasibility only)
    brows = []
    for s in SCALES:
        b, refp = M.feasibility_protocol_b(df, s, w_ref["entropy"][1])
        b.insert(1, "scale", s); brows.append(b)
    bl = pd.concat(brows, ignore_index=True)
    # B at the reference scale must equal Protocol A at the reference scale
    a25 = scored[REF]
    chk = {sc: float(np.abs(bl[bl.scale == REF][f"feasibility_B_{sc}"].to_numpy() - a25[f"feasibility_score_{sc}"].to_numpy()).max())
           for sc in ["equal", "literature_informed", "entropy_ref"]}
    assert max(chk.values()) < 1e-12, chk
    bl.to_csv(RESULTS / "protocolB_feasibility_long.csv", index=False)

    pd.DataFrame(wrows).to_csv(RESULTS / "weights_by_scale.csv", index=False)
    pd.DataFrame(ties).to_csv(RESULTS / "raw_score_ties.csv", index=False)
    json.dump({"gate_max_abs_diff": max(gate.values()), "gate_detail": gate, "protocolB_ref_check": chk,
               "n_facilities": n, "n_countries": int(df.country.nunique()), "top_decile_k": M.n_top(n),
               "tolerance": TOL}, open(RESULTS / "c4_reproduction_gate.json", "w"), indent=1)
    print("wrote multiscale_scores_long.csv", long.shape, "| protocolB", bl.shape, "| ties", pd.DataFrame(ties).n_tied_raw_scores.max())

if __name__ == "__main__":
    main()
