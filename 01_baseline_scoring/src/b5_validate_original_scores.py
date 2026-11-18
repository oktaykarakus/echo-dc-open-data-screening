"""Validation of the recomputed scores against the literature-weighted scores of the initial implementation of the index
(manuscript Section 2.2): rank correlation and top-decile overlap. Reports only; nothing is tuned.
Run from 01_baseline_scoring/src:  python b5_validate_original_scores.py"""
import json
import numpy as np, pandas as pd
from common import *

sc = pd.read_csv(RESULTS / "scores_B0QA.csv")
ref = pd.read_csv(DATA / "recovered_literature_scores.csv")
d = sc.merge(ref, on="facility_id", validate="1:1")
n10 = int(np.ceil(len(d) * 0.10))
top_ref = set(d.loc[d.lit_rank_recovered <= n10, "facility_id"])
top_new = set(d.loc[d.early_rank <= n10, "facility_id"])
out = {"n_records": len(d),
       "spearman_rank": float(d.early_rank.corr(d.lit_rank_recovered, method="spearman")),
       "opportunity_scores_identical_share": float(((d.early_opp - d.lit_opp_recovered).abs() < 1e-9).mean()),
       "top_decile_n": n10, "top_decile_overlap": len(top_ref & top_new)}
print(json.dumps(out, indent=1))
assert out["n_records"] == 1216 and out["top_decile_overlap"] == 120 and out["spearman_rank"] > 0.998
