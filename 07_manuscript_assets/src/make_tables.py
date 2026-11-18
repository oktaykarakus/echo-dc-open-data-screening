"""Build manuscript tables (CSV + Markdown fragments) from existing result tables (read-only)."""
import json, os
import numpy as np, pandas as pd

H1 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V2 = os.path.abspath(os.path.join(H1, ".."))
R = lambda p: os.path.join(V2, p)
TAB = os.path.join(H1, "Tables"); os.makedirs(TAB, exist_ok=True)

def md(df):
    cols = list(df.columns)
    wid = []
    for c in cols:
        L = max([len(str(c)) // 2] + [len(str(v)) for v in df[c]])
        wid.append(max(5, min(48, L)))
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["-" * w for w in wid]) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)

def emit(name, df, stem):
    df.to_csv(os.path.join(TAB, f"{name}_{stem}.csv"), index=False)
    open(os.path.join(TAB, f"{name}.md"), "w").write(md(df) + "\n")

f3 = lambda x: f"{x + 1e-9:.3f}" if x >= 0 else f"{x - 1e-9:.3f}"
f4 = lambda x: f"{x:.4f}"

# ---- Table 1: indicators (specification; static, taken from the scoring implementation)
t1 = pd.DataFrame([
 ["Opportunity", "Heat demand", "Sum of Hotmaps heat-demand density in the buffer (2015)", "log(1+x)", "Higher = better"],
 ["Opportunity", "Population density (also in Feasibility)", "GHS-POP population ÷ buffer area", "log(1+x)", "Higher = better"],
 ["Opportunity", "Heating degree days", "Annual HDD 2024, mean of a 15 km buffer (ERA5-Land)", "none", "Higher = better"],
 ["Opportunity", "Anchor count", "Hospitals, universities and colleges, care facilities and sports centres in the buffer", "log(1+x)", "Higher = better"],
 ["Opportunity", "Anchor-type diversity", "Number of anchor categories present in the buffer (0–4)", "none", "Higher = better"],
 ["Opportunity", "Nearest-anchor distance", "Distance to the nearest anchor in the pooled national layers (not radius-limited)", "log(1+x), reversed", "Lower = better"],
 ["Feasibility", "Urban compactness", "Mean of normalised GHSL built-up share and normalised Dynamic World built-up share", "each: none", "Higher = better"],
 ["Feasibility", "Road density", "Length of motorway, trunk and primary roads (with links) within the buffer ÷ buffer area", "none", "Higher = better"],
 ["Feasibility", "Nearest-road distance", "Distance to the nearest such road (not radius-limited)", "log(1+x), reversed", "Lower = better"],
 ["Feasibility", "Population density", "As above (second membership)", "log(1+x)", "Higher = better"],
], columns=["Component", "Indicator", "Definition (2.5 km reference buffer)", "Transform before scaling", "Direction"])
emit("Table1", t1, "indicators")

# ---- Table 2: composite vs simple indicators
d1 = pd.read_csv(R("03_robustness/results/d1_composite_vs_simple_summary.csv")).set_index("comparator")
names = {"heat_hotmaps": "Hotmaps heat demand", "population_density": "Population density", "opportunity": "Opportunity component", "feasibility": "Feasibility component"}
rows = []
for k, n in names.items():
    r = d1.loc[k]
    rows.append([n, f3(r.spearman), f3(r.kendall_tau_b), f"{int(r.top_decile_overlap)}", f"{int(r.top20_overlap)}", f"{int(r.median_abs_drank + 0.5)}", f"{r.share_abs_drank_ge100*100:.0f}", f"{int(r.composite_only_top_decile)}"])
t2 = pd.DataFrame(rows, columns=["Comparator", "Spearman ρ", "Kendall τ-b", "Top-decile overlap (of 122)", "Top-20 overlap", "Median |Δrank|", "Share |Δrank| ≥ 100 (%)", "Composite-only top-decile records"])
emit("Table2", t2, "composite_vs_simple")

# ---- Table 3: ablations and alternative specifications
ab = pd.read_csv(R("03_robustness/results/d3_ablation_summary.csv")).set_index("family")
d2 = pd.read_csv(R("03_robustness/results/d2_population_comparisons.csv"))
clip = pd.read_csv(R("03_robustness/results/d4a_clipping_summary.csv")).set_index("variant")
def rowab(label, r, nominal):
    return [label, nominal, f3(r.spearman), f"{int(r.top_decile_overlap)}", f"{int(r.top20_overlap)}", f"{int(r.median_abs_drank + 0.5)}", f"{int(r.n_leave_top_decile)}"]
rows = [
 rowab("Road accessibility removed (density, nearest distance)", ab.loc["road_accessibility"], "0.250"),
 rowab("Anchors removed (count, diversity, nearest distance)", ab.loc["anchors"], "0.250"),
 rowab("Urban compactness removed", ab.loc["urban_compactness"], "0.125"),
 rowab("Heat removed (Hotmaps and HDD)", ab.loc["heat_demand"], "0.167"),
 rowab("HDD removed", ab.loc["[sub] hdd_only"], "0.083"),
 rowab("Hotmaps heat removed", ab.loc["[sub] hotmaps_only"], "0.083"),
]
for alt, lab in [("D2-B", "Population density removed from Opportunity"), ("D2-C", "Population density removed from Feasibility")]:
    r = d2[(d2.base == "D2-A") & (d2.alt == alt)].iloc[0]
    rows.append([lab, "0.083 / 0.125", f3(r.spearman), f"{int(r.top_decile_overlap)}", f"{int(r.top20_overlap)}", f"{int(r.median_abs_drank + 0.5)}", f"{int(r.n_leave_top_decile)}"])
for k, lab in [("none_min_max", "No percentile clipping (min–max)"), ("modest_2p5_97p5", "Clipping at 2.5th–97.5th percentiles")]:
    r = clip.loc[k]
    rows.append([lab, "–", f4(r.spearman), f"{int(r.top_decile_overlap)}", f"{int(r.top20_overlap)}", f"{int(r.median_abs_drank + 0.5)}", f"{int(r.n_leave_top_decile)}"])
t3 = pd.DataFrame(rows, columns=["Alternative specification", "Linearised weight removed", "Spearman ρ", "Top-decile overlap (of 122)", "Top-20 overlap", "Median |Δrank|", "Records leaving top decile"])
emit("Table3", t3, "ablation")

# ---- Table 4: multiscale agreement by weighting scenario
c6 = pd.read_csv(R("02_multiscale/results/c6_weighting_scenarios_summary.csv")).set_index("scenario")
lab = {"equal": "Equal weights (primary)", "literature_informed": "Literature-informed weights", "entropy": "Entropy weights, refitted per radius", "entropy_ref": "Entropy weights, held at 2.5 km values"}
rows = []
for k, n in lab.items():
    r = c6.loc[k]
    rows.append([n, f"{r.rho_1km:.3f} / {r.rho_5km:.3f} / {r.rho_10km:.3f}", f"{int(r.top_overlap_1km)} / {int(r.top_overlap_5km)} / {int(r.top_overlap_10km)}",
                 f"{int(r.n_top_all_four)}", f"{int(r.n_top_any_scale)}", f"{int(r.top20_all_four)}", f"{int(r.n_range_ge200)}", f"{r.rho_1km_vs_10km:.3f}"])
cm = pd.read_csv(R("04_operational_projects/results/e4_campus_view_metrics.csv"))
r = cm[cm.view == "primary"].iloc[0]
rows.append(["Campus units (1,017 units; equal weights)", f"{r.spearman_1km_vs_2p5km:.3f} / {r.spearman_5km_vs_2p5km:.3f} / {r.spearman_10km_vs_2p5km:.3f}",
             f"{int(r.top_decile_overlap_1km_vs_2p5km)} / {int(r.top_decile_overlap_5km_vs_2p5km)} / {int(r.top_decile_overlap_10km_vs_2p5km)} (of 102)",
             f"{int(r.top_decile_all4_scales)} (of 102)", f"{int(r.top_decile_ge1_scale)}", "–", f"{r.share_rank_range_ge_scaled_threshold*100:.1f}% of units", "–"])
t4 = pd.DataFrame(rows, columns=["Specification", "Spearman ρ with 2.5 km (1 / 5 / 10 km)", "Top-decile overlap with 2.5 km (1 / 5 / 10 km; of 122)", "Top decile at all four radii", "Top decile at ≥ 1 radius", "Top 20 at all four radii", "Rank range ≥ 200", "ρ, 1 km vs 10 km"])
emit("Table4", t4, "multiscale")

# ---- Table 5: benchmark (selected statistics) + supplementary full table
bm = pd.read_csv(R("06_gaussian_benchmark/results/benchmark_summary.csv")).set_index("statistic")
nm = {"top_decile_all4": "Top decile at all four radii", "top_decile_at_least_once": "Top decile at ≥ 1 radius", "top20_all4": "Top 20 at all four radii",
      "range_ge_100": "Rank range ≥ 100", "range_ge_200": "Rank range ≥ 200", "range_ge_400": "Rank range ≥ 400",
      "overlap_top_decile_1km_vs_2p5km": "Top-decile overlap, 1 vs 2.5 km", "overlap_top_decile_2p5km_vs_5km": "Top-decile overlap, 2.5 vs 5 km",
      "overlap_top_decile_2p5km_vs_10km": "Top-decile overlap, 2.5 vs 10 km", "overlap_top_decile_1km_vs_10km": "Top-decile overlap, 1 vs 10 km",
      "rank_range_median": "Median rank range", "max_rank_displacement": "Maximum rank displacement"}
def brow(k, n):
    r = bm.loc[k]
    return [n, f"{r.observed:g}", f"{r.benchmark_mean + 1e-9:.1f}", f"{r.central90_lo + 1e-9:.0f}–{r.central90_hi + 1e-9:.0f}", f"{r.observed_percentile_in_simulation:.0f}", "yes" if r.observed_inside_central90 else "no"]
cols5 = ["Statistic", "Observed", "Benchmark mean", "Central 90% of benchmark", "Observed percentile", "Inside interval"]
t5 = pd.DataFrame([brow(k, n) for k, n in nm.items()], columns=cols5)
emit("Table5", t5, "gaussian_benchmark")
full = pd.DataFrame([brow(k, k) for k in bm.index], columns=cols5)
full.to_csv(os.path.join(TAB, "TableS1_gaussian_benchmark_all17.csv"), index=False)

# ---- Table 6: French evaluation
five = pd.read_csv(R("05_france_network_proximity/results/f2b_five_arm_auc.csv")).set_index("arm")
arms = [("A", "ECHO-DC (equal weights)"), ("P", "Population density"), ("H", "Hotmaps heat demand"), ("B*", "Construct-balanced alternative (exploratory)"),
        ("C", "Pareto-layer alternative (exploratory)"), ("Opportunity", "Opportunity component (descriptive)"), ("Feasibility", "Feasibility component (descriptive)")]
rows = []
for k, n in arms:
    r = five.loc[k]
    sg = lambda v: ("+" if v >= 0.0005 else ("−" if v <= -0.0005 else "")) + f"{abs(v) + 1e-9:.3f}"
    d = "–" if k == "P" else f"{sg(r.dAUC_vs_P)} ({sg(r.dAUC_ci90_lo_50km)} to {sg(r.dAUC_ci90_hi_50km)})"
    rows.append([n, f3(r.AUC), f"{f3(r.AUC_ci90_lo_50km)}–{f3(r.AUC_ci90_hi_50km)}", d])
t6 = pd.DataFrame(rows, columns=["Ranking", "AUC", "90% interval", "ΔAUC versus population density (90% interval)"])
emit("Table6", t6, "france_auc")

# ---- Table 7: operational projects
pp = pd.read_csv(R("04_operational_projects/results/e3_project_positions.csv"))
ev = pd.read_csv(R("04_operational_projects/results/e1_external_evidence_table.csv"))
pp = pp[pp.match_confidence == "High"].merge(ev[["case_id", "status_evidence_date", "heat_reuse_partner", "n_matched_records", "in_primary_operational_set"]], on="case_id")
pp = pp[pp.in_primary_operational_set == True].sort_values("rank_2p5km")
assert len(pp) == 8 and pp.n_matched_records_x.sum() == 13 if "n_matched_records_x" in pp else True
fmt = lambda v: f"{v:g}"
rows = []
for r in pp.itertuples():
    rows.append([r.facility.replace("Mantsala", "Mäntsälä").replace("Digital Realty VIE1 & VIE2", "Digital Realty Vienna (Floridsdorf)"), r.country, (str(r.status_evidence_date)[:4] + " (year only)") if str(r.status_evidence_date).endswith("12-31") else str(r.status_evidence_date)[:10],
                 str(int(getattr(r, "n_records", getattr(r, "n_matched_records", 1)))), fmt(r.rank_1km), fmt(r.rank_2p5km), fmt(r.rank_5km), fmt(r.rank_10km)])
t7 = pd.DataFrame(rows, columns=["Project", "Country", "Evidence date", "Records", "Rank, 1 km", "Rank, 2.5 km", "Rank, 5 km", "Rank, 10 km"])
emit("Table7", t7, "operational_projects")
print("tables written:", sorted(f for f in os.listdir(TAB)))
print(t7.to_string()); print("median 2.5km rank of the 8:", float(np.median(pp.rank_2p5km)))
