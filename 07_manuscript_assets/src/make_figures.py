"""Manuscript figures (Fig. 1-6) built only from existing result tables (read-only).
H4 version: identical to the H2 script except (i) paths, (ii) the Vienna project label in Figure 6 (no unsupported facility codes), (iii) the colour-bar label of Figure 4a moved up so that it no longer touches the panel title.
Run: bash src/run_figures.sh   (sets all temp/cache paths inside the H1 folder)."""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection
from pyproj import Transformer
from scipy.stats import rankdata

H1 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V2 = os.path.abspath(os.path.join(H1, ".."))
ROOT = V2
FIG = os.path.join(H1, "Figures")
os.makedirs(FIG, exist_ok=True)
R = lambda p: os.path.join(V2, p)

# Okabe-Ito colour-blind-safe palette
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#8c8c8c")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 300,
})

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

def panel(ax, letter, x=-0.02, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")

# ------------------------------------------------------------------ data
sc = pd.read_csv(R("01_baseline_scoring/results/scores_B0QA.csv"))
master = pd.read_csv(os.path.join(ROOT, "data", "inputs", "inventory_indicator_master_87col.csv"),
                     usecols=["facility_id", "pop_density_2p5km", "hotmaps_heat_sum_2p5km"])
assert len(sc) == 1216 and sc.facility_id.is_unique
d1 = pd.read_csv(R("03_robustness/results/d1_facility_ranks.csv"))
K = 122

# ================================================================== Fig 1: map + country composition
gj = json.load(open(os.path.join(H1, "data", "ne_50m_admin_0_countries.geojson")))
tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
XL, YL = (2.45e6, 6.05e6), (1.25e6, 5.45e6)
patches = []
def add_ring(ring):
    a = np.asarray(ring)
    x, y = tr.transform(a[:, 0], a[:, 1])
    x, y = np.asarray(x), np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3: return
    if x[ok].max() < XL[0] or x[ok].min() > XL[1] or y[ok].max() < YL[0] or y[ok].min() > YL[1]: return
    patches.append(MplPoly(np.c_[x[ok], y[ok]], closed=True))
for f in gj["features"]:
    g = f["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    for p in polys:
        lon = np.asarray(p[0])[:, 0]; lat = np.asarray(p[0])[:, 1]
        if lon.max() < -25 or lon.min() > 45 or lat.max() < 30 or lat.min() > 75: continue
        add_ring(p[0])
x, y = tr.transform(sc.lon.values, sc.lat.values)
sc["x"], sc["y"] = x, y

fig = plt.figure(figsize=(7.3, 4.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.22)
ax = fig.add_subplot(gs[0])
ax.add_collection(PatchCollection(patches, facecolor="#f0f0f0", edgecolor="#b0b0b0", linewidth=0.3, zorder=0))
o = np.argsort(sc.echo_dc_score_raw_equal.values)
top = sc.echo_dc_rank_equal.values <= K
s1 = ax.scatter(sc.x.values[o], sc.y.values[o], c=sc.echo_dc_score_raw_equal.values[o], cmap="viridis", s=7,
                linewidths=0.15, edgecolors="white", vmin=0, vmax=1, zorder=2)
ax.scatter(sc.x.values[top], sc.y.values[top], s=22, facecolors="none", edgecolors=OI["verm"], linewidths=0.6, zorder=3)
ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal"); ax.axis("off")
cb = fig.colorbar(s1, ax=ax, orientation="horizontal", fraction=0.04, pad=0.01, shrink=0.7)
cb.set_label("ECHO-DC score (equal weights, 2.5 km)"); cb.outline.set_linewidth(0.4)
ax.scatter([], [], s=22, facecolors="none", edgecolors=OI["verm"], linewidths=0.6, label="Top decile (122 records)")
ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.98))
panel(ax, "a", x=0.02, y=0.96)

ax2 = fig.add_subplot(gs[1])
cc = sc.groupby("country").agg(n=("facility_id", "size"), top=("echo_dc_rank_equal", lambda r: int((r <= K).sum())),
                               med=("echo_dc_score_raw_equal", "median")).sort_values("n")
yy = np.arange(len(cc))
ax2.barh(yy, cc.n, color="#d0d0d0", height=0.7, label="Records")
ax2.barh(yy, cc.top, color=OI["verm"], height=0.7, label="Top-decile records")
for yi, (n, t) in enumerate(zip(cc.n, cc.top)):
    ax2.text(n + 4, yi, f"{n}" + (f" ({t})" if t else ""), va="center", fontsize=6.3)
ax2.set_yticks(yy); ax2.set_yticklabels(cc.index); ax2.set_xlabel("Number of records")
ax2.set_xlim(0, 410); ax2.legend(frameon=False, loc="lower right")
panel(ax2, "b", x=-0.12, y=1.0)
save(fig, "Fig1_screening_overview")

# ================================================================== Fig 2: composite vs simple indicators
fig, axs = plt.subplots(1, 3, figsize=(7.3, 2.75), gridspec_kw=dict(width_ratios=[1, 1, 0.95], wspace=0.5))
for ax, col, lab, letter in [(axs[0], "rank_population_density", "Population-density rank", "a"),
                             (axs[1], "rank_heat_hotmaps", "Hotmaps heat-demand rank", "b")]:
    inc = d1.composite_rank <= K; ins = d1[col] <= K
    cls = np.where(inc & ins, 0, np.where(inc & ~ins, 1, np.where(~inc & ins, 2, 3)))
    cols = [OI["blue"], OI["verm"], OI["orange"], "#c8c8c8"]
    for k in (3, 2, 0, 1):
        m = cls == k
        ax.scatter(d1.composite_rank[m], d1[col][m], s=3.2 if k == 3 else 5, color=cols[k], linewidths=0, alpha=0.9 if k != 3 else 0.6,
                   rasterized=True, label=["Both top decile", "Composite only", "Indicator only", "Neither"][k])
    ax.axvline(K + 0.5, color="k", lw=0.4, ls=":"); ax.axhline(K + 0.5, color="k", lw=0.4, ls=":")
    ax.set_xlabel("Composite rank (1 = highest)"); ax.set_ylabel(lab)
    ax.set_xlim(0, 1216); ax.set_ylim(0, 1216); ax.set_aspect("equal")
    panel(ax, letter, x=-0.2, y=1.02)
    n_only = int((cls == 1).sum()); ax.text(0.97, 0.03, f"composite-only: {n_only}", transform=ax.transAxes, ha="right", fontsize=6.5, color=OI["verm"])
axs[0].legend(frameon=False, loc="upper left", fontsize=6, markerscale=2, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.25, bbox_to_anchor=(0.12, 0.98))
ax = axs[2]
j = json.load(open(R("03_robustness/results/d1_union_and_linear_reproduction.json")))
both, neither = j["composite_top_decile_in_heat_and_pop"], j["composite_top_decile_in_neither"]
one = j["composite_top_decile_in_union_heat_or_pop_top_decile"] - both
vals = [both, one, neither]; labs = ["In both simple\ntop deciles", "In one simple\ntop decile", "In neither"]
colz = [OI["blue"], OI["sky"], OI["verm"]]
left = 0
for v, l, c in zip(vals, labs, colz):
    ax.barh(0, v, left=left, color=c, height=0.5); ax.text(left + v / 2, 0, str(v), ha="center", va="center", color="white" if c != OI["sky"] else "k", fontsize=8, fontweight="bold")
    ax.text(left + v / 2, -0.43 if l != labs[1] else 0.43, l, ha="center", va="center" if l != labs[1] else "bottom", fontsize=6.3)
    left += v
ax.set_xlim(0, 122); ax.set_ylim(-0.95, 0.95); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("Records in composite\ntop decile (n = 122)")
ax.text(0.5, 0.97, f"Linear $R^2$ of composite score\non heat + population: {j['R2_composite_score_on_heat_pop_indicators']:.2f}",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
panel(ax, "c", x=-0.02, y=1.02)
save(fig, "Fig2_composite_vs_simple")

# ================================================================== Fig 3: ablation and effect-size ladder
ab = pd.read_csv(R("03_robustness/results/d3_ablation_summary.csv")).set_index("family")
print("ablation families:", list(ab.index))
d2 = pd.read_csv(R("03_robustness/results/d2_population_comparisons.csv"))
clip = pd.read_csv(R("03_robustness/results/d4a_clipping_summary.csv")).set_index("variant")
sp = pd.read_csv(R("02_multiscale/results/c5_global_spearman_equal.csv"), index_col=0)
ov = pd.read_csv(R("02_multiscale/results/c5_global_top_overlap_equal.csv"), index_col=0)
d1s = pd.read_csv(R("03_robustness/results/d1_composite_vs_simple_summary.csv")).set_index("comparator")
rowA = lambda a, b: d2[(d2.base == a) & (d2.alt == b)].iloc[0]
def abrow(name):
    for k in ab.index:
        if name in k: return ab.loc[k]
    raise KeyError(name)
ladder = [
    ("Clipping window 2.5–97.5%", clip.loc["modest_2p5_97p5", "spearman"], clip.loc["modest_2p5_97p5", "top_decile_overlap"], "m"),
    ("Hotmaps heat removed", abrow("hotmaps").spearman, abrow("hotmaps").top_decile_overlap, "m"),
    ("Population removed from Opportunity", rowA("D2-A", "D2-B").spearman, rowA("D2-A", "D2-B").top_decile_overlap, "m"),
    ("Population removed from Feasibility", rowA("D2-A", "D2-C").spearman, rowA("D2-A", "D2-C").top_decile_overlap, "m"),
    ("Heat family removed (Hotmaps + HDD)", ab.loc["heat_demand"].spearman, ab.loc["heat_demand"].top_decile_overlap, "m"),
    ("Urban compactness removed", abrow("compact").spearman, abrow("compact").top_decile_overlap, "m"),
    ("No clipping (min–max)", clip.loc["none_min_max", "spearman"], clip.loc["none_min_max", "top_decile_overlap"], "m"),
    ("Anchors removed", ab.loc["anchors"].spearman, ab.loc["anchors"].top_decile_overlap, "m"),
    ("Road accessibility removed", abrow("road").spearman, abrow("road").top_decile_overlap, "m"),
    ("Composite vs Hotmaps heat alone", d1s.loc["heat_hotmaps", "spearman"], d1s.loc["heat_hotmaps", "top_decile_overlap"], "c"),
    ("Composite vs population density alone", d1s.loc["population_density", "spearman"], d1s.loc["population_density", "top_decile_overlap"], "c"),
    ("Radius 5 km vs 2.5 km", sp.loc["2p5km", "5km"], ov.loc["2p5km", "5km"], "r"),
    ("Radius 1 km vs 2.5 km", sp.loc["1km", "2p5km"], ov.loc["1km", "2p5km"], "r"),
    ("Radius 10 km vs 2.5 km", sp.loc["2p5km", "10km"], ov.loc["2p5km", "10km"], "r"),
]
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.7), gridspec_kw=dict(width_ratios=[0.85, 1.15], wspace=0.95))
ax = axs[0]
order = ["Road accessibility", "Anchors", "Urban compactness", "Heat (Hotmaps + HDD)", "Heating degree days", "Hotmaps heat sum"]
keys = [abrow("road"), ab.loc["anchors"], abrow("compact"), ab.loc["heat_demand"], abrow("hdd"), abrow("hotmaps")]
yy = np.arange(len(order))[::-1]
cols = [OI["verm"], OI["orange"], OI["sky"], OI["blue"], "#7aa6c9", "#7aa6c9"]
for yi, r, c, nm in zip(yy, keys, cols, order):
    ax.barh(yi, r.n_leave_top_decile, color=c, height=0.62)
    ax.text(r.n_leave_top_decile + 0.6, yi, f"{int(r.n_leave_top_decile)}   (ρ = {r.spearman:.3f})", va="center", fontsize=6.8)
ax.set_yticks(yy); ax.set_yticklabels(order); ax.set_xlim(0, 52)
ax.set_xlabel("Top-decile records replaced (of 122)")
ax.set_title("Indicator family or indicator removed", loc="left", fontsize=8)
panel(ax, "a", x=-0.02, y=1.0)
ax = axs[1]
ladder_s = sorted(ladder, key=lambda t: t[1], reverse=True)
yy = np.arange(len(ladder_s))[::-1]
gcol = {"m": OI["blue"], "c": OI["green"], "r": OI["verm"]}
for yi, (lab, rho, ovr, g) in zip(yy, ladder_s):
    ax.plot([0.8, rho], [yi, yi], color="#d6d6d6", lw=0.8, zorder=1)
    ax.scatter(rho, yi, color=gcol[g], s=22, zorder=3, marker={"m": "o", "c": "s", "r": "D"}[g])
ax.set_yticks(yy); ax.set_yticklabels([t[0] for t in ladder_s], fontsize=6.6)
ax.set_xlim(0.8, 1.0); ax.set_xlabel("Spearman ρ with reference ranking")
ax.axvline(0.999, color="k", ls=":", lw=0.5); ax.text(0.9985, yy[0] + 0.55, "0.999 agreement\nwith original", ha="right", va="bottom", fontsize=5.8)
for g, lab, m in [("m", "Method change", "o"), ("c", "Composite vs single indicator", "s"), ("r", "Change of radius", "D")]:
    ax.scatter([], [], color=gcol[g], marker=m, s=18, label=lab)
ax.legend(frameon=False, loc="center left", fontsize=6.2, borderaxespad=0.1)
ax.set_title("Size of ranking changes", loc="left", fontsize=8)
panel(ax, "b", x=-0.02, y=1.0)
ax.yaxis.set_ticks_position("none")
save(fig, "Fig3_ablation_and_effect_sizes")
pd.DataFrame(ladder_s, columns=["change", "spearman", "top_decile_overlap", "group"]).to_csv(os.path.join(H1, "Tables/fig3_effect_sizes_source.csv"), index=False)

# ================================================================== Fig 4: cross-radius stability and benchmark
bm = pd.read_csv(R("06_gaussian_benchmark/results/benchmark_summary.csv"))
dd = pd.read_csv(R("06_gaussian_benchmark/results/benchmark_decile_diagnostic.csv"))
fig = plt.figure(figsize=(7.3, 5.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.55, wspace=0.55)
ax = fig.add_subplot(gs[0, 0])
M = np.array(json.load(open(R("06_gaussian_benchmark/results/benchmark_meta.json")))["observed_spearman_matrix"], dtype=float)
im = ax.imshow(M, cmap="cividis", vmin=0.75, vmax=1.0)
lbl = ["1 km", "2.5 km", "5 km", "10 km"]
ax.set_xticks(range(4)); ax.set_xticklabels(lbl); ax.set_yticks(range(4)); ax.set_yticklabels(lbl)
for i in range(4):
    for j2 in range(4):
        ax.text(j2, i, f"{M[i, j2]:.3f}", ha="center", va="center", fontsize=7, color="white" if M[i, j2] < 0.93 else "k")
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0); cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[0.8, 0.9, 1.0]); cb.ax.set_title("ρ", fontsize=7, pad=14); cb.outline.set_linewidth(0.4)
ax.set_title("Rank agreement between radii", loc="left", fontsize=8); panel(ax, "a", x=-0.12, y=1.04)

ax = fig.add_subplot(gs[0, 1])
x = np.arange(1, 11)
ax.fill_between(x, dd.benchmark_p05, dd.benchmark_p95, color=OI["sky"], alpha=0.35, lw=0, label="Benchmark 5–95%")
ax.plot(x, dd.benchmark_mean_of_median_range, color=OI["blue"], lw=1, label="Benchmark mean")
ax.plot(x, dd.observed_median_range, color=OI["verm"], marker="o", ms=3.5, lw=1.1, label="Observed")
ax.set_xticks(x); ax.set_xlabel("Decile of 2.5 km rank (1 = highest)"); ax.set_ylabel("Median rank range\nacross four radii")
ax.legend(frameon=False, loc="lower center", fontsize=6.3, ncol=1, bbox_to_anchor=(0.5, -0.02))
ax.set_title("Instability by rank position", loc="left", fontsize=8); panel(ax, "b", x=-0.2, y=1.04)

ax = fig.add_subplot(gs[1, :])
names = {
 "top_decile_all4": "Top decile at all four radii", "top_decile_at_least_once": "Top decile at ≥ 1 radius",
 "top20_all4": "Top 20 at all four radii", "range_ge_50": "Rank range ≥ 50", "range_ge_100": "Rank range ≥ 100",
 "range_ge_200": "Rank range ≥ 200", "range_ge_400": "Rank range ≥ 400",
 "overlap_top_decile_1km_vs_2p5km": "Top-decile overlap 1 vs 2.5 km", "overlap_top_decile_1km_vs_5km": "Top-decile overlap 1 vs 5 km",
 "overlap_top_decile_1km_vs_10km": "Top-decile overlap 1 vs 10 km", "overlap_top_decile_2p5km_vs_5km": "Top-decile overlap 2.5 vs 5 km",
 "overlap_top_decile_2p5km_vs_10km": "Top-decile overlap 2.5 vs 10 km", "overlap_top_decile_5km_vs_10km": "Top-decile overlap 5 vs 10 km",
 "rank_range_median": "Rank range, median", "rank_range_p90": "Rank range, 90th percentile",
 "rank_range_p99": "Rank range, 99th percentile", "max_rank_displacement": "Maximum rank displacement"}
assert set(bm.statistic) == set(names)
bm = bm.reset_index(drop=True)
yy = np.arange(len(bm))[::-1]
ax.axvspan(5, 95, color=OI["sky"], alpha=0.25, lw=0, label="Central 90% of benchmark")
for yi, r in zip(yy, bm.itertuples()):
    ins = bool(r.observed_inside_central90)
    ax.scatter(r.observed_percentile_in_simulation, yi, s=26, color=OI["blue"] if ins else OI["verm"], zorder=3,
               marker="o" if ins else "D", edgecolor="white", linewidth=0.4)
    ax.text(103, yi, f"{r.observed:g} vs {r.benchmark_mean + 1e-9:.1f}", va="center", fontsize=6.3)
ax.set_yticks(yy); ax.set_yticklabels([names[s] for s in bm.statistic], fontsize=6.6)
ax.set_xlim(0, 100); ax.set_xlabel("Position of observed value within the simulated distribution (percentile)")
ax.axvline(50, color="k", lw=0.4, ls=":")
ax.text(103, yy[0] + 1.1, "observed vs\nbenchmark mean", fontsize=6.3, va="bottom", style="italic")
ax.scatter([], [], color=OI["blue"], s=22, label="Inside central 90% (16 of 17)"); ax.scatter([], [], color=OI["verm"], marker="D", s=22, label="Outside (1 of 17)")
ax.legend(frameon=False, loc="upper center", fontsize=6.5, ncol=3, bbox_to_anchor=(0.45, -0.17))
ax.set_title("Observed rank turnover against a dependence-preserving Gaussian-copula benchmark", loc="left", fontsize=8)
panel(ax, "c", x=-0.30, y=1.04)
save(fig, "Fig4_scale_stability_benchmark")

# ================================================================== Fig 5: French network-proximity comparison
lab = pd.read_csv(R("05_france_network_proximity/results/f2b_labels.csv"))
lab = lab[lab.status_2500.isin(["positive", "mapped_negative"])]
dfm = lab[["facility_id", "status_2500"]].merge(sc[["facility_id", "echo_dc_score_raw_equal"]], on="facility_id").merge(master, on="facility_id")
assert len(dfm) == 322 and (dfm.status_2500 == "positive").sum() == 247
y = (dfm.status_2500 == "positive").values.astype(int)

def auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def roc(s, y):
    o = np.argsort(-s, kind="mergesort"); ys = y[o]; ss = s[o]
    tps = np.cumsum(ys); fps = np.cumsum(1 - ys)
    keep = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    return np.r_[0, fps[keep] / fps[-1]], np.r_[0, tps[keep] / tps[-1]]

arms = {"ECHO-DC": dfm.echo_dc_score_raw_equal.values, "Population density": dfm.pop_density_2p5km.values,
        "Hotmaps heat": dfm.hotmaps_heat_sum_2p5km.values}
A = {k: auc(v, y) for k, v in arms.items()}
five = pd.read_csv(R("05_france_network_proximity/results/f2b_five_arm_auc.csv")).set_index("arm")
print("AUC check:", {k: round(v, 4) for k, v in A.items()}, "reported:", five.loc[["A", "P", "H"], "AUC"].round(4).tolist())
assert abs(A["ECHO-DC"] - five.loc["A", "AUC"]) < 6e-4 and abs(A["Population density"] - five.loc["P", "AUC"]) < 6e-4 and abs(A["Hotmaps heat"] - five.loc["H", "AUC"]) < 6e-4

sens = pd.read_csv(R("05_france_network_proximity/results/f2b_sensitivity.csv"))
fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], hspace=0.5, wspace=0.55)
ax = fig.add_subplot(gs[0, 0])
for (k, s), c, ls in zip(arms.items(), [OI["verm"], OI["blue"], OI["green"]], ["-", "--", ":"]):
    fx, tx = roc(s, y); ax.plot(fx, tx, color=c, lw=1.2, ls=ls, label=f"{k} (AUC {A[k]:.3f})")
ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.6)
ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate"); ax.set_box_aspect(1)
ax.legend(frameon=False, loc="lower right", fontsize=6.3, handlelength=1.8, borderaxespad=0.2)
ax.set_title("ROC curves", loc="left", fontsize=8); panel(ax, "a", x=-0.2, y=1.04)

ax = fig.add_subplot(gs[0, 1])
arm_lab = [("A", "ECHO-DC"), ("P", "Population density"), ("H", "Hotmaps heat"), ("B*", "Construct-balanced*"), ("C", "Pareto-layer*")]
yy = np.arange(len(arm_lab))[::-1]
for yi, (k, nm) in zip(yy, arm_lab):
    r = five.loc[k]
    c = OI["verm"] if k == "A" else (OI["blue"] if k == "P" else "#6a6a6a")
    ax.plot([r.AUC_ci90_lo_50km, r.AUC_ci90_hi_50km], [yi, yi], color=c, lw=1.4)
    ax.scatter(r.AUC, yi, color=c, s=22, zorder=3)
    ax.text(1.005, yi, f"{r.AUC:.3f}", ha="left", va="center", fontsize=6.8, color=c)
ax.set_yticks(yy); ax.set_yticklabels([n for _, n in arm_lab]); ax.set_xlim(0.84, 1.0); ax.set_box_aspect(1); ax.spines["right"].set_visible(False)
ax.set_xlabel("AUC with 90% block-\nbootstrap interval")
ax.set_title("Discrimination", loc="left", fontsize=8); panel(ax, "b", x=-0.05, y=1.04)

ax = fig.add_subplot(gs[1, :])
vlabels = {"primary_2500m": "Primary (2.5 km, n = 322)", "S3_threshold_1000m": "1 km threshold", "S1_campus_units": "Campus units",
           "S2_exclude_13_salle_technique": "Excl. 13 technical rooms", "S4a_indeterminate_scored_negative": "Indeterminate as negative",
           "S4b_exclude_largest_50km_block": "Excl. Paris-region block", "S5_tier2_traces_as_existing": "Unidentified traces as existing"}
print("sensitivity variants:", list(sens.variant))
sv = sens[sens.variant.isin(vlabels)].copy() if sens.variant.isin(vlabels).any() else sens.copy()
yy = np.arange(len(sv))[::-1]
for yi, r in zip(yy, sv.itertuples()):
    prim = r.variant.startswith("primary")
    c = OI["verm"] if prim else "#555555"
    ax.plot([r.dAUC_A_lo, r.dAUC_A_hi], [yi, yi], color=c, lw=1.4 if prim else 1.0)
    ax.scatter(r.dAUC_A, yi, color=c, s=22 if prim else 14, zorder=3)
ax.axvline(0, color="k", lw=0.5); ax.axvline(0.03, color=OI["green"], lw=0.8, ls="--")
ax.text(0.031, yy[-1] - 0.35, "+0.03 margin", color=OI["green"], fontsize=6.3, ha="left", va="center")
ax.set_yticks(yy); ax.set_yticklabels([vlabels.get(v, v) for v in sv.variant], fontsize=6.8)
ax.set_xlim(-0.05, 0.05); ax.set_ylim(yy[-1] - 0.7, yy[0] + 0.6); ax.set_xlabel("ΔAUC (ECHO-DC minus population density), 90% block-bootstrap interval")
ax.set_title("Incremental discrimination over population density", loc="left", fontsize=8); panel(ax, "c", x=-0.3, y=1.04)
save(fig, "Fig5_french_network_proximity")

# ================================================================== Fig 6: operational projects
pp = pd.read_csv(R("04_operational_projects/results/e3_project_positions.csv"))
pp = pp[pp.match_confidence == "High"].copy()
ev = pd.read_csv(R("04_operational_projects/results/e1_external_evidence_table.csv"))
pp = pp.sort_values("rank_2p5km")
print("operational set:", len(pp), pp.case_id.tolist())
assert len(pp) >= 8
fig, ax = plt.subplots(figsize=(6.2, 3.1))
yy = np.arange(len(pp))[::-1]
mk = {"rank_1km": ("o", "1 km"), "rank_2p5km": ("D", "2.5 km"), "rank_5km": ("s", "5 km"), "rank_10km": ("^", "10 km")}
cmap = {"rank_1km": OI["sky"], "rank_2p5km": OI["verm"], "rank_5km": OI["orange"], "rank_10km": OI["blue"]}
for xv in (122, 244, 608):
    ax.axvline(xv, color="#b5b5b5", lw=0.6, ls=":")
for xv, t in [(122, "top 10%"), (244, "top 20%"), (608, "top 50%")]:
    ax.text(xv, -0.85, t, ha="center", fontsize=6.3, color="#555555")
for yi, r in zip(yy, pp.itertuples()):
    xs = [r.rank_1km, r.rank_2p5km, r.rank_5km, r.rank_10km]
    ax.plot([min(xs), max(xs)], [yi, yi], color="#cfcfcf", lw=1.0, zorder=1)
for k, (m, l) in mk.items():
    ax.scatter(pp[k], yy, marker=m, color=cmap[k], s=26 if k != "rank_2p5km" else 34, zorder=3, label=l, edgecolor="white", linewidth=0.4)
LABEL = {"Digital Realty VIE1 & VIE2": "Digital Realty Vienna (Floridsdorf)"}   # sources do not name facility codes
ax.set_yticks(yy); ax.set_yticklabels([f"{LABEL.get(f, f).replace('Mantsala', 'Mäntsälä')} ({c})" for f, c in zip(pp.facility, pp.country)], fontsize=6.6)
ax.set_xlim(1216, 0); ax.set_ylim(-1.1, len(pp) - 0.4); ax.set_xlabel("Rank among 1,216 records (1 = highest)")
ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=6.6, handletextpad=0.2, columnspacing=1.0)
save(fig, "Fig6_operational_projects")
print("done")
