"""D7: three compact multipanel figures from the saved stage 03 result tables (no recomputation).
Palette: documented validated default (slots 1-5 in fixed order; light surface). Direct labels + legends; no dual axes."""
import json
from d_lib import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4df"
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.axisbelow": True, "pdf.fonttype": 42, "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300})
RAD = {"1km": "1", "2p5km": "2.5", "5km": "5", "10km": "10"}

def panel(ax, tag, title):
    ax.set_title(f"({tag}) {title}", loc="left", fontsize=8.5, fontweight="bold", color=INK)

def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

def rank_scatter(ax, x, y, k, xlabel, ylabel, hl, hl_label):
    ax.scatter(x[~hl], y[~hl], s=3, color=C1, alpha=0.35, lw=0, rasterized=True)
    ax.scatter(x[hl], y[hl], s=7, color=C2, alpha=0.9, lw=0, label=hl_label, rasterized=True)
    ax.axvline(k, color=INK2, lw=0.6, ls=":"); ax.axhline(k, color=INK2, lw=0.6, ls=":")
    ax.set_xlim(0, 1216); ax.set_ylim(0, 1216); ax.set_aspect("equal"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(loc="lower right", fontsize=7, markerscale=1.4, handletextpad=0.2)

def fig1():
    fr = pd.read_csv(RESULTS / "d1_facility_ranks.csv"); sm = pd.read_csv(RESULTS / "d1_composite_vs_simple_summary.csv").set_index("comparator")
    un = json.load(open(RESULTS / "d1_union_and_linear_reproduction.json")); by = pd.read_csv(RESULTS / "d6_composite_vs_simple_by_radius.csv")
    k = 122
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.6), constrained_layout=True)
    for a, col, nm, tag in ((ax[0, 0], "rank_heat_hotmaps", "Hotmaps heat-demand indicator rank", "a"), (ax[0, 1], "rank_population_density", "Population-density indicator rank", "b")):
        hl = ((fr.composite_rank <= k) & (fr[col] > k)).to_numpy()
        rank_scatter(a, fr.composite_rank.to_numpy(), fr[col].to_numpy(), k, "ECHO-DC composite rank (1 = highest)", nm, hl, f"in composite top decile only (n={hl.sum()})")
        panel(a, tag, f"Composite vs {'heat' if tag == 'a' else 'population'} (2.5 km)")
    a = ax[1, 0]
    rows = [("Hotmaps heat", sm.loc["heat_hotmaps"]), ("Population density", sm.loc["population_density"]), ("Opportunity", sm.loc["opportunity"]), ("Feasibility", sm.loc["feasibility"])]
    labels = [r[0] for r in rows] + ["Heat or population\n(either top decile)"]
    shared = [int(r[1].both_top_decile) for r in rows] + [un["composite_top_decile_in_union_heat_or_pop_top_decile"]]
    only = [int(r[1].composite_only_top_decile) for r in rows] + [un["composite_top_decile_in_neither"]]
    y = np.arange(len(labels))[::-1]
    a.barh(y, shared, color=C1, height=0.55, label="also top decile in comparator")
    a.barh(y, only, left=shared, color=C2, height=0.55, label="composite only", hatch="////", edgecolor="white", lw=0)
    for yi, s, o in zip(y, shared, only):
        a.text(s / 2, yi, str(s), ha="center", va="center", color="white", fontsize=7.5); a.text(s + o / 2, yi, str(o), ha="center", va="center", color=INK, fontsize=7.5)
    a.set_yticks(y); a.set_yticklabels(labels); a.set_xlim(0, 122); a.set_xlabel("Composite top-decile sites (n = 122)"); a.grid(axis="y", visible=False)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7); panel(a, "c", "Screening decisions at 2.5 km")
    a = ax[1, 1]
    for nm, lab, c, mk in (("heat_hotmaps", "Hotmaps heat", C1, "o"), ("population_density", "Population", C2, "s"), ("opportunity", "Opportunity", C3, "^"), ("feasibility", "Feasibility", C4, "D")):
        t = by[by.comparator == nm].set_index("scale").loc[SCALES]
        a.plot(range(4), t.spearman, color=c, marker=mk, ms=4.5, lw=1.4, label=lab)
    a.set_xticks(range(4)); a.set_xticklabels([RAD[s] for s in SCALES]); a.set_xlabel("Catchment radius (km)"); a.set_ylabel("Spearman ρ with composite")
    a.set_ylim(0.88, 0.98); a.legend(loc="lower right", fontsize=7, ncol=2); panel(a, "d", "Agreement rises with radius")
    save(fig, "figD1_composite_vs_simple")

def fig2():
    s = pd.read_csv(RESULTS / "d3_ablation_summary.csv"); fk = pd.read_csv(RESULTS / "d3_facility_ranks.csv"); hd = json.load(open(RESULTS / "d3_heat_diagnosis.json"))
    pt = pd.read_csv(RESULTS / "d5_weight_perturbations.csv")
    nice = {"road_accessibility": "Road accessibility", "anchors": "Anchors", "urban_compactness": "Urban compactness", "heat_demand": "Heat (Hotmaps + HDD)",
            "[sub] hdd_only": "HDD only", "[sub] hotmaps_only": "Hotmaps only"}
    s = s.assign(replaced=122 - s.top_decile_overlap).sort_values("replaced", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.6), constrained_layout=True)
    a = ax[0, 0]; y = np.arange(len(s))[::-1]
    a.barh(y, s.replaced, height=0.55, color=[C2 if f.startswith("[sub]") else C1 for f in s.family], hatch=None)
    for yi, r, rho in zip(y, s.replaced, s.spearman): a.text(r + 0.6, yi, f"{r}   (ρ = {rho:.3f})", va="center", fontsize=7.5, color=INK2)
    a.set_yticks(y); a.set_yticklabels([nice[f] for f in s.family]); a.set_xlim(0, 44); a.set_xlabel("Top-decile sites replaced (of 122)"); a.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color=C1, label="indicator family"), Patch(color=C2, label="sub-ablation of heat family")], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7)
    panel(a, "a", "Leave-one-family-out (2.5 km)")
    a = ax[0, 1]
    order = [f for f in s.family]
    data = [np.abs(fk[f"rank_wo_{f}"] - fk.rank_full).to_numpy() for f in order]
    bp = a.boxplot(data, vert=False, whis=(0, 99), showfliers=False, widths=0.5, patch_artist=True, medianprops=dict(color=INK, lw=1.2), boxprops=dict(lw=0.6, edgecolor=INK2),
                   whiskerprops=dict(lw=0.6, color=INK2), capprops=dict(lw=0.6, color=INK2))
    for p, f in zip(bp["boxes"], order): p.set_facecolor(C2 if f.startswith("[sub]") else C1); p.set_alpha(0.75)
    a.set_yticks(range(1, len(order) + 1)); a.set_yticklabels([nice[f] for f in order]); a.invert_yaxis(); a.grid(axis="y", visible=False)
    a.set_xlabel("|rank change|  (box = IQR, bar = median, whisker = 99th pct)", fontsize=7.5); panel(a, "b", "Size of rank changes")
    a = ax[1, 0]
    pc = hd["per_scale_heat_diagnostics"]; rs = ["1km", "5km", "10km"]
    rows = [("stage 02 driver label\n(winner-take-all, pooled)", [0.0, 0.0, 0.0], False),
            ("Strict single-radius winner\n(% of 1,216 sites)", [100 * pc[r]["heat_strict_wins"] / 1216 for r in rs], True),
            ("Mean share of summed\nsingle-factor shifts", [100 * pc[r]["heat_mean_share_of_summed_single_factor_shifts"] for r in rs], True),
            ("stage 02 revert-based share\nof mean |rank change|", [100 * hd["phaseC_revert_share_of_total_mean_abs_dRank_heat"][r] for r in rs], True)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, rng) in zip(y, rows):
        a.plot([min(v), max(v)], [yi, yi], color=C1, lw=1.2)
        for val, mk in zip(v, ("o", "s", "^")): a.plot(val, yi, marker=mk, color=C1, ms=5, mec="white", mew=0.6)
        a.text(max(v) + 0.5, yi, (f"{min(v):.1f}–{max(v):.1f}%" if rng else "0% (all radii)"), va="center", fontsize=7.5, color=INK2)
    a.set_yticks(y); a.set_yticklabels([r[0] for r in rows], fontsize=7); a.set_xlim(-0.5, 13); a.set_xlabel("Heat demand's apparent share of scale sensitivity (%)"); a.grid(axis="y", visible=False)
    a.text(0.98, 0.97, "● 1 km   ■ 5 km   ▲ 10 km", transform=a.transAxes, ha="right", va="top", fontsize=7, color=INK2); a.set_ylim(-0.6, len(rows) - 0.4)
    panel(a, "c", "How the “0%” arises (Hotmaps group)")
    a = ax[1, 1]
    gl = {"anchors": ("Anchors", C2, "s"), "road_accessibility": ("Road access.", C3, "^"), "urban_compactness": ("Compactness", C4, "D"), "heat_demand": ("Heat (HM + HDD)", C1, "o"),
          "population_density": ("Population", C5, "v")}
    xs = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
    for g, (lab, c, mk) in gl.items():
        t = pt[pt.group == g].set_index("multiplier").spearman
        vals = {0: t[0.0], 1: t[0.5], 2: 1.0, 3: t[2.0]}
        a.plot(list(vals), list(vals.values()), color=c, marker=mk, ms=4.5, lw=1.3, label=lab)
    a.set_xticks(range(4)); a.set_xticklabels(["× 0\n(removed)", "× 0.5", "× 1\n(baseline)", "× 2"]); a.set_ylim(0.955, 1.002)
    a.set_ylabel("Spearman ρ with baseline"); a.set_xlabel("Weight multiplier on indicator group"); a.legend(loc="lower right", fontsize=7, ncol=1); panel(a, "d", "Removal vs modest re-weighting")
    save(fig, "figD2_indicator_family_ablation")

def fig3():
    r = pd.read_csv(RESULTS / "d2_facility_ranks.csv"); rel = pd.read_csv(RESULTS / "d2_component_relationships.csv").set_index("config"); ef = pd.read_csv(RESULTS / "d6_alternative_effects_by_radius.csv")
    k = 122
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.6), constrained_layout=True)
    for a, col, tag, ttl in ((ax[0, 0], "rank_B", "a", "Population removed from Opportunity"), (ax[0, 1], "rank_C", "b", "Population removed from Feasibility")):
        hl = (np.abs(r[col] - r.rank_A) >= 50).to_numpy()
        rank_scatter(a, r.rank_A.to_numpy(), r[col].to_numpy(), k, "D2-A (original) rank", f"{'D2-B' if tag == 'a' else 'D2-C'} rank", hl, f"|rank change| ≥ 50 (n={hl.sum()})")
        a.plot([0, 1216], [0, 1216], color=INK2, lw=0.5); panel(a, tag, ttl)
    a = ax[1, 0]
    mets = [("spearman_opp_feas", "Opportunity vs\nFeasibility"), ("spearman_opp_pop", "Opportunity vs\npopulation indicator"),
            ("spearman_feas_pop", "Feasibility vs\npopulation indicator"), ("spearman_composite_pop", "Composite vs\npopulation indicator")]
    y = np.arange(len(mets))[::-1]
    for cfg, c, mk, lab in (("D2-A", C1, "o", "A: original"), ("D2-B", C2, "s", "B: no pop in Opp."), ("D2-C", C3, "^", "C: no pop in Feas.")):
        a.plot([rel.loc[cfg, m] for m, _ in mets], y, ls="none", marker=mk, color=c, ms=6, mec="white", mew=0.6, label=lab)
    a.set_yticks(y); a.set_yticklabels([m[1] for m in mets], fontsize=7); a.set_ylim(-0.6, len(mets) - 0.4); a.set_xlim(0.6, 1.0); a.set_xlabel("Spearman ρ (2.5 km)"); a.grid(axis="y", visible=False)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6.5, handletextpad=0.2, columnspacing=0.8); panel(a, "c", "Component relationships")
    a = ax[1, 1]
    for cfg, c, mk, lab in (("D2-B (no pop in Opp)", C2, "s", "D2-B"), ("D2-C (no pop in Feas)", C3, "^", "D2-C")):
        t = ef[ef.config == cfg].set_index("scale").loc[SCALES]
        a.plot(range(4), t.top_decile_overlap, color=c, marker=mk, ms=5, lw=1.4, label=lab)
        for i, v in enumerate(t.top_decile_overlap): a.text(i, v + (1.2 if cfg.startswith("D2-B") else -2.6), str(int(v)), ha="center", fontsize=7, color=INK2)
    a.set_xticks(range(4)); a.set_xticklabels([RAD[s] for s in SCALES]); a.set_xlabel("Catchment radius (km)"); a.set_ylabel("Top-decile overlap with D2-A (of 122)")
    a.set_ylim(106, 123); a.set_xlim(-0.35, 3.35); a.set_yticks(range(108, 123, 2)); a.legend(loc="lower right", fontsize=7); panel(a, "d", "Effect across radii")
    save(fig, "figD3_population_double_counting")

if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print(sorted(p.name for p in FIGURES.iterdir()))
