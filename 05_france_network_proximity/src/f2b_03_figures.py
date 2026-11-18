"""F2-B figures (2). Colours: documented default categorical slots in fixed order (palette validator needs node, which is absent here;
identity is also carried by direct labels, row position and marker shape)."""
import os, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
PH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); R = os.path.join(PH, "results"); FG = os.path.join(PH, "figures")
INK, SUB, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SL = {"A": "#2a78d6", "P": "#eb6834", "H": "#1baf7a", "B*": "#eda100", "C": "#e87ba4"}
MK = {"A": "o", "P": "s", "H": "^", "B*": "D", "C": "v"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
def clean(ax):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)

a = pd.read_csv(os.path.join(R, "f2b_five_arm_auc.csv")).set_index("arm")
fig, ax = plt.subplots(figsize=(6.4, 3.6)); clean(ax)
arms = ["A", "P", "H", "B*", "C"]; names = {"A": "A  ECHO-DC (original)", "P": "P  population density", "H": "H  Hotmaps heat", "B*": "B*  construct-balanced", "C": "C  Pareto profile"}
for i, k in enumerate(arms[::-1]):
    lo, hi = a.loc[k, "AUC_ci90_lo_50km"], a.loc[k, "AUC_ci90_hi_50km"]
    ax.plot([lo, hi], [i, i], color=SL[k], lw=2, solid_capstyle="round"); ax.plot(a.loc[k, "AUC"], i, MK[k], color=SL[k], ms=7, mec=SURF, mew=1.5)
    ax.text(0.995, i, f"{a.loc[k,'AUC']:.3f}", transform=ax.get_yaxis_transform(), ha="right", va="center", color=INK, fontsize=8.5)
ax.set_yticks(range(5)); ax.set_yticklabels([names[k] for k in arms[::-1]]); ax.set_xlim(0.85, 1.0); ax.set_ylim(-0.6, 4.6)
ax.set_xlabel("AUC for an existing heat-network trace within 2.5 km\n(France, n = 322; 90% interval, 50 km spatial blocks)")
fig.suptitle("ECHO-DC and population density rank mapped-network\nproximity almost identically", x=0.02, ha="left", color=INK, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.9)); [fig.savefig(os.path.join(FG, f"f2b_fig1_auc_by_arm.{e}"), dpi=200) for e in ("png", "pdf")]

s = pd.read_csv(os.path.join(R, "f2b_sensitivity.csv")).set_index("variant")
rows = [("Primary, 2.5 km, 50 km blocks", a.loc["A", "dAUC_vs_P"], a.loc["A", "dAUC_ci90_lo_50km"], a.loc["A", "dAUC_ci90_hi_50km"], a.loc["B*", "dAUC_vs_P"], a.loc["B*", "dAUC_ci90_lo_50km"], a.loc["B*", "dAUC_ci90_hi_50km"]),
        ("Primary, 25 km blocks", a.loc["A", "dAUC_vs_P"], a.loc["A", "dAUC_ci90_lo_25km"], a.loc["A", "dAUC_ci90_hi_25km"], a.loc["B*", "dAUC_vs_P"], a.loc["B*", "dAUC_ci90_lo_25km"], a.loc["B*", "dAUC_ci90_hi_25km"]),
        ("Primary, 100 km blocks", a.loc["A", "dAUC_vs_P"], a.loc["A", "dAUC_ci90_lo_100km"], a.loc["A", "dAUC_ci90_hi_100km"], a.loc["B*", "dAUC_vs_P"], a.loc["B*", "dAUC_ci90_lo_100km"], a.loc["B*", "dAUC_ci90_hi_100km"])]
lbl = {"S3_threshold_1000m": "S3  1,000 m threshold", "S1_campus_units": "S1  campus units (n = 268)", "S2_exclude_13_salle_technique": "S2  excl. 13 'salle technique'",
       "S2b_exclude_4_broadcaster (descriptive)": "S2b excl. 4 broadcaster rooms", "S4a_indeterminate_scored_negative": "S4a indeterminate = negative",
       "S4b_exclude_largest_50km_block": "S4b excl. largest 50 km block", "S5_tier2_traces_as_existing": "S5  untagged traces = existing"}
for k, t in lbl.items():
    r = s.loc[k]; rows.append((t, r.dAUC_A, r.dAUC_A_lo, r.dAUC_A_hi, r["dAUC_B*"], r["dAUC_B*_lo"], r["dAUC_B*_hi"]))
fig, ax = plt.subplots(figsize=(6.6, 4.6)); clean(ax)
for i, (t, pa, la, ha, pb, lb, hb) in enumerate(rows[::-1]):
    for off, k, (p, l, h) in ((0.13, "A", (pa, la, ha)), (-0.13, "B*", (pb, lb, hb))):
        ax.plot([l, h], [i + off, i + off], color=SL[k], lw=1.8, solid_capstyle="round"); ax.plot(p, i + off, MK[k], color=SL[k], ms=5.5, mec=SURF, mew=1.2)
ax.axvline(0, color=SUB, lw=1); ax.axvline(0.03, color=SUB, lw=1, ls=(0, (4, 3)))
ax.text(0.0295, len(rows) - 0.3, "+0.03 practical threshold", color=SUB, fontsize=8, va="bottom", ha="right")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]]); ax.set_ylim(-0.7, len(rows) - 0.2); ax.set_xlim(-0.05, 0.045)
ax.set_xlabel("ΔAUC vs population density (90% spatial-block interval)")
h = [plt.Line2D([0], [0], color=SL["A"], marker=MK["A"], lw=1.8, label="A  ECHO-DC"), plt.Line2D([0], [0], color=SL["B*"], marker=MK["B*"], lw=1.8, label="B*  construct-balanced")]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.35, -0.16), ncol=2, frameon=False, fontsize=8, labelcolor=INK)
fig.suptitle("No point estimate reaches +0.03;\nall intervals for A include 0", x=0.02, ha="left", color=INK, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.92)); [fig.savefig(os.path.join(FG, f"f2b_fig2_delta_auc_sensitivity.{e}"), dpi=200) for e in ("png", "pdf")]
