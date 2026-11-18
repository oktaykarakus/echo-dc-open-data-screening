"""C7: preliminary figures (equal-weight, Protocol A). Reads results/*.csv written by c4-c6; writes PNG (300 dpi) + PDF."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cfg import *
import multiscale as M, stability as ST

# tokens (reference palette, light surface)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"1km": "#2a78d6", "5km": "#eb6834", "10km": "#1baf7a"}          # categorical slots 1-3, fixed order
CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
LAB = [SCALE_LABEL[s] for s in SCALES]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "font.size": 9, "axes.titlesize": 10,
                     "axes.titleweight": "regular", "axes.titlelocation": "left", "font.family": "DejaVu Sans"})

def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)

def heat(ax, mat, vmin, vmax, fmt, title):
    v = mat.to_numpy(float)
    ax.imshow(v, cmap=CMAP, vmin=vmin, vmax=vmax)
    for i in range(4):
        for j in range(4):
            dark = (v[i, j] - vmin) / (vmax - vmin) > 0.5
            ax.text(j, i, fmt(v[i, j]), ha="center", va="center", color="#ffffff" if dark else INK, fontsize=9)
    ax.set_xticks(range(4), LAB); ax.set_yticks(range(4), LAB); ax.set_title(title)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)

def main():
    FIGURES.mkdir(exist_ok=True)
    long = pd.read_csv(RESULTS / "multiscale_scores_long.csv"); df = M.load_table(); n = len(df); k = M.n_top(n)
    R = ST.ranks_of(long, "equal"); g = ST.global_metrics(R, k)
    # --- Fig 1: rank-correlation matrices
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.5))
    heat(ax[0], g["spearman"], 0.75, 1.0, lambda x: f"{x:.3f}", "(a) Spearman ρ of European rank")
    heat(ax[1], g["kendall"], 0.55, 1.0, lambda x: f"{x:.3f}", "(b) Kendall τ-b of European rank")
    fig.suptitle("Fig. 1  Four-scale rank agreement, equal weights (n = 1,216)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.text(0.01, -0.03, "Same single-hue scale in both panels (darker = stronger agreement); panel scales differ in range. Protocol A: normalisation re-fitted within each radius.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, "fig1_rank_correlation_matrix")
    # --- Fig 2: top-decile overlap
    sens = pd.read_csv(RESULTS / "c5_facility_sensitivity_equal.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.5), gridspec_kw={"width_ratios": [1.1, 1]})
    share = g["top_overlap"] / k
    v = share.to_numpy(float); ax[0].imshow(v, cmap=CMAP, vmin=0.5, vmax=1.0)
    for i in range(4):
        for j in range(4):
            ax[0].text(j, i, f"{int(g['top_overlap'].iloc[i, j])}\n{100*v[i,j]:.0f}%", ha="center", va="center", fontsize=8.5, color="#ffffff" if (v[i, j]-0.5)/0.5 > 0.5 else INK)
    ax[0].set_xticks(range(4), LAB); ax[0].set_yticks(range(4), LAB); ax[0].tick_params(length=0)
    for sp in ax[0].spines.values(): sp.set_visible(False)
    ax[0].set_title(f"(a) Shared top-decile sites (of {k})")
    c = sens[sens.n_top_decile > 0].n_top_decile.value_counts().reindex([1, 2, 3, 4]).fillna(0).astype(int)
    b = ax[1].bar(range(4), c.values, color="#3987e5", width=0.62)
    for x, y in enumerate(c.values): ax[1].text(x, y + 2, str(y), ha="center", color=INK, fontsize=9)
    ax[1].set_xticks(range(4), ["1", "2", "3", "4"]); ax[1].set_xlabel("Number of radii at which the site is in the top decile")
    ax[1].set_ylim(0, 78); ax[1].set_ylabel("Facilities"); ax[1].set_title(f"(b) {int(c.sum())} sites reach the top decile at ≥1 radius")
    ax[1].yaxis.grid(True, color=GRID, lw=0.6); ax[1].set_axisbelow(True)
    for sp in ["top", "right", "left"]: ax[1].spines[sp].set_visible(False)
    ax[1].tick_params(length=0)
    fig.suptitle("Fig. 2  Priority-site (top-decile) stability across catchment radii, equal weights", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, "fig2_top_decile_overlap")
    # --- Fig 3: |Δrank| distributions vs 2.5 km
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for s in ["1km", "5km", "10km"]:
        d = np.sort(np.abs(R[s] - R[REF])); y = np.arange(1, n + 1) / n
        ax.plot(d, y, color=SERIES[s], lw=2, label=SCALE_LABEL[s])
        med = np.median(d); ax.plot([med], [0.5], "o", color=SERIES[s], ms=6, mec=SURF, mew=2)
    noise = json.load(open(RESULTS / "c5_summary_equal.json"))["reconstruction_noise_floor"]["p99"]
    ax.axvline(noise, color=MUTED, lw=1, ls=(0, (3, 3))); ax.text(noise + 8, 0.03, f"stage 01 reconstruction\nnoise, 99th pct ({noise:.0f})", fontsize=7.5, color=INK2, va="bottom")
    ax.axvline(100, color=MUTED, lw=0.8, ls=":"); ax.text(104, 0.34, "100", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 700); ax.set_ylim(0, 1.0); ax.set_xlabel("Absolute change in European rank vs 2.5 km (positions, of 1,216)")
    ax.set_ylabel("Cumulative share of facilities"); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.legend(title="Radius vs 2.5 km", frameon=False, loc="lower right", fontsize=8, title_fontsize=8)
    ax.set_title("Fig. 3  Distribution of absolute ranking changes relative to 2.5 km (dots = median)", fontsize=10)
    fig.tight_layout(); save(fig, "fig3_abs_rank_change_distribution")
    # --- Fig 4: where is sensitivity? (a) by 2.5 km rank decile, (b) by country
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    bd = pd.read_csv(RESULTS / "c5_range_by_rank_decile_2p5km.csv")
    x = np.arange(len(bd)); ax[0].bar(x, bd.median_range, color="#3987e5", width=0.62)
    for xi, y in zip(x, bd.median_range): ax[0].text(xi, y + 5, f"{y:.0f}", ha="center", fontsize=8, color=INK)
    ax[0].set_xticks(x, bd.band); ax[0].set_xlabel("Decile of European rank at 2.5 km (D1 = highest-ranked)")
    ax[0].set_ylabel("Median rank range across 4 radii (positions)"); ax[0].set_title("(a) Sensitivity peaks mid-table")
    ax[0].yaxis.grid(True, color=GRID, lw=0.6); ax[0].set_axisbelow(True)
    for sp in ["top", "right", "left"]: ax[0].spines[sp].set_visible(False)
    ax[0].tick_params(length=0)
    ct = pd.read_csv(RESULTS / "c5_country_patterns_equal.csv"); ct = ct[ct.n >= 15].sort_values("10km_pct_abs_ge100")
    y = np.arange(len(ct))
    for s in ["1km", "5km", "10km"]:
        ax[1].plot(ct[f"{s}_pct_abs_ge100"], y, "o", color=SERIES[s], ms=7, mec=SURF, mew=1.5, label=SCALE_LABEL[s])
    ax[1].set_yticks(y, [f"{c} (n={n_})" for c, n_ in zip(ct.country, ct.n)])
    ax[1].set_xlabel("% of facilities moving ≥100 ranks vs 2.5 km"); ax[1].set_xlim(0, 80)
    ax[1].xaxis.grid(True, color=GRID, lw=0.6); ax[1].set_axisbelow(True); ax[1].tick_params(length=0)
    for sp in ["top", "right", "left"]: ax[1].spines[sp].set_visible(False)
    ax[1].legend(frameon=False, loc="lower right", fontsize=8, title="Radius", title_fontsize=8)
    ax[1].set_title("(b) Countries with n ≥ 15 (smaller samples in CSV)")
    fig.suptitle("Fig. 4  Where scale sensitivity concentrates, equal weights", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, "fig4_scale_sensitivity_patterns")
    print("figures written:", sorted(p.name for p in FIGURES.glob("*.png")))

if __name__ == "__main__":
    main()
