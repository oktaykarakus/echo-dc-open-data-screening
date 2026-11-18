"""Figure E1: positions of the primary operational set (8 projects) across scales (a) and in Opportunity-Feasibility rank space (b).
No Figure E2 is produced: the campus-level view changes almost nothing (see report section 7), so a table is sufficient."""
from e_lib import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 of the reference palette (top-half vs bottom-half at 2.5 km)
SHORT = {"E01": "Equinix PA10 (FR)", "E02": "Telia HDC (FI)", "E03": "Meta Odense (DK)", "E04": "NorthC Aalsmeer (NL)", "E05": "Digital Realty VIE (AT)",
         "E06": "Google Hamina (FI)", "E07": "NorthC Eindhoven (NL)", "E08": "Yandex Mäntsälä (FI)"}
MARK = dict(zip(SHORT, ["o", "s", "^", "D", "v", "P", "X", "h"]))


def spread(vals, gap, lo, hi):
    order = np.argsort(vals); pos = np.array(vals, float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    if pos[-1] > hi:
        pos -= pos[-1] - hi
        for i in range(len(pos) - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - gap)
    out = np.empty_like(pos); out[order] = pos
    return out


def main():
    pos = pd.read_csv(RESULTS / "e3_project_positions.csv").set_index("case_id")
    comp = pd.read_csv(RESULTS / "e3_components_and_roads.csv").set_index("case_id")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.7), gridspec_kw={"width_ratios": [1.25, 1]})
    xs = np.arange(4); labs = ["1 km", "2.5 km", "5 km", "10 km"]
    for ymax, txt in [(TOP10, "top 10 %"), (TOP20, "top 20 %"), (TOP50, "top 50 %")]:
        a.axhline(ymax, color=GRID, lw=1, ls=(0, (4, 3)), zorder=0)
        a.text(-0.72, ymax, txt, color=INK2, fontsize=7, va="bottom")
    ends = []
    for cid in SHORT:
        r = [pos.loc[cid, f"rank_{s}"] for s in SCALES]
        col = BLUE if pos.loc[cid, "rank_2p5km"] <= TOP50 else ORANGE
        a.plot(xs, r, color=col, lw=1.6, marker=MARK[cid], ms=6.5, mec=SURF, mew=1.2, zorder=3)
        ends.append(r[-1])
    ypos = spread(ends, 62, 40, 1200)
    for (cid, e), y in zip(zip(SHORT, ends), ypos):
        a.annotate(SHORT[cid], (3, e), xytext=(3.12, y), fontsize=7.5, color=INK, va="center", ha="left",
                   arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8, shrinkA=0, shrinkB=3))
    a.set_xlim(-0.75, 4.35); a.set_ylim(1216, 1)
    a.set_xticks(xs); a.set_xticklabels(labs); a.set_xlabel("Catchment radius")
    a.set_ylabel("Rank among 1,216 records (1 = best), equal weights")
    a.set_title("(a) Rank of each project across spatial scales", loc="left", fontsize=9, color=INK)
    a.grid(axis="y", color=GRID, lw=0.6, alpha=.6); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    # (b) Opportunity vs Feasibility rank at 2.5 km
    for cid in SHORT:
        o, f = comp.loc[cid, "opp_rank_2p5km"], comp.loc[cid, "feas_rank_2p5km"]
        col = BLUE if pos.loc[cid, "rank_2p5km"] <= TOP50 else ORANGE
        b.scatter(o, f, s=46, color=col, marker=MARK[cid], edgecolor=SURF, linewidth=1.0, zorder=3)
    lab_xy = {c: (comp.loc[c, "opp_rank_2p5km"], comp.loc[c, "feas_rank_2p5km"]) for c in SHORT}
    ly = spread([v[1] for v in lab_xy.values()], 62, 40, 1200)
    for (c, (o, f)), y in zip(lab_xy.items(), ly):
        dx = 70 if o < 600 else -70           # x axis is inverted: positive dx moves the label LEFT on screen
        b.annotate(SHORT[c], (o, f), xytext=(o + dx, y), fontsize=7.2, color=INK, va="center", ha="right" if dx > 0 else "left",
                   arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8, shrinkA=0, shrinkB=3))
    b.plot([1, 1216], [1, 1216], color=GRID, lw=1, ls=(0, (4, 3)), zorder=0)
    b.set_xlim(1260, -520); b.set_xticks(range(1200, -1, -200)); b.set_ylim(1260, 1)
    b.set_xlabel("Opportunity-component rank, 2.5 km (1 = best)"); b.set_ylabel("Feasibility-component rank, 2.5 km (1 = best)")
    b.set_title("(b) Opportunity vs Feasibility", loc="left", fontsize=9, color=INK)
    b.grid(color=GRID, lw=0.6, alpha=.6); b.set_axisbelow(True)
    for s in ("top", "right"): b.spines[s].set_visible(False)
    h = [plt.Line2D([], [], color=BLUE, lw=1.6, marker="o", ms=6, mec=SURF, label="top half of records at 2.5 km (rank ≤ 608)"),
         plt.Line2D([], [], color=ORANGE, lw=1.6, marker="o", ms=6, mec=SURF, label="bottom half (rank > 608)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.055))
    fig.text(0.01, 0.005, "n = 8 operational, high-confidence projects; multi-record projects use the median rank of their records (Odense 2, Hamina 4, Mäntsälä 2).\n"
             "External plausibility only - not a predictive test. Dashed diagonal in (b): equal rank in both components.", fontsize=6.6, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "figE1_operational_cases.png", dpi=200, facecolor=SURF); fig.savefig(FIGURES / "figE1_operational_cases.pdf", facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    main(); print("ok")
