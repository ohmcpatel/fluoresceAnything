"""Figures for the OpenCell taggability grid.

Per-cell diagnostics (PR curves with fold spread, calibration, confusion matrix) and the
isolated-axis master comparisons that are the actual payoff of the factorial.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve

REPO = "/workspace/fluoresceAnything"
RESULTS = f"{REPO}/results"
FIGURES = f"{REPO}/figures"

# reference palette (light mode): categorical slots 1-3, which validate all-pairs
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)

CELL_ORDER = [
    "OpenCell/protein/300m",
    "OpenCell/protein+tag/300m",
    "OpenCell/protein/6b",
    "OpenCell/protein+tag/6b",
]
CELL_LABEL = {
    "OpenCell/protein/300m": "protein\n300M",
    "OpenCell/protein+tag/300m": "protein+tag\n300M",
    "OpenCell/protein/6b": "protein\n6B",
    "OpenCell/protein+tag/6b": "protein+tag\n6B",
}


def tidy(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    path = f"{FIGURES}/{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def best_config(master, task, cell):
    sel = master[(master["task"] == task) & (master["cell"] == cell)]
    return sel.loc[sel["pr_auc_mean"].idxmax()]


def config_id(row):
    return (
        f"{row['task']}|{row['cell']}|{row['feature_set']}|{row['classifier']}|"
        f"{'w' if row['weighted'] else 'u'}"
    )


def threshold_at_precision(y, p, target=0.90):
    """See run_grid.threshold_at_precision -- precision[i] pairs with thresholds[i]."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[: len(thresholds)], recall[: len(thresholds)]
    ok = np.where(precision >= target)[0]
    if len(ok) == 0:
        return float(np.max(p)), False
    return float(thresholds[ok[np.argmax(recall[ok])]]), True


# ------------------------------------------------------------------ per-cell figures
def fig_pr_curves(oof, row, name):
    sub = oof[oof["config_id"] == config_id(row)]
    grid = np.linspace(0, 1, 201)
    curves = []
    for f in sorted(sub["fold"].unique()):
        d = sub[sub["fold"] == f]
        precision, recall, _ = precision_recall_curve(d["y"], d["p"])
        order = np.argsort(recall)
        curves.append(np.interp(grid, recall[order], precision[order]))
    curves = np.array(curves)
    mean, std = curves.mean(0), curves.std(0, ddof=1)
    base = sub["y"].mean()

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.fill_between(grid, mean - std, mean + std, color=BLUE, alpha=0.16, linewidth=0)
    ax.plot(grid, mean, color=BLUE, linewidth=2, label="mean of 5 folds")
    ax.axhline(base, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(0.02, base + 0.012, f"no skill = base rate {base:.3f}", color=MUTED, fontsize=8.5)
    ax.axhline(0.90, color=ORANGE, linewidth=1.2, linestyle=(0, (1, 2)))
    ax.text(0.98, 0.912, "precision target 0.90", color=ORANGE, fontsize=8.5, ha="right")
    ax.set_xlim(0, 1)
    ax.set_ylim(min(base - 0.08, 0.4), 1.02)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(
        f"{row['cell']}\n{row['feature_set']} / {row['classifier']}"
        f"{' / balanced' if row['weighted'] else ''}   "
        f"PR-AUC {row['pr_auc_mean']:.3f} ± {row['pr_auc_std']:.3f}",
        loc="left",
        color=INK,
    )
    ax.legend(loc="lower left")
    tidy(ax)
    return save(fig, name)


def fig_calibration(oof, row, name, bins=10):
    sub = oof[oof["config_id"] == config_id(row)]
    p, y = sub["p"].values, sub["y"].values
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    xs, ys, ns = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        xs.append(p[m].mean())
        ys.append(y[m].mean())
        ns.append(m.sum())

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="perfect")
    ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.6, label="observed")
    ax.axhline(y.mean(), color=MUTED, linewidth=0.9, alpha=0.6)
    ax.set_xlabel("predicted probability (equal-count bins)")
    ax.set_ylabel("observed frequency")
    lo = min(min(xs), min(ys)) - 0.06
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(lo, 1.0)
    ax.set_title(
        f"Calibration — {row['cell']}\n{row['feature_set']} / {row['classifier']}"
        f"   Brier {row['brier_mean']:.3f}",
        loc="left",
        color=INK,
    )
    ax.legend(loc="upper left")
    tidy(ax)
    return save(fig, name)


def fig_confusion(oof, row, name):
    sub = oof[oof["config_id"] == config_id(row)]
    y, p = sub["y"].values, sub["p"].values
    thr, reached = threshold_at_precision(y, p, 0.90)
    pred = (p >= thr).astype(int)
    cm = np.array(
        [
            [((pred == 0) & (y == 0)).sum(), ((pred == 1) & (y == 0)).sum()],
            [((pred == 0) & (y == 1)).sum(), ((pred == 1) & (y == 1)).sum()],
        ]
    )
    precision = cm[1, 1] / max(cm[:, 1].sum(), 1)
    recall = cm[1, 1] / max(cm[1].sum(), 1)

    cmap = LinearSegmentedColormap.from_list("blues", BLUE_RAMP[:5])
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.imshow(cm / cm.sum(), cmap=cmap, vmin=0, vmax=0.7)
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / cm.sum()
            ax.text(j, i, f"{cm[i, j]}\n{frac:.1%}", ha="center", va="center",
                    color="#ffffff" if frac > 0.35 else INK, fontsize=11)
    ax.set_xticks([0, 1], ["predicted\nunsuccessful", "predicted\nsuccessful"])
    ax.set_yticks([0, 1], ["actually\nunsuccessful", "actually\nsuccessful"])
    ax.set_xticks(np.arange(-0.5, 2), minor=True)
    ax.set_yticks(np.arange(-0.5, 2), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    note = "" if reached else "  (precision 0.90 unreachable — top-scoring cutoff used)"
    ax.set_title(
        f"{row['cell']} at threshold {thr:.3f}\n"
        f"precision {precision:.3f}, recall {recall:.3f}{note}",
        loc="left",
        color=INK,
        fontsize=10.5,
    )
    return save(fig, name)


# ------------------------------------------------------------------ master comparisons
def cell_best(master, task, metric="pr_auc_mean"):
    sel = master[(master["task"] == task) & (master["cell"] != "baseline")]
    out = {}
    for cell in CELL_ORDER:
        c = sel[sel["cell"] == cell]
        if len(c):
            out[cell] = c.loc[c[metric].idxmax()]
    return out


def baseline_value(master, task, feature_set, metric="pr_auc_mean"):
    sel = master[(master["task"] == task) & (master["feature_set"] == feature_set)]
    return sel[metric].max() if len(sel) else np.nan


def fig4_all_cells(master, metric, ylabel, name):
    best = cell_best(master, "filter", metric)
    cells = [c for c in CELL_ORDER if c in best]
    vals = [best[c][metric] for c in cells]
    errs = [best[c][metric.replace("_mean", "_std")] for c in cells]
    colors = [ORANGE if "protein+tag" in c else BLUE for c in cells]

    floor = baseline_value(master, "filter", "FLOOR", metric)
    term = baseline_value(master, "filter", "TERMINUS_ONLY", metric)
    base_rate = master[master["task"] == "filter"]["base_rate"].iloc[0]
    if metric.startswith("pr_auc_minority"):
        base_rate = 1 - base_rate

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(len(cells))
    bars = ax.bar(x, vals, width=0.6, color=colors, zorder=3)
    ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1.4, capsize=5, zorder=4)
    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, v + e + 0.008, f"{v:.3f}", ha="center", color=INK, fontsize=9.5, zorder=5)

    # reference lines are labelled in a reserved right-hand margin, never over the bars
    right = len(cells) - 0.5
    for value, label, style in [
        (floor, f"FLOOR  {floor:.3f}", (0, (5, 3))),
        (term, f"terminus only  {term:.3f}", (0, (1, 2))),
        (base_rate, f"no skill  {base_rate:.3f}", (0, (2, 2))),
    ]:
        ax.hlines(value, -0.6, right, color=MUTED, linewidth=1.2, linestyle=style, zorder=2)
        ax.text(right + 0.1, value, label, ha="left", va="center", color=MUTED, fontsize=8.5)
    ax.set_xlim(-0.6, right + 1.35)

    ax.set_xticks(x, [CELL_LABEL[c] for c in cells])
    ax.set_ylabel(ylabel)
    lo = min(min(vals) - 0.06, base_rate - 0.02, term - 0.02)
    ax.set_ylim(lo, max(np.array(vals) + np.array(errs)).max() + 0.035)
    ax.set_title(
        "Taggability filter — best configuration per cell, 5-fold grouped CV",
        loc="left",
        color=INK,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE),
    ]
    ax.legend(handles, ["protein", "protein + tag"], loc="upper left", ncol=2)
    tidy(ax)
    return save(fig, name)


def _paired(master, metric, group_of, x_of, x_order, title, xlabel, name, colors,
            legend_loc="upper left"):
    """Dumbbell: one line per group across the isolated axis, with fold-spread bars."""
    sel = master[(master["task"] == "filter") & (master["cell"] != "baseline")].copy()
    sel["grp"] = sel["cell"].map(group_of)
    sel["xax"] = sel["cell"].map(x_of)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    groups = [g for g in colors if g in set(sel["grp"])]
    xpos = {v: i for i, v in enumerate(x_order)}
    series = {}
    for gi, g in enumerate(groups):
        pts, errs, xs = [], [], []
        for v in x_order:
            c = sel[(sel["grp"] == g) & (sel["xax"] == v)]
            if not len(c):
                continue
            best = c.loc[c[metric].idxmax()]
            pts.append(best[metric])
            errs.append(best[metric.replace("_mean", "_std")])
            xs.append(xpos[v] + (gi - (len(groups) - 1) / 2) * 0.06)
        series[g] = (xs, pts, errs)
        ax.plot(xs, pts, color=colors[g], linewidth=2, marker="o", markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=1.8, label=g, zorder=3)
        ax.errorbar(xs, pts, yerr=errs, fmt="none", ecolor=colors[g], elinewidth=1.4,
                    capsize=6, alpha=0.85, zorder=2)
    # the series sit ~0.002 apart, so separate labels in point space: upper series above,
    # lower series below, regardless of plotting order
    ranked = sorted(series, key=lambda g: -np.mean(series[g][1]))
    for rank, g in enumerate(ranked):
        dy = 9 if rank == 0 else -15
        for xi, v in zip(series[g][0], series[g][1]):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(9, dy),
                        color=INK2, fontsize=9)

    floor = baseline_value(master, "filter", "FLOOR", metric)
    ax.hlines(floor, -0.3, len(x_order) - 0.9, color=MUTED, linewidth=1.2, linestyle=(0, (5, 3)))
    ax.text(len(x_order) - 0.85, floor, f"FLOOR  {floor:.3f}", ha="left", va="center",
            color=MUTED, fontsize=8.5)

    ax.set_xticks(range(len(x_order)), x_order)
    ax.set_xlim(-0.35, len(x_order) - 0.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PR-AUC (mean ± std across folds)")
    ax.set_title(title, loc="left", color=INK)
    # surface-coloured mask so the legend never reads through the FLOOR rule
    ax.legend(loc=legend_loc, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=1)
    tidy(ax)
    return save(fig, name)


def fig5_scale(master, metric):
    return _paired(
        master,
        metric,
        group_of=lambda c: "protein + tag" if "protein+tag" in c else "protein",
        x_of=lambda c: "6B" if c.endswith("6b") else "300M",
        x_order=["300M", "6B"],
        title="SCALE axis isolated — does 6B beat 300M beyond the fold spread?",
        xlabel="ESM-C model size",
        name="fig5_scale_axis",
        colors={"protein": BLUE, "protein + tag": ORANGE},
    )


def fig6_tag(master, metric):
    return _paired(
        master,
        metric,
        group_of=lambda c: "ESM-C 6B" if c.endswith("6b") else "ESM-C 300M",
        x_of=lambda c: "protein + tag" if "protein+tag" in c else "protein",
        x_order=["protein", "protein + tag"],
        title="TAG axis isolated — does the fusion construct add anything?",
        xlabel="input sequence",
        name="fig6_tag_axis",
        colors={"ESM-C 300M": BLUE, "ESM-C 6B": AQUA},
        legend_loc="lower left",
    )


def fig8_terminus(master):
    sel = master[master["task"] == "terminus"].copy()
    if sel.empty:
        print("  skipping fig8: no terminus rows")
        return None
    order = ["FLOOR", "TOPOLOGY_SP", "TOPOLOGY_FULL", "OpenCell/protein/300m", "OpenCell/protein/6b"]
    labels = {
        "FLOOR": "FLOOR\nlen + aa",
        "TOPOLOGY_SP": "signal\npeptide",
        "TOPOLOGY_FULL": "topology\nSP+TM+KD",
        "OpenCell/protein/300m": "ESM-C\n300M",
        "OpenCell/protein/6b": "ESM-C\n6B",
    }
    rows = []
    for key in order:
        c = sel[sel["feature_set"] == key] if key in ("FLOOR", "TOPOLOGY_SP", "TOPOLOGY_FULL") else sel[sel["cell"] == key]
        if len(c):
            rows.append((key, c.loc[c["pr_auc_mean"].idxmax()]))
    if not rows:
        return None

    base = sel["base_rate"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, metric, ylab, ref, reflab in [
        (axes[0], "pr_auc", "PR-AUC (positive = N-terminus)", base, f"no skill {base:.3f}"),
        (axes[1], "balanced_accuracy", "balanced accuracy at p = 0.5", 0.5, "chance 0.500"),
    ]:
        vals = [r[1][f"{metric}_mean"] for r in rows]
        errs = [r[1][f"{metric}_std"] for r in rows]
        colors = [MUTED if r[0] in ("FLOOR", "TOPOLOGY_SP", "TOPOLOGY_FULL") else BLUE for r in rows]
        x = np.arange(len(rows))
        ax.bar(x, vals, width=0.62, color=colors, zorder=3)
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1.3, capsize=5, zorder=4)
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + e + 0.006, f"{v:.3f}", ha="center", color=INK, fontsize=9, zorder=5)
        right = len(rows) - 0.5
        ax.hlines(ref, -0.6, right, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.text(right + 0.1, ref, reflab, ha="left", va="center", color=MUTED, fontsize=8.5)
        ax.set_xlim(-0.6, right + 1.2)
        ax.set_xticks(x, [labels[r[0]] for r in rows], fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.set_ylim(min(min(vals) - 0.08, ref - 0.05), max(max(vals) + 0.09, ref + 0.05))
        tidy(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MUTED), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    axes[0].legend(handles, ["baseline", "ESM-C embedding"], loc="upper left", ncol=2)
    fig.suptitle(
        "Terminus selector (Model 2) — is it just a signal-peptide detector?",
        x=0.09, ha="left", color=INK, fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save(fig, "fig8_terminus_selector")


def main():
    master = pd.read_csv(f"{RESULTS}/master_results.csv")
    oof = pd.read_csv(f"{RESULTS}/oof_predictions.csv")
    oof["config_id"] = (
        oof["task"] + "|" + oof["cell"] + "|" + oof["feature_set"] + "|" + oof["classifier"]
        + "|" + np.where(oof["weighted"], "w", "u")
    )
    os.makedirs(FIGURES, exist_ok=True)

    print("per-cell diagnostics")
    for cell in CELL_ORDER:
        if cell not in set(master["cell"]):
            print(f"  skipping {cell}: not in results")
            continue
        row = best_config(master, "filter", cell)
        slug = cell.replace("OpenCell/", "").replace("/", "_").replace("+", "plus")
        fig_pr_curves(oof, row, f"pr_{slug}")
        fig_calibration(oof, row, f"calibration_{slug}")
        fig_confusion(oof, row, f"confusion_{slug}")

    print("master comparisons")
    fig4_all_cells(master, "pr_auc_mean", "PR-AUC (positive = successful)", "fig4_all_cells")
    fig4_all_cells(
        master,
        "pr_auc_minority_mean",
        "PR-AUC (positive = unsuccessful)",
        "fig4b_all_cells_minority",
    )
    fig5_scale(master, "pr_auc_mean")
    fig6_tag(master, "pr_auc_mean")
    fig8_terminus(master)
    print("\nfig7 (PaperClip generalization) NOT produced: no PaperClip data in the repo")


if __name__ == "__main__":
    main()
