"""Figures for the revised paper (static PNG, light surface, print-safe).

Colours follow the validated reference palette in fixed slot order; series are
always identified by a legend or a direct label, never by colour alone.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import configs

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#b9b8b1"

SHORT = {
    "c0_none": "c0 none", "c1_A_smotetomek": "c1 A", "c2_B_weights": "c2 B",
    "c3_C_threshold": "c3 C", "c4_AB_smotetomek_weights": "c4 A+B",
    "c5_AC_smotetomek_threshold": "c5 A+C", "c6_BC_weights_threshold": "c6 B+C",
    "c7_ABC_full": "c7 A+B+C", "c8_A_Brecalc": "c8 A+B'",
}
TITLES = {"oilspill": "Oil Spill", "smsspam": "SMS Spam", "cicids2017": "CIC-IDS2017",
          "aps": "APS", "creditfraud": "Credit Fraud"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.grid": True,
        "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 200,
    })


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main_bars(table, path):
    """Macro-F1 (mean over classifiers) per config, one panel per dataset."""
    style()
    ds_list = list(table["dataset"].unique())
    fig, axes = plt.subplots(1, len(ds_list), figsize=(4.2 * len(ds_list), 3.2), squeeze=False)
    order = list(configs.CONFIGS)
    for ax, ds in zip(axes[0], ds_list):
        t = table[table["dataset"] == ds].set_index("config_id").reindex(order)
        # Dots + SD whiskers rather than bars: the axis does not start at 0,
        # and truncated bars would exaggerate the differences.
        x = np.arange(len(order))
        best = t["mean"].idxmax()
        colors = [SERIES[0] if c != best else SERIES[1] for c in order]
        ax.errorbar(x, t["mean"], yerr=t["std"], fmt="none", ecolor=AXIS, elinewidth=2)
        ax.scatter(x, t["mean"], s=48, color=colors, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_xticks(x, [SHORT[c] for c in order], rotation=45, ha="right")
        ax.set_title(TITLES.get(ds, ds))
        ax.set_ylabel("Macro-F1 (mean ± SD over folds)")
        ax.grid(axis="x", visible=False)
        ax.annotate(f"best {t.loc[best, 'mean']:.3f}", (order.index(best), t.loc[best, "mean"]),
                    textcoords="offset points", xytext=(8, 0), va="center", color=INK2, fontsize=8)
    return _save(fig, path)


def heatmap(by_clf, path):
    """Macro-F1 per classifier x config, one panel per dataset."""
    style()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    ds_list = list(by_clf["dataset"].unique())
    order = list(configs.CONFIGS)
    fig, axes = plt.subplots(1, len(ds_list), figsize=(5.2 * len(ds_list), 3.4), squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        m = by_clf[by_clf["dataset"] == ds].pivot(index="classifier", columns="config_id",
                                                   values="mean").reindex(columns=order)
        im = ax.imshow(m.to_numpy(), cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(order)), [SHORT[c] for c in order], rotation=45, ha="right")
        ax.set_yticks(range(len(m.index)), m.index)
        ax.grid(False)
        vmin, vmax = np.nanmin(m.to_numpy()), np.nanmax(m.to_numpy())
        for i in range(m.shape[0]):
            row_best = np.nanargmax(m.to_numpy()[i])
            for j in range(m.shape[1]):
                v = m.iat[i, j]
                dark = (v - vmin) / (vmax - vmin + 1e-9) > 0.55
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if dark else INK,
                        fontweight="bold" if j == row_best else "normal")
        ax.set_title(f"{TITLES.get(ds, ds)}: macro-F1 (bold = best per classifier)")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    return _save(fig, path)


def effects_forest(effects, path):
    """Factorial effects with 95% CI (averaged over classifiers), per dataset."""
    style()
    e = effects[effects["classifier"] == "ALL"]
    ds_list = list(e["dataset"].unique())
    terms = ["A", "B", "C", "AB", "AC", "BC", "ABC"]
    fig, axes = plt.subplots(1, len(ds_list), figsize=(3.8 * len(ds_list), 3.0), squeeze=False,
                             sharey=True)
    for ax, ds in zip(axes[0], ds_list):
        t = e[e["dataset"] == ds].set_index("term").reindex(terms)
        y = np.arange(len(terms))[::-1]
        sig = t["significant"].to_numpy()
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.hlines(y, t["ci_low"], t["ci_high"], color=[SERIES[0] if s else NEUTRAL for s in sig],
                  linewidth=2)
        ax.scatter(t["effect"], y, s=36, color=[SERIES[0] if s else NEUTRAL for s in sig],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_yticks(y, terms)
        ax.set_title(TITLES.get(ds, ds))
        ax.set_xlabel("Effect on macro-F1 (95% CI)")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_ylabel("Factor / interaction")
    fig.text(0.99, 0.01, "blue = CI excludes 0; grey = not significant", ha="right",
             color=MUTED, fontsize=7)
    return _save(fig, path)


def interaction(df_main, path, metric="macro_f1"):
    """A x B interaction: macro-F1 against B, one line per A, panels C=0 / C=1."""
    style()
    d = df_main[df_main["config_id"] != "c8_A_Brecalc"]
    ds_list = list(d["dataset"].unique())
    fig, axes = plt.subplots(len(ds_list), 2, figsize=(6.4, 2.6 * len(ds_list)), squeeze=False)
    for r, ds in enumerate(ds_list):
        g = d[d["dataset"] == ds].groupby(["A", "B", "C"])[metric].mean()
        for c in (0, 1):
            ax = axes[r][c]
            for a, color, label in ((0, SERIES[0], "no resampling"), (1, SERIES[1], "SMOTETomek")):
                ys = [g[(a, b, c)] for b in (0, 1)]
                ax.plot([0, 1], ys, color=color, marker="o", markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
                ax.annotate(label, (1, ys[1]), textcoords="offset points", xytext=(6, 0),
                            va="center", color=INK2, fontsize=7)
            ax.set_xticks([0, 1], ["no weights", "class weights"])
            ax.set_xlim(-0.2, 1.6)
            ax.set_title(f"{TITLES.get(ds, ds)}, threshold tuning {'on' if c else 'off'}")
            ax.set_ylabel("Macro-F1")
    axes[0][0].legend(loc="lower left", fontsize=7)
    return _save(fig, path)


def cd_diagram(ranks, cd, n_blocks, p, path):
    """Average rank of each config (1 = best) with the Nemenyi critical difference."""
    style()
    fig, ax = plt.subplots(figsize=(7, 2.9))
    r = ranks.sort_values()
    y = np.arange(len(r))[::-1]
    best = r.iloc[0]
    colors = [SERIES[0] if v - best < cd else NEUTRAL for v in r]
    ax.hlines(y, 1, r.values, color=GRID, linewidth=1)
    ax.scatter(r.values, y, s=40, color=colors, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for yi, (cid, v) in zip(y, r.items()):
        ax.annotate(f"{v:.2f}", (v, yi), textcoords="offset points", xytext=(7, 0), va="center",
                    color=INK2, fontsize=7)
    ax.set_yticks(y, [SHORT[c] for c in r.index])
    ax.axvspan(best, best + cd, color=SERIES[0], alpha=0.08)
    ax.set_xlabel(f"Average rank over {n_blocks} dataset x classifier blocks (1 = best)")
    ax.set_title(f"Config ranking on macro-F1  (Friedman p = {p:.2g}, Nemenyi CD = {cd:.2f})")
    ax.set_xlim(1, len(r) + 0.5)
    ax.grid(axis="y", visible=False)
    fig.text(0.99, 0.01, "shaded band = within one critical difference of the best config",
             ha="right", color=MUTED, fontsize=7)
    return _save(fig, path)


def e2_curves(curves, path):
    """Macro-F1, minority precision and predicted-positive ratio against the
    effective correction ratio rho_eff (log2 scale), per dataset."""
    style()
    ds_list = list(curves["dataset"].unique())
    metrics = [("macro_f1", "Macro-F1"), ("precision_1", "Minority precision"),
               ("ppr_ratio", "Predicted / true positive rate")]
    fig, axes = plt.subplots(len(ds_list), 3, figsize=(10.5, 2.7 * len(ds_list)), squeeze=False)
    groups = [(0.0, "no resampling"), (0.5, "SMOTETomek to 1:2"), (1.0, "SMOTETomek to 1:1")]
    for r, ds in enumerate(ds_list):
        c = curves[(curves["dataset"] == ds) & (curves["weight_mode"] == "orig")]
        for j, (m, label) in enumerate(metrics):
            ax = axes[r][j]
            ax.axvline(0, color=AXIS, linewidth=1)
            for k, (sr, name) in enumerate(groups):
                for C, ls in ((0, "-"), (1, "--")):
                    g = c[(c["sampling_ratio"] == sr) & (c["C"] == C)].sort_values("log2_rho")
                    ax.plot(g["log2_rho"], g[m], color=SERIES[k], linestyle=ls, marker="o",
                            markersize=4, label=f"{name}, threshold {'tuned' if C else 'default'}")
            if m == "ppr_ratio":
                ax.axhline(1, color=MUTED, linewidth=1, linestyle=":")
            ax.set_title(f"{TITLES.get(ds, ds)}: {label}")
            ax.set_xlabel("log2 effective correction ratio (0 = balanced)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    return _save(fig, path)


def over_correction(mech, path):
    """Predicted-positive ratio per config (1 = as many positives predicted as exist)."""
    style()
    m = mech.reset_index()
    ds_list = list(m["dataset"].unique())
    order = list(configs.CONFIGS)
    fig, axes = plt.subplots(1, len(ds_list), figsize=(4.2 * len(ds_list), 3.0), squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        t = m[m["dataset"] == ds].set_index("config_id").reindex(order)
        x = np.arange(len(order))
        ax.bar(x, t["ppr_ratio"], width=0.7, color=SERIES[0], edgecolor=SURFACE)
        ax.axhline(1, color=INK2, linewidth=1, linestyle=":")
        ax.set_xticks(x, [SHORT[c] for c in order], rotation=45, ha="right")
        ax.set_ylabel("Predicted / true positive rate")
        ax.set_title(TITLES.get(ds, ds))
        ax.grid(axis="x", visible=False)
    return _save(fig, path)
