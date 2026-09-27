"""Statistics for the revised paper (design section 8).

Blocks = dataset x classifier. Fold-level scores are paired across configs,
because every config uses the same outer folds.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from common import configs, settings

FACTORIAL = [c for c in configs.CONFIGS if c != "c8_A_Brecalc"]
TERMS = ["A", "B", "C", "AB", "AC", "BC", "ABC"]

# Comparisons the reviewers asked about, as (baseline, other, question).
KEY_PAIRS = [
    ("c0_none", "c1_A_smotetomek", "Resampling alone vs no correction"),
    ("c0_none", "c2_B_weights", "Class weights alone vs no correction"),
    ("c0_none", "c3_C_threshold", "Threshold tuning alone vs no correction"),
    ("c2_B_weights", "c4_AB_smotetomek_weights", "Adding resampling on top of weights"),
    ("c1_A_smotetomek", "c4_AB_smotetomek_weights", "Adding original-ratio weights on top of resampling"),
    ("c4_AB_smotetomek_weights", "c8_A_Brecalc", "Original vs recomputed weights after resampling (R7)"),
    ("c4_AB_smotetomek_weights", "c7_ABC_full", "Does threshold tuning repair stacked corrections?"),
]


def main_only(df):
    return df[df["experiment"] == "main"].copy()


def fold_matrix(df, metric, dataset, classifier, config_ids):
    """Rows = (repeat, fold), columns = configs, values = metric."""
    d = df[(df["dataset"] == dataset) & (df["classifier"] == classifier)
           & df["config_id"].isin(config_ids)]
    m = d.pivot_table(index=["repeat", "fold"], columns="config_id", values=metric)
    return m.reindex(columns=config_ids).dropna()


def _signs(term):
    """+1/-1 contrast of every factorial cell for a Yates effect term."""
    out = []
    for cid in FACTORIAL:
        cfg = configs.CONFIGS[cid]
        s = 1
        for f in term:
            s *= 1 if cfg[f] else -1
        out.append(s)
    return np.array(out)


SIGNS = {t: _signs(t) for t in TERMS}


def yates(cell_means):
    """Effects of a 2^3 design: mean(high) - mean(low) for each term."""
    return {t: float(SIGNS[t] @ cell_means / 4) for t in TERMS}


def factorial_effects(df, metric="macro_f1", n_boot=2000, seed=settings.MASTER_SEED):
    """Main and interaction effects per dataset x classifier, plus the average over
    classifiers per dataset. 95% CIs by bootstrapping the outer folds (paired)."""
    df = main_only(df)
    rng = np.random.default_rng(seed)
    rows, per_ds = [], {}
    for (ds, clf), _ in df.groupby(["dataset", "classifier"]):
        m = fold_matrix(df, metric, ds, clf, FACTORIAL).to_numpy()
        if len(m) == 0:
            continue
        est = yates(m.mean(0))
        idx = rng.integers(0, len(m), size=(n_boot, len(m)))
        boot = np.array([[SIGNS[t] @ m[i].mean(0) / 4 for t in TERMS] for i in idx])
        per_ds.setdefault(ds, []).append((m, idx))
        for j, t in enumerate(TERMS):
            rows.append(dict(dataset=ds, classifier=clf, term=t, effect=est[t],
                             ci_low=np.percentile(boot[:, j], 2.5),
                             ci_high=np.percentile(boot[:, j], 97.5)))
    for ds, items in per_ds.items():   # average over classifiers, same bootstrap folds
        n = min(len(m) for m, _ in items)
        stack = np.stack([m[:n] for m, _ in items])          # clf x folds x cells
        est = yates(stack.mean((0, 1)))
        idx = rng.integers(0, n, size=(n_boot, n))
        boot = np.array([[SIGNS[t] @ stack[:, i].mean((0, 1)) / 4 for t in TERMS] for i in idx])
        for j, t in enumerate(TERMS):
            rows.append(dict(dataset=ds, classifier="ALL", term=t, effect=est[t],
                             ci_low=np.percentile(boot[:, j], 2.5),
                             ci_high=np.percentile(boot[:, j], 97.5)))
    out = pd.DataFrame(rows)
    out["significant"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    return out


def pooled_model(df, metric="macro_f1"):
    """OLS on fold-level scores with effect-coded factors (-1/+1), dataset and
    classifier fixed effects, and standard errors clustered by outer fold.
    Reported 'effect' = 2 x coefficient, i.e. on the same scale as Yates effects."""
    d = main_only(df)
    d = d[d["config_id"].isin(FACTORIAL)].copy()
    for f in "ABC":
        d[f + "e"] = 2 * d[f].astype(int) - 1
    # a column named "C" would hide the formula's C(...) categorical helper
    d = d.drop(columns=["A", "B", "C"])
    d["cluster"] = d["dataset"] + "_" + d["repeat"].astype(str) + "_" + d["fold"].astype(str)
    fit = smf.ols(f"{metric} ~ Ae * Be * Ce + C(classifier) + C(dataset)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["cluster"])[0]})
    names = {"Ae": "A", "Be": "B", "Ce": "C", "Ae:Be": "AB", "Ae:Ce": "AC", "Be:Ce": "BC",
             "Ae:Be:Ce": "ABC"}
    ci = fit.conf_int()
    rows = [dict(term=names[k], effect=2 * fit.params[k], ci_low=2 * ci.loc[k, 0],
                 ci_high=2 * ci.loc[k, 1], p_value=fit.pvalues[k]) for k in names]
    return pd.DataFrame(rows), fit


def block_means(df, metric="macro_f1", config_ids=None):
    """dataset x classifier blocks (rows) by config (columns), mean over folds."""
    d = main_only(df)
    if config_ids is not None:
        d = d[d["config_id"].isin(config_ids)]
    return d.pivot_table(index=["dataset", "classifier"], columns="config_id", values=metric)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def five_by_two_f(a, b):
    """Alpaydin's combined 5x2cv F-test on paired fold scores (repeat-major order)."""
    d = (np.asarray(b) - np.asarray(a)).reshape(-1, 2)
    s2 = ((d - d.mean(1, keepdims=True)) ** 2).sum(1)
    if s2.sum() == 0:
        return np.nan
    f = (d ** 2).sum() / (2 * s2.sum())
    return float(stats.f.sf(f, d.size, len(d)))


def pairwise_tests(df, metric="macro_f1", pairs=KEY_PAIRS, n_boot=5000,
                   seed=settings.MASTER_SEED):
    """Wilcoxon signed-rank over blocks (Holm-corrected), rank-biserial r,
    bootstrap CI of the median difference, and how many blocks the within-block
    5x2cv F-test finds significant."""
    rng = np.random.default_rng(seed)
    bm = block_means(df, metric)
    main = main_only(df)
    rows = []
    for a, b, question in pairs:
        if a not in bm or b not in bm:
            continue
        pair = bm[[a, b]].dropna()
        diff = (pair[b] - pair[a]).to_numpy()
        nz = diff[diff != 0]
        if len(nz) == 0:
            p, r = 1.0, 0.0
        else:
            p = stats.wilcoxon(nz).pvalue
            ranks = stats.rankdata(np.abs(nz))
            r = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
        boot = np.median(diff[rng.integers(0, len(diff), (n_boot, len(diff)))], axis=1)
        sig_blocks, better = 0, 0
        for ds, clf in pair.index:
            m = fold_matrix(main, metric, ds, clf, [a, b])
            pv = five_by_two_f(m[a], m[b]) if len(m) == 10 else np.nan
            if pv == pv and pv < 0.05:
                sig_blocks += 1
                better += int(m[b].mean() > m[a].mean())
        rows.append(dict(comparison=f"{b} vs {a}", question=question, blocks=len(diff),
                         median_diff=float(np.median(diff)),
                         ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                         wins=int((diff > 0).sum()), losses=int((diff < 0).sum()),
                         p_wilcoxon=p, rank_biserial=r,
                         blocks_sig_5x2cv=sig_blocks, of_which_better=better))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_wilcoxon"])
    return out


def friedman_nemenyi(df, metric="macro_f1", alpha=0.05):
    """Average ranks of the 9 configs over blocks (rank 1 = best), Friedman test
    and the Nemenyi critical difference."""
    bm = block_means(df, metric, list(configs.CONFIGS)).dropna()
    ranks = bm.rank(axis=1, ascending=False).mean().sort_values()
    k, n = bm.shape[1], bm.shape[0]
    p = stats.friedmanchisquare(*[bm[c] for c in bm.columns]).pvalue
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return ranks, float(p), float(cd), n


def mechanism_summary(df):
    """How each config shifts the decision towards the minority class.
    Mean over folds per classifier, then the MEDIAN over classifiers, so one
    badly calibrated classifier (e.g. Gaussian NB) cannot dominate."""
    d = main_only(df)
    cols = ["rho_eff", "ppr_ratio", "precision_1", "recall_1", "macro_f1"]
    per_clf = d.groupby(["dataset", "config_id", "classifier"])[cols].mean()
    return per_clf.groupby(["dataset", "config_id"]).median().round(3)


def e2_curves(df):
    """Mean metrics per E2 setting, averaged over classifiers and folds."""
    d = df[df["experiment"] == "e2"].copy()
    d["log2_rho"] = np.log2(d["rho_eff"])
    keys = ["dataset", "sampling_ratio", "weight_mode", "weight_power", "C"]
    agg = d.groupby(keys).agg(log2_rho=("log2_rho", "mean"), macro_f1=("macro_f1", "mean"),
                              macro_f1_sd=("macro_f1", "std"), precision_1=("precision_1", "mean"),
                              recall_1=("recall_1", "mean"), ppr_ratio=("ppr_ratio", "mean"),
                              n=("macro_f1", "size"))
    return agg.reset_index()


def mechanism_tests(df):
    """Over-correction test on E2 (C = 0, rho_eff >= 1): per dataset x classifier,
    slope of P1 and of the predicted-positive ratio against log2(rho_eff).
    Over-correction predicts P1 slope < 0 and PPR slope > 0. With C = 1 the same
    slopes should be closer to 0 if threshold tuning undoes the over-weighting."""
    d = df[df["experiment"] == "e2"].copy()
    d["x"] = np.log2(d["rho_eff"])
    rows = []
    for (ds, clf, C), g in d[d["rho_eff"] >= 0.999].groupby(["dataset", "classifier", "C"]):
        if g["x"].nunique() < 3:
            continue
        row = dict(dataset=ds, classifier=clf, C=C, points=len(g))
        for y in ("precision_1", "recall_1", "ppr_ratio", "macro_f1"):
            res = stats.linregress(g["x"], g[y])
            row[f"slope_{y}"] = res.slope
            row[f"p_{y}"] = res.pvalue
        rows.append(row)
    return pd.DataFrame(rows)


def macro_f1_peak(df):
    """Where macro-F1 peaks along log2(rho_eff) (quadratic fit), per dataset and C."""
    d = df[df["experiment"] == "e2"].copy()
    d["x"] = np.log2(d["rho_eff"])
    rows = []
    for (ds, C), g in d.groupby(["dataset", "C"]):
        a, b, c = np.polyfit(g["x"], g["macro_f1"], 2)
        peak = -b / (2 * a) if a < 0 else np.nan
        rows.append(dict(dataset=ds, C=C, curvature=a, peak_log2_rho=peak,
                         peak_rho=2 ** peak if peak == peak else np.nan))
    return pd.DataFrame(rows)


def e3_table(df, metric="macro_f1"):
    """E3: config x minority share, mean over classifiers and folds."""
    d = df[df["experiment"] == "e3"]
    return d.pivot_table(index="config_id", columns="ir_level", values=metric).reindex(
        list(configs.CONFIGS)).sort_index(axis=1, ascending=False)


def e3_effects(df, metric="macro_f1", n_boot=2000):
    """Factorial effects at each E3 imbalance level (averaged over classifiers),
    computed exactly like the main factorial with the level in place of the dataset."""
    d = df[df["experiment"] == "e3"].copy()
    d["dataset"] = "level_" + d["ir_level"].astype(float).map(lambda v: f"{v:.2f}")
    d["experiment"] = "main"
    eff = factorial_effects(d, metric, n_boot=n_boot)
    eff = eff[eff["classifier"] == "ALL"].copy()
    eff["minority_share"] = eff["dataset"].str.replace("level_", "").astype(float)
    return eff.drop(columns=["dataset", "classifier"])


def results_table(df, metric="macro_f1", by_classifier=False):
    """Mean Â± SD over folds: config x dataset (averaged over classifiers) or
    config x classifier per dataset."""
    d = main_only(df)
    if by_classifier:
        g = d.groupby(["dataset", "config_id", "classifier"])[metric]
    else:
        # average over classifiers within each fold first, then over folds
        d = d.groupby(["dataset", "config_id", "repeat", "fold"])[metric].mean().reset_index()
        g = d.groupby(["dataset", "config_id"])[metric]
    return g.agg(["mean", "std"]).reset_index()
