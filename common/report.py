"""Per-notebook summaries and merging of all result files."""
from pathlib import Path

import pandas as pd

from . import configs, io, settings
from .paths import results_root

SUMMARY_METRICS = ["macro_f1", "mcc", "pr_auc", "roc_auc", "precision_1", "recall_1", "ppr_ratio"]


def summarize(df: pd.DataFrame, by=("classifier",)) -> pd.DataFrame:
    """Mean ± SD over outer folds."""
    by = list(by)
    g = df.groupby(by)[SUMMARY_METRICS]
    mean, sd = g.mean(), g.std().fillna(0)
    out = mean.round(4).astype(str) + " ± " + sd.round(4).astype(str)
    out.insert(0, "folds", g.size())
    return out


def merge_results(root: Path | None = None, include_smoke=False) -> pd.DataFrame:
    root = Path(root or results_root())
    files = sorted(root.glob("*/*/results/*.csv"))
    if not include_smoke:
        files = [f for f in files if not f.stem.endswith("__SMOKE")]
    frames = [io.read_results(f) for f in files]
    if not frames:
        return pd.DataFrame(columns=io.SCHEMA)
    df = pd.concat(frames, ignore_index=True)
    dup = df.duplicated(subset=["dataset", "experiment"] + io.KEY_COLS).sum()
    if dup:
        print(f"WARNING: {dup} duplicated result rows")
    return df


def expected_rows() -> dict:
    """Rows a complete run produces per (experiment, config_id)."""
    folds = settings.OUTER_REPEATS * settings.OUTER_SPLITS
    from .cv import e2_grid
    exp = {("main", c): len(settings.CLASSIFIERS) * folds for c in configs.CONFIGS}
    exp[("e2", "e2")] = len(settings.E2_CLASSIFIERS) * folds * len(e2_grid())
    for c in configs.CONFIGS:
        exp[("e3", c)] = len(settings.E3_CLASSIFIERS) * folds * len(settings.E3_LEVELS)
    return exp


def completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Row count vs expected per dataset/experiment/config, plus commits used."""
    cols = ["rows", "expected", "complete", "git_commits", "metric_nans"]
    if df.empty:
        print("No results found yet.")
        return pd.DataFrame(columns=cols)
    exp = expected_rows()
    keys = ["dataset", "experiment", "config_id"]
    g = df.groupby(keys)
    out = g.size().rename("rows").to_frame()
    out["expected"] = [exp.get((e, c)) for _, e, c in out.index]
    out["complete"] = out["rows"] == out["expected"]
    out["git_commits"] = g["git_commit"].agg(lambda s: ",".join(sorted(set(map(str, s)))))
    out["metric_nans"] = df[io.METRIC_COLS].isna().sum(axis=1).groupby([df[k] for k in keys]).sum()
    return out[cols]
