"""Loading processed data and the committed outer-fold splits.

Processed files hold the output of the stateless preprocessing only (label
mapping, de-duplication, inf -> NaN, subsampling). Anything that is fitted
(imputation, scaling, TF-IDF) happens inside the pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from . import settings
from .datasets import info
from .paths import PROCESSED_DIR, SPLITS_DIR

TARGET = "y"
META_PREFIX = "meta_"   # columns kept for description only, never used as features


def processed_path(dataset: str):
    return PROCESSED_DIR / f"{dataset}.parquet"


def splits_path(dataset: str, level: float | None = None):
    tag = "outer" if level is None else f"level{level:.2f}"
    return SPLITS_DIR / f"{dataset}_{tag}.npz"


def save_processed(dataset: str, df: pd.DataFrame) -> None:
    if TARGET not in df.columns:
        raise ValueError(f"Processed data needs a '{TARGET}' column with 0/1 labels.")
    if not set(pd.unique(df[TARGET])) <= {0, 1}:
        raise ValueError("Labels must be 0 (majority) / 1 (minority).")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(processed_path(dataset), index=False)


def load_processed(dataset: str):
    """Return X (DataFrame, or Series of strings for text) and y (int array)."""
    df = pd.read_parquet(processed_path(dataset))
    y = df[TARGET].to_numpy().astype(int)
    if info(dataset)["modality"] == "text":
        X = df["text"].astype(str)
    else:
        feats = [c for c in df.columns if c != TARGET and not c.startswith(META_PREFIX)]
        X = df[feats]
    return X, y


def make_outer_splits(y, dataset: str, level: float | None = None, subset=None) -> None:
    """Create and save the 5x2 outer folds once, so every run uses identical folds.

    `subset` (E3 only) holds the pool rows that form this imbalance level; fold
    indices then refer to positions inside that subset.
    """
    y = np.asarray(y)
    cv = RepeatedStratifiedKFold(n_splits=settings.OUTER_SPLITS,
                                 n_repeats=settings.OUTER_REPEATS,
                                 random_state=settings.MASTER_SEED)
    arrays = {}
    for i, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        r, f = divmod(i, settings.OUTER_SPLITS)
        arrays[f"train_{r}_{f}"] = tr
        arrays[f"test_{r}_{f}"] = te
    if subset is not None:
        arrays["subset"] = np.asarray(subset)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(splits_path(dataset, level), **arrays)


def load_outer_splits(dataset: str, n_repeats: int, level: float | None = None):
    """Yield (repeat, fold, train_idx, test_idx) for the first `n_repeats` repeats."""
    z = np.load(splits_path(dataset, level))
    out = []
    for r in range(n_repeats):
        for f in range(settings.OUTER_SPLITS):
            out.append((r, f, z[f"train_{r}_{f}"], z[f"test_{r}_{f}"]))
    subset = z["subset"] if "subset" in z.files else None
    return out, subset


def summary(dataset: str) -> dict:
    X, y = load_processed(dataset)
    n_pos = int(y.sum())
    out = {
        "dataset": dataset,
        "n": len(y),
        "d": 1 if isinstance(X, pd.Series) else X.shape[1],
        "n_minority": n_pos,
        "minority_pct": round(100 * n_pos / len(y), 3),
        "IR": round((len(y) - n_pos) / n_pos, 2),
    }
    if isinstance(X, pd.DataFrame):
        out["nan_cells"] = int(X.isna().sum().sum())
    return out
