"""Dataset meta-features for relating effects to dataset properties (R1, R4).

Descriptive only: never used for model selection. Computed on a stratified
sample of at most 5,000 rows, after median imputation and standardisation (and
TF-IDF + SVD for text).
"""
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import settings

MAX_ROWS = 5000


def _prepare(X, modality):
    if modality == "text":
        Z = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(X)
        Z = TruncatedSVD(100, random_state=settings.MASTER_SEED).fit_transform(Z)
    else:
        Z = SimpleImputer(strategy="median").fit_transform(X)
        Z = VarianceThreshold(0.0).fit_transform(Z)
    return StandardScaler().fit_transform(Z)


def fisher_f1(Z, y):
    """F1 = 1 / (1 + max Fisher discriminant ratio). Lower = more separable."""
    a, b = Z[y == 1], Z[y == 0]
    num = (a.mean(0) - b.mean(0)) ** 2
    den = a.var(0) + b.var(0)
    ratio = np.where(den > 0, num / np.where(den > 0, den, 1), 0)
    return 1 / (1 + ratio.max())


def n1(Z, y):
    """Share of points joined to the other class in the minimum spanning tree."""
    mst = minimum_spanning_tree(squareform(pdist(Z))).tocoo()
    cross = y[mst.row] != y[mst.col]
    borderline = np.unique(np.concatenate([mst.row[cross], mst.col[cross]]))
    return len(borderline) / len(y)


def n3_and_1nn(Z, y):
    """Leave-one-out 1-NN error rate (N3) and 1-NN macro-F1."""
    nn = NearestNeighbors(n_neighbors=2).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1]
    pred = y[idx]
    return float((pred != y).mean()), f1_score(y, pred, average="macro")


def meta_features(X, y, modality, dataset) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    n, n_pos = len(y), int(y.sum())
    d = 1 if modality == "text" else X.shape[1]
    if n > MAX_ROWS:
        idx, _ = train_test_split(np.arange(n), train_size=MAX_ROWS, stratify=y,
                                  random_state=settings.MASTER_SEED)
        X = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        ys = y[idx]
    else:
        ys = y
    Z = _prepare(X, modality)
    n3, nn_f1 = n3_and_1nn(Z, ys)
    return pd.DataFrame([{
        "dataset": dataset,
        "n": n,
        "d_raw": d,
        "d_used": Z.shape[1],
        "n_minority": n_pos,
        "minority_pct": round(100 * n_pos / n, 3),
        "IR": round((n - n_pos) / n_pos, 2),
        "F1_fisher": round(fisher_f1(Z, ys), 4),
        "N1_mst": round(n1(Z, ys), 4),
        "N3_1nn_error": round(n3, 4),
        "macro_f1_1nn": round(nn_f1, 4),
        "sample_size": len(ys),
    }])
