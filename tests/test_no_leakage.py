"""Leakage checks (reviewer R10) and a fit/evaluate smoke test for every
classifier and every config on small synthetic data."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from common import configs, settings
from common.cv import fit_evaluate
from common.pipeline import build_pipeline

RS = settings.SMOKE


def _data(n=600, weights=(0.85,), seed=0):
    X, y = make_classification(n_samples=n, n_features=8, n_informative=4,
                               weights=list(weights), random_state=seed)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    X.iloc[::17, 0] = np.nan   # exercise the in-fold imputer
    return X, y


class Spy(ClassifierMixin, BaseEstimator):
    """Records every row it is trained on."""
    def fit(self, X, y, sample_weight=None):
        self.seen_ = np.asarray(X).copy()
        self.inner_ = LogisticRegression().fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.inner_.classes_
        return self

    def predict(self, X):
        return self.inner_.predict(X)

    def predict_proba(self, X):
        return self.inner_.predict_proba(X)


@pytest.mark.parametrize("sampling_ratio", [0.0, 1.0])
def test_classifier_never_sees_test_rows(sampling_ratio):
    X, y = _data()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    tr, te = idx[:400], idx[400:]
    pipe = build_pipeline("tabular", Spy(), sampling_ratio, "orig", 1.0, seed=0)
    pipe.fit(X.iloc[tr], y[tr])
    seen = pipe.named_steps["clf"].base_.seen_
    test_rows = pipe[:-1].transform(X.iloc[te])
    train_rows = pipe[:-1].transform(X.iloc[tr])
    seen_set = {r.tobytes() for r in np.round(seen, 10)}
    assert not any(r.tobytes() in seen_set for r in np.round(test_rows, 10))
    assert all(r.tobytes() in seen_set for r in np.round(train_rows, 10)
               ) or sampling_ratio > 0   # Tomek links may drop some training rows
    if sampling_ratio > 0:
        assert len(seen) > len(tr) * 0.9


def test_preprocessing_statistics_come_from_training_fold_only():
    X, y = _data()
    tr = np.arange(300)
    X_shift = X.copy()
    X_shift.iloc[300:] += 100.0   # test part very different from training part
    pipe = build_pipeline("tabular", LogisticRegression(), 0.0, "none", 0.0, seed=0)
    pipe.fit(X_shift.iloc[tr], y[tr])
    train_median = X_shift.iloc[tr].median().to_numpy()
    assert np.allclose(pipe.named_steps["impute"].statistics_, train_median)
    assert np.all(np.abs(pipe.named_steps["scale"].mean_) < 5)


def test_permuted_labels_give_chance_auc():
    X, y = _data(n=1000)
    y_perm = np.random.default_rng(1).permutation(y)
    idx = np.random.default_rng(2).permutation(len(y))
    res = fit_evaluate(X, y_perm, idx[:500], idx[500:], "tabular", "LR", 1.0, "orig", 1.0,
                       1, 0, RS, RS.n_iter)
    assert abs(res["roc_auc"] - 0.5) < 0.1


@pytest.mark.parametrize("clf", settings.CLASSIFIERS)
def test_every_classifier_runs(clf):
    X, y = _data()
    idx = np.random.default_rng(3).permutation(len(y))
    res = fit_evaluate(X, y, idx[:300], idx[300:], "tabular", clf, 1.0, "orig", 1.0,
                       1, 0, RS, RS.n_iter)
    assert 0 <= res["macro_f1"] <= 1 and 0 <= res["roc_auc"] <= 1


@pytest.mark.parametrize("config_id", list(configs.CONFIGS))
def test_every_config_runs_and_rho_is_as_designed(config_id):
    X, y = _data(n=800, weights=(0.9,))
    cfg = configs.CONFIGS[config_id]
    idx = np.random.default_rng(4).permutation(len(y))
    tr, te = idx[:500], idx[500:]
    res = fit_evaluate(X, y, tr, te, "tabular", "LR", configs.sampling_ratio(cfg),
                       cfg["weight_mode"], configs.weight_power(cfg), cfg["C"], 0, RS, RS.n_iter)
    ir = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    expected = {"c0_none": 1 / ir, "c3_C_threshold": 1 / ir,
                "c4_AB_smotetomek_weights": ir, "c7_ABC_full": ir}.get(config_id, 1.0)
    assert res["rho_eff"] == pytest.approx(expected, rel=0.15)
    if not cfg["C"]:
        assert res["threshold"] == 0.5


def test_pipeline_cache_does_not_change_results():
    from common import pipeline as pl
    X, y = _data()
    idx = np.random.default_rng(5).permutation(len(y))
    args = (X, y, idx[:400], idx[400:], "tabular", "RF", 1.0, "orig", 1.0, 1, 0, RS, RS.n_iter)
    cached = fit_evaluate(*args)
    saved, pl._CACHE = pl._CACHE, None      # Pipeline(memory=None) = no caching
    try:
        uncached = fit_evaluate(*args)
    finally:
        pl._CACHE = saved
    for k in ("macro_f1", "roc_auc", "threshold", "rho_eff", "best_params_json"):
        assert cached[k] == uncached[k]


def test_text_pipeline_runs():
    # Vocabulary large enough for the 300 SVD components used on the real SMS data.
    rng = np.random.default_rng(0)
    spam, ham = [f"spam{i}" for i in range(150)], [f"ham{i}" for i in range(250)]
    y = (rng.random(800) < 0.15).astype(int)
    texts = [" ".join(rng.choice(spam if t else ham, 12)) for t in y]
    X = pd.Series(texts)
    idx = rng.permutation(800)
    res = fit_evaluate(X, y, idx[:600], idx[600:], "text", "GNB", 1.0, "none", 0.0, 0, 0,
                       RS, RS.n_iter)
    assert res["macro_f1"] > 0.5
