"""CorrectedClassifier: factors A (resampling) and B (weighting) inside fit().

Because both corrections happen inside `fit`, they only ever see the rows the
estimator is being trained on: the inner-CV training folds during the search,
the outer training fold at refit, and the threshold-CV folds when C = 1. This
is what rules out resampling/weighting leakage (reviewer R10).
"""
import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.metaestimators import available_if


def _base_has(attr):
    return lambda self: hasattr(self.base, attr)


class CorrectedClassifier(ClassifierMixin, BaseEstimator):
    """Wraps a binary classifier (labels 0/1, positive = minority = 1).

    Parameters
    ----------
    base : estimator supporting ``fit(X, y, sample_weight=...)``.
    sampling_ratio : 0.0 disables resampling (A = 0). Otherwise SMOTETomek
        oversamples the minority to ``sampling_ratio`` x majority, then removes
        Tomek links. If the data is already at or above that ratio only the
        Tomek-link cleaning is applied.
    weight_mode : "none", "orig" (ratio of the data BEFORE resampling, as in
        v1) or "recalc" (ratio of the data AFTER resampling).
    weight_power : positive-class weight = ratio ** weight_power.
    """

    def __init__(self, base=None, sampling_ratio=0.0, weight_mode="none",
                 weight_power=1.0, random_state=None):
        self.base = base
        self.sampling_ratio = sampling_ratio
        self.weight_mode = weight_mode
        self.weight_power = weight_power
        self.random_state = random_state

    def _resample(self, X, y):
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        if n_pos / n_neg >= self.sampling_ratio:
            return TomekLinks().fit_resample(X, y)
        # The ratio must be set on the SMOTE object: when `smote` is passed,
        # SMOTETomek ignores its own sampling_strategy and uses SMOTE's.
        smote = SMOTE(sampling_strategy=self.sampling_ratio, k_neighbors=min(5, n_pos - 1),
                      random_state=self.random_state)
        sampler = SMOTETomek(smote=smote, random_state=self.random_state)
        return sampler.fit_resample(X, y)

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
        self.ir_train_ = n_neg / n_pos

        if self.sampling_ratio and self.sampling_ratio > 0:
            X, y = self._resample(X, y)
        n_pos2, n_neg2 = (y == 1).sum(), (y == 0).sum()

        if self.weight_mode == "none":
            w_pos = 1.0
        elif self.weight_mode == "orig":
            w_pos = self.ir_train_ ** self.weight_power
        elif self.weight_mode == "recalc":
            w_pos = (n_neg2 / n_pos2) ** self.weight_power
        else:
            raise ValueError(f"Unknown weight_mode '{self.weight_mode}'")

        sample_weight = np.where(y == 1, w_pos, 1.0)
        # Mean weight 1 keeps the effective regularisation strength (LR/SVM C,
        # tree min_child_weight) comparable across weighting settings.
        sample_weight = sample_weight / sample_weight.mean()

        self.base_ = clone(self.base)
        self.base_.fit(X, y, sample_weight=sample_weight)

        self.w_pos_ = w_pos
        self.n_fit_ = len(y)
        self.train_pos_ratio_after_ = n_pos2 / len(y)
        # Effective positive:negative loss mass the classifier actually sees.
        self.rho_eff_ = w_pos * n_pos2 / n_neg2
        return self

    def predict(self, X):
        return self.base_.predict(X).astype(int)

    @available_if(_base_has("predict_proba"))
    def predict_proba(self, X):
        return self.base_.predict_proba(X)

    @available_if(_base_has("decision_function"))
    def decision_function(self, X):
        return self.base_.decision_function(X)
