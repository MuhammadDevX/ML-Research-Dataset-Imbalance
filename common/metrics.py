"""Evaluation metrics (design section 7). Positive class = minority = 1."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)


def scores(model, X):
    """Continuous score for AUCs: P(y=1) if available, else the decision function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def default_threshold(model) -> float:
    return 0.5 if hasattr(model, "predict_proba") else 0.0


def compute(y_true, y_pred, y_score) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p = dict(zero_division=0)
    r1 = recall_score(y_true, y_pred, pos_label=1, **p)
    r0 = recall_score(y_true, y_pred, pos_label=0, **p)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "precision_1": precision_score(y_true, y_pred, pos_label=1, **p),
        "recall_1": r1,
        "f1_1": f1_score(y_true, y_pred, pos_label=1, **p),
        "precision_0": precision_score(y_true, y_pred, pos_label=0, **p),
        "recall_0": r0,
        "f1_0": f1_score(y_true, y_pred, pos_label=0, **p),
        "macro_f1": f1_score(y_true, y_pred, average="macro", **p),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "gmean": float(np.sqrt(r1 * r0)),
        # > 1 means the model predicts more positives than exist: over-correction.
        "ppr_ratio": y_pred.mean() / y_true.mean(),
    }
