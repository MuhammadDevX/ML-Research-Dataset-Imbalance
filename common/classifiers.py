"""The 7 base classifiers and their random-search spaces (design section 6)."""
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def make_base(name: str, seed: int):
    """Every estimator uses 1 thread; parallelism comes from the search."""
    if name == "LR":
        return LogisticRegression(max_iter=3000, random_state=seed)
    if name == "LinearSVM":
        return LinearSVC(max_iter=5000, random_state=seed)
    if name == "GNB":
        return GaussianNB()
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed)
    if name == "RF":
        return RandomForestClassifier(random_state=seed, n_jobs=1)
    if name == "XGB":
        return XGBClassifier(tree_method="hist", eval_metric="logloss",
                             random_state=seed, n_jobs=1)
    if name == "LGBM":
        return LGBMClassifier(subsample_freq=1, random_state=seed, n_jobs=1, verbose=-1)
    raise KeyError(f"Unknown classifier '{name}'")


SEARCH_SPACES = {
    "LR": {"C": loguniform(1e-3, 1e2)},
    "LinearSVM": {"C": loguniform(1e-3, 1e2)},
    "GNB": {"var_smoothing": loguniform(1e-11, 1e-5)},
    "DT": {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "criterion": ["gini", "entropy"],
    },
    "RF": {
        "n_estimators": [200, 400],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", 0.3],
    },
    "XGB": {
        "n_estimators": [200, 400, 800],
        "max_depth": [3, 4, 6, 8],
        "learning_rate": loguniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": [1, 3, 5],
        "reg_lambda": loguniform(1e-2, 10),
    },
    "LGBM": {
        "n_estimators": [200, 400, 800],
        "num_leaves": [15, 31, 63],
        "learning_rate": loguniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_samples": [10, 20, 50],
    },
}


def search_space(name: str) -> dict:
    """Search space with pipeline prefixes (step 'clf' -> CorrectedClassifier.base)."""
    return {f"clf__base__{k}": v for k, v in SEARCH_SPACES[name].items()}
