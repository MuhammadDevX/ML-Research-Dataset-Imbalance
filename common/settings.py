"""Frozen experiment settings. Every notebook of every dataset reads these values.

Do not change anything here on a personal branch: a change must go through a
reviewed pull request, because it changes the protocol for both team members.
"""
import os
from dataclasses import dataclass, replace

MASTER_SEED = 2026

# Outer evaluation: 5 x 2 repeated stratified CV (10 test estimates per cell).
OUTER_SPLITS = 2
OUTER_REPEATS = 5

# Inner model selection (nested inside every outer training fold).
INNER_FOLDS = 3
N_ITER = 20            # random-search candidates per outer fold
THRESHOLD_CV = 3       # folds used by TunedThresholdClassifierCV when C = 1
N_ITER_E3 = 10         # smaller budget for the IR sweep (5 levels x 9 configs)

CLASSIFIERS = ["LR", "LinearSVM", "GNB", "DT", "RF", "XGB", "LGBM"]
E2_CLASSIFIERS = ["LR", "RF", "XGB"]
E3_CLASSIFIERS = ["LR", "RF", "XGB"]

# Weight-sensitivity grid (E2). The positive-class weight is IR ** power, so
# power 0 = no weighting, 0.5 = sqrt-inverse, 1 = inverse frequency (the B used
# in the main factorial), 2 = inverse frequency squared.
E2_WEIGHT_POWERS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
E2_SAMPLING_RATIOS = [0.0, 0.5, 1.0]   # 0.0 = no resampling

# Controlled imbalance sweep on CIC-IDS2017 (E3): minority share of a fixed N.
E3_LEVELS = [0.15, 0.10, 0.05, 0.02, 0.01]
E3_N = 10_000

# Parallel workers for the inner search (estimators themselves use 1 thread).
# Only affects speed, never results. Override with the N_JOBS environment variable.
N_JOBS = int(os.environ.get("N_JOBS", "-1"))


@dataclass(frozen=True)
class RunSettings:
    outer_repeats: int
    inner_folds: int
    n_iter: int
    n_iter_e3: int
    threshold_cv: int
    classifiers: tuple
    e2_classifiers: tuple
    e3_classifiers: tuple
    n_jobs: int
    smoke: bool


FULL = RunSettings(
    outer_repeats=OUTER_REPEATS,
    inner_folds=INNER_FOLDS,
    n_iter=N_ITER,
    n_iter_e3=N_ITER_E3,
    threshold_cv=THRESHOLD_CV,
    classifiers=tuple(CLASSIFIERS),
    e2_classifiers=tuple(E2_CLASSIFIERS),
    e3_classifiers=tuple(E3_CLASSIFIERS),
    n_jobs=N_JOBS,
    smoke=False,
)

# Smoke mode: proves a notebook runs end to end in minutes. Its results are
# written to separate *__SMOKE.csv files and must never be used in the paper.
SMOKE = replace(
    FULL,
    outer_repeats=1,
    inner_folds=2,
    n_iter=2,
    n_iter_e3=2,
    threshold_cv=2,
    classifiers=("LR", "DT"),
    e2_classifiers=("LR",),
    e3_classifiers=("LR",),
    smoke=True,
)


def run_settings(smoke: bool) -> RunSettings:
    return SMOKE if smoke else FULL


def fold_seed(repeat: int, fold: int) -> int:
    """Seed shared by every config for the same outer fold, so twins match."""
    return MASTER_SEED + 1000 * repeat + fold
