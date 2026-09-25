"""Nested, leakage-free evaluation (design section 5) for the main factorial,
the weight-sensitivity study (E2) and the imbalance sweep (E3).

Every (classifier, outer fold) result is appended to the CSV as soon as it is
finished. Re-running a notebook skips finished rows, so a Colab disconnect only
loses the fold that was running.
"""
import os

# Silence convergence/deprecation warnings, including in parallel workers.
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import time
import warnings
from dataclasses import replace

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     TunedThresholdClassifierCV)

from . import configs, io, settings
from .classifiers import make_base, search_space
from .data import load_outer_splits, load_processed
from .datasets import info
from .metrics import compute, default_threshold, scores
from .paths import results_path
from .pipeline import build_pipeline, clear_cache

warnings.filterwarnings("ignore")


def _rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def fit_evaluate(X, y, train_idx, test_idx, modality, clf_name, sampling_ratio,
                 weight_mode, weight_power, C, seed, rs, n_iter, fixed_params=None):
    """Fit on the outer training fold only, evaluate on the outer test fold.

    Hyperparameters come from a random search on inner folds of the training
    fold, or from `fixed_params` (the twin's tuned values). When C = 1 the
    decision threshold is tuned by cross-validation inside the training fold.
    """
    t0 = time.time()
    pipe = build_pipeline(modality, make_base(clf_name, seed), sampling_ratio,
                          weight_mode, weight_power, seed)
    X_tr, y_tr = _rows(X, train_idx), y[train_idx]
    X_te, y_te = _rows(X, test_idx), y[test_idx]

    if fixed_params is None:
        search = RandomizedSearchCV(
            pipe, search_space(clf_name), n_iter=n_iter, scoring="f1_macro",
            cv=StratifiedKFold(rs.inner_folds, shuffle=True, random_state=seed),
            n_jobs=rs.n_jobs, random_state=seed, refit=True, error_score="raise")
        search.fit(X_tr, y_tr)
        model, params = search.best_estimator_, search.best_params_
    else:
        model, params = pipe.set_params(**fixed_params), fixed_params
        if not C:
            model.fit(X_tr, y_tr)

    if C:
        final = TunedThresholdClassifierCV(
            model, scoring="f1_macro",
            cv=StratifiedKFold(rs.threshold_cv, shuffle=True, random_state=seed),
            refit=True, n_jobs=rs.n_jobs, random_state=seed)
        try:
            final.fit(X_tr, y_tr)
            fitted_pipe, threshold = final.estimator_, float(final.best_threshold_)
        except ValueError as e:
            # Extreme weights can make a model's scores constant on a threshold-CV
            # fold, so no threshold can be tuned. Keep the default threshold and
            # record NaN, so these cases are visible and countable in the results.
            if "constant predictions" not in str(e):
                raise
            final = fitted_pipe = model.fit(X_tr, y_tr)
            threshold = float("nan")
    else:
        final, fitted_pipe, threshold = model, model, default_threshold(model)

    clf_step = fitted_pipe.named_steps["clf"]
    out = compute(y_te, final.predict(X_te), scores(final, X_te))
    out.update(
        n_train=len(train_idx), n_test=len(test_idx), n_test_pos=int(y_te.sum()),
        n_iter=0 if fixed_params is not None else n_iter,
        train_pos_ratio_after=clf_step.train_pos_ratio_after_, rho_eff=clf_step.rho_eff_,
        threshold=threshold, best_params_json=io.params_to_json(params),
        fit_time_s=round(time.time() - t0, 2),
    )
    clear_cache()
    return out


def _make_row(dataset, experiment, config_id, A, B, C, weight_mode, weight_power,
              sampling_ratio, ir_level, clf, repeat, fold, seed, res, smoke):
    meta = info(dataset)
    row = dict(dataset=dataset, domain=meta["domain"], owner=meta["owner"],
               experiment=experiment, config_id=config_id, A=A, B=B, C=C,
               weight_mode=weight_mode, weight_power=weight_power,
               sampling_ratio=sampling_ratio, ir_level="" if ir_level is None else ir_level,
               classifier=clf, repeat=repeat, fold=fold, seed=seed,
               smoke=smoke, git_commit=io.git_commit(),
               package_versions=io.package_versions(), timestamp=io.now())
    row.update(res)
    return row


def _params_lookup(path, config_id, ir_level=None) -> dict:
    """{(classifier, repeat, fold): params} from another config's results."""
    if not path.exists():
        return {}
    df = io.read_results(path)
    df = df[df["config_id"] == config_id]
    if ir_level is not None:
        df = df[np.isclose(df["ir_level"].astype(float), ir_level)]
    return {(r.classifier, int(r.repeat), int(r.fold)): io.params_from_json(r.best_params_json)
            for r in df.itertuples()}


def _progress(k, total, clf, r, f, res, note=""):
    print(f"[{k:>3}/{total}] {clf:<9} repeat {r} fold {f}  macro-F1={res['macro_f1']:.4f}  "
          f"P1={res['precision_1']:.3f} R1={res['recall_1']:.3f}  ({res['fit_time_s']:.0f}s){note}",
          flush=True)


# --------------------------------------------------------------------------
# Main factorial: one notebook per (dataset, config)
# --------------------------------------------------------------------------
def run_config(dataset, config_id, smoke=False, classifiers=None, reuse_twin=True):
    rs = settings.run_settings(smoke)
    cfg = configs.get(config_id)
    modality = info(dataset)["modality"]
    X, y = load_processed(dataset)
    splits, _ = load_outer_splits(dataset, rs.outer_repeats)
    path = results_path(dataset, config_id, smoke)
    done = io.done_keys(path)

    twin = {}
    if reuse_twin and cfg["twin"]:
        twin = _params_lookup(results_path(dataset, cfg["twin"], smoke), cfg["twin"])
        print(f"Twin {cfg['twin']}: tuned hyperparameters found for {len(twin)} folds "
              f"(the rest are tuned here with the same seed, giving the same result).")

    sr, wp = configs.sampling_ratio(cfg), configs.weight_power(cfg)
    clfs = list(classifiers or rs.classifiers)
    total, k = len(clfs) * len(splits), 0
    print(f"{dataset} / {config_id}: {configs.DESCRIPTIONS[config_id]}")
    print(f"Writing to {path}\n")
    for clf in clfs:
        for r, f, tr, te in splits:
            k += 1
            key = dict(config_id=config_id, classifier=clf, repeat=r, fold=f,
                       weight_mode=cfg["weight_mode"], weight_power=wp,
                       sampling_ratio=sr, C=cfg["C"], ir_level=None)
            if io.row_key(key) in done:
                print(f"[{k:>3}/{total}] {clf:<9} repeat {r} fold {f}  already done, skipped")
                continue
            seed = settings.fold_seed(r, f)
            fixed = twin.get((clf, r, f))
            res = fit_evaluate(X, y, tr, te, modality, clf, sr, cfg["weight_mode"], wp,
                               cfg["C"], seed, rs, rs.n_iter, fixed_params=fixed)
            io.append_row(path, _make_row(dataset, "main", config_id, cfg["A"], cfg["B"],
                                          cfg["C"], cfg["weight_mode"], wp, sr, None, clf,
                                          r, f, seed, res, rs.smoke))
            _progress(k, total, clf, r, f, res, "  [twin params]" if fixed else "")
    return io.read_results(path)


# --------------------------------------------------------------------------
# E2: weight sensitivity (hyperparameters reused from c0 / c1)
# --------------------------------------------------------------------------
def e2_grid():
    grid = []
    for sr in settings.E2_SAMPLING_RATIOS:
        settings_ = [("orig", p) for p in settings.E2_WEIGHT_POWERS]
        if sr > 0:   # without resampling "recalc" is identical to orig, power 1
            settings_.append(("recalc", 1.0))
        for mode, power in settings_:
            for C in (0, 1):
                grid.append((sr, mode, power, C))
    return grid


def run_e2(dataset, smoke=False, classifiers=None):
    rs = settings.run_settings(smoke)
    modality = info(dataset)["modality"]
    X, y = load_processed(dataset)
    splits, _ = load_outer_splits(dataset, rs.outer_repeats)
    path = results_path(dataset, "e2_weight_sensitivity", smoke)
    done = io.done_keys(path)
    source = {0: _params_lookup(results_path(dataset, "c0_none", smoke), "c0_none"),
              1: _params_lookup(results_path(dataset, "c1_A_smotetomek", smoke), "c1_A_smotetomek")}
    print(f"Hyperparameters found: c0 {len(source[0])} folds, c1 {len(source[1])} folds. "
          "Missing ones are tuned here once with the same seed.")
    print(f"Writing to {path}\n")

    grid = e2_grid()
    clfs = list(classifiers or rs.e2_classifiers)
    serial = replace(rs, n_jobs=1)   # parallelism is across grid points instead
    total, k = len(clfs) * len(splits) * len(grid), 0
    for clf in clfs:
        for r, f, tr, te in splits:
            seed = settings.fold_seed(r, f)
            todo = []
            for sr, mode, power, C in grid:
                key = dict(config_id="e2", classifier=clf, repeat=r, fold=f, weight_mode=mode,
                           weight_power=power, sampling_ratio=sr, C=C, ir_level=None)
                if io.row_key(key) not in done:
                    todo.append((sr, mode, power, C))
            k += len(grid) - len(todo)
            if not todo:
                continue
            for src in {int(sr > 0) for sr, *_ in todo}:
                if (clf, r, f) not in source[src]:
                    tuned = fit_evaluate(X, y, tr, te, modality, clf, 1.0 if src else 0.0,
                                         "none", 0.0, 0, seed, rs, rs.n_iter)
                    source[src][(clf, r, f)] = io.params_from_json(tuned["best_params_json"])
            results = Parallel(n_jobs=rs.n_jobs)(
                delayed(fit_evaluate)(X, y, tr, te, modality, clf, sr, mode, power, C, seed,
                                      serial, rs.n_iter,
                                      fixed_params=source[int(sr > 0)][(clf, r, f)])
                for sr, mode, power, C in todo)
            for (sr, mode, power, C), res in zip(todo, results):
                k += 1
                B = int(mode == "recalc" or power > 0)
                io.append_row(path, _make_row(dataset, "e2", "e2", int(sr > 0), B, C, mode,
                                              power, sr, None, clf, r, f, seed, res, rs.smoke))
                _progress(k, total, clf, r, f, res,
                          f"  [ratio {sr}, {mode}^{power}, C={C}, rho={res['rho_eff']:.2f}]")
    return io.read_results(path)


# --------------------------------------------------------------------------
# E3: controlled imbalance sweep on CIC-IDS2017 (BENIGN vs Bot)
# --------------------------------------------------------------------------
def run_e3(levels=None, smoke=False, classifiers=None):
    rs = settings.run_settings(smoke)
    pool = "cicids2017_e3"
    modality = info(pool)["modality"]
    X_pool, y_pool = load_processed(pool)
    path = results_path("cicids2017", "e3_ir_sweep", smoke)
    clfs = list(classifiers or rs.e3_classifiers)
    levels = list(levels or settings.E3_LEVELS)
    print(f"Writing to {path}\n")

    for level in levels:
        splits, subset = load_outer_splits(pool, rs.outer_repeats, level=level)
        X, y = _rows(X_pool, subset), y_pool[subset]
        X = X.reset_index(drop=True) if hasattr(X, "reset_index") else X
        print(f"=== minority share {level:.0%}: n={len(y)}, minority={int(y.sum())} ===")
        for config_id in configs.RUN_ORDER:
            cfg = configs.get(config_id)
            sr, wp = configs.sampling_ratio(cfg), configs.weight_power(cfg)
            done = io.done_keys(path)
            twin = _params_lookup(path, cfg["twin"], level) if cfg["twin"] else {}
            total, k = len(clfs) * len(splits), 0
            print(f"--- {config_id}")
            for clf in clfs:
                for r, f, tr, te in splits:
                    k += 1
                    key = dict(config_id=config_id, classifier=clf, repeat=r, fold=f,
                               weight_mode=cfg["weight_mode"], weight_power=wp,
                               sampling_ratio=sr, C=cfg["C"], ir_level=level)
                    if io.row_key(key) in done:
                        continue
                    seed = settings.fold_seed(r, f)
                    fixed = twin.get((clf, r, f))
                    res = fit_evaluate(X, y, tr, te, modality, clf, sr, cfg["weight_mode"], wp,
                                       cfg["C"], seed, rs, rs.n_iter_e3, fixed_params=fixed)
                    io.append_row(path, _make_row("cicids2017", "e3", config_id, cfg["A"],
                                                  cfg["B"], cfg["C"], cfg["weight_mode"], wp,
                                                  sr, level, clf, r, f, seed, res, rs.smoke))
                    _progress(k, total, clf, r, f, res, "  [twin params]" if fixed else "")
    return io.read_results(path)
