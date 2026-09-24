"""Results schema and resume-safe CSV writing.

Every row of every experiment, from both team members, has exactly these
columns in this order, so all result files can be concatenated directly.
"""
import csv
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version

import pandas as pd

from .paths import REPO_ROOT

ID_COLS = ["dataset", "domain", "owner", "experiment", "config_id", "A", "B", "C",
           "weight_mode", "weight_power", "sampling_ratio", "ir_level", "classifier",
           "repeat", "fold", "seed"]
INFO_COLS = ["n_train", "n_test", "n_test_pos", "n_iter", "train_pos_ratio_after",
             "rho_eff", "threshold", "best_params_json"]
METRIC_COLS = ["accuracy", "balanced_acc", "precision_1", "recall_1", "f1_1",
               "precision_0", "recall_0", "f1_0", "macro_f1", "mcc", "roc_auc",
               "pr_auc", "gmean", "ppr_ratio"]
BOOK_COLS = ["fit_time_s", "smoke", "git_commit", "package_versions", "timestamp"]
SCHEMA = ID_COLS + INFO_COLS + METRIC_COLS + BOOK_COLS

# Columns that identify one unit of work; used to skip finished rows on resume.
KEY_COLS = ["config_id", "classifier", "repeat", "fold", "weight_mode", "weight_power",
            "sampling_ratio", "C", "ir_level"]


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "common"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def package_versions() -> str:
    pkgs = ["scikit-learn", "imbalanced-learn", "xgboost", "lightgbm", "numpy", "pandas"]
    return ";".join(f"{p}={version(p)}" for p in pkgs)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def params_to_json(params: dict) -> str:
    clean = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
    return json.dumps(clean, sort_keys=True)


def params_from_json(s: str) -> dict:
    return json.loads(s)


def _norm(v) -> str:
    """Same text for a value whether it comes from memory or back from the CSV."""
    if v is None or (isinstance(v, float) and v != v) or v == "":
        return ""
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def row_key(row) -> tuple:
    return tuple(_norm(row[c]) for c in KEY_COLS)


def read_results(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SCHEMA)
    df = pd.read_csv(path)
    missing = set(SCHEMA) - set(df.columns)
    if missing or list(df.columns) != SCHEMA:
        raise ValueError(f"{path} does not match the results schema (missing: {sorted(missing)}).")
    return df


def done_keys(path) -> set:
    df = read_results(path)
    return {row_key(r) for _, r in df.iterrows()}


def append_row(path, row: dict) -> None:
    extra, missing = set(row) - set(SCHEMA), set(SCHEMA) - set(row)
    if extra or missing:
        raise ValueError(f"Row does not match schema. Missing: {sorted(missing)}, extra: {sorted(extra)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SCHEMA)
        if new:
            w.writeheader()
        w.writerow(row)
