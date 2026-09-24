"""Checks that this session uses the pinned package versions."""
import importlib
import re

from .paths import REPO_ROOT

MODULES = {"scikit-learn": "sklearn", "imbalanced-learn": "imblearn",
           "xgboost": "xgboost", "lightgbm": "lightgbm"}

RESTART = "On Colab: Runtime > Restart session, then Run all again."


def pinned() -> dict:
    pins = {}
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s#]+)", line)
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


def check_environment() -> None:
    """Compare the versions actually loaded in this session (not just installed)
    with requirements.txt. After pip upgrades a package that was already
    imported, only a restart loads the new version."""
    loaded = {}
    for pkg in pinned():
        try:
            loaded[pkg] = importlib.import_module(MODULES.get(pkg, pkg)).__version__
        except Exception as e:  # e.g. numpy upgraded underneath a loaded module
            raise RuntimeError(f"Could not import {pkg} ({e}). {RESTART}") from e
    bad = {p: (v, loaded[p]) for p, v in pinned().items() if loaded[p] != v}
    if bad:
        lines = "\n".join(f"  {p}: need {need}, loaded {have}" for p, (need, have) in bad.items())
        raise RuntimeError("Package versions differ from requirements.txt, so results would "
                           f"not be comparable:\n{lines}\n{RESTART}")
    print("Environment OK:", ", ".join(f"{p} {v}" for p, v in loaded.items()))
