"""Run every experiment notebook of a dataset in the right order, resuming
where the last run stopped. Works on Windows, Linux and macOS.

    python scripts/run_experiments.py --status                     # what is done / missing
    python scripts/run_experiments.py --dataset cicids2017         # run everything missing
    python scripts/run_experiments.py --dataset cicids2017 --only e2 e3
    python scripts/run_experiments.py --dataset pima --dataset creditfraud --compile

Each notebook is executed in place (outputs are saved in it). Results are
appended to <dataset>/<experiment>/results/*.csv after every fold, so an
interrupted run loses at most the fold in progress; running this script again
skips finished work. A notebook that fails is retried (default 3 attempts),
then the script moves on and reports it at the end.

Speed: set the N_JOBS environment variable to the number of parallel workers
(default: all cores). It never changes results. Memory is the usual limit:
SMOTETomek configs on CIC-IDS2017 need about 1 GB per worker.
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbformat  # noqa: E402
import pandas as pd  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

from common import configs, settings  # noqa: E402
from common.cv import e2_grid  # noqa: E402
from common.datasets import DATASETS  # noqa: E402
from common.paths import results_path  # noqa: E402

FOLDS = settings.OUTER_REPEATS * settings.OUTER_SPLITS


def jobs(dataset):
    """(key, notebook, results csv, expected rows) in run order."""
    out = []
    for cid in configs.RUN_ORDER:
        out.append((cid, ROOT / dataset / cid / f"{dataset}_{cid}.ipynb",
                    results_path(dataset, cid, False), len(settings.CLASSIFIERS) * FOLDS))
    out.append(("e2", ROOT / dataset / "e2_weight_sensitivity" / f"{dataset}_e2_weight_sensitivity.ipynb",
                results_path(dataset, "e2_weight_sensitivity", False),
                len(settings.E2_CLASSIFIERS) * FOLDS * len(e2_grid())))
    if dataset == "cicids2017":
        out.append(("e3", ROOT / "cicids2017" / "e3_ir_sweep" / "cicids2017_e3_ir_sweep.ipynb",
                    results_path("cicids2017", "e3_ir_sweep", False),
                    len(settings.E3_CLASSIFIERS) * FOLDS * len(settings.E3_LEVELS) * len(configs.CONFIGS)))
    return out


def rows(csv):
    return len(pd.read_csv(csv)) if csv.exists() else 0


def log(msg):
    print(f"{datetime.now():%m-%d %H:%M}  {msg}", flush=True)


def execute(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": str(nb_path.parent)}})
    try:
        client.execute()
        ok = True
    except Exception as e:  # keep the outputs written so far for diagnosis
        log(f"   error: {type(e).__name__}: {str(e).splitlines()[-1][:200] if str(e) else ''}")
        ok = False
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbformat.write(nb, nb_path)
    return ok


def status(datasets):
    for ds in datasets:
        print(f"\n{ds}")
        for key, nb, csv, n in jobs(ds):
            have = rows(csv)
            mark = "done" if have >= n else ("missing" if have == 0 else "partial")
            print(f"  {key:<28} {have:>5}/{n:<5} {mark}{'' if nb.exists() else '   (notebook not generated)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--only", nargs="*", help="subset of keys, e.g. c0_none e2 e3")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--compile", action="store_true", help="rebuild analysis/RESULTS.md at the end")
    a = ap.parse_args()

    datasets = a.dataset or [d for d in DATASETS if d != "cicids2017_e3"]
    if a.status:
        status(datasets)
        return
    os.environ.pop("SMOKE", None)
    failed = []
    for ds in a.dataset:
        for key, nb, csv, n in jobs(ds):
            if a.only and key not in a.only:
                continue
            if not nb.exists():
                log(f"{ds}/{key}: notebook missing - run scripts/make_notebooks.py --dataset {ds}")
                failed.append(f"{ds}/{key}")
                continue
            for attempt in range(1, a.attempts + 1):
                have = rows(csv)
                if have >= n:
                    log(f"{ds}/{key}: complete ({have}/{n})")
                    break
                log(f"{ds}/{key}: running, {have}/{n} rows so far (attempt {attempt})")
                t = time.time()
                execute(nb)
                log(f"{ds}/{key}: {rows(csv)}/{n} rows after {(time.time() - t) / 60:.1f} min")
            if rows(csv) < n:
                failed.append(f"{ds}/{key}")
    if a.compile:
        from analysis.compile import compile_results
        compile_results()
    log("finished" + (f"; INCOMPLETE: {', '.join(failed)}" if failed else "; everything complete"))


if __name__ == "__main__":
    main()
