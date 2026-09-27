"""Run a dataset's experiment notebooks locally, several at a time, in dependency order,
resuming where the last run stopped. Works on Windows, Linux and macOS.

    python scripts/run_experiments.py --status                          # what is done / missing
    python scripts/run_experiments.py --dataset aps --dataset creditfraud
    python scripts/run_experiments.py --dataset cicids2017 --only e2 e3 --compile
    python scripts/run_experiments.py --dataset oilspill --parallel 2 --workers 8 --no-e2

Each notebook is executed in place with nbconvert, so the executed notebook can be
committed next to its results. A single notebook spends much of its time in
single-threaded refits; running a few notebooks side by side, each with a share
of the cores (N_JOBS), keeps the machine busy. N_JOBS changes only speed, never
results (see common/settings.py). Memory is the usual limit: SMOTETomek configs
on CIC-IDS2017 need about 1 GB per worker.

Order: the C = 0 configs first (heaviest, the resampling ones, first); each config
that reuses tuned hyperparameters (the C = 1 twins, and c8 via params_from) starts
once its source is finished, and E2 once c0_none and c1_A_smotetomek are finished.
E3 (CIC-IDS2017 only) is one notebook covering all levels and configs.

Resuming: results are saved after every (classifier, outer fold), so a stop loses at
most the fold in progress. On a restart, notebooks whose results are complete are not
executed again (their saved outputs are kept), a half-written last CSV line from a hard
stop is removed, and a failed notebook is retried (--retries) before its dependents
are given up.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import configs, io  # noqa: E402
from common.datasets import DATASETS  # noqa: E402
from common.paths import results_path  # noqa: E402
from common.report import expected_rows  # noqa: E402

LOG_DIR = ROOT / "logs"
E2, E3 = "e2_weight_sensitivity", "e3_ir_sweep"
SHORT = {E2: "e2", E3: "e3"}   # names accepted by --only
# Resampling configs are the slowest (Random Forest on the doubled data), so start them first.
PRIORITY = ["c1_A_smotetomek", "c4_AB_smotetomek_weights", "c8_A_Brecalc", "c0_none",
            "c2_B_weights", E2, E3, "c5_AC_smotetomek_threshold",
            "c7_ABC_full", "c3_C_threshold", "c6_BC_weights_threshold"]


def tasks_for(ds, with_e2):
    tasks = []
    for cid, cfg in configs.CONFIGS.items():
        source = cfg["twin"] or cfg.get("params_from")
        tasks.append(((ds, cid), [(ds, source)] if source else []))
    if with_e2:
        tasks.append(((ds, E2), [(ds, "c0_none"), (ds, "c1_A_smotetomek")]))
    if ds == "cicids2017":
        tasks.append(((ds, E3), []))
    return tasks


def notebook(ds, exp):
    return ROOT / ds / exp / f"{ds}_{exp}.ipynb"


def expected(exp):
    exp_rows = expected_rows()
    if exp == E2:
        return exp_rows[("e2", "e2")]
    if exp == E3:
        return sum(exp_rows[("e3", c)] for c in configs.CONFIGS)
    return exp_rows[("main", exp)]


def repair(path):
    """Remove a partial last line left by a hard stop (power loss, killed process)."""
    if not path.exists():
        return
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        path.write_bytes(raw[:raw.rfind(b"\n") + 1])
        print(f"repaired  {path.name} (removed a partial last line)", flush=True)


def have_rows(ds, exp):
    path = results_path(ds, exp, smoke=False)
    repair(path)
    return len(io.done_keys(path)) if path.exists() else 0


def is_complete(ds, exp):
    return have_rows(ds, exp) >= expected(exp)


def status(datasets):
    for ds in datasets:
        print(f"\n{ds}")
        for (_, exp), _ in tasks_for(ds, True):
            have, n = have_rows(ds, exp), expected(exp)
            mark = "done" if have >= n else ("missing" if have == 0 else "partial")
            note = "" if notebook(ds, exp).exists() else "   (notebook not generated)"
            print(f"  {exp:<28} {have:>5}/{n:<5} {mark}{note}")


def start(ds, exp, workers):
    LOG_DIR.mkdir(exist_ok=True)
    log = open(LOG_DIR / f"{ds}__{exp}.log", "a", encoding="utf-8")
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start =====\n")
    log.flush()
    # Estimators are single-threaded by protocol. Without these limits every kernel and
    # worker also starts BLAS/OpenMP pools sized to all cores, whose reserved memory
    # exhausted the Windows commit limit with 4 notebooks x 6 workers.
    threads = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    env = dict(os.environ, N_JOBS=str(workers), SMOKE="0", PYTHONUNBUFFERED="1", **threads)
    cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
           "--inplace", "--ExecutePreprocessor.timeout=-1", str(notebook(ds, exp))]
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--only", nargs="*", help="subset of experiments, e.g. c0_none e2 e3")
    ap.add_argument("--parallel", type=int, default=4, help="notebooks running at once")
    ap.add_argument("--workers", type=int, default=6, help="N_JOBS for each notebook")
    ap.add_argument("--no-e2", action="store_true")
    ap.add_argument("--retries", type=int, default=2, help="re-runs of a failed notebook")
    ap.add_argument("--status", action="store_true", help="print what is done / missing and exit")
    ap.add_argument("--compile", action="store_true", help="rebuild analysis/RESULTS.md at the end")
    a = ap.parse_args()
    for ds in a.dataset:
        if ds not in DATASETS:
            raise SystemExit(f"Unknown dataset '{ds}'")
    if a.status:
        status(a.dataset or [d for d in DATASETS if d != "cicids2017_e3"])
        return
    if not a.dataset:
        raise SystemExit("Give at least one --dataset (or use --status)")

    pending = {}
    for ds in a.dataset:
        for key, deps in tasks_for(ds, not a.no_e2):
            if a.only and SHORT.get(key[1], key[1]) not in a.only and key[1] not in a.only:
                continue
            pending[key] = deps
    # Interleave datasets by priority so both make progress on the heavy configs first.
    order = sorted(pending, key=lambda k: (PRIORITY.index(k[1]), a.dataset.index(k[0])))
    running, done, failed, attempts = {}, set(), set(), {}

    def stamp():
        return time.strftime("%H:%M:%S")

    for key in list(order):
        if not notebook(*key).exists():
            order.remove(key)
            failed.add(key)
            print(f"{stamp()} missing  {key[0]}/{key[1]} (notebook not generated: "
                  f"python scripts/make_notebooks.py --dataset {key[0]})", flush=True)
    # Dependencies outside the selected subset (--only) must already be complete.
    for key in order:
        for d in pending[key]:
            if d not in pending and is_complete(*d):
                done.add(d)
    for key in list(order):
        if is_complete(*key):
            order.remove(key)
            done.add(key)
            print(f"{stamp()} complete {key[0]}/{key[1]} (all rows present, not re-run)", flush=True)

    while order or running:
        blocked = [k for k in order if any(d in failed or (d not in pending and d not in done)
                                           for d in pending[k])]
        for key in blocked:
            order.remove(key)
            failed.add(key)
            print(f"{stamp()} skipped  {key[0]}/{key[1]} (a dependency failed or is incomplete)",
                  flush=True)
        for key in list(order):
            if len(running) >= a.parallel:
                break
            if all(d in done for d in pending[key]):
                order.remove(key)
                repair(results_path(*key, smoke=False))
                attempts[key] = attempts.get(key, 0) + 1
                running[key] = (*start(*key, a.workers), time.time())
                print(f"{stamp()} started  {key[0]}/{key[1]} "
                      f"({have_rows(*key)}/{expected(key[1])} rows so far)", flush=True)
        time.sleep(5)
        for key, (proc, log, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[key]
            mins = (time.time() - t0) / 60
            if proc.returncode == 0 and is_complete(*key):
                done.add(key)
                print(f"{stamp()} finished {key[0]}/{key[1]} in {mins:.0f} min", flush=True)
            elif attempts[key] <= a.retries:
                order.insert(0, key)   # resumes from the last saved fold
                print(f"{stamp()} RETRY    {key[0]}/{key[1]} after {mins:.0f} min "
                      f"(attempt {attempts[key]} failed; see logs/{key[0]}__{key[1]}.log)",
                      flush=True)
            else:
                failed.add(key)
                print(f"{stamp()} FAILED   {key[0]}/{key[1]} after {mins:.0f} min "
                      f"(see logs/{key[0]}__{key[1]}.log)", flush=True)
    if a.compile:
        from analysis.compile import compile_results
        compile_results()
    print(f"{stamp()} all done: {len(done & set(pending))} finished, "
          f"{len(failed)} failed or skipped", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
