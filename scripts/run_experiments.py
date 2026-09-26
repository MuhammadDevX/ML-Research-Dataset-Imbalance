"""Run a dataset's experiment notebooks locally, several at a time, in dependency order.

    python scripts/run_experiments.py --dataset aps --dataset creditfraud
    python scripts/run_experiments.py --dataset oilspill --parallel 2 --workers 8 --no-e2

Each notebook is executed in place with nbconvert, so the executed notebook can be
committed next to its results. A single notebook spends much of its time in
single-threaded refits; running a few notebooks side by side, each with a share
of the cores (N_JOBS), keeps the machine busy. N_JOBS changes only speed, never
results (see common/settings.py).

Order: the C = 0 configs first (heaviest, the resampling ones, first); each C = 1
config starts once its twin is finished, and E2 once c0_none and c1_A_smotetomek
are finished, so they reuse the tuned hyperparameters. Finished rows are skipped
on a re-run, so the script can be stopped and restarted at any time.

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
# Resampling configs are the slowest (Random Forest on the doubled data), so start them first.
PRIORITY = ["c1_A_smotetomek", "c4_AB_smotetomek_weights", "c8_A_Brecalc", "c0_none",
            "c2_B_weights", "e2_weight_sensitivity", "c5_AC_smotetomek_threshold",
            "c7_ABC_full", "c3_C_threshold", "c6_BC_weights_threshold"]


def tasks_for(ds, with_e2):
    tasks = []
    for cid, cfg in configs.CONFIGS.items():
        deps = [(ds, cfg["twin"])] if cfg["twin"] else []
        tasks.append(((ds, cid), deps))
    if with_e2:
        tasks.append(((ds, "e2_weight_sensitivity"),
                      [(ds, "c0_none"), (ds, "c1_A_smotetomek")]))
    return tasks


def notebook(ds, exp):
    return ROOT / ds / exp / f"{ds}_{exp}.ipynb"


def expected(exp):
    key = ("e2", "e2") if exp == "e2_weight_sensitivity" else ("main", exp)
    return expected_rows()[key]


def repair(path):
    """Remove a partial last line left by a hard stop (power loss, killed process)."""
    if not path.exists():
        return
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        path.write_bytes(raw[:raw.rfind(b"\n") + 1])
        print(f"repaired  {path.name} (removed a partial last line)", flush=True)


def is_complete(ds, exp):
    path = results_path(ds, exp, smoke=False)
    repair(path)
    return len(io.done_keys(path)) >= expected(exp)


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
    ap.add_argument("--dataset", action="append", required=True)
    ap.add_argument("--parallel", type=int, default=4, help="notebooks running at once")
    ap.add_argument("--workers", type=int, default=6, help="N_JOBS for each notebook")
    ap.add_argument("--no-e2", action="store_true")
    ap.add_argument("--retries", type=int, default=2, help="re-runs of a failed notebook")
    a = ap.parse_args()
    for ds in a.dataset:
        if ds not in DATASETS:
            raise SystemExit(f"Unknown dataset '{ds}'")

    pending = {}
    for ds in a.dataset:
        for key, deps in tasks_for(ds, not a.no_e2):
            pending[key] = deps
    # Interleave datasets by priority so both make progress on the heavy configs first.
    order = sorted(pending, key=lambda k: (PRIORITY.index(k[1]), a.dataset.index(k[0])))
    running, done, failed, attempts = {}, set(), set(), {}

    def stamp():
        return time.strftime("%H:%M:%S")

    for key in list(order):
        if is_complete(*key):
            order.remove(key)
            done.add(key)
            print(f"{stamp()} complete {key[0]}/{key[1]} (all rows present, not re-run)", flush=True)

    while order or running:
        for key in [k for k in order if any(d in failed for d in pending[k])]:
            order.remove(key)
            failed.add(key)
            print(f"{stamp()} skipped  {key[0]}/{key[1]} (a dependency failed)", flush=True)
        for key in list(order):
            if len(running) >= a.parallel:
                break
            if all(d in done for d in pending[key]):
                order.remove(key)
                repair(results_path(*key, smoke=False))
                attempts[key] = attempts.get(key, 0) + 1
                running[key] = (*start(*key, a.workers), time.time())
                print(f"{stamp()} started  {key[0]}/{key[1]}", flush=True)
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
    print(f"{stamp()} all done: {len(done)} finished, {len(failed)} failed or skipped", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
