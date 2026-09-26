# Team guide: keeping both halves of the study identical

The paper compares 9 configurations across 5 datasets. That comparison is only valid if
every dataset goes through exactly the same protocol. Member 1 runs Oil Spill, SMS Spam
and CIC-IDS2017; member 2 runs APS Failure (Scania
trucks) and Credit Card Fraud. Follow these rules.

## 1. Never change the protocol on your own
- `common/`, `requirements.txt` and `scripts/make_notebooks.py` define the protocol.
  Do not edit them on your branch. If something must change, open a separate pull request
  that the other person reviews, and then **both** re-run whatever it affects.
- Frozen values (in `common/settings.py`): seed 2026, outer 5x2 repeated stratified CV,
  inner 3-fold, 20 random-search candidates, the 7 classifiers, the E2 grid.
- Every result row stores the git commit and package versions. A commit ending in
  `+dirty` means `common/` was edited locally: those rows are not valid.

## 2. Adding a dataset
Member 2's datasets, APS Failure (`aps`) and Credit Card Fraud (`creditfraud`), are
already added this way; their preprocessing notebooks are the worked examples.
1. Put the raw file(s) in `data/raw/` (never commit them; `data/raw/` is gitignored).
   The expected file names and sources are in `data/README.md`.
2. Register the dataset in `common/datasets.py` (title, domain, owner, modality) and add
   its preprocessing cells to `PREPROCESS` in `scripts/make_notebooks.py`.
3. Generate the notebooks:
   `python scripts/make_notebooks.py --dataset <name>`
   This creates the same folders as for the other datasets. **Never hand-edit a generated
   experiment notebook.**
4. Run `<name>/00_preprocessing/<name>_preprocessing.ipynb`. Rules:
   - `y` = 1 for the minority class, 0 for the majority.
   - Stateless steps only: map labels, drop identifier columns, turn invalid values into
     NaN, remove duplicate rows. **No** imputation, scaling or outlier removal, because
     those are fitted inside each training fold by the pipeline.
   - Large data: take a **stratified sample** with `random_state=settings.MASTER_SEED`
     (CIC-IDS2017 50,000 rows, Credit Card Fraud 100,000, APS 20,000).
   - Finish with `data.save_processed(...)` and `data.make_outer_splits(...)`, then
     commit `data/processed/<name>.parquet` and `data/splits/<name>_outer.npz`.
5. Run the meta-features notebook (1 minute) and commit its CSV.

## 3. Running experiments on Colab
- Open a notebook from GitHub (the "Open in Colab" badge), then *Runtime > Run all*.
  Allow Google Drive access. Do not edit any cell except `SMOKE`.
- Results are written to `MyDrive/ML-Research-Dataset-Imbalance/<dataset>/<config>/results/`
  after every classifier/fold. If Colab disconnects, *Run all* again: finished folds
  are skipped.
- Do not `pip install` anything else. The setup cell stops the notebook if the package
  versions differ from `requirements.txt`. If that happens: *Runtime > Restart session*,
  then *Run all*.
- Run order per dataset:
  1. `c0_none`, `c1_A_smotetomek`, `c2_B_weights`, `c4_AB_smotetomek_weights`, `c8_A_Brecalc`
  2. `c3_C_threshold`, `c5_AC_…`, `c6_BC_…`, `c7_ABC_full` (these reuse the tuned
     hyperparameters of their twin from step 1, which makes them much faster)
  3. `e2_weight_sensitivity` (reuses c0 and c1)
- You can run several notebooks at the same time in separate Colab tabs.
- Setting `SMOKE = True` gives a quick test run (a few minutes; up to about 15 for E2).
  Its files end in `__SMOKE.csv`: never commit them or use them in the paper.

## 4. Handing results back
1. Copy the finished CSVs from Drive into the same paths in the repository (not needed
   when running locally with `scripts/run_experiments.py`).
2. Run `analysis/01_merge_results.ipynb`. Every experiment must show `complete = True`,
   `metric_nans = 0`, and one git commit that is not `+dirty`.
3. Run `python -m pytest tests`.
4. Commit on your own branch (e.g. `data/aps-creditfraud`) and open a pull request.
   The other person reviews it. Do not force-push.

## 5. Checklist before each pull request
- [ ] `pytest tests` passes
- [ ] `01_merge_results` shows every experiment complete, no NaN metrics
- [ ] no raw data, `logs/` or `__SMOKE.csv` files committed
- [ ] no commits while a run is in progress (every row records the commit it ran on)
- [ ] `data/splits/` files unchanged after the first commit of a dataset
- [ ] no edits to `common/`, `requirements.txt` or generated notebooks
