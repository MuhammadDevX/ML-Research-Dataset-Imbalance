# Factorial ablation of imbalance corrections across five domains

Revised experiments for *An Ablation Study of Hybrid Strategies for Minority-Class
Misclassification*. The full design and how it answers each reviewer comment is in
[docs/REVISED_DESIGN.md](docs/REVISED_DESIGN.md). Team rules are in
[TEAM_GUIDE.md](TEAM_GUIDE.md). **Taking over the remaining runs? Start with
[HANDOFF.md](HANDOFF.md).** Compiled results: [analysis/RESULTS.md](analysis/RESULTS.md).

## What is compared
Three corrections, switched on and off in a full 2x2x2 factorial, plus one extra arm:

| Config | A: SMOTETomek | B: class weights | C: threshold tuning |
|---|---|---|---|
| `c0_none` | - | - | - |
| `c1_A_smotetomek` | yes | - | - |
| `c2_B_weights` | - | yes | - |
| `c3_C_threshold` | - | - | yes |
| `c4_AB_smotetomek_weights` | yes | yes (original ratio) | - |
| `c5_AC_smotetomek_threshold` | yes | - | yes |
| `c6_BC_weights_threshold` | - | yes | yes |
| `c7_ABC_full` | yes | yes (original ratio) | yes |
| `c8_A_Brecalc` | yes | yes (recomputed after resampling) | - |

Each config is run with 7 classifiers (LR, Linear SVM, Gaussian NB, Decision Tree,
Random Forest, XGBoost, LightGBM) under nested, leakage-free 5x2 cross-validation.
Two extra experiments: **E2** varies the class weight systematically, and **E3** varies
only the imbalance ratio (CIC-IDS2017, BENIGN vs Bot).

## Datasets
| Folder | Dataset | Domain | Rows used | Minority | Owner |
|---|---|---|---|---|---|
| `oilspill/` | Oil Spill | remote sensing | 937 | 4.4% | member 1 |
| `smsspam/` | SMS Spam Collection | text | 5,171 | 12.6% | member 1 |
| `cicids2017/` | CIC-IDS2017 (50k sample, attacks down-sampled to 2%) | network security | 50,000 | 2.0% | member 1 |
| `pima/` | Pima Indians Diabetes | medicine | 768 | 34.9% | member 2 |
| `creditfraud/` | Credit Card Fraud (100k stratified sample) | finance | 100,000 | ~0.17% | member 2 |

## Layout
```
common/                 shared protocol code (pipeline, CV, metrics, results schema)
scripts/make_notebooks.py   generates every notebook from one template
data/processed/, data/splits/   preprocessed data and fixed outer folds (committed)
data/raw/               raw downloads (not committed, see data/README.md)
<dataset>/
  00_preprocessing/     stateless cleaning -> data/processed + data/splits (run locally)
  meta_features/        size, imbalance ratio, class-overlap measures
  c0_none/ ... c8_A_Brecalc/   one notebook per config; results/ holds its CSV
  e2_weight_sensitivity/
cicids2017/e3_ir_sweep/ controlled imbalance sweep
analysis/               merging results, statistics, figures
tests/                  leakage, schema and config tests
legacy_v1_pima/         the original (rejected) v1 notebooks, unchanged
```

## Running on Google Colab
1. Open any experiment notebook on GitHub and click **Open in Colab** (or go to
   colab.research.google.com > GitHub > this repository).
2. *Runtime > Run all*, and allow Google Drive access. The first cell clones this
   repository, installs the pinned packages and mounts Drive.
3. Results are saved after every classifier/fold to
   `MyDrive/ML-Research-Dataset-Imbalance/<dataset>/<config>/results/`. After a
   disconnect, *Run all* again and finished folds are skipped.
4. When a config is complete, copy its CSV into the same path here and commit it.

Run the configs in the order given in [TEAM_GUIDE.md](TEAM_GUIDE.md): the threshold
configs and E2 reuse hyperparameters from earlier ones and are much faster then.

## Running locally
```
python -m venv .venv
.venv\Scripts\activate            (Windows)   or   source .venv/bin/activate
pip install -r requirements.txt pytest
python -m pytest tests
python scripts/run_experiments.py --status                 # what is done / missing
python scripts/run_experiments.py --dataset <name> --compile   # run what is missing, resumable
```
Notebooks work the same way locally; results go into the repository folders.
`N_JOBS` sets the number of parallel workers (speed only, never results).
Set the environment variable `SMOKE=1` for a quick test run.

## Reproducibility
- One master seed (2026) drives the samples, outer folds, inner folds, resampling and
  every estimator. The outer folds are stored in `data/splits/`, so every run and both
  team members use identical folds.
- `scikit-learn`, `imbalanced-learn`, `xgboost` and `lightgbm` are pinned; each notebook
  refuses to run with other versions.
- Every result row stores the git commit and package versions it was produced with.
- Anything fitted (imputation, scaling, TF-IDF/SVD, resampling, class weights, the
  decision threshold) is fitted only on training data inside each fold;
  `tests/test_no_leakage.py` checks this.
