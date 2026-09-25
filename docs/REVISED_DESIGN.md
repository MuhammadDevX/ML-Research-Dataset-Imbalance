# Revised experimental design: factorial ablation of imbalance corrections across 5 domains

## Context
The ICoDT2 paper (AblationStudy_V3.docx) was rejected. Reviewers raised ten concerns, R1–R10: one small dataset, no true factorial, fixed hyperparameters, overclaiming, no uncorrected baseline, no recomputed weights, XGBoost used as both a "strategy" and a classifier, a single split, and unclear leakage control. This document is the methodology design that answers each of them. It is laid out so two people can run it in parallel in `MuhammadDevX/ML-Research-Dataset-Imbalance`.

Dataset facts below were checked on the local files where possible:
- `oil_spill.csv`: 937 rows, 49 features, 41 positives (the label column is `class`, with values `'1'` / `'-1'`).
- `SMSSpamCollection`: a TSV inside the zip.
- `CIC-IDS2017/`: 8 MachineLearningCVE CSVs.

## Implementation notes (where the code differs from the plan below)
The code in `common/` is the reference. It differs from the plan in these places:
- **Weight grid (E2):** weights are `IR ^ power`, power ∈ {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2},
  instead of multipliers. Power 0 is "no weight", 0.5 is sqrt-inverse and 1 is inverse
  frequency, so all of them lie on one log scale (which is also the x-axis of the ρ_eff plot).
  Sample weights are normalised to mean 1, so the regularisation strength stays comparable.
  Result column: `weight_power` (not `weight_mult`).
- **Resampling ratio 0.5** (E2): if a training fold is already above that minority:majority
  ratio (e.g. Pima), only Tomek-link cleaning is applied.
- **E3** uses 10 search candidates per fold instead of 20 (5 levels × 9 configs). Every row
  records `n_iter` (0 = hyperparameters reused from the twin).
- **Random Forest search:** `max_features` ∈ {sqrt, log2} instead of {sqrt, 0.3}. The 30% option made each
  fit about 5x slower on the 300-dimensional SMS features, and sqrt is the standard default.
- **Settings** are in `common/settings.py` and `common/configs.py`, not YAML files.
- **Meta-features** are computed by `common/complexity.py` (F1 Fisher ratio, N1, N3, 1-NN
  macro-F1), not with `problexity`.
- **Measured dataset facts:** Oil Spill 937 × 47 after dropping `attr1` (patch ID) and
  `attr23` (constant); SMS 5,171 after removing 403 duplicates (12.6% spam); CIC-IDS2017
  16.87% attacks after removing 308,381 duplicate and 1,396 label-conflicting flows;
  E3 pool has 1,953 Bot flows.

---

## 1. Reviewer-to-design mapping

| # | Concern | Design change | Where in repo | Evidence in revised paper |
|---|---|---|---|---|
| R1 | Single small dataset | 5 datasets from 5 domains, minority rate 0.17%–34.9%, plus a controlled imbalance-ratio (IR) sweep on CIC-IDS2017 (E3) | `pima/`, `oilspill/`, `smsspam/`, `cicids2017/`, `creditfraud/`, `cicids2017/e3_ir_sweep/` | Table 1 (dataset meta-features), Fig. 5 (IR sweep) |
| R2 | Not a true factorial | Full 2³ design over A, B, C (8 cells) crossed with 7 classifiers. Main and interaction effects are estimated by a linear mixed model (LMM) and by classical factorial effect contrasts | `<dataset>/c0…c7/`, `analysis/02_factorial_effects.ipynb` | Table 3 (A, B, C, AB, AC, BC, ABC effects with 95% CI), Fig. 3 (interaction plots) |
| R3 | Fixed hyperparameters, single weight | Nested random search (20 iterations × 3 inner folds) per cell. Weight-sensitivity study E2 with 8 weight powers (including none and sqrt-inverse), recomputed weights, and 2 resampling ratios | `common/search_spaces.py`, `<dataset>/e2_weight_sensitivity/` | Fig. 4 (performance vs. effective correction ratio) |
| R4 | Overgeneralization | Claims limited to "the five benchmarks evaluated". Effects related to IR and class overlap (N3) are reported as exploratory. Specific sentences are rewritten (§12) | `analysis/07_meta_regression.ipynb` | Discussion wording, new Limitations paragraph |
| R5 | Limited novelty | Three contributions: (i) a quantified interaction decomposition; (ii) the effective correction ratio ρ_eff as a mechanism for over-correction, tested formally; (iii) weight recalibration and threshold tuning as fixes, shown empirically | §9 | Contributions list, Fig. 4, Table 4 |
| R6 | No uncorrected baseline | Cell `c0_none` | `<dataset>/c0_none/` | Row c0 in every table |
| R7 | No recomputed weights | Extra arm `c8_A_Brecalc`: weights recomputed on the resampled data. Also included in E2 | `<dataset>/c8_A_Brecalc/` | Table 4: c4 vs. c8 Wilcoxon test |
| R8 | XGBoost as both strategy and classifier | XGBoost is now only a classifier. Factor C becomes decision-threshold tuning, which is orthogonal to A and B | `common/configs.py` | Table 2 (configuration matrix) |
| R9 | Single split, 154-sample test set | Outer 5×2 repeated stratified CV: 10 test estimates per cell, each row tested 5 times. No reliance on one split | `common/cv.py` | Mean ± SD, bootstrap CIs, and 5×2cv F-tests |
| R10 | Possible leakage | Every stateful step (imputer, scaler, TF-IDF/SVD, resampling, weight computation, threshold) is fitted inside `fit()` of one pipeline object on training folds only. Deduplication happens before splitting. An automated leakage test is included | `common/pipeline.py`, `tests/test_no_leakage.py` | Methods subsection "Leakage control" and Fig. 1 (nested-CV diagram) |

---

## 2. Redefined factors (fixes R8)

- **A: data-level resampling.** SMOTETomek with `sampling_strategy=1.0` and `k_neighbors=min(5, n_min−1)`, fitted only on the training part of each fold.
- **B: cost-sensitive weighting.** Weights are passed to every classifier as `sample_weight`. The positive/negative weight ratio is `(n_neg / n_pos) ^ power`, where power = 1 in the main design. In "orig" mode the ratio is computed from the training fold *before* resampling, which reproduces the v1 behaviour.
- **C: decision-threshold tuning.** `TunedThresholdClassifierCV(scoring="f1_macro", cv=3)` is fitted on the outer-training fold only. C = 0 means the default threshold (0.5 for probabilities, 0 for decision functions).
  - *Why C:* it is a third, genuinely different correction mechanism at the post-hoc decision level, and unlike an XGBoost "strategy" it does not overlap with the classifier axis (R8).

The **classifier** is a separate crossed factor with 7 levels (§3).

| ID / folder | A | B | C | Meaning |
|---|---|---|---|---|
| `c0_none` | 0 | 0 | 0 | Uncorrected baseline (R6) |
| `c1_A_smotetomek` | 1 | 0 | 0 | Resampling only |
| `c2_B_weights` | 0 | 1 | 0 | Weights only |
| `c3_C_threshold` | 0 | 0 | 1 | Threshold only |
| `c4_AB_smotetomek_weights` | 1 | 1 | 0 | v1 "Test 1" (weights from the original ratio) |
| `c5_AC_smotetomek_threshold` | 1 | 0 | 1 | |
| `c6_BC_weights_threshold` | 0 | 1 | 1 | |
| `c7_ABC_full` | 1 | 1 | 1 | Full hybrid |
| `c8_A_Brecalc` (extra arm, R7) | 1 | recalc | 0 | Weights recomputed on the resampled y. With a 1:1 resample these weights are ≈ 1, so c8 ≈ c1. That near-equality is the expected result and shows that recalculation removes over-correction |

The C = 1 cells run the *same* seeded hyperparameter search as their C = 0 twin (c3↔c0, c5↔c1, c6↔c2, c7↔c4). As a result, the only difference between twins is the threshold, which isolates C (R2).

---

## 3. Classifier pool (7)

| Classifier | Status | How B is applied (all through `sample_weight`) | Score used for AUC and threshold |
|---|---|---|---|
| Logistic Regression | kept | `fit(sample_weight=)`, equivalent to `class_weight` | `predict_proba` |
| Linear SVM (`LinearSVC`) | replaces RBF-SVC. RBF does not scale to 50–100k rows × 630 outer runs, and a linear model suits TF-IDF | `fit(sample_weight=)` | `decision_function` |
| Gaussian NB | kept, as a weak probabilistic learner | `fit(sample_weight=)`, which reweights the class priors and likelihoods | `predict_proba` |
| Decision Tree | kept | `fit(sample_weight=)` | `predict_proba` |
| Random Forest | **added**, as the bagging representative | `fit(sample_weight=)` | `predict_proba` |
| XGBoost | kept, now only as a classifier | `fit(sample_weight=)`, equivalent to `scale_pos_weight` | `predict_proba` |
| LightGBM | **added**, as a second boosting implementation | `fit(sample_weight=)` | `predict_proba` |

Using one mechanism for all classifiers makes "B" mean exactly the same thing everywhere. The paper states its equivalence to `class_weight` / `scale_pos_weight`.

---

## 4. Dataset protocol

A common preprocessing chain is used for all datasets. Everything except the stateless load-time fixes runs inside the pipeline:
- load-time fixes: `inf → NaN`, deduplicate, fix label
- `SimpleImputer(median)` → `VarianceThreshold(0)` → [text: `TfidfVectorizer` → `TruncatedSVD(300)`] → `StandardScaler` → corrected classifier
- `StandardScaler` replaces v1's Min-Max because the CIC and fraud features are heavy-tailed. SMOTE still gets comparable feature scales.

| Dataset (owner) | Domain | Raw → used | Minority % (IR) | Specific handling |
|---|---|---|---|---|
| **Oil Spill** (you) | Remote sensing | 937 × 49, full | 4.4% (21.9) | Label `'1'`→1, `'-1'`→0. Drop the patch-ID column if present; constant columns are removed by VarianceThreshold. The minority class is tiny (41), which is why the outer CV is 2-fold (~20 positives per test fold) |
| **SMS Spam** (you) | Text / telecom | 5,574 → ~5,169 after dedup | ~12.6% (≈6.9) | Dedup exact messages **before** splitting, because duplicates across folds cause leakage (R10). TF-IDF (1–2-grams, `min_df=2`, `sublinear_tf`) → SVD(300), fitted inside folds. SVD gives a dense space where SMOTE interpolation and GNB are meaningful |
| **CIC-IDS2017** (you) | Network security | 2.83M × 78 → dedup → **stratified 50,000** | Natural attack rate after dedup (≈17–20%, report the exact value) | Strip column names. Drop the duplicate `Fwd Header Length.1`. Convert `inf`→NaN (imputed inside folds). Target: BENIGN = 0, any attack = 1. Save the subsample indices (seed 2026) to `data/splits/`. Optional external check: evaluate final c0/c2/c4/c8 models on a disjoint 200k stratified hold-out |
| **Pima Diabetes** (teammate, recommended) | Medicine | 768 × 8, full | 34.9% (1.87) | Keeps continuity with v1: it shows whether the v1 finding survives the new protocol. Zeros → NaN, imputed inside folds, which fixes v1's global median imputation |
| **Credit Card Fraud (ULB)** (teammate, recommended) | Finance | 284,807 × 30 → dedup → **stratified 100,000** | ~0.17% (≈580) | Covers the extreme-IR end. Scale `Time`/`Amount` inside folds. About 167 frauds after subsampling, so ~80 per test fold |

If the teammate prefers larger or other datasets, possible substitutes are:
- AI4I 2020 Predictive Maintenance (manufacturing, 10k rows, 3.4%)
- Bank Marketing (marketing, 45k rows, 11.7%)

Together the recommended set spans minority rates of 35 / ~18 / 12.6 / 4.4 / 0.17%.

**Meta-features** (`<dataset>/meta_features/`, computed with `problexity` on a stratified subsample of at most 5k rows): N, d, n_min, IR, F1 (maximum Fisher discriminant ratio), N1, N3 (1-NN error rate), and a 1-NN baseline macro-F1. These are descriptive only and are never used in model selection. They feed the exploratory meta-regression (R4).

**E3: controlled IR sweep (CIC-IDS2017, you).** BENIGN vs. Bot, N = 10,000 fixed, minority rate ∈ {15, 10, 5, 2, 1}% (1,500 → 100 Bot rows). All 9 configs are run with LR, RF, and XGBoost. Because only the IR changes, the "effect vs. IR" curves are not confounded by domain. Bot is chosen because DDoS and PortScan are almost perfectly separable, which would cause ceiling effects.

---

## 5. Leakage-free evaluation (R9, R10)

```
for dataset:
  dedup, stateless fixes, subsample (saved indices)
  outer = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=2026)  # 10 outer folds
  for (train, test) in outer:
     pipe = Pipeline([impute, varthresh, (tfidf, svd), scaler,
                      CorrectedClassifier(base_clf, A, weight_mode, mult, seed)])
     search = RandomizedSearchCV(pipe, space, n_iter=20,
                 cv=StratifiedKFold(3, shuffle=True, random_state=fold_seed),
                 scoring="f1_macro", refit=True)
     search.fit(X[train], y[train])
     model = TunedThresholdClassifierCV(search.best_estimator_, cv=3,
                 scoring="f1_macro").fit(X[train], y[train])   if C=1 else search
     score on X[test] only → one results row
```

- `CorrectedClassifier.fit(X, y)` does three things internally: (1) record the pre-resampling ratio; (2) run SMOTETomek if A = 1; (3) compute `sample_weight` ("orig" = from ratio (1), "recalc" = from the resampled y); then call `base.fit`. Because this all happens in `fit`, it is automatically nested in every inner fold, in the refit, and in the threshold CV.
- **5×2 is chosen over 10-fold** because it keeps ≥ 20 minority rows per test fold for Oil Spill, and it enables Alpaydin's 5×2cv F-test for comparisons within a dataset.
- **No separate fixed held-out set.** Every row is tested 5 times, and a small fixed test set was exactly R9's complaint. CIC-IDS2017 gets the optional external 200k hold-out mentioned in §4.
- **Outer fold indices** are saved to `data/splits/<dataset>_outer.npz`, so both team members use identical folds.

---

## 6. Hyperparameters and weight sensitivity (R3)

`RandomizedSearchCV` with `n_iter=20`, inner 3-fold, `scoring="f1_macro"`, and a fixed seed. Each outer fold runs 61 fits (20 candidates × 3 inner folds, plus the refit), and 3 more for threshold tuning when C = 1.

| Classifier | Search space |
|---|---|
| LR | `C` loguniform[1e-3, 1e2]; l2; `max_iter=3000` |
| LinearSVM | `C` loguniform[1e-3, 1e2]; `max_iter=5000` |
| GNB | `var_smoothing` loguniform[1e-11, 1e-5] |
| DT | `max_depth` {3, 5, 8, 12, None}; `min_samples_leaf` {1, 2, 5, 10, 20}; `criterion` {gini, entropy} |
| RF | `n_estimators` {200, 400}; `max_depth` {None, 8, 16}; `min_samples_leaf` {1, 2, 5}; `max_features` {sqrt, log2} |
| XGBoost | `n_estimators` {200, 400, 800}; `max_depth` {3, 4, 6, 8}; `learning_rate` loguniform[0.01, 0.3]; `subsample`, `colsample_bytree` U[0.6, 1]; `min_child_weight` {1, 3, 5}; `reg_lambda` loguniform[1e-2, 10]; `tree_method=hist` |
| LightGBM | `n_estimators` {200, 400, 800}; `num_leaves` {15, 31, 63}; `learning_rate` loguniform[0.01, 0.3]; `subsample`, `colsample_bytree` U[0.6, 1]; `min_child_samples` {10, 20, 50} |

SMOTETomek parameters are fixed rather than tuned, so that A remains a defined intervention.

**E2: weight sensitivity** (all 5 datasets; LR, RF, XGBoost; C ∈ {0, 1})
- Resampling: none / SMOTETomek at ratio 1.0 / SMOTETomek at ratio 0.5.
- Weight modes:
  - "orig": weight = IR ^ power, power ∈ {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2} (0 = none, 0.5 = sqrt-inverse, 1 = inverse frequency)
  - "recalc"
- Hyperparameters are reused from the matching c0/c1 fold, which keeps the cost manageable. The paper states this.
- **Reporting:** x-axis = log₂ ρ_eff, where ρ_eff = (w_pos·n_pos′)/(w_neg·n_neg′) is the positive-to-negative loss mass the classifier actually sees:
  - c0: ρ_eff = 1/IR
  - c2, c1, and c8: ρ_eff ≈ 1
  - c4: ρ_eff ≈ IR
- Y-axes: macro-F1, P1, R1, and predicted-positive rate ÷ prevalence. Plot one panel per dataset, show the mean with a 95% CI band, and overlay the C = 0 and C = 1 curves.

---

## 7. Metrics

- **Macro-F1 (primary):** weights both classes equally, so it penalizes majority-favouring and over-corrected models alike.
- **MCC:** uses all four confusion-matrix cells and stays informative at a 0.17% minority rate.
- **PR-AUC (average precision):** a threshold-free metric that focuses on the minority class. ROC-AUC is optimistic at extreme IR.
- **ROC-AUC:** kept for continuity with v1. Both AUCs should be *unchanged* by C, which serves as a built-in sanity check.
- **Balanced accuracy and G-mean** (√(TPR·TNR)): the standard imbalanced-learning summaries reviewers expect.
- **Minority precision and recall (P1, R1):** needed to demonstrate the over-correction mechanism, which shows up as precision loss.
- **Predicted-positive rate / prevalence:** the direct diagnostic for over-correction; values ≫ 1 indicate it.
- **Accuracy:** reported for completeness only.

---

## 8. Statistical analysis

1. **Factorial effects per dataset (R2).** Compute the classical 2³ effect contrasts (A, B, C, AB, AC, BC, ABC) on macro-F1 for each dataset × classifier. Give 95% CIs by bootstrapping over the 10 outer folds (10,000 resamples).
2. **Pooled model.** `macro_f1 ~ A*B*C + C(classifier)` with a random intercept for `dataset` and a variance component for `dataset:repeat:fold` (statsmodels `MixedLM`). Report coefficients with CIs. The AB coefficient is the formal over-correction interaction, and ABC tests whether threshold tuning absorbs it.
3. **Across datasets.** Friedman test over the 9 configs, with blocks = dataset × classifier (35 blocks, using fold-mean scores), then Nemenyi post-hoc tests and a critical-difference diagram (`autorank`).
4. **Key pairwise claims** (Wilcoxon signed-rank over the 35 blocks, Holm-corrected, with rank-biserial r and the bootstrap CI of the median Δ):
   - c2 vs. c4 (weights alone vs. stacked)
   - c4 vs. c8 (original vs. recomputed weights, R7)
   - c0 vs. each single factor
   - c4 vs. c7 (does C rescue over-correction?)
5. **Within one dataset:** Alpaydin's combined 5×2cv F-test for the same pairs. This is correct for dependent CV folds; a naive t-test is not.
6. **Mechanism test (R5).** For each block, regress P1 and PPR/prevalence on log ρ_eff over the E2 points with ρ_eff ≥ 1. Test whether the slopes are < 0 for P1 and > 0 for PPR/prevalence (one-sided Wilcoxon over blocks). Fit a quadratic of macro-F1 on log ρ_eff and report where the peak falls relative to ρ_eff = 1. The prediction is that with C = 1 the curve flattens, i.e. the slope magnitude is reduced.
7. **Exploratory meta-regression (R4).** Model the per-block effect Δ_A, Δ_B, Δ_AB against log IR and N3, using the 5 datasets plus the 5 E3 levels. Label it exploratory and do not claim a general rule.

---

## 9. Contributions the design can honestly support (R5)

1. A full 2³ factorial decomposition of resampling × cost weighting × threshold tuning across 5 domains and 7 classifiers. Main and interaction effects are quantified with confidence intervals under leakage-free nested CV.
2. The **effective correction ratio ρ_eff** as an explanatory variable. Stacking SMOTETomek with prior-based weights raises ρ_eff to ≈ IR. This inflates the predicted-positive rate and lowers precision, and the effect is measured and tested rather than asserted.
3. Two simple fixes, evaluated head to head:
   - recomputing weights after resampling (c8)
   - threshold tuning (C)

   The design also includes a controlled IR sweep showing how each factor's benefit changes with imbalance on the benchmarks studied.

---

## 10. Repository design

```
ML-Research-Dataset-Imbalance/
├── README.md                      # protocol summary + reproducibility section
├── requirements.txt               # pinned via pip freeze after pilot (py3.11, sklearn>=1.5, imbalanced-learn, xgboost 2.x, lightgbm 4.x, statsmodels, autorank, problexity)
├── configs/
│   ├── experiment.yaml            # MASTER_SEED=2026, outer 5x2, inner 3, n_iter=20, classifier list
│   └── datasets.yaml              # paths, target col, subsample size, owner, domain
├── common/
│   ├── data.py                    # load_pima(), load_oilspill(), load_sms(), load_cicids(), load_creditfraud() -> X, y, meta
│   ├── configs.py                 # CONFIGS = {"c0_none": dict(A=0,B=0,C=0,weight_mode="none"), ...}
│   ├── corrected.py               # CorrectedClassifier (resample + weights inside fit)
│   ├── pipeline.py                # build_pipeline(dataset, clf, config, mult=1.0, ratio=1.0)
│   ├── search_spaces.py
│   ├── cv.py                      # run_config(dataset, config_id, classifiers) -> DataFrame
│   ├── metrics.py                 # all §7 metrics + ppr_ratio, rho_eff
│   ├── complexity.py              # meta-features
│   └── io.py                      # save_results() validates schema, adds git hash
├── data/                          # raw data gitignored; README with download links
│   └── splits/                    # committed: subsample idx + outer fold idx per dataset
├── legacy_v1_pima/                # current 5 folders moved here unchanged
├── pima/
│   ├── c0_none/pima_c0_none.ipynb
│   ├── c1_A_smotetomek/pima_c1_A_smotetomek.ipynb
│   ├── … c2 … c7 …
│   ├── c8_A_Brecalc/pima_c8_A_Brecalc.ipynb
│   ├── e2_weight_sensitivity/pima_e2.ipynb
│   └── meta_features/pima_meta.ipynb
├── oilspill/     (same sub-folders)
├── smsspam/      (same)
├── cicids2017/   (same + e3_ir_sweep/cicids2017_e3.ipynb)
├── creditfraud/  (same)
├── analysis/
│   ├── 01_merge_results.ipynb   02_factorial_effects.ipynb   03_cd_diagrams.ipynb
│   ├── 04_pairwise_tests.ipynb  05_weight_sensitivity.ipynb  06_ir_sweep.ipynb
│   ├── 07_meta_regression.ipynb
│   └── figures/  tables/
└── tests/  test_no_leakage.py  test_schema.py  test_configs.py
```

**Conventions**
- Each config folder name is the config ID from §2.
- Notebooks are named `<dataset>_<config_id>.ipynb`. Each one is a thin wrapper, e.g. `run_config("oilspill", "c4_AB_smotetomek_weights")`, with **no local protocol code**.
- Results are written to `<dataset>/<config_id>/results/<dataset>__<config_id>__<YYYYMMDD>.csv`.

**Results schema** (one row per dataset × config × classifier × outer fold):
- Identifiers: `dataset, domain, owner, config_id, experiment (main|e2|e3), A, B, C, weight_mode, weight_mult, sampling_ratio, ir_level, classifier, repeat, fold, seed`
- Fold and model info: `n_train, n_test, n_test_pos, train_pos_ratio_after, rho_eff, threshold, best_params_json`
- Metrics: `accuracy, balanced_acc, precision_1, recall_1, f1_1, precision_0, recall_0, f1_0, macro_f1, mcc, roc_auc, pr_auc, gmean, ppr_ratio`
- Bookkeeping: `fit_time_s, git_commit, timestamp`

`io.save_results` rejects any file that has missing or extra columns, so both members' CSVs can be concatenated directly with `pd.concat`.

**Reproducibility**
- One master seed feeds every other seed: outer CV, inner CV, sampler, and estimator `random_state`.
- Fold indices are committed.
- `requirements.txt` is pinned.
- The README states "run notebooks top-to-bottom from a clean kernel" and gives the dataset download steps.

**Leakage test** (`tests/test_no_leakage.py`)
- (a) A spy estimator records the hashes of rows passed to `fit`, and the test asserts no test-fold row ever appears.
- (b) With labels permuted, ROC-AUC must be ≈ 0.5 within ±0.05.
- (c) Every synthetic SMOTE point must be interpolated only from training-fold rows.

---

## 11. Work split and run plan

**You:** Oil Spill, SMS Spam, CIC-IDS2017 (+ E3).
**Teammate:** Pima, Credit Fraud, and owner of `analysis/`. This balances the load, since CIC-IDS2017 is the heaviest dataset.

**Cost:** each dataset's main factorial is 9 configs × 7 classifiers × 10 outer folds = 630 runs × ~61 fits ≈ 38k fits.
- Pima, Oil Spill, SMS: under 2 h each on 8 cores.
- CIC-IDS2017 (50k) and Credit Fraud (100k): tree ensembles plus Tomek neighbour search take roughly 10–30 CPU-hours each, about 2–4 h of wall time on 8 cores with `n_jobs=-1`.
- E2 and E3 add roughly 30% more.
- Time one config on each large dataset before launching everything, and reduce `n_iter` to 15 for the large datasets only if needed. If you do, document it.

**Order**
1. Both together: build `common/`, the tests, and the Pima pilot (c0–c8). Check that the pilot reproduces v1's direction. Tag `common-v1.0`. After this, `common/` changes only through a reviewed PR.
2. In parallel, the small datasets: Oil Spill and SMS (you); Pima complete (teammate).
3. The large datasets: CIC-IDS2017 (you); Credit Fraud (teammate).
4. E2 on each owner's datasets, then E3 (you).
5. Analysis notebooks, figures, and tables (teammate), and paper revision (both).

**Checklist before merging (per PR)**
- [ ] `pytest tests/` passes
- [ ] Row count = configs × classifiers × 10 for each dataset
- [ ] No NaN in the metric columns
- [ ] `git_commit` in the results matches the `common-v1.0` tag or later
- [ ] Notebook runs top-to-bottom from a clean kernel
- [ ] No raw data committed
- [ ] Split files unchanged
- [ ] The other member has reviewed the PR

---

## 12. Paper changes

**Tables**
- T1: dataset meta-features
- T2: configuration matrix
- T3: factorial effects (per dataset + pooled LMM) with 95% CI
- T4: key pairwise tests (Δ, CI, p_Holm, r)
- T5: macro-F1 mean ± SD, config × dataset, averaged over classifiers. Per-classifier tables go on GitHub.

**Figures**
- F1: pipeline with nested-CV and leakage-control diagram
- F2: critical-difference diagram
- F3: A×B interaction plots, C = 0 vs. C = 1
- F4: macro-F1, P1, and PPR/prevalence vs. log ρ_eff (the mechanism)
- F5: E3 effect of each factor vs. IR
- Drop the Pima feature-distribution and correlation figures (old Figs. 2–3) to save space.

**Sentences to rewrite or remove (R4 and consistency)**
- Abstract: "Adding resampling on top of this makes things worse, not better." Scope it to "on the evaluated benchmarks" and report it with test statistics.
- Introduction: "the findings are meant to speak to the broader imbalanced classification problem". Remove.
- Contribution 2: "class weighting without resampling outperforms all combined configurations". Replace with the measured AB interaction and how consistent it was across datasets.
- §V-C: "Test 3 shows a measurable improvement [over [8]], so Tomek Links is contributing". Remove. This compares against a different paper's protocol.
- §V-D: "Tests 1 and 4 give identical numbers… a structural result". Remove, since the design flaw is fixed.
- §V-E: "confirming that the over-correction effect works through precision, not through a shift in the decision threshold". Rewrite. Over-weighting *is* an effective threshold shift, and C now tests this explicitly.
- §VI-C: "gives a direct rule for moderate binary imbalance on tabular data" and "This mismatch is not specific to this dataset". Replace with "on the five benchmarks studied, we observed…", plus a Limitations paragraph: binary tasks only; SMOTETomek as the only resampler; subsampled large datasets; 5 datasets are too few for general rules.
- §III-B / §III-D: global median imputation and Min-Max scaling before the split. Restate both as fitted inside the folds.
- §IV-D: "ten-fold stratified cross-validation is run on the training partition". Replace with the nested protocol from §5.

---

## Verification (after approval)
- Check that every R1–R10 row in §1 maps to a folder, notebook, or test that exists.
- Run the Pima pilot. c4 vs. c2 should show v1's direction, and c8 ≈ c1 should hold.
- `tests/test_no_leakage.py` must pass before any large run starts.
