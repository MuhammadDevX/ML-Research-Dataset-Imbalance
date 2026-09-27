# Notes for the revised paper (all 5 datasets; APS and Credit Card details)

Facts, numbers and decisions from the member-2 runs that the paper needs. Every number
comes from `analysis/RESULTS.md`, compiled 2026-09-28 from **all 5 datasets** (Oil Spill,
SMS Spam, CIC-IDS2017, APS, Credit Card Fraud) with every experiment complete (main configs,
E2 on all 5, E3 on CIC-IDS2017). Pooled statistics in §5 are final; §5b covers the
cross-dataset findings and §5c the CIC-IDS2017 Linear SVM anomaly.

---

## 1. Datasets (Methods, Table 1)

| | APS Failure at Scania Trucks | Credit Card Fraud (ULB) |
|---|---|---|
| Domain | manufacturing / automotive maintenance | finance |
| Raw size | 76,000 x 170 (official train 60,000 + test 16,000, pooled) | 284,807 x 30 |
| Cleaning | `na` -> missing (8.3% of cells); 0 duplicates; 0 label conflicts | 1,081 duplicate transactions removed; 0 label conflicts |
| Used | **stratified 20,000-row sample** (seed 2026) | **stratified 100,000-row sample** (seed 2026) |
| Minority | 362 APS failures (1.81%), IR 54.3 | 167 frauds (0.167%), IR 597.8 |
| Features | 170 anonymised sensor counters/histograms; 168 after removing constant columns in-fold | V1-V28 (PCA), Time, Amount |
| Per outer fold | 10,000 train / 10,000 test, 181 positives in each test fold | 50,000 train / 50,000 test, 83-84 positives in each test fold |
| Meta-features (5k-row sample) | F1 = 0.304, N1 = 0.020, N3 = 0.012, 1-NN macro-F1 = 0.814 | F1 = 0.159, N1 = 0.0004, N3 = 0.000, 1-NN macro-F1 = 1.000 (**unreliable, see §7**) |

With Oil Spill (4.38%, IR 21.9), SMS Spam (12.6%, IR 6.9) and CIC-IDS2017 (2.0%, IR 49;
attacks down-sampled from the natural 16.9%), the five datasets span minority rates of
0.167%-12.6% (IR 6.9-598).

**Why APS replaced Pima** (say this in the paper; reviewers saw Pima in v1):
APS adds what the set lacked: many features (170), heavy missingness (8.3%), and an IR
between Oil Spill and Credit Card. Pima results stay only in `legacy_v1_pima/`.

**Why samples, not full data:** keeps the full nested factorial plus E2 feasible on one
workstation. On all 76,000 APS rows a single Random Forest fit after SMOTE on a
25,000-row inner fold took 6.5 min single-threaded. The samples keep the natural class
ratio (stratified). CIC-IDS2017 (50k) is subsampled the same way.

**Suggested wording:** "APS Failure at Scania Trucks [cite] and Credit Card Fraud [cite]
were stratified-subsampled to 20,000 and 100,000 rows respectively, preserving their
natural minority rates (1.81% and 0.167%)."

---

## 2. Protocol (Methods; identical for every dataset)

- 2^3 factorial: A = SMOTETomek to 1:1, B = class weights w_pos = IR (original ratio),
  C = decision-threshold tuning (TunedThresholdClassifierCV, macro-F1, 3-fold). Plus
  c8 = A + weights recomputed after resampling (R7). Classifier is a separate factor.
- 7 classifiers: LR, Linear SVM, Gaussian NB, Decision Tree, Random Forest, XGBoost, LightGBM.
- Outer: 5x2 repeated stratified CV (10 test estimates per cell), fixed folds committed in
  `data/splits/`. Inner: 20-candidate random search, 3-fold, macro-F1.
- Everything fitted (imputation, scaling, resampling, weights, threshold) is fitted inside
  training folds only; the leakage tests in `tests/` check this.
- E2 (R3): w_pos = IR^p, p in {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2}, SMOTETomek ratio in
  {none, 1:2, 1:1}, plus recomputed weights, each with and without C; LR, RF, XGB.
- Sample weights are normalised to mean 1.

---

## 3. Compute environment (Reproducibility paragraph)

- Member-2 runs: one laptop, Intel Core i7-13650HX (14 cores / 20 threads), 24 GB RAM,
  Windows 11, Python 3.12.11 (uv environment). GPU (RTX 4060 Laptop) **not used**.
- Pinned: scikit-learn 1.9.1, imbalanced-learn 0.14.2, xgboost 3.4.1, lightgbm 4.7.0,
  numpy 2.5.3, pandas 3.0.6: the same versions as the Colab runs of the other datasets.
- Wall clock: about 14 h for both datasets (26-27 Sep 2026), 4 notebooks in parallel x
  6 search workers each. Summed per-fold time: APS 19.4 h, Credit Card 56.7 h.
- Estimators single-threaded; parallelism across search candidates only, so results do
  not depend on the number of workers.
- Result rows record git commits `cbfb9d6` and `4ca1c69`. They differ only by commit
  `5ff0aa1` (reusing the SMOTETomek output across search candidates), which was verified
  to give identical results: a unit test, plus 66 real folds recomputed with identical
  metrics and tuned hyperparameters.

**Why the GPU was not used (if a reviewer asks):** the bottlenecks, Random Forest and
SMOTETomek's nearest-neighbour search, are CPU-only in scikit-learn / imbalanced-learn;
GPU XGBoost would also give numerically different results from the CPU runs of the other
datasets.

---

## 4. Results on APS and Credit Card (Results section)

Macro-F1, mean over 7 classifiers, mean ± SD over 10 outer folds:

| Config | APS | Credit Card |
|---|---|---|
| c0 none | 0.8155 ± 0.0099 | 0.8482 ± 0.0141 |
| c1 A | 0.7952 ± 0.0090 | 0.7462 ± 0.0135 |
| c2 B | 0.7979 ± 0.0069 | 0.7573 ± 0.0064 |
| **c3 C (best)** | **0.8195 ± 0.0088** | **0.8643 ± 0.0089** |
| **c4 A+B (worst)** | **0.7278 ± 0.0116** | **0.6077 ± 0.0145** |
| c5 A+C | 0.8079 ± 0.0078 | 0.8423 ± 0.0059 |
| c6 B+C | 0.8058 ± 0.0056 | 0.8486 ± 0.0067 |
| c7 A+B+C | 0.7817 ± 0.0122 | 0.7981 ± 0.0146 |
| c8 A+B recomputed | 0.7952 ± 0.0090 | 0.7462 ± 0.0135 |

Factorial effects on macro-F1 (Yates, 95% bootstrap CI; all significant):

| Effect | APS | Credit Card |
|---|---|---|
| A | -0.032 [-0.034, -0.029] | -0.081 [-0.089, -0.073] |
| B | -0.031 [-0.034, -0.029] | -0.072 [-0.078, -0.067] |
| C | +0.020 [0.017, 0.022] | +0.099 [0.094, 0.103] |
| AB | -0.016 [-0.019, -0.011] | -0.019 [-0.023, -0.015] |
| AC | +0.014 [0.013, 0.015] | +0.045 [0.041, 0.048] |
| BC | +0.011 [0.009, 0.013] | +0.042 [0.040, 0.046] |

Observations worth writing up (phrase as observations on these benchmarks, R4):

1. **Over-correction is strongest at extreme imbalance.** Stacking A and B (c4) gives an
   effective loss ratio rho_eff = IR: 54 on APS, 598 on Credit Card. On Credit Card the
   median classifier then predicts 14.7x as many frauds as exist (ppr_ratio), with minority
   precision 0.065 and recall 0.886; macro-F1 falls from 0.896 (c0) to 0.554. APS: ppr 2.91,
   precision 0.305.
2. **Threshold tuning repairs most of it.** c7 (A+B+C) vs c4: Credit Card ppr 14.7 -> 1.22,
   precision 0.065 -> 0.597. Across all 35 dataset x classifier blocks, c7 beats c4 in 32.
3. **Threshold tuning alone is the best single correction on both datasets** (c3), and C
   is the only main effect that is positive.
4. **Classifier matters (keep the classifier axis, R8).** On Credit Card, tree ensembles
   tolerate A or B alone (RF: c0 0.910, c1 0.913, c2 0.916), but linear models collapse
   (LR: c0 0.863 -> c1 0.570, c2 0.588; Linear SVM 0.874 -> 0.563) and recover with
   threshold tuning (LR c5 0.856, c6 0.881). With A+B, even RF (0.554) and XGB (0.605) fail.
5. **c8 = c1 exactly.** SMOTETomek balances to exactly 1:1 (its Tomek step removes pairs
   from both classes), so recomputed weights are 1 and c8 equals resampling alone. Say this
   explicitly: recomputing the weights (R7) removes the over-correction entirely at a 1:1
   ratio; E2's 1:2 setting shows the partial-balance case.
6. **E2:** without threshold tuning, precision falls and ppr rises as rho_eff increases, for
   every classifier on both datasets (all p < 0.001). Threshold tuning flattens this for LR
   and RF (APS RF ppr slope 2.55 -> 0.06; Credit Card RF 34.0 -> 0.45) but **not for XGBoost
   on Credit Card** (27.9 -> 25.1). Report this honestly as a limit of post-hoc
   thresholding. With all 5 datasets this XGBoost limit is **not Credit Card-only**: Oil
   Spill XGB 2.29 -> 2.05 and APS XGB 1.40 -> 1.04 are also barely flattened (§5b).

---

## 5. Pooled results over all 5 datasets (final)

Macro-F1 by config (mean over 7 classifiers, ± SD over 10 outer folds):

| Config | Oil Spill | SMS Spam | CIC-IDS2017 | APS | Credit Card |
|---|---|---|---|---|---|
| c0 none | 0.6417 | 0.9180 | **0.8752** | 0.8155 | 0.8482 |
| c1 A | 0.6514 | 0.8940 | 0.8288 | 0.7952 | 0.7462 |
| c2 B | 0.6378 | 0.9209 | 0.8278 | 0.7979 | 0.7573 |
| c3 C | **0.6975** | 0.9242 | 0.8404 | **0.8195** | **0.8643** |
| c4 A+B | *0.6154* | *0.8653* | *0.7775* | *0.7278* | *0.6077* |
| c5 A+C | 0.6934 | 0.9086 | 0.8332 | 0.8079 | 0.8423 |
| c6 B+C | 0.6962 | **0.9257** | 0.8303 | 0.8058 | 0.8486 |
| c7 A+B+C | 0.6803 | 0.8905 | 0.8183 | 0.7817 | 0.7981 |
| c8 A+B recomputed | 0.6514 | 0.8940 | 0.8288 | 0.7952 | 0.7462 |

Bold = best, italics = worst per dataset. **c4 (A+B) is the worst config on all 5 datasets.**

- Pooled effects (OLS, dataset and classifier fixed effects, SEs clustered by fold; effect =
  2 x coefficient): A -0.037 [-0.044, -0.029], B -0.032 [-0.038, -0.025],
  **C +0.038 [0.028, 0.048]**, AB -0.012 [-0.015, -0.009], AC +0.017, BC +0.016,
  ABC +0.004; all p < 0.001.
- Friedman over 35 blocks (dataset x classifier): p = 4.18e-16; Nemenyi CD = 2.03 ranks.
  Average ranks: c3 3.11, c6 3.31, c5 4.11, c0 4.57, c2 4.63, c1 5.66, c8 5.66, c7 5.74,
  **c4 8.20**.
- Wilcoxon over 35 blocks (Holm):

| Comparison | Wins / losses | Median diff | p_holm |
|---|---|---|---|
| c4 vs c2: resampling added on top of weights | 3 / 32 | -0.031 | < 1e-4 |
| c4 vs c1: original-ratio weights added on top of resampling | 4 / 31 | -0.035 | < 1e-4 |
| c8 vs c4: recomputed vs original weights (R7) | 31 / 4 | +0.035 | < 1e-4 |
| c7 vs c4: does threshold tuning repair stacking? | 32 / 3 | +0.031 | < 1e-4 |
| c3 vs c0: threshold tuning alone | 25 / 10 | +0.002 | 0.012 |
| c2 vs c0: weights alone | 16 / 19 | -0.001 | 0.163 (n.s.) |
| c1 vs c0: resampling alone | 13 / 22 | -0.008 | 0.027 |

## 5b. Cross-dataset findings (what the paper can claim, on these 5 benchmarks)

1. **Stacking resampling and original-ratio weights hurts, everywhere.** c4 is last on every
   dataset, and the AB interaction is negative on 4 of 5 (CIC n.s. in the per-dataset
   Yates effects) and pooled (-0.012).
2. **The mechanism is over-correction.** In E2, without threshold tuning the precision slope
   against log2(rho_eff) is negative and the ppr slope positive in all 15 dataset x
   classifier blocks (p < 0.001). Recall does rise (APS c0 -> c4: 0.611 -> 0.798); it rises
   less than precision falls (0.745 -> 0.305). **v1's "precision drops without any gain in
   recall" is wrong and must be rewritten.**
3. **The harm grows with imbalance (E3, the cleanest trend).** On CIC-IDS2017 BENIGN vs Bot,
   N = 10,000, only the minority share varies:

| Effect | 15% | 10% | 5% | 2% | 1% |
|---|---|---|---|---|---|
| A | -0.002 | -0.005 | -0.009 | -0.027 | -0.061 |
| B | -0.005 | -0.007 | -0.012 | -0.031 | -0.050 |
| C | +0.006 | +0.009 | +0.015 | +0.040 | +0.059 |
| AB | -0.001 | -0.003 | -0.007 | -0.028 | -0.042 |

   Each effect moves monotonically with imbalance. c4 macro-F1 falls 0.960 -> 0.673 while
   c3 falls only 0.972 -> 0.871. Label it exploratory (one dataset pair, R4).
4. **Threshold tuning is the most useful single correction**, not class weights. C is the
   only positive main effect; c3 ranks first overall and is best on Oil Spill, APS and
   Credit Card, second on SMS Spam (c6 best). **v1's headline "class weights alone is best"
   is not supported**: c2 vs c0 is 16/19, n.s.
5. **Threshold tuning repairs stacked corrections** (c7 vs c4, 32/35), but not for XGBoost
   in E2 on Oil Spill, APS or Credit Card (ppr slope barely reduced).
6. **"Recompute weights after resampling" (v1 takeaway 3) holds but is trivial here:** c8
   beats c4 in 31/35 blocks only because c8 = c1 (SMOTETomek reaches exactly 1:1). The honest
   form of the advice is "do not stack original-ratio weights on resampled data".

## 5c. CIC-IDS2017: the Linear SVM anomaly (decide how to report it)

On CIC-IDS2017, c0 (no correction) is the best config and C is not significant
(+0.003, CI [-0.003, 0.012]). This is caused by **one classifier**:

- Linear SVM's tuned decision threshold lands at 15-130 (on the `decision_function`
  scale) in most CIC folds, so it predicts no attacks (macro-F1 0.495). On the other four
  datasets its tuned threshold stays within [-0.7, 0.6]. Folds collapsed to ~0.495:
  c3 8 of 10, c5 8 of 10, c6 9 of 10.
- LinearSVM on CIC: c0 0.857 -> c3 0.568, c5 0.518, c6 0.500. All other classifiers match
  or improve with C.
- Without Linear SVM, CIC follows the other datasets: c3 0.886 (best), c5 0.886, c6 0.885,
  c0 0.878, ..., c4 0.820 (worst); C effect +0.029.
- Likely cause: heavy-tailed flow features (values up to ~1e7) give unbounded margin scores
  whose scale differs between the inner folds, where the threshold is tuned, and the refit
  model. Not a bug in the run: the protocol is frozen and the rows are valid.

Suggested handling: report it as a limitation of threshold tuning on uncalibrated margin
scores, and give the CIC and pooled results without Linear SVM as a sensitivity analysis.
Agree this with member 1 before writing.

---

## 6. Where each reviewer point is answered by these runs

| Reviewer | Evidence from APS / Credit Card |
|---|---|
| R1 more datasets | two new domains (manufacturing, finance), IR 54 and 598 |
| R2 true factorial | all 8 cells + c8 for both; Yates effects with CIs (§4) |
| R3 tuning, weight grid | nested 20-candidate search; E2 weight grid (§4 obs. 6) |
| R4 overclaiming | effects differ in size by dataset and classifier (§4 obs. 4) |
| R6 no-correction baseline | c0 on both |
| R7 recomputed weights | c8 vs c4, 31/35 wins (§5) |
| R8 XGBoost confusion | XGBoost only a classifier; C = threshold tuning |
| R9 single split | 5x2 CV; 181 (APS) / 83-84 (Credit Card) positives per test fold |
| R10 leakage | all fitting in-fold; dedup before splitting; leakage tests |

---

## 7. Limitations to state

- **Subsampling:** APS and Credit Card use stratified samples (20k, 100k); results are for
  those samples.
- **Few minority cases on Credit Card:** 167 frauds in total, about 83 per test fold, so
  minority metrics are noisier there; the fold SDs in §4 show it.
- **Credit Card meta-features are not usable as computed.** The shared meta-feature code
  samples 5,000 rows, which leaves about 8 frauds, giving N3 = 0 and 1-NN macro-F1 = 1.0.
  Recompute them on all minority rows plus a majority sample before Table 1 (open item).
- **Anonymised features** (APS counters, Credit Card PCA): no domain interpretation of
  feature effects.
- **Threshold tuning does not fully correct XGBoost** in E2 (Credit Card, Oil Spill, APS;
  §5b point 5).
- **Threshold tuning fails for Linear SVM on CIC-IDS2017** (tuned margin thresholds do not
  transfer; §5c).
- E3 (the imbalance trend) uses one dataset pair (CIC BENIGN vs Bot) and 3 classifiers:
  exploratory only.
- One run per cell (seeded); variation is across the 10 outer folds only.

---

## 8. References to add

Dataset and method citations. Verify the exact bibliographic details before submission.

- APS: Scania CV AB, "APS Failure at Scania Trucks," UCI Machine Learning Repository,
  2016 (GPL-3.0; released for the IDA 2016 Industrial Challenge).
- Credit Card: A. Dal Pozzolo, O. Caelen, R. A. Johnson, G. Bontempi, "Calibrating
  probability with undersampling for unbalanced classification," IEEE SSCI, 2015
  (the dataset's requested citation; ULB Machine Learning Group, Kaggle).
- SMOTE: Chawla et al., JAIR 2002. Tomek links: Tomek, IEEE Trans. SMC 1976.
  SMOTE + Tomek: Batista, Prati, Monard, SIGKDD Explorations 2004.
- imbalanced-learn: Lemaître, Nogueira, Aridas, JMLR 2017. scikit-learn: Pedregosa et al.,
  JMLR 2011. XGBoost: Chen & Guestrin, KDD 2016. LightGBM: Ke et al., NeurIPS 2017.
- 5x2cv F-test: Alpaydin, Neural Computation 1999. Friedman/Nemenyi and critical
  difference: Demšar, JMLR 2006.
- The code repository itself: `CITATION.cff` (confirm the author names).

---

## 9. Open items before writing

- [x] CIC-IDS2017 results (main configs, E2, E3), recompiled; §5 updated (2026-09-28).
- [ ] Agree with member 1 how to report the CIC Linear SVM anomaly (§5c); if a sensitivity
      table without Linear SVM is wanted, add it to `analysis/compile.py`.
- [ ] Fix the Credit Card meta-features (§7) before building Table 1 (still N3 = 0,
      1-NN macro-F1 = 1.0 in `analysis/RESULTS.md` §8).
- [ ] Optional: the exploratory meta-regression planned in `REVISED_DESIGN.md` §8.7 was
      never built; E3 (§5b point 3) already shows the imbalance trend.
- [ ] Confirm the author names in `CITATION.cff`; choose a licence.
- [ ] Rewrite the v1 claims the data contradicts: "class weights alone is best" (abstract,
      contribution 2, conclusion), "precision drops without any gain in recall", and the
      "rule for moderate imbalance" (R4). Lead with §5b instead.
