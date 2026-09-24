# Data

`processed/` and `splits/` are committed: they are the output of each dataset's
`00_preprocessing` notebook and are all an experiment needs. `raw/` is not committed.
To re-run preprocessing, put the raw files here:

| Dataset | Put in `data/raw/` | Source |
|---|---|---|
| Oil Spill | `oil_spill.csv` | Kubat, Holte & Matwin (1998); CSV with header `class, attr1..attr49` |
| SMS Spam | `sms+spam+collection.zip` | https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
| CIC-IDS2017 | `CIC-IDS2017/*.csv` (the 8 `MachineLearningCVE` files) | https://www.unb.ca/cic/datasets/ids-2017.html (registration form); mirror used: https://huggingface.co/datasets/c01dsnap/CIC-IDS2017 |
| Pima | `diabetes.csv` | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
| Credit Fraud | `creditcard.csv` | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |

## What preprocessing does (stateless only)
| Dataset | Steps | Result |
|---|---|---|
| Oil Spill | label `'1'`->1, `'-1'`->0; drop `attr1` (patch ID) and `attr23` (constant); no duplicates found | 937 rows, 47 features, 41 positive |
| SMS Spam | spam->1; 403 duplicate messages removed | 5,171 messages, 653 spam |
| CIC-IDS2017 | strip column names; drop `Fwd Header Length.1`; 4,376 inf -> NaN; 308,381 duplicate flows and 1,396 label-conflicting flows removed; BENIGN->0, attack->1; stratified 50,000 sample | 50,000 flows, 77 features, 16.87% attacks |
| CIC-IDS2017 E3 pool | all 1,953 Bot flows + 10,000 BENIGN; nested levels of N = 10,000 at 15/10/5/2/1% Bot | `cicids2017_e3.parquet` + one split file per level |

Columns starting with `meta_` (e.g. the attack name) are descriptive and never used as
features.
